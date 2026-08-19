from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from voxdelta.domain.models import StageName, StageStatus
from voxdelta.jobs.repository import JobRepository


def test_job_stage_survives_repository_reopen(tmp_path: Path) -> None:
    database = tmp_path / "voxdelta.sqlite3"
    repository = JobRepository(database)
    job_id = repository.create_job("sample.wav", diagnostic_capture=True)
    repository.set_stage(
        job_id,
        StageName.NORMALIZE,
        StageStatus.COMPLETED,
        "normalize.v1.json",
        {"code": "recovered", "message": "public detail"},
    )

    job = JobRepository(database).get_job(job_id)

    assert job["source_name"] == "sample.wav"
    assert job["diagnostic_capture"] == 1
    assert job["stages"]["normalize"]["status"] == "completed"
    assert job["stages"]["normalize"]["artifact_path"] == "normalize.v1.json"
    assert job["stages"]["normalize"]["error_json"] == (
        '{"code": "recovered", "message": "public detail"}'
    )


def test_create_job_initializes_every_stage_as_pending(tmp_path: Path) -> None:
    repository = JobRepository(tmp_path / "voxdelta.sqlite3")

    job = repository.get_job(repository.create_job("sample.wav"))

    assert list(job["stages"]) == [stage.value for stage in StageName]
    assert {stage["status"] for stage in job["stages"].values()} == {"pending"}


def test_set_stage_rejects_an_unknown_job(tmp_path: Path) -> None:
    repository = JobRepository(tmp_path / "voxdelta.sqlite3")

    with pytest.raises(KeyError, match="missing"):
        repository.set_stage("missing", StageName.NORMALIZE, StageStatus.RUNNING)


def test_set_stage_rejects_a_missing_stage_row(tmp_path: Path) -> None:
    database = tmp_path / "voxdelta.sqlite3"
    repository = JobRepository(database)
    job_id = repository.create_job("sample.wav")
    with sqlite3.connect(database) as connection:
        connection.execute(
            "DELETE FROM stages WHERE job_id = ? AND stage = ?",
            (job_id, StageName.NORMALIZE.value),
        )

    with pytest.raises(KeyError, match="normalize"):
        repository.set_stage(job_id, StageName.NORMALIZE, StageStatus.RUNNING)


def test_set_stage_rejects_an_unknown_stage_value(tmp_path: Path) -> None:
    repository = JobRepository(tmp_path / "voxdelta.sqlite3")
    job_id = repository.create_job("sample.wav")

    with pytest.raises(KeyError, match="unknown"):
        repository.set_stage(job_id, cast(StageName, "unknown"), StageStatus.RUNNING)


def test_repository_migrates_cache_key_for_an_existing_database(tmp_path: Path) -> None:
    database = tmp_path / "voxdelta.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE jobs (
              id TEXT PRIMARY KEY, source_name TEXT NOT NULL, status TEXT NOT NULL,
              diagnostic_capture INTEGER NOT NULL DEFAULT 0,
              created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE stages (
              job_id TEXT NOT NULL, stage TEXT NOT NULL, status TEXT NOT NULL,
              artifact_path TEXT, error_json TEXT,
              PRIMARY KEY (job_id, stage)
            );
            """
        )

    JobRepository(database)

    with sqlite3.connect(database) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(stages)")}
    assert {
        "cache_key",
        "generation",
        "claim_token",
        "claimed_at",
        "artifact_hash",
        "role_confirmed",
    } <= columns


def test_concurrent_repository_initialization_serializes_old_schema_migration(
    tmp_path: Path,
) -> None:
    database = tmp_path / "voxdelta.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE jobs (
              id TEXT PRIMARY KEY, source_name TEXT NOT NULL, status TEXT NOT NULL,
              diagnostic_capture INTEGER NOT NULL DEFAULT 0,
              created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE stages (
              job_id TEXT NOT NULL, stage TEXT NOT NULL, status TEXT NOT NULL,
              artifact_path TEXT, error_json TEXT,
              PRIMARY KEY (job_id, stage)
            );
            """
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        repositories = list(executor.map(lambda _: JobRepository(database), range(16)))

    assert len(repositories) == 16
    with sqlite3.connect(database) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(stages)")}
    assert {"generation", "claim_token", "artifact_hash", "role_confirmed"} <= columns


def test_claim_stage_does_not_steal_a_live_claim_and_recovers_an_expired_claim(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 19, tzinfo=UTC)

    def clock() -> datetime:
        return now

    repository = JobRepository(
        tmp_path / "voxdelta.sqlite3",
        clock=clock,
        claim_lease_seconds=30,
    )
    job_id = repository.create_job("sample.wav")

    first = repository.claim_stage(job_id, StageName.NORMALIZE)
    assert first is not None
    assert first.generation == 0
    assert repository.claim_stage(job_id, StageName.NORMALIZE) is None

    now += timedelta(seconds=29)
    assert repository.claim_stage(job_id, StageName.NORMALIZE) is None

    now += timedelta(seconds=2)
    recovered = repository.claim_stage(job_id, StageName.NORMALIZE)
    assert recovered is not None
    assert recovered.generation == first.generation + 1
    assert recovered.token != first.token


def test_claim_renewal_extends_only_the_current_token_generation(tmp_path: Path) -> None:
    now = datetime(2026, 8, 19, tzinfo=UTC)
    repository = JobRepository(
        tmp_path / "voxdelta.sqlite3",
        clock=lambda: now,
        claim_lease_seconds=10,
    )
    job_id = repository.create_job("sample.wav")
    original = repository.claim_stage(job_id, StageName.DIARIZE)
    assert original is not None
    first_timestamp = repository.get_job(job_id)["stages"]["diarize"]["claimed_at"]

    now += timedelta(seconds=6)
    assert repository.renew_claim(original)
    renewed_timestamp = repository.get_job(job_id)["stages"]["diarize"]["claimed_at"]
    assert renewed_timestamp != first_timestamp

    now += timedelta(seconds=11)
    recovered = repository.claim_stage(job_id, StageName.DIARIZE)
    assert recovered is not None
    assert not repository.renew_claim(original)
    assert repository.get_job(job_id)["stages"]["diarize"]["claim_token"] == recovered.token


def test_stale_claim_cannot_publish_or_change_current_stage_state(tmp_path: Path) -> None:
    now = datetime(2026, 8, 19, tzinfo=UTC)
    repository = JobRepository(
        tmp_path / "voxdelta.sqlite3",
        clock=lambda: now,
        claim_lease_seconds=10,
    )
    job_id = repository.create_job("sample.wav")
    stale = repository.claim_stage(job_id, StageName.NORMALIZE)
    assert stale is not None
    now += timedelta(seconds=11)
    current = repository.claim_stage(job_id, StageName.NORMALIZE)
    assert current is not None
    publish_calls: list[str] = []

    published = repository.publish_claimed_stage(
        stale,
        status=StageStatus.COMPLETED,
        artifact_path="normalize.v1.json",
        cache_key="a" * 64,
        artifact_hash="b" * 64,
        role_confirmed=False,
        publish=lambda: publish_calls.append("published"),
    )

    assert published is False
    assert publish_calls == []
    row = repository.get_job(job_id)["stages"]["normalize"]
    assert row["status"] == "running"
    assert row["generation"] == current.generation
    assert row["claim_token"] == current.token


@pytest.mark.parametrize(
    ("stage", "stage_status", "expected_job_status"),
    [
        (StageName.DIARIZE, StageStatus.COMPLETED, StageStatus.RUNNING),
        (StageName.CONFIRM_ROLES, StageStatus.PAUSED, StageStatus.PAUSED),
        (StageName.REPORT, StageStatus.COMPLETED, StageStatus.COMPLETED),
    ],
)
def test_claimed_publication_sets_only_terminal_or_paused_job_status(
    stage: StageName,
    stage_status: StageStatus,
    expected_job_status: StageStatus,
    tmp_path: Path,
) -> None:
    repository = JobRepository(tmp_path / "voxdelta.sqlite3")
    job_id = repository.create_job("sample.wav")
    claim = repository.claim_stage(job_id, stage)
    assert claim is not None

    assert repository.publish_claimed_stage(
        claim,
        status=stage_status,
        artifact_path=f"{stage.value}.v1.json",
        cache_key="a" * 64,
        artifact_hash="b" * 64,
        role_confirmed=False,
        publish=lambda: None,
    )

    assert repository.get_job(job_id)["status"] == expected_job_status.value


def test_claim_failure_sets_failed_job_status(tmp_path: Path) -> None:
    repository = JobRepository(tmp_path / "voxdelta.sqlite3")
    job_id = repository.create_job("sample.wav")
    claim = repository.claim_stage(job_id, StageName.DIARIZE)
    assert claim is not None

    assert repository.fail_claimed_stage(
        claim,
        {"code": "failed", "message": "public"},
    )

    assert repository.get_job(job_id)["status"] == StageStatus.FAILED.value


def test_invalidate_stages_resets_exact_rows_and_job_status_atomically(tmp_path: Path) -> None:
    repository = JobRepository(tmp_path / "voxdelta.sqlite3")
    job_id = repository.create_job("sample.wav")
    for stage in StageName:
        if stage == StageName.CONFIRM_ROLES:
            continue
        repository.set_stage(
            job_id,
            stage,
            StageStatus.COMPLETED,
            f"{stage.value}.v1.json",
            cache_key=stage.value * 8,
        )
    with sqlite3.connect(repository.path) as database:
        database.execute(
            """
            UPDATE stages SET status = 'completed', artifact_path = 'confirm_roles.v1.json',
                cache_key = ?, role_confirmed = 1
            WHERE job_id = ? AND stage = 'confirm_roles'
            """,
            (StageName.CONFIRM_ROLES.value * 8, job_id),
        )

    repository.invalidate_stages(
        job_id,
        (
            StageName.TRANSCRIBE,
            StageName.CONFIRM_ROLES,
            StageName.EMOTION,
            StageName.RESPONSE_STRATEGY,
            StageName.TRANSITIONS,
            StageName.REPORT,
        ),
    )

    job = repository.get_job(job_id)
    assert job["status"] == "pending"
    assert job["stages"]["normalize"]["status"] == "completed"
    assert job["stages"]["diarize"]["status"] == "completed"
    for stage in (
        "transcribe",
        "confirm_roles",
        "emotion",
        "response_strategy",
        "transitions",
        "report",
    ):
        row = job["stages"][stage]
        assert row["status"] == "pending"
        assert row["artifact_path"] is None
        assert row["cache_key"] is None
        assert row["error_json"] is None


def test_invalidate_stages_rejects_unknown_job_without_partial_changes(tmp_path: Path) -> None:
    repository = JobRepository(tmp_path / "voxdelta.sqlite3")

    with pytest.raises(KeyError, match="missing"):
        repository.invalidate_stages("missing", (StageName.REPORT,))


def test_delete_job_cascades_to_stages_in_one_database_transaction(tmp_path: Path) -> None:
    database = tmp_path / "voxdelta.sqlite3"
    repository = JobRepository(database)
    job_id = repository.create_job("sample.wav")

    repository.delete_job(job_id)

    with pytest.raises(KeyError, match=job_id):
        repository.get_job(job_id)
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM stages WHERE job_id = ?", (job_id,)
        ).fetchone() == (0,)


def test_delete_job_rejects_an_unknown_job(tmp_path: Path) -> None:
    repository = JobRepository(tmp_path / "voxdelta.sqlite3")

    with pytest.raises(KeyError, match="missing"):
        repository.delete_job("missing")


def test_create_job_accepts_one_explicit_validated_id_and_rejects_reuse(tmp_path: Path) -> None:
    repository = JobRepository(tmp_path / "voxdelta.sqlite3")
    selected = "a" * 32

    created = repository.create_job("source.wav", job_id=selected)

    assert created == selected
    assert repository.get_job(selected)["source_name"] == "source.wav"
    with pytest.raises(sqlite3.IntegrityError):
        repository.create_job("other.wav", job_id=selected)
    with pytest.raises(ValueError, match="job ID"):
        repository.create_job("escape.wav", job_id="../escape")


@pytest.mark.parametrize(
    "job_id",
    ["j1", "A" * 32, "a" * 31, "a" * 33, "g" * 32, ".deleted", ".locks", ".incoming"],
)
def test_create_job_rejects_noncanonical_explicit_ids(job_id: str, tmp_path: Path) -> None:
    repository = JobRepository(tmp_path / "voxdelta.sqlite3")

    with pytest.raises(ValueError, match="32 lowercase hexadecimal|reserved"):
        repository.create_job("source.wav", job_id=job_id)


def test_resolve_create_after_error_distinguishes_commit_absence_and_mismatch(
    tmp_path: Path,
) -> None:
    repository = JobRepository(tmp_path / "voxdelta.sqlite3")
    committed = "a" * 32
    absent = "b" * 32
    claimed = "c" * 32
    repository.create_job("committed.wav", diagnostic_capture=True, job_id=committed)
    repository.create_job("claimed.wav", job_id=claimed)
    assert repository.claim_stage(claimed, StageName.NORMALIZE) is not None
    discarded: list[str] = []

    assert repository.resolve_create_after_error(
        committed,
        source_name="committed.wav",
        diagnostic_capture=True,
        discard_if_absent=lambda proof: discarded.append("committed"),
    )
    assert not repository.resolve_create_after_error(
        absent,
        source_name="absent.wav",
        diagnostic_capture=False,
        discard_if_absent=lambda proof: discarded.append("absent"),
    )
    with pytest.raises(ValueError, match="ambiguous"):
        repository.resolve_create_after_error(
            claimed,
            source_name="claimed.wav",
            diagnostic_capture=False,
            discard_if_absent=lambda proof: discarded.append("claimed"),
        )

    assert discarded == ["absent"]


@pytest.mark.parametrize(
    "job_id",
    [
        "",
        ".",
        "..",
        "../escape",
        "nested/job",
        r"nested\job",
        "/absolute",
        "C:",
        "C:..",
        "C:foo",
    ],
)
def test_repository_delete_rejects_unsafe_job_ids(job_id: str, tmp_path: Path) -> None:
    repository = JobRepository(tmp_path / "voxdelta.sqlite3")

    with pytest.raises(ValueError, match="job ID"):
        repository.delete_job(job_id)


@pytest.mark.parametrize("job_id", ["C:", "C:..", "C:foo"])
@pytest.mark.parametrize("operation", ["get_job", "set_stage"])
def test_repository_operations_reject_drive_qualified_job_ids(
    job_id: str, operation: str, tmp_path: Path
) -> None:
    repository = JobRepository(tmp_path / "voxdelta.sqlite3")

    with pytest.raises(ValueError, match="job ID"):
        if operation == "get_job":
            repository.get_job(job_id)
        else:
            repository.set_stage(job_id, StageName.NORMALIZE, StageStatus.RUNNING)
