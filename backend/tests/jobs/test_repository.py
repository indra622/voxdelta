from __future__ import annotations

import sqlite3
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
    assert "cache_key" in columns


def test_invalidate_stages_resets_exact_rows_and_job_status_atomically(tmp_path: Path) -> None:
    repository = JobRepository(tmp_path / "voxdelta.sqlite3")
    job_id = repository.create_job("sample.wav")
    for stage in StageName:
        repository.set_stage(
            job_id,
            stage,
            StageStatus.COMPLETED,
            f"{stage.value}.v1.json",
            cache_key=stage.value * 8,
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
