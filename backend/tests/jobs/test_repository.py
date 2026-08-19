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
    "job_id", ["", ".", "..", "../escape", "nested/job", r"nested\job", "/absolute"]
)
def test_repository_delete_rejects_unsafe_job_ids(job_id: str, tmp_path: Path) -> None:
    repository = JobRepository(tmp_path / "voxdelta.sqlite3")

    with pytest.raises(ValueError, match="job ID"):
        repository.delete_job(job_id)
