"""SQLite persistence for jobs and their pipeline stage state."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath

from voxdelta.domain.models import StageName, StageStatus


def _validate_job_id(job_id: str) -> None:
    if (
        not job_id
        or job_id in {".", ".."}
        or "/" in job_id
        or "\\" in job_id
        or Path(job_id).is_absolute()
        or PureWindowsPath(job_id).is_absolute()
    ):
        raise ValueError("job ID must be a non-empty path component")


class JobRepository:
    """Persist job metadata and resumable stage state in SQLite."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._connect() as database:
            database.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                  id TEXT PRIMARY KEY,
                  source_name TEXT NOT NULL,
                  status TEXT NOT NULL,
                  diagnostic_capture INTEGER NOT NULL DEFAULT 0,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS stages (
                  job_id TEXT NOT NULL,
                  stage TEXT NOT NULL,
                  status TEXT NOT NULL,
                  artifact_path TEXT,
                  error_json TEXT,
                  PRIMARY KEY (job_id, stage),
                  FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        database = sqlite3.connect(self.path)
        database.row_factory = sqlite3.Row
        database.execute("PRAGMA foreign_keys = ON")
        return database

    def create_job(self, source_name: str, diagnostic_capture: bool = False) -> str:
        """Create a pending job and all canonical pending stage rows atomically."""

        job_id = uuid.uuid4().hex
        now = datetime.now(UTC).isoformat()
        with self._connect() as database:
            database.execute(
                """
                INSERT INTO jobs (
                  id, source_name, status, diagnostic_capture, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (job_id, source_name, StageStatus.PENDING.value, int(diagnostic_capture), now, now),
            )
            database.executemany(
                "INSERT INTO stages (job_id, stage, status) VALUES (?, ?, ?)",
                [(job_id, stage.value, StageStatus.PENDING.value) for stage in StageName],
            )
        return job_id

    def set_stage(
        self,
        job_id: str,
        stage: StageName,
        status: StageStatus,
        artifact_path: str | None = None,
        error: dict[str, str] | None = None,
    ) -> None:
        """Persist a stage transition, rejecting unknown jobs or stage rows."""

        if not isinstance(stage, StageName):
            raise KeyError(str(stage))
        now = datetime.now(UTC).isoformat()
        with self._connect() as database:
            updated_stage = database.execute(
                """
                UPDATE stages
                SET status = ?, artifact_path = ?, error_json = ?
                WHERE job_id = ? AND stage = ?
                """,
                (
                    status.value,
                    artifact_path,
                    json.dumps(error) if error is not None else None,
                    job_id,
                    stage.value,
                ),
            )
            if updated_stage.rowcount != 1:
                job_exists = database.execute(
                    "SELECT 1 FROM jobs WHERE id = ?", (job_id,)
                ).fetchone()
                if job_exists is None:
                    raise KeyError(job_id)
                raise KeyError(f"{job_id}:{stage.value}")
            updated_job = database.execute(
                "UPDATE jobs SET updated_at = ? WHERE id = ?", (now, job_id)
            )
            if updated_job.rowcount != 1:
                raise KeyError(job_id)

    def get_job(self, job_id: str) -> dict[str, object]:
        """Return one job and its ordered stage mapping."""

        with self._connect() as database:
            job = database.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if job is None:
                raise KeyError(job_id)
            stages = database.execute(
                "SELECT * FROM stages WHERE job_id = ? ORDER BY rowid", (job_id,)
            ).fetchall()
        return {**dict(job), "stages": {row["stage"]: dict(row) for row in stages}}

    def delete_job(self, job_id: str) -> None:
        """Delete a job and cascade its stages in one transaction."""

        _validate_job_id(job_id)
        with self._connect() as database:
            deleted = database.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            if deleted.rowcount != 1:
                raise KeyError(job_id)
