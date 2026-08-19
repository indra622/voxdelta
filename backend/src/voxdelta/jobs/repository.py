"""SQLite persistence for jobs and their pipeline stage state."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from voxdelta.domain.models import StageName, StageStatus
from voxdelta.jobs._ids import validate_job_id


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
                  cache_key TEXT,
                  error_json TEXT,
                  PRIMARY KEY (job_id, stage),
                  FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
                );
                """
            )
            columns = {
                str(row[1]) for row in database.execute("PRAGMA table_info(stages)").fetchall()
            }
            if "cache_key" not in columns:
                database.execute("ALTER TABLE stages ADD COLUMN cache_key TEXT")

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
        cache_key: str | None = None,
    ) -> None:
        """Persist a stage transition, rejecting unknown jobs or stage rows."""

        validate_job_id(job_id)
        if not isinstance(stage, StageName):
            raise KeyError(str(stage))
        now = datetime.now(UTC).isoformat()
        with self._connect() as database:
            updated_stage = database.execute(
                """
                UPDATE stages
                SET status = ?, artifact_path = ?, cache_key = ?, error_json = ?
                WHERE job_id = ? AND stage = ?
                """,
                (
                    status.value,
                    artifact_path,
                    cache_key,
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
            if status in {StageStatus.RUNNING, StageStatus.PAUSED, StageStatus.FAILED}:
                job_status = status
            elif stage == StageName.REPORT and status == StageStatus.COMPLETED:
                job_status = StageStatus.COMPLETED
            elif status == StageStatus.PENDING:
                job_status = StageStatus.PENDING
            else:
                job_status = StageStatus.RUNNING
            updated_job = database.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE id = ?",
                (job_status.value, now, job_id),
            )
            if updated_job.rowcount != 1:
                raise KeyError(job_id)

    def invalidate_stages(self, job_id: str, stages: tuple[StageName, ...]) -> None:
        """Reset an exact stage set and the job state in one database transaction."""

        validate_job_id(job_id)
        if not stages or len(set(stages)) != len(stages):
            raise ValueError("stages must be a non-empty unique tuple")
        if any(not isinstance(stage, StageName) for stage in stages):
            raise KeyError("invalid stage")
        now = datetime.now(UTC).isoformat()
        placeholders = ", ".join("?" for _ in stages)
        with self._connect() as database:
            database.execute("BEGIN IMMEDIATE")
            if database.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,)).fetchone() is None:
                raise KeyError(job_id)
            updated = database.execute(
                f"""
                UPDATE stages
                SET status = ?, artifact_path = NULL, cache_key = NULL, error_json = NULL
                WHERE job_id = ? AND stage IN ({placeholders})
                """,  # noqa: S608 - placeholders are generated, never user-controlled
                (StageStatus.PENDING.value, job_id, *(stage.value for stage in stages)),
            )
            if updated.rowcount != len(stages):
                raise KeyError(f"{job_id}:stage rows")
            database.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE id = ?",
                (StageStatus.PENDING.value, now, job_id),
            )

    def try_start_stage(self, job_id: str, stage: StageName) -> bool:
        """Atomically claim one pending stage for a worker."""

        validate_job_id(job_id)
        if not isinstance(stage, StageName):
            raise KeyError(str(stage))
        now = datetime.now(UTC).isoformat()
        with self._connect() as database:
            database.execute("BEGIN IMMEDIATE")
            if database.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,)).fetchone() is None:
                raise KeyError(job_id)
            claimed = database.execute(
                """
                UPDATE stages
                SET status = ?, artifact_path = NULL, cache_key = NULL, error_json = NULL
                WHERE job_id = ? AND stage = ? AND status = ?
                """,
                (
                    StageStatus.RUNNING.value,
                    job_id,
                    stage.value,
                    StageStatus.PENDING.value,
                ),
            )
            if claimed.rowcount == 1:
                database.execute(
                    "UPDATE jobs SET status = ?, updated_at = ? WHERE id = ?",
                    (StageStatus.RUNNING.value, now, job_id),
                )
                return True
            return False

    def get_job(self, job_id: str) -> dict[str, object]:
        """Return one job and its ordered stage mapping."""

        validate_job_id(job_id)
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

        validate_job_id(job_id)
        with self._connect() as database:
            deleted = database.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            if deleted.rowcount != 1:
                raise KeyError(job_id)
