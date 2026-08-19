"""SQLite persistence for jobs and generation-fenced pipeline stage state."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from voxdelta.domain.models import StageName, StageStatus
from voxdelta.jobs._ids import validate_job_id

Clock = Callable[[], datetime]
Publisher = Callable[[], None]


@dataclass(frozen=True, slots=True)
class StageClaim:
    """Opaque authority for one stage generation."""

    job_id: str
    stage: StageName
    generation: int
    token: str


class JobRepository:
    """Persist job metadata and resumable, fenced stage state in SQLite."""

    def __init__(
        self,
        path: Path,
        *,
        clock: Clock | None = None,
        claim_lease_seconds: float = 300.0,
    ) -> None:
        if claim_lease_seconds <= 0:
            raise ValueError("claim lease must be positive")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._clock = clock or (lambda: datetime.now(UTC))
        self._claim_lease = timedelta(seconds=claim_lease_seconds)
        self._initialize_schema()

    @property
    def claim_lease_seconds(self) -> float:
        """Expose the configured lease so runners can choose a shorter heartbeat interval."""

        return self._claim_lease.total_seconds()

    def _initialize_schema(self) -> None:
        """Serialize creation and additive migrations across concurrent constructors."""

        with self._connect() as database:
            database.execute("BEGIN IMMEDIATE")
            database.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                  id TEXT PRIMARY KEY,
                  source_name TEXT NOT NULL,
                  status TEXT NOT NULL,
                  diagnostic_capture INTEGER NOT NULL DEFAULT 0,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                )
                """
            )
            database.execute(
                """
                CREATE TABLE IF NOT EXISTS stages (
                  job_id TEXT NOT NULL,
                  stage TEXT NOT NULL,
                  status TEXT NOT NULL,
                  artifact_path TEXT,
                  cache_key TEXT,
                  artifact_hash TEXT,
                  error_json TEXT,
                  generation INTEGER NOT NULL DEFAULT 0,
                  claim_token TEXT,
                  claimed_at TEXT,
                  role_confirmed INTEGER NOT NULL DEFAULT 0,
                  PRIMARY KEY (job_id, stage),
                  FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
                )
                """
            )
            migrations = {
                "cache_key": "TEXT",
                "artifact_hash": "TEXT",
                "generation": "INTEGER NOT NULL DEFAULT 0",
                "claim_token": "TEXT",
                "claimed_at": "TEXT",
                "role_confirmed": "INTEGER NOT NULL DEFAULT 0",
            }
            for name, declaration in migrations.items():
                columns = {
                    str(row[1]) for row in database.execute("PRAGMA table_info(stages)").fetchall()
                }
                if name not in columns:
                    database.execute(f"ALTER TABLE stages ADD COLUMN {name} {declaration}")

    def _connect(self) -> sqlite3.Connection:
        database = sqlite3.connect(self.path)
        database.row_factory = sqlite3.Row
        database.execute("PRAGMA foreign_keys = ON")
        return database

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("repository clock must return a timezone-aware datetime")
        return now

    def create_job(self, source_name: str, diagnostic_capture: bool = False) -> str:
        """Create a pending job and all canonical pending stage rows atomically."""

        job_id = uuid.uuid4().hex
        now = self._now().isoformat()
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

    def update_source_name(self, job_id: str, source_name: str) -> None:
        """Bind a newly streamed job upload before any pipeline work is scheduled."""

        validate_job_id(job_id)
        if not source_name:
            raise ValueError("source name must not be empty")
        now = self._now().isoformat()
        with self._connect() as database:
            updated = database.execute(
                "UPDATE jobs SET source_name = ?, updated_at = ? WHERE id = ?",
                (source_name, now, job_id),
            )
            if updated.rowcount != 1:
                raise KeyError(job_id)

    def set_stage(
        self,
        job_id: str,
        stage: StageName,
        status: StageStatus,
        artifact_path: str | None = None,
        error: dict[str, str] | None = None,
        cache_key: str | None = None,
        artifact_hash: str | None = None,
    ) -> None:
        """Administrative transition used outside claimed runner publication.

        Role pause/completion must use the fenced publication APIs so an arbitrary row
        update cannot bypass explicit confirmation.
        """

        validate_job_id(job_id)
        if not isinstance(stage, StageName):
            raise KeyError(str(stage))
        if stage == StageName.CONFIRM_ROLES and status in {
            StageStatus.RUNNING,
            StageStatus.PAUSED,
            StageStatus.COMPLETED,
        }:
            raise ValueError("role stage transitions require fenced publication")
        now = self._now().isoformat()
        with self._connect() as database:
            database.execute("BEGIN IMMEDIATE")
            job = database.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if job is None:
                raise KeyError(job_id)
            if job["status"] == "deleting":
                raise ValueError("job is being deleted")
            updated_stage = database.execute(
                """
                UPDATE stages
                SET status = ?, artifact_path = ?, cache_key = ?, artifact_hash = ?,
                    error_json = ?, claim_token = NULL, claimed_at = NULL,
                    role_confirmed = 0
                WHERE job_id = ? AND stage = ?
                """,
                (
                    status.value,
                    artifact_path,
                    cache_key,
                    artifact_hash,
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
        """Fence active workers and reset an exact stage set transactionally."""

        validate_job_id(job_id)
        if not stages or len(set(stages)) != len(stages):
            raise ValueError("stages must be a non-empty unique tuple")
        if any(not isinstance(stage, StageName) for stage in stages):
            raise KeyError("invalid stage")
        now = self._now().isoformat()
        placeholders = ", ".join("?" for _ in stages)
        with self._connect() as database:
            database.execute("BEGIN IMMEDIATE")
            job = database.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if job is None:
                raise KeyError(job_id)
            if job["status"] == "deleting":
                raise ValueError("job is being deleted")
            updated = database.execute(
                f"""
                UPDATE stages
                SET status = ?, artifact_path = NULL, cache_key = NULL,
                    artifact_hash = NULL, error_json = NULL,
                    generation = generation + 1, claim_token = NULL, claimed_at = NULL,
                    role_confirmed = 0
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

    def _claim_is_expired(self, claimed_at: object, now: datetime) -> bool:
        if not isinstance(claimed_at, str):
            return True
        try:
            claimed = datetime.fromisoformat(claimed_at)
        except ValueError:
            return True
        if claimed.tzinfo is None or claimed.utcoffset() is None:
            return True
        return now - claimed >= self._claim_lease

    def claim_stage(self, job_id: str, stage: StageName) -> StageClaim | None:
        """Claim pending work or recover only an expired running claim."""

        validate_job_id(job_id)
        if not isinstance(stage, StageName):
            raise KeyError(str(stage))
        now_value = self._now()
        now = now_value.isoformat()
        token = uuid.uuid4().hex
        with self._connect() as database:
            database.execute("BEGIN IMMEDIATE")
            job = database.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if job is None:
                raise KeyError(job_id)
            if job["status"] == "deleting":
                return None
            row = database.execute(
                "SELECT status, generation, claimed_at FROM stages WHERE job_id = ? AND stage = ?",
                (job_id, stage.value),
            ).fetchone()
            if row is None:
                if (
                    database.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,)).fetchone()
                    is None
                ):
                    raise KeyError(job_id)
                raise KeyError(f"{job_id}:{stage.value}")
            generation = int(row["generation"])
            if row["status"] == StageStatus.PENDING.value:
                next_generation = generation
            elif row["status"] == StageStatus.RUNNING.value and self._claim_is_expired(
                row["claimed_at"], now_value
            ):
                next_generation = generation + 1
            else:
                return None
            claimed = database.execute(
                """
                UPDATE stages
                SET status = ?, generation = ?, claim_token = ?, claimed_at = ?,
                    artifact_path = NULL, cache_key = NULL, artifact_hash = NULL,
                    error_json = NULL, role_confirmed = 0
                WHERE job_id = ? AND stage = ? AND generation = ? AND status = ?
                """,
                (
                    StageStatus.RUNNING.value,
                    next_generation,
                    token,
                    now,
                    job_id,
                    stage.value,
                    generation,
                    row["status"],
                ),
            )
            if claimed.rowcount != 1:
                return None
            database.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE id = ?",
                (StageStatus.RUNNING.value, now, job_id),
            )
            return StageClaim(job_id, stage, next_generation, token)

    def renew_claim(self, claim: StageClaim) -> bool:
        """Extend only the still-current running claim's lease."""

        now = self._now().isoformat()
        with self._connect() as database:
            renewed = database.execute(
                """
                UPDATE stages SET claimed_at = ?
                WHERE job_id = ? AND stage = ? AND status = ?
                  AND generation = ? AND claim_token = ?
                """,
                (
                    now,
                    claim.job_id,
                    claim.stage.value,
                    StageStatus.RUNNING.value,
                    claim.generation,
                    claim.token,
                ),
            )
            return renewed.rowcount == 1

    def publish_claimed_stage(
        self,
        claim: StageClaim,
        *,
        status: StageStatus,
        artifact_path: str,
        cache_key: str,
        artifact_hash: str,
        role_confirmed: bool,
        publish: Publisher,
    ) -> bool:
        """Publish and index a prepared artifact under the claim's write fence."""

        if status not in {StageStatus.COMPLETED, StageStatus.PAUSED}:
            raise ValueError("claimed publication must complete or pause a stage")
        if claim.stage != StageName.CONFIRM_ROLES and role_confirmed:
            raise ValueError("only the role stage can be confirmed")
        if claim.stage == StageName.CONFIRM_ROLES and status == StageStatus.COMPLETED:
            raise ValueError("role completion requires confirmation publication")
        now = self._now().isoformat()
        with self._connect() as database:
            database.execute("BEGIN IMMEDIATE")
            row = database.execute(
                """
                SELECT status, generation, claim_token
                FROM stages WHERE job_id = ? AND stage = ?
                """,
                (claim.job_id, claim.stage.value),
            ).fetchone()
            if (
                row is None
                or row["status"] != StageStatus.RUNNING.value
                or int(row["generation"]) != claim.generation
                or row["claim_token"] != claim.token
            ):
                return False
            publish()
            updated = database.execute(
                """
                UPDATE stages
                SET status = ?, artifact_path = ?, cache_key = ?, artifact_hash = ?,
                    error_json = NULL, claim_token = NULL, claimed_at = NULL,
                    role_confirmed = ?
                WHERE job_id = ? AND stage = ? AND status = ?
                  AND generation = ? AND claim_token = ?
                """,
                (
                    status.value,
                    artifact_path,
                    cache_key,
                    artifact_hash,
                    int(role_confirmed),
                    claim.job_id,
                    claim.stage.value,
                    StageStatus.RUNNING.value,
                    claim.generation,
                    claim.token,
                ),
            )
            if updated.rowcount != 1:
                raise RuntimeError("stage claim changed while database write lock was held")
            if claim.stage == StageName.REPORT and status == StageStatus.COMPLETED:
                job_status = StageStatus.COMPLETED
            elif status == StageStatus.PAUSED:
                job_status = StageStatus.PAUSED
            else:
                job_status = StageStatus.RUNNING
            database.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE id = ?",
                (job_status.value, now, claim.job_id),
            )
            return True

    def fail_claimed_stage(self, claim: StageClaim, error: dict[str, str]) -> bool:
        """Mark only the still-current claim failed."""

        now = self._now().isoformat()
        with self._connect() as database:
            database.execute("BEGIN IMMEDIATE")
            updated = database.execute(
                """
                UPDATE stages
                SET status = ?, artifact_path = NULL, cache_key = NULL,
                    artifact_hash = NULL, error_json = ?, claim_token = NULL,
                    claimed_at = NULL, role_confirmed = 0
                WHERE job_id = ? AND stage = ? AND status = ?
                  AND generation = ? AND claim_token = ?
                """,
                (
                    StageStatus.FAILED.value,
                    json.dumps(error),
                    claim.job_id,
                    claim.stage.value,
                    StageStatus.RUNNING.value,
                    claim.generation,
                    claim.token,
                ),
            )
            if updated.rowcount != 1:
                return False
            database.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE id = ?",
                (StageStatus.FAILED.value, now, claim.job_id),
            )
            return True

    def fail_unhandled_job(self, job_id: str) -> bool:
        """Fence active claims and persist one generic failure when no stage did so."""

        validate_job_id(job_id)
        terminal = {
            StageStatus.PAUSED.value,
            StageStatus.COMPLETED.value,
            StageStatus.FAILED.value,
            "deleting",
        }
        now = self._now().isoformat()
        generic_error = json.dumps(
            {"code": "pipeline_failed", "message": "The pipeline stage failed."}
        )
        with self._connect() as database:
            database.execute("BEGIN IMMEDIATE")
            job = database.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if job is None:
                raise KeyError(job_id)
            if job["status"] in terminal:
                return False
            rows = {
                str(row["stage"]): row
                for row in database.execute(
                    "SELECT stage, status FROM stages WHERE job_id = ?", (job_id,)
                ).fetchall()
            }
            active = next(
                (
                    stage
                    for stage in StageName
                    if rows.get(stage.value) is not None
                    and rows[stage.value]["status"] == StageStatus.RUNNING.value
                ),
                None,
            )
            if active is None:
                active = next(
                    (
                        stage
                        for stage in StageName
                        if rows.get(stage.value) is not None
                        and rows[stage.value]["status"] == StageStatus.PENDING.value
                    ),
                    None,
                )
            if active is None:
                return False
            for stage in StageName:
                row = rows.get(stage.value)
                if row is None:
                    raise KeyError(f"{job_id}:{stage.value}")
                previous = str(row["status"])
                status = (
                    StageStatus.PENDING.value if previous == StageStatus.RUNNING.value else previous
                )
                error_json = None
                if stage == active:
                    status = StageStatus.FAILED.value
                    error_json = generic_error
                database.execute(
                    """
                    UPDATE stages
                    SET status = ?, error_json = ?, generation = generation + 1,
                        claim_token = NULL, claimed_at = NULL
                    WHERE job_id = ? AND stage = ?
                    """,
                    (status, error_json, job_id, stage.value),
                )
            database.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE id = ?",
                (StageStatus.FAILED.value, now, job_id),
            )
            return True

    def publish_role_confirmation(
        self,
        job_id: str,
        *,
        expected_generation: int,
        expected_candidate_hash: str,
        artifact_path: str,
        cache_key: str,
        artifact_hash: str,
        publish: Publisher,
    ) -> bool:
        """Replace exactly the current paused candidate with confirmed roles."""

        validate_job_id(job_id)
        now = self._now().isoformat()
        with self._connect() as database:
            database.execute("BEGIN IMMEDIATE")
            row = database.execute(
                """
                SELECT status, generation, artifact_hash, role_confirmed
                FROM stages WHERE job_id = ? AND stage = ?
                """,
                (job_id, StageName.CONFIRM_ROLES.value),
            ).fetchone()
            if row is None:
                if (
                    database.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,)).fetchone()
                    is None
                ):
                    raise KeyError(job_id)
                raise KeyError(f"{job_id}:{StageName.CONFIRM_ROLES.value}")
            if (
                row["status"] != StageStatus.PAUSED.value
                or int(row["generation"]) != expected_generation
                or row["artifact_hash"] != expected_candidate_hash
                or int(row["role_confirmed"]) != 0
            ):
                return False
            publish()
            updated = database.execute(
                """
                UPDATE stages
                SET status = ?, artifact_path = ?, cache_key = ?, artifact_hash = ?,
                    error_json = NULL, claim_token = NULL, claimed_at = NULL,
                    role_confirmed = 1
                WHERE job_id = ? AND stage = ? AND status = ? AND generation = ?
                  AND artifact_hash = ? AND role_confirmed = 0
                """,
                (
                    StageStatus.COMPLETED.value,
                    artifact_path,
                    cache_key,
                    artifact_hash,
                    job_id,
                    StageName.CONFIRM_ROLES.value,
                    StageStatus.PAUSED.value,
                    expected_generation,
                    expected_candidate_hash,
                ),
            )
            if updated.rowcount != 1:
                raise RuntimeError("role candidate changed while database write lock was held")
            database.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE id = ?",
                (StageStatus.RUNNING.value, now, job_id),
            )
            return True

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

    def begin_delete(self, job_id: str) -> None:
        """Fence every stage and retain a retryable deleting row."""

        validate_job_id(job_id)
        now = self._now().isoformat()
        with self._connect() as database:
            database.execute("BEGIN IMMEDIATE")
            job = database.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if job is None:
                raise KeyError(job_id)
            if job["status"] != "deleting":
                updated = database.execute(
                    """
                    UPDATE stages
                    SET status = 'deleting', generation = generation + 1,
                        claim_token = NULL, claimed_at = NULL, error_json = NULL
                    WHERE job_id = ?
                    """,
                    (job_id,),
                )
                if updated.rowcount != len(StageName):
                    raise KeyError(f"{job_id}:stage rows")
                database.execute(
                    "UPDATE jobs SET status = 'deleting', updated_at = ? WHERE id = ?",
                    (now, job_id),
                )

    def finalize_delete(self, job_id: str) -> None:
        """Delete only a previously fenced job after artifact removal succeeds."""

        validate_job_id(job_id)
        with self._connect() as database:
            deleted = database.execute(
                "DELETE FROM jobs WHERE id = ? AND status = 'deleting'", (job_id,)
            )
            if deleted.rowcount != 1:
                if database.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,)).fetchone():
                    raise ValueError("job deletion was not fenced")
                raise KeyError(job_id)

    def discard_unstarted_job(self, job_id: str) -> None:
        """CAS-delete only a pristine pending job that has never entered pipeline work."""

        validate_job_id(job_id)
        with self._connect() as database:
            database.execute("BEGIN IMMEDIATE")
            deleted = database.execute(
                """
                DELETE FROM jobs
                WHERE id = ? AND status = ?
                  AND (SELECT COUNT(*) FROM stages WHERE job_id = ?) = ?
                  AND NOT EXISTS (
                    SELECT 1 FROM stages
                    WHERE job_id = ? AND (
                      status != ? OR artifact_path IS NOT NULL OR cache_key IS NOT NULL
                      OR artifact_hash IS NOT NULL OR error_json IS NOT NULL
                      OR claim_token IS NOT NULL OR claimed_at IS NOT NULL OR role_confirmed != 0
                    )
                  )
                """,
                (
                    job_id,
                    StageStatus.PENDING.value,
                    job_id,
                    len(StageName),
                    job_id,
                    StageStatus.PENDING.value,
                ),
            )
            if deleted.rowcount == 1:
                return
            if database.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,)).fetchone() is None:
                raise KeyError(job_id)
            raise ValueError("job is not an unstarted pending job")

    def delete_job(self, job_id: str) -> None:
        """Delete a job and cascade its stages in one transaction."""

        validate_job_id(job_id)
        with self._connect() as database:
            deleted = database.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            if deleted.rowcount != 1:
                raise KeyError(job_id)


__all__ = ["JobRepository", "StageClaim"]
