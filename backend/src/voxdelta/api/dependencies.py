"""Construct production dependencies without loading credentials or remote models."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import SecretStr

from voxdelta.audio.service import AudioService
from voxdelta.config import Settings
from voxdelta.jobs.artifacts import ArtifactStore
from voxdelta.jobs.repository import JobRepository
from voxdelta.pipeline.runner import PipelineRunner


@dataclass(frozen=True, slots=True)
class ApiDependencies:
    repository: JobRepository
    artifacts: ArtifactStore
    runner: PipelineRunner
    max_upload_bytes: int
    admission_reconciliation_lease_seconds: int
    max_active_jobs: int
    api_capability_token: SecretStr | None


def build_dependencies(settings: Settings | None = None) -> ApiDependencies:
    selected = settings or Settings()
    jobs_root = selected.data_root / "jobs"
    repository = JobRepository(selected.database_path)
    artifacts = ArtifactStore(jobs_root)
    audio = AudioService(
        jobs_root,
        min_seconds=selected.min_audio_seconds,
        max_seconds=selected.max_audio_seconds,
    )
    return ApiDependencies(
        repository=repository,
        artifacts=artifacts,
        runner=PipelineRunner(repository, artifacts, audio),
        max_upload_bytes=selected.max_upload_bytes,
        admission_reconciliation_lease_seconds=selected.admission_reconciliation_lease_seconds,
        max_active_jobs=selected.max_active_jobs,
        api_capability_token=selected.api_capability_token,
    )


__all__ = ["ApiDependencies", "build_dependencies"]
