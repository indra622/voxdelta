"""Atomic, job-scoped persistence for versioned Pydantic artifacts."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path, PureWindowsPath
from typing import TypeVar

import orjson
from pydantic import BaseModel

from voxdelta.domain.models import StageName

T = TypeVar("T", bound=BaseModel)


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


class ArtifactStore:
    """Store one versioned JSON artifact per pipeline stage and job."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _job_path(self, job_id: str) -> Path:
        _validate_job_id(job_id)
        return self.root / job_id

    def job_dir(self, job_id: str) -> Path:
        """Return the validated job directory, creating it when necessary."""

        path = self._job_path(job_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_model(self, job_id: str, stage: StageName, value: BaseModel) -> Path:
        """Atomically persist a model without dropping its version fields."""

        target = self.job_dir(job_id) / f"{stage.value}.v1.json"
        payload = orjson.dumps(value.model_dump(mode="json"), option=orjson.OPT_INDENT_2)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as file:
                file.write(payload)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return target

    def read_model(self, job_id: str, stage: StageName, model_type: type[T]) -> T:
        """Read and validate a versioned stage artifact."""

        path = self._job_path(job_id) / f"{stage.value}.v1.json"
        return model_type.model_validate_json(path.read_bytes())

    def delete_job(self, job_id: str) -> None:
        """Delete exactly one validated job directory beneath the configured root."""

        unresolved_target = self._job_path(job_id)
        resolved_root = self.root.resolve()
        resolved_target = unresolved_target.resolve()
        expected_target = resolved_root / job_id
        if resolved_target != expected_target or resolved_target.parent != resolved_root:
            raise ValueError("job directory must resolve directly beneath the configured jobs root")
        if resolved_target.exists():
            shutil.rmtree(resolved_target)
