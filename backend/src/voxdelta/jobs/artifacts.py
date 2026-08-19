"""Atomic, job-scoped persistence for versioned Pydantic artifacts."""

from __future__ import annotations

import errno
import hashlib
import os
import shutil
import stat
import tempfile
from pathlib import Path
from typing import TypeVar

import orjson
from pydantic import BaseModel

from voxdelta.domain.models import StageName
from voxdelta.jobs._ids import validate_job_id

T = TypeVar("T", bound=BaseModel)


_UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS = {
    errno.EBADF,
    errno.EINVAL,
    getattr(errno, "ENOTSUP", errno.EINVAL),
    getattr(errno, "EOPNOTSUPP", errno.EINVAL),
}
_LINKED_JOB_DIRECTORY_ERROR = (
    "job directory beneath jobs root must not be a symlink or reparse point"
)


def _is_link_like(path: Path) -> bool:
    """Detect symlinks and Windows reparse points without following them."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(file_attributes & reparse_attribute)


def _fsync_directory(directory: Path) -> None:
    """Persist a rename on POSIX, falling back when directory fsync is unsupported.

    ``os.replace`` remains atomic on non-POSIX platforms and filesystems that do not expose
    syncable directory descriptors, but those platforms cannot provide this extra durability
    barrier through Python's portable APIs.
    """

    if os.name != "posix":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(directory, flags)
    except OSError as error:
        if error.errno in _UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS:
            return
        raise
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            if error.errno not in _UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS:
                raise
    finally:
        os.close(descriptor)


class PreparedArtifact:
    """A durable temporary artifact that is harmless until its fenced publish."""

    def __init__(
        self,
        store: ArtifactStore,
        job_id: str,
        target: Path,
        temporary: Path,
        content_hash: str,
    ) -> None:
        self._store = store
        self._job_id = job_id
        self.target = target
        self.temporary = temporary
        self.content_hash = content_hash
        self._published = False

    def publish(self) -> None:
        """Atomically replace the target after revalidating its job directory."""

        directory = self._store._job_path(  # noqa: SLF001 - paired capability type
            self._job_id, create=False, require_directory=True
        )
        if directory != self.target.parent or self.temporary.parent != directory:
            raise ValueError("prepared artifact directory changed before publication")
        os.replace(self.temporary, self.target)
        _fsync_directory(directory)
        self._published = True

    def discard(self) -> None:
        """Remove an unpublished temporary artifact."""

        if not self._published:
            self.temporary.unlink(missing_ok=True)


class ArtifactStore:
    """Store one versioned JSON artifact per pipeline stage and job."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _job_path(self, job_id: str, *, create: bool, require_directory: bool) -> Path:
        validate_job_id(job_id)
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
        resolved_root = self.root.resolve()
        unresolved = self.root / job_id
        if _is_link_like(unresolved):
            raise ValueError(_LINKED_JOB_DIRECTORY_ERROR)
        if create:
            try:
                unresolved.mkdir()
            except FileExistsError:
                pass
        if _is_link_like(unresolved):
            raise ValueError(_LINKED_JOB_DIRECTORY_ERROR)
        try:
            resolved = unresolved.resolve()
        except RuntimeError as error:
            raise ValueError("job directory could not be resolved safely") from error
        expected = resolved_root / job_id
        if resolved != expected or resolved.parent != resolved_root:
            raise ValueError("job directory must resolve directly beneath the configured jobs root")
        if require_directory and not resolved.is_dir():
            raise FileNotFoundError(resolved)
        if _is_link_like(unresolved) or unresolved.resolve() != resolved:
            raise ValueError("job directory changed while it was being validated")
        return resolved

    def job_dir(self, job_id: str) -> Path:
        """Return the validated job directory, creating it when necessary."""

        return self._job_path(job_id, create=True, require_directory=True)

    def write_model(self, job_id: str, stage: StageName, value: BaseModel) -> Path:
        """Atomically persist a model without dropping its version fields."""

        prepared = self.prepare_model(job_id, stage, value)
        try:
            prepared.publish()
        finally:
            prepared.discard()
        return prepared.target

    def prepare_model(self, job_id: str, stage: StageName, value: BaseModel) -> PreparedArtifact:
        """Validate and durably stage a model without replacing the fixed artifact."""

        directory = self.job_dir(job_id)
        target = directory / f"{stage.value}.v1.json"
        validated = type(value).model_validate(value.model_dump(mode="python"))
        payload = orjson.dumps(validated.model_dump(mode="json"), option=orjson.OPT_INDENT_2)
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
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return PreparedArtifact(
            self,
            job_id,
            target,
            temporary,
            hashlib.sha256(payload).hexdigest(),
        )

    def artifact_path(self, job_id: str, stage: StageName) -> Path:
        """Return the exact path reserved for one validated stage artifact."""

        if not isinstance(stage, StageName):
            raise KeyError(str(stage))
        directory = self._job_path(job_id, create=False, require_directory=True)
        return directory / f"{stage.value}.v1.json"

    def content_hash(self, job_id: str, stage: StageName) -> str:
        """Hash the exact persisted artifact bytes with SHA-256."""

        digest = hashlib.sha256()
        with self.artifact_path(job_id, stage).open("rb") as artifact:
            while chunk := artifact.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def delete_stage(self, job_id: str, stage: StageName) -> None:
        """Delete only one stage JSON, preserving audio and all other job files."""

        path = self.artifact_path(job_id, stage)
        path.unlink(missing_ok=True)
        _fsync_directory(path.parent)

    def read_model(self, job_id: str, stage: StageName, model_type: type[T]) -> T:
        """Read and validate a versioned stage artifact."""

        directory = self._job_path(job_id, create=False, require_directory=True)
        path = directory / f"{stage.value}.v1.json"
        return model_type.model_validate_json(path.read_bytes())

    def delete_job(self, job_id: str) -> None:
        """Delete exactly one validated job directory beneath the configured root."""

        resolved_target = self._job_path(job_id, create=False, require_directory=False)
        if resolved_target.exists():
            shutil.rmtree(resolved_target)
