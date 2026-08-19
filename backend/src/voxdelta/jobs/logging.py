"""Structured, redacted pipeline event and diagnostic persistence."""

from __future__ import annotations

import math
import os
import re
import tempfile
import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import orjson

from voxdelta.jobs.artifacts import ArtifactStore
from voxdelta.security import is_credential_key, key_tokens

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised only on non-POSIX platforms
    fcntl = None  # type: ignore[assignment]

_REDACTED = "[REDACTED]"
_UNSERIALIZABLE = "[UNSERIALIZABLE]"
_SENSITIVE_CONTENT_TOKENS = frozenset({"transcript", "payload"})
_SAFE_NAME = re.compile(r"[^a-z0-9_-]+")
_PATH_LOCKS: dict[Path, threading.Lock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


def _path_lock(path: Path) -> threading.Lock:
    """Return a process-wide fallback lock; POSIX also takes an OS flock."""

    resolved = path.resolve()
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(resolved, threading.Lock())


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("pipeline log write made no progress")
        remaining = remaining[written:]


def redact(value: object) -> Any:
    """Return a recursively JSON-safe value with sensitive keyed values removed."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _UNSERIALIZABLE
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                continue
            result[key] = (
                _REDACTED
                if is_credential_key(key) or _SENSITIVE_CONTENT_TOKENS.intersection(key_tokens(key))
                else redact(item)
            )
        return result
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    return _UNSERIALIZABLE


class PipelineLogger:
    """Append bounded JSON events and optional private diagnostic objects."""

    def __init__(self, artifacts: ArtifactStore) -> None:
        self._artifacts = artifacts

    def event(
        self,
        *,
        job_id: str,
        stage: str,
        event: str,
        duration_ms: float,
        provider: str | None,
        model: str | None,
        error_code: str | None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Append exactly one newline-delimited, safely serializable event."""

        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "job_id": job_id,
            "stage": stage,
            "event": event,
            "duration_ms": round(max(0.0, duration_ms), 3),
            "provider": provider,
            "model": model,
            "error_code": error_code,
            "metadata": redact(metadata or {}),
        }
        encoded = orjson.dumps(record) + b"\n"
        target = self._artifacts.job_dir(job_id) / "pipeline.jsonl"
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        with _path_lock(target):
            descriptor = os.open(target, flags, 0o600)
            try:
                os.fchmod(descriptor, 0o600)
                if fcntl is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                _write_all(descriptor, encoded)
            finally:
                if fcntl is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def diagnostic(
        self,
        job_id: str,
        enabled: bool,
        name: str,
        value: object,
    ) -> Path | None:
        """Atomically write one opted-in redacted diagnostic with mode 0600."""

        if not enabled:
            return None
        job_directory = self._artifacts.job_dir(job_id)
        diagnostics = job_directory / "diagnostics"
        diagnostics.mkdir(mode=0o700, exist_ok=True)
        if diagnostics.is_symlink() or diagnostics.resolve().parent != job_directory:
            raise ValueError("diagnostics directory must remain inside the job directory")
        safe_stem = _SAFE_NAME.sub("-", name.casefold()).strip("-_") or "diagnostic"
        target = diagnostics / f"{safe_stem}-{uuid4().hex}.json"
        payload = orjson.dumps(redact(value), option=orjson.OPT_SORT_KEYS)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=diagnostics,
            prefix=f".{safe_stem}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, target)
            os.chmod(target, 0o600)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return target


__all__ = ["PipelineLogger", "redact"]
