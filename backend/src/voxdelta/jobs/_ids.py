"""Shared validation for opaque job identifiers."""

from __future__ import annotations

from pathlib import Path, PureWindowsPath


def validate_job_id(job_id: str) -> None:
    """Reject IDs that can acquire path semantics on POSIX or Windows."""

    windows_path = PureWindowsPath(job_id)
    if (
        not job_id
        or job_id in {".", ".."}
        or "/" in job_id
        or "\\" in job_id
        or Path(job_id).is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or bool(windows_path.anchor)
    ):
        raise ValueError("job ID must be a non-empty path component")
