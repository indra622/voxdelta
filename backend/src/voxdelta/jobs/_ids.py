"""Shared validation for opaque job identifiers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

_RESERVED_JOB_COMPONENTS = frozenset({".deleted", ".incoming", ".locks"})
_CANONICAL_JOB_ID = re.compile(r"[0-9a-f]{32}")
_ABSENCE_PROOF_AUTHORITY = object()


@dataclass(frozen=True, slots=True)
class JobAbsenceProof:
    job_id: str
    _authority: object


def validate_job_id(job_id: str) -> None:
    """Reject IDs that can acquire path semantics on POSIX or Windows."""

    windows_path = PureWindowsPath(job_id)
    if (
        not job_id
        or job_id in _RESERVED_JOB_COMPONENTS
        or job_id in {".", ".."}
        or "/" in job_id
        or "\\" in job_id
        or Path(job_id).is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or bool(windows_path.anchor)
    ):
        if job_id in _RESERVED_JOB_COMPONENTS:
            raise ValueError("job ID uses a reserved control component")
        raise ValueError("job ID must be a non-empty path component")


def validate_canonical_job_id(job_id: str) -> None:
    """Require the API/repository identity format emitted by ``uuid4().hex``."""

    validate_job_id(job_id)
    if _CANONICAL_JOB_ID.fullmatch(job_id) is None:
        raise ValueError("job ID must be exactly 32 lowercase hexadecimal characters")


def _issue_job_absence_proof(job_id: str) -> JobAbsenceProof:
    validate_canonical_job_id(job_id)
    return JobAbsenceProof(job_id, _ABSENCE_PROOF_AUTHORITY)


def validate_job_absence_proof(proof: object, job_id: str) -> None:
    validate_canonical_job_id(job_id)
    if (
        not isinstance(proof, JobAbsenceProof)
        or proof.job_id != job_id
        or proof._authority is not _ABSENCE_PROOF_AUTHORITY
    ):
        raise ValueError("a repository-issued job absence proof is required")


__all__ = [
    "JobAbsenceProof",
    "validate_canonical_job_id",
    "validate_job_absence_proof",
    "validate_job_id",
]
