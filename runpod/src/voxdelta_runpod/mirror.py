"""Durably mirror completed remote epoch checkpoints to private local storage.

Active run state lives on the Pod's container disk, which does not survive a Pod
stop. Only immutable, fully published epoch directories are mirrored, each verified
against its own ``state.json`` before it is allowed to replace anything locally, so
a lost Pod costs at most the epoch that was still in flight.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from voxdelta_runpod.checkpoint import (
    CheckpointError,
    CheckpointStage,
    CheckpointState,
    verify_checkpoint_directory,
)

_EPOCH = re.compile(r"^epoch-(\d{4})$")

#: Minimal non-audio evidence an exact resume needs alongside the checkpoints.
MIRROR_EVIDENCE: tuple[str, ...] = (
    "state/run-identity.json",
    "state/environment.json",
    "state/preflight-complete.json",
    "state/batch-profile.json",
)


class MirrorError(ValueError):
    """Stable failure for an unusable or unsafe mirror operation."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def completed_epochs(root: Path) -> tuple[Path, ...]:
    """Published epoch directories only; staging directories are still in flight."""
    if not root.is_absolute() or root.is_symlink():
        raise MirrorError("invalid_mirror_root")
    if not root.is_dir():
        return ()
    found: list[tuple[int, Path]] = []
    for path in root.iterdir():
        match = _EPOCH.fullmatch(path.name)
        if match is None or path.is_symlink() or not path.is_dir():
            continue
        if (path / "state.json").is_file():
            found.append((int(match.group(1)), path))
    return tuple(path for _, path in sorted(found))


def verify_mirrored_checkpoint(path: Path, *, expected_stage: CheckpointStage) -> CheckpointState:
    """Re-read and re-hash every payload; a partial or tampered copy is rejected."""
    try:
        return verify_checkpoint_directory(path, expected_stage)
    except CheckpointError as error:
        raise MirrorError(error.code) from error
    except Exception as error:
        raise MirrorError("invalid_mirror_checkpoint") from error


def publish_mirror(staging: Path, target: Path) -> Path:
    """Move a verified staging copy into place atomically, refusing to overwrite."""
    if not target.is_absolute() or target.is_symlink() or target.exists():
        raise MirrorError("mirror_exists")
    if not staging.is_dir() or staging.is_symlink():
        raise MirrorError("invalid_mirror_staging")
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(staging, 0o700)
    for item in staging.iterdir():
        if item.is_symlink() or not item.is_file():
            raise MirrorError("invalid_mirror_staging")
        os.chmod(item, 0o600)
    os.replace(staging, target)
    parent = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    return target


def latest_verified_epoch(
    root: Path, *, expected_stage: CheckpointStage
) -> tuple[int, Path, CheckpointState] | None:
    """Highest epoch whose local copy still verifies; unusable copies are skipped."""
    for path in reversed(completed_epochs(root)):
        try:
            state = verify_mirrored_checkpoint(path, expected_stage=expected_stage)
        except MirrorError:
            continue
        match = _EPOCH.fullmatch(path.name)
        if match is None or state.epoch != int(match.group(1)):
            continue
        return state.epoch, path, state
    return None


def discard_staging(staging: Path) -> None:
    shutil.rmtree(staging, ignore_errors=True)


__all__ = [
    "MIRROR_EVIDENCE",
    "MirrorError",
    "completed_epochs",
    "discard_staging",
    "latest_verified_epoch",
    "publish_mirror",
    "verify_mirrored_checkpoint",
]
