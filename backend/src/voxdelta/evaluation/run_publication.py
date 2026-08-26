"""Publish one evidence run directory so a visible report is always a complete report.

Both the readiness runner and the shadow replay evaluator publish the same shape: a new
private directory holding a JSON manifest and its checksum. Getting that publication wrong
is a safety problem rather than a cosmetic one — a half-written run that an operator reads
as evidence is worse than no run at all — so the logic lives here once instead of being
copied per report type.

The ordering is deliberate. The directory is created exclusively, so two writers racing
for the same path cannot merge into one another's output. The checksum is written first
and the manifest last, which makes the manifest the commit marker: any manifest a reader
can see already has a checksum beside it. On failure only a directory this call actually
created is cleaned up; if another writer won the creation race, its files are not ours to
remove.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from voxdelta.evaluation.emotion_experiment import publish_private_file


class RunPublicationError(ValueError):
    """A run directory could not be published atomically and left nothing partial."""


def canonical_payload(document: object) -> bytes:
    """Encode one report deterministically, so its digest is stable across runs."""

    return (
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        + b"\n"
    )


def publish_run_directory(
    directory: Path,
    *,
    manifest_name: str,
    checksums_name: str,
    payload: bytes,
) -> Path:
    """Create one new private run directory containing ``payload`` and its checksum.

    Raises ``RunPublicationError`` and leaves nothing partial behind.
    """

    run_directory = Path(directory)
    if not run_directory.is_absolute() or run_directory.exists() or run_directory.is_symlink():
        raise RunPublicationError("run_publication_failed")
    target = run_directory / manifest_name
    checksums = run_directory / checksums_name
    digest = hashlib.sha256(payload).hexdigest()
    created = False
    try:
        run_directory.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        run_directory.mkdir(mode=0o700)
        created = True
        publish_private_file(
            checksums, f"{digest}  {manifest_name}\n".encode(), "run_publication_failed"
        )
        # This is the commit marker: never make the manifest visible before its checksum.
        publish_private_file(target, payload, "run_publication_failed")
    except Exception:
        # If another process won the race to create this path, its directory and its
        # contents are not ours to touch.
        if created:
            target.unlink(missing_ok=True)
            checksums.unlink(missing_ok=True)
            try:
                run_directory.rmdir()
            except OSError:
                pass
        raise RunPublicationError("run_publication_failed") from None
    return target


__all__ = ["RunPublicationError", "canonical_payload", "publish_run_directory"]
