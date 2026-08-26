"""Publish an immutable local encoder bundle from a package cache, without trusting it.

The cache snapshot is read, never written and never deleted from: this module only copies
out the four files the loader needs. Weights arrive through a symlink into a second cache,
so ``model.pt`` — and only ``model.pt`` — is dereferenced into a real file inside the
bundle; a bundle whose contents can change when someone prunes an unrelated cache is not
immutable. The other three files must already be real regular files, because a symlink
there is a way to read something the snapshot does not contain.

Every copied byte is checked against the pinned encoder identity *before* anything becomes
visible at the target, so wrong content is destroyed in staging.

**Publication commits by marker, not by directory rename.** Renaming a directory onto an
existing *empty* directory succeeds on POSIX, so a check-then-rename would silently
destroy another writer's freshly reserved target. Instead ownership is reserved
atomically with an exclusive ``mkdir``: exactly one writer can create the target, and a
loser preserves whatever is already there. The verified payloads are then moved in, the
checksum file is written, and ``ENCODER.json`` is written **last** as the logical commit
marker — the verifier cannot accept a directory without it, so a reserved-but-incomplete
bundle is never mistaken for a finished one.

Nothing derived from evaluation data is copied.
"""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

from voxdelta.evaluation.run_publication import canonical_payload
from voxdelta.providers._attested_tree import digest_and_size
from voxdelta.providers.encoder_bundle import (
    CHECKSUMS_NAME,
    INIT_PARAM_NAME,
    MANIFEST_NAME,
    PINNED_BUNDLE_TREE_SHA256,
    PINNED_PAYLOADS,
    REQUIRED_FILES,
    VerifiedEncoderBundle,
    encoder_bundle_manifest,
    verify_encoder_bundle,
)


class EncoderPublicationError(ValueError):
    """A local encoder bundle could not be published and left nothing partial behind."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _copy_dereferenced(source: Path, target: Path) -> None:
    """Copy the weights through their symlink into a real file inside the bundle."""

    resolved = source.resolve(strict=True)
    if not stat.S_ISREG(resolved.stat().st_mode):
        raise EncoderPublicationError("encoder_source_not_a_regular_file")
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with open(descriptor, "wb") as sink, resolved.open("rb") as stream:
        shutil.copyfileobj(stream, sink, length=1024 * 1024)


def _copy_regular(source: Path, target: Path) -> None:
    """Copy one file that must already be a real regular file, following no link."""

    if source.is_symlink():
        raise EncoderPublicationError("encoder_source_is_a_symlink")
    metadata = source.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode):
        raise EncoderPublicationError("encoder_source_not_a_regular_file")
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with open(descriptor, "wb") as sink, source.open("rb") as stream:
        shutil.copyfileobj(stream, sink, length=1024 * 1024)


def _reserve(target: Path) -> None:
    """Claim the target atomically. Exactly one writer can win; a loser must not touch it.

    ``mkdir`` is the primitive that makes this safe: it fails with ``FileExistsError``
    whether the existing target is empty or not, unlike a directory rename, which happily
    replaces an empty one.
    """

    try:
        target.mkdir(mode=0o700)
    except FileExistsError:
        raise EncoderPublicationError("encoder_bundle_already_exists") from None


def publish_encoder_bundle(snapshot: Path, output: Path) -> VerifiedEncoderBundle:
    """Copy the loader's file set out of ``snapshot`` into a new pinned, verified bundle.

    Raises ``EncoderPublicationError`` and leaves nothing partial or invalid behind — not
    in staging, and not at the target, even when the final verification is what fails. A
    target belonging to another writer is never modified or removed.
    """

    source = Path(snapshot)
    target = Path(output)
    if not source.is_absolute() or not target.is_absolute():
        raise EncoderPublicationError("encoder_bundle_path_not_absolute")
    if not source.is_dir():
        raise EncoderPublicationError("encoder_snapshot_missing")
    if target.exists() or target.is_symlink():
        raise EncoderPublicationError("encoder_bundle_already_exists")

    for name in REQUIRED_FILES:
        if not (source / name).exists():
            raise EncoderPublicationError("encoder_snapshot_incomplete")

    staging = target.with_name(f".{target.name}.staging")
    if staging.exists() or staging.is_symlink():
        raise EncoderPublicationError("encoder_bundle_already_exists")
    reserved = False
    try:
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        staging.mkdir(mode=0o700)
        for name in REQUIRED_FILES:
            if name == INIT_PARAM_NAME:
                _copy_dereferenced(source / name, staging / name)
            else:
                _copy_regular(source / name, staging / name)

        # The pin is enforced on the copied bytes, while they are still only in staging.
        payloads: list[dict[str, object]] = []
        for name in sorted(REQUIRED_FILES):
            digest, size = digest_and_size(staging / name)
            expected_digest, expected_bytes = PINNED_PAYLOADS[name]
            if digest != expected_digest or size != expected_bytes:
                raise EncoderPublicationError("encoder_source_does_not_match_pin")
            payloads.append({"path": name, "sha256": digest, "bytes": size})

        manifest = encoder_bundle_manifest(payloads)
        if manifest["bundle_tree_sha256"] != PINNED_BUNDLE_TREE_SHA256:
            raise EncoderPublicationError("encoder_source_does_not_match_pin")
        manifest_bytes = canonical_payload(manifest)
        lines = [f"{_digest_bytes(manifest_bytes)}  {MANIFEST_NAME}"]
        lines.extend(f"{record['sha256']}  {record['path']}" for record in payloads)
        checksums_bytes = "".join(
            f"{line}\n" for line in sorted(lines, key=lambda line: line.split("  ", 1)[1])
        ).encode()

        # Everything is verified; only now is the target claimed.
        _reserve(target)
        reserved = True
        for name in sorted(REQUIRED_FILES):
            os.rename(staging / name, target / name)
        staging.rmdir()
        _write_private(target / CHECKSUMS_NAME, checksums_bytes)
        # The commit marker: a bundle without this cannot be accepted by the verifier.
        _write_private(target / MANIFEST_NAME, manifest_bytes)

        # A bundle that cannot pass the production verifier must not survive this call.
        return verify_encoder_bundle(target)
    except EncoderPublicationError:
        _discard(staging, target if reserved else None)
        raise
    except Exception:
        _discard(staging, target if reserved else None)
        raise EncoderPublicationError("encoder_bundle_publication_failed") from None


def _discard(staging: Path, target: Path | None) -> None:
    """Remove only what this invocation exclusively created.

    ``target`` is passed only when this call won the reservation, so a directory another
    writer owns is never removed here.
    """

    shutil.rmtree(staging, ignore_errors=True)
    if target is not None:
        shutil.rmtree(target, ignore_errors=True)


def _digest_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _write_private(path: Path, payload: bytes | str) -> None:
    data = payload.encode() if isinstance(payload, str) else payload
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with open(descriptor, "wb") as stream:
        stream.write(data)


__all__ = ["EncoderPublicationError", "publish_encoder_bundle"]
