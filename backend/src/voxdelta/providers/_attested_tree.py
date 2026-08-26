"""Shared primitives for verifying an immutable, self-attesting artifact directory.

A release bundle and a post-hoc calibration artifact are the same shape: a directory
whose ``SHA256SUMS`` and JSON manifest must agree with one another and with the bytes on
disk, holding no file the manifest does not attest and no link of any kind. Both fail
closed with a fixed, path-free error so a rejected artifact never discloses a host path.

These helpers are deliberately free of any release- or calibration-specific knowledge;
identity checks belong to the caller.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

CHECKSUMS_NAME = "SHA256SUMS"
MAX_METADATA_BYTES = 1024 * 1024
MAX_CHECKSUM_ENTRIES = 4096


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def safe_relative(value: object, reserved: frozenset[str]) -> str:
    """Accept one plain, forward-slashed relative path that escapes nothing."""

    if not isinstance(value, str) or not value or value in reserved:
        raise ValueError("unsafe relative path")
    parts = value.split("/")
    if (
        value.startswith("/")
        or value != Path(value).as_posix()
        or "" in parts
        or "." in parts
        or ".." in parts
    ):
        raise ValueError("unsafe relative path")
    return value


def _reject_json_constant(_value: str) -> None:
    raise ValueError("json constant is not allowed")


def load_strict_json(raw: bytes) -> object:
    """Parse bounded JSON that admits no NaN or Infinity constant."""

    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError("metadata too large")
    return json.loads(raw, parse_constant=_reject_json_constant)


def resolved_artifact_root(path: str | Path) -> Path:
    """Resolve an absolute directory whose every component is a real directory."""

    candidate = Path(path)
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("unsafe artifact root")
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError("symlinked artifact root")
    resolved = candidate.resolve(strict=True)
    if not stat.S_ISDIR(resolved.stat(follow_symlinks=False).st_mode):
        raise ValueError("artifact root is not a directory")
    return resolved


def tree_files(root: Path) -> set[str]:
    """List every regular file under the artifact, rejecting links and special files."""

    found: set[str] = set()
    for directory, directory_names, filenames in os.walk(root, followlinks=False):
        current = Path(directory)
        for name in sorted(directory_names):
            metadata = (current / name).stat(follow_symlinks=False)
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError("unexpected directory entry")
        for name in sorted(filenames):
            child = current / name
            if not stat.S_ISREG(child.stat(follow_symlinks=False).st_mode):
                raise ValueError("unexpected file entry")
            found.add(child.relative_to(root).as_posix())
            if len(found) > MAX_CHECKSUM_ENTRIES:
                raise ValueError("too many files")
    return found


def digest_and_size(path: Path) -> tuple[str, int]:
    """Hash one regular file without following a link or racing a replacement."""

    digest = hashlib.sha256()
    size = 0
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("not a regular file")
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError("file changed while being read")
    return digest.hexdigest(), size


def parse_checksums(raw: bytes, *, manifest_name: str, reserved: frozenset[str]) -> dict[str, str]:
    """Parse a ``SHA256SUMS`` file that must attest the manifest and nothing twice."""

    text = raw.decode("utf-8")
    entries: dict[str, str] = {}
    for line in text.splitlines():
        if not line:
            raise ValueError("blank checksum line")
        head, separator, tail = line.partition("  ")
        if not separator or not is_sha256(head):
            raise ValueError("malformed checksum line")
        relative = tail if tail == manifest_name else safe_relative(tail, reserved)
        if relative in entries:
            raise ValueError("duplicate checksum entry")
        entries[relative] = head
        if len(entries) > MAX_CHECKSUM_ENTRIES:
            raise ValueError("too many checksum entries")
    if manifest_name not in entries:
        raise ValueError("checksums do not attest the manifest")
    return entries


def canonical_sha256(payload: object) -> str:
    """SHA-256 over one canonical JSON encoding, so key order cannot change identity."""

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


__all__ = [
    "CHECKSUMS_NAME",
    "MAX_CHECKSUM_ENTRIES",
    "MAX_METADATA_BYTES",
    "canonical_sha256",
    "digest_and_size",
    "is_positive_int",
    "is_sha256",
    "load_strict_json",
    "parse_checksums",
    "resolved_artifact_root",
    "safe_relative",
    "tree_files",
]
