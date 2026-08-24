"""Verify OCI images and publish a credential-free manual registry handoff."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tarfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Final

from voxdelta.evaluation.manifest import read_trusted_regular_file

from voxdelta_runpod.config import BASE_IMAGE

_DIGEST: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT: Final = re.compile(r"^[0-9a-f]{40}$")
_LOCK: Final = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID: Final = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_IMAGE_PLATFORM: Final = "linux/amd64"
_ATTESTATION_TYPE: Final = "attestation-manifest"
_FORBIDDEN_HISTORY: Final = re.compile(
    r"(?i)(password|passwd|secret|token|private[_ -]?key|ssh[_ -]?host|/users/|/volumes/)"
)


class ImageHandoffError(ValueError):
    """Stable failure for an invalid image or unsafe handoff."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class OciImageEvidence:
    manifest_digest: str
    config_digest: str
    git_commit: str
    lock_sha256: str
    sbom: Mapping[str, Any]


def _sha256(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts


def _json_object(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
        raise ImageHandoffError("invalid_oci_archive") from None
    if not isinstance(value, dict):
        raise ImageHandoffError("invalid_oci_archive")
    return value


class _OciReader:
    def __init__(self, archive: Path) -> None:
        if not archive.is_absolute():
            raise ImageHandoffError("invalid_oci_archive")
        self._archive = archive

    def __enter__(self) -> _OciReader:
        try:
            self._tar = tarfile.open(self._archive, mode="r:")
        except (OSError, tarfile.TarError):
            raise ImageHandoffError("invalid_oci_archive") from None
        members = self._tar.getmembers()
        if any(not _safe_member(member.name) or not member.isfile() for member in members):
            self._tar.close()
            raise ImageHandoffError("invalid_oci_archive")
        self._members = {member.name: member for member in members}
        if len(self._members) != len(members):
            self._tar.close()
            raise ImageHandoffError("invalid_oci_archive")
        return self

    def __exit__(self, *_args: object) -> None:
        self._tar.close()

    def read(self, name: str) -> bytes:
        member = self._members.get(name)
        if member is None:
            raise ImageHandoffError("invalid_oci_archive")
        stream = self._tar.extractfile(member)
        if stream is None:
            raise ImageHandoffError("invalid_oci_archive")
        return stream.read()

    def blob(self, digest: str) -> bytes:
        if not _DIGEST.fullmatch(digest):
            raise ImageHandoffError("invalid_oci_archive")
        payload = self.read(f"blobs/sha256/{digest.removeprefix('sha256:')}")
        if _sha256(payload) != digest:
            raise ImageHandoffError("invalid_oci_digest")
        return payload


def _descriptor_digest(descriptor: object) -> str:
    if not isinstance(descriptor, dict):
        raise ImageHandoffError("invalid_oci_archive")
    digest = descriptor.get("digest")
    if not isinstance(digest, str):
        raise ImageHandoffError("invalid_oci_archive")
    return digest


def _image_descriptor(manifests: object) -> dict[str, Any]:
    if not isinstance(manifests, list):
        raise ImageHandoffError("invalid_oci_archive")
    candidates: list[dict[str, Any]] = []
    for descriptor in manifests:
        if not isinstance(descriptor, dict):
            raise ImageHandoffError("invalid_oci_archive")
        annotations = descriptor.get("annotations", {})
        platform = descriptor.get("platform", {})
        if not isinstance(annotations, dict) or not isinstance(platform, dict):
            raise ImageHandoffError("invalid_oci_archive")
        is_attestation = annotations.get("vnd.docker.reference.type") == _ATTESTATION_TYPE
        if (
            not is_attestation
            and platform.get("os") == "linux"
            and platform.get("architecture") == "amd64"
        ):
            candidates.append(descriptor)
    if len(candidates) != 1:
        raise ImageHandoffError("invalid_oci_platform")
    return candidates[0]


def _extract_sbom(reader: _OciReader, manifests: object) -> Mapping[str, Any]:
    if not isinstance(manifests, list):
        raise ImageHandoffError("invalid_image_sbom")
    predicates: list[Mapping[str, Any]] = []
    for descriptor in manifests:
        if not isinstance(descriptor, dict):
            continue
        annotations = descriptor.get("annotations", {})
        if not isinstance(annotations, dict):
            continue
        if annotations.get("vnd.docker.reference.type") != _ATTESTATION_TYPE:
            continue
        attestation = _json_object(reader.blob(_descriptor_digest(descriptor)))
        layers = attestation.get("layers")
        if not isinstance(layers, list):
            raise ImageHandoffError("invalid_image_sbom")
        for layer in layers:
            payload = _json_object(reader.blob(_descriptor_digest(layer)))
            predicate_type = payload.get("predicateType")
            predicate = payload.get("predicate")
            if (
                isinstance(predicate_type, str)
                and "spdx" in predicate_type.lower()
                and isinstance(predicate, dict)
            ):
                predicates.append(predicate)
    if len(predicates) != 1:
        raise ImageHandoffError("invalid_image_sbom")
    return predicates[0]


def inspect_oci_archive(
    archive: Path, *, expected_git_commit: str, expected_lock_sha256: str
) -> OciImageEvidence:
    """Validate the single-platform image identity, labels, history, and SPDX attestation."""

    if not _COMMIT.fullmatch(expected_git_commit) or not _LOCK.fullmatch(expected_lock_sha256):
        raise ImageHandoffError("invalid_build_identity")
    with _OciReader(archive) as reader:
        index = _json_object(reader.read("index.json"))
        manifests = index.get("manifests")
        image_descriptor = _image_descriptor(manifests)
        manifest_digest = _descriptor_digest(image_descriptor)
        manifest = _json_object(reader.blob(manifest_digest))
        config_digest = _descriptor_digest(manifest.get("config"))
        config = _json_object(reader.blob(config_digest))
        labels = config.get("config", {}).get("Labels", {})
        if not isinstance(labels, dict):
            raise ImageHandoffError("invalid_image_labels")
        if (
            labels.get("org.opencontainers.image.revision") != expected_git_commit
            or labels.get("io.voxdelta.runpod.lock-sha256") != expected_lock_sha256
            or labels.get("io.voxdelta.runpod.platform") != _IMAGE_PLATFORM
        ):
            raise ImageHandoffError("invalid_image_labels")
        history = config.get("history", [])
        if not isinstance(history, list):
            raise ImageHandoffError("invalid_image_history")
        history_text = json.dumps(history, sort_keys=True, separators=(",", ":"))
        if _FORBIDDEN_HISTORY.search(history_text):
            raise ImageHandoffError("unsafe_image_history")
        sbom = _extract_sbom(reader, manifests)
    return OciImageEvidence(
        manifest_digest=manifest_digest,
        config_digest=config_digest,
        git_commit=expected_git_commit,
        lock_sha256=expected_lock_sha256,
        sbom=sbom,
    )


def _push_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ $# -ne 1 || "$1" != */* || "$1" == *@* ]]; then
  echo "usage: push-image.sh <registry/repository:tag>" >&2
  exit 64
fi
for command in shasum zstd skopeo mktemp; do
  command -v "$command" >/dev/null 2>&1 || { echo "missing command: $command" >&2; exit 69; }
done

root="$(cd "$(dirname "$0")" && pwd -P)"
cd "$root"
shasum -a 256 -c SHA256SUMS
expected="$(tr -d '\\n' < image-digest.txt)"
temporary="$(mktemp -d "${TMPDIR:-/tmp}/voxdelta-image.XXXXXX")"
trap 'rm -rf -- "$temporary"' EXIT
zstd --quiet --decompress --stdout voxdelta-runpod.oci.tar.zst > "$temporary/image.oci.tar"
skopeo copy --preserve-digests "oci-archive:$temporary/image.oci.tar" "docker://$1"
remote="$(skopeo inspect --format '{{.Digest}}' "docker://$1")"
if [[ "$remote" != "$expected" ]]; then
  echo "remote image digest mismatch" >&2
  exit 65
fi
printf '%s@%s\\n' "${1%%:*}" "$remote" > image-reference.txt
chmod 600 image-reference.txt
echo "image push verified: $remote"
"""


def _write_bytes(path: Path, payload: bytes, mode: int = 0o600) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(mode)


def _compress_archive(source: Path, target: Path) -> None:
    with target.open("xb") as output:
        completed = subprocess.run(
            ["zstd", "-19", "--threads=0", "--quiet", "--stdout", str(source)],
            stdout=output,
            stderr=subprocess.PIPE,
            check=False,
        )
        output.flush()
        os.fsync(output.fileno())
    if completed.returncode != 0:
        raise ImageHandoffError("image_compression_failed")
    target.chmod(0o600)


def _checksums(paths: list[Path]) -> bytes:
    lines = []
    for path in sorted(paths, key=lambda item: item.name):
        digest = hashlib.sha256(read_trusted_regular_file(path)).hexdigest()
        lines.append(f"{digest}  {path.name}")
    return ("\n".join(lines) + "\n").encode()


def publish_image_handoff(
    archive: Path,
    output_root: Path,
    *,
    run_id: str,
    git_commit: str,
    lock_sha256: str,
    created_at: datetime | None = None,
) -> Path:
    """Publish an immutable private image packet beneath an absolute output root."""

    if (
        not archive.is_absolute()
        or not output_root.is_absolute()
        or not _RUN_ID.fullmatch(run_id)
        or output_root.is_symlink()
    ):
        raise ImageHandoffError("invalid_handoff_target")
    evidence = inspect_oci_archive(
        archive,
        expected_git_commit=git_commit,
        expected_lock_sha256=lock_sha256,
    )
    output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    output_root.chmod(0o700)
    target = output_root / run_id
    staging = output_root / f".{run_id}.staging"
    if target.exists() or staging.exists():
        raise ImageHandoffError("handoff_exists")
    staging.mkdir(mode=0o700)
    try:
        compressed = staging / "voxdelta-runpod.oci.tar.zst"
        _compress_archive(archive, compressed)
        digest_path = staging / "image-digest.txt"
        _write_bytes(digest_path, f"{evidence.manifest_digest}\n".encode())
        timestamp = created_at or datetime.now(UTC)
        metadata = {
            "schema_version": "1",
            "run_id": run_id,
            "created_at": timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "platform": _IMAGE_PLATFORM,
            "base_image": BASE_IMAGE,
            "manifest_digest": evidence.manifest_digest,
            "config_digest": evidence.config_digest,
            "git_commit": evidence.git_commit,
            "runpod_lock_sha256": evidence.lock_sha256,
        }
        metadata_path = staging / "image-build.json"
        _write_bytes(
            metadata_path,
            (json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n").encode(),
        )
        sbom_path = staging / "image-sbom.spdx.json"
        _write_bytes(
            sbom_path,
            (json.dumps(evidence.sbom, sort_keys=True, separators=(",", ":")) + "\n").encode(),
        )
        push_path = staging / "push-image.sh"
        _write_bytes(push_path, _push_script().encode(), mode=0o700)
        sums_path = staging / "SHA256SUMS"
        _write_bytes(
            sums_path,
            _checksums([compressed, digest_path, metadata_path, sbom_path, push_path]),
        )
        os.replace(staging, target)
        parent_fd = os.open(output_root, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if stat.S_IMODE(target.stat().st_mode) != 0o700:
        raise ImageHandoffError("unsafe_handoff_mode")
    return target


__all__ = [
    "ImageHandoffError",
    "OciImageEvidence",
    "inspect_oci_archive",
    "publish_image_handoff",
]
