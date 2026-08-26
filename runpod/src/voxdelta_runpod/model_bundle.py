"""Deterministic, allowlisted transfer bundles for pinned model inputs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from collections.abc import Mapping
from pathlib import Path
from typing import IO, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from voxdelta.evaluation.manifest import read_trusted_regular_file
from voxdelta.evaluation.wav2vec_base import validate_wav2vec_base
from voxdelta.providers._emotion_runtime import validate_checkpoint

BundleKind = Literal["xls-r-base", "emotion2vec-baseline"]

_FILES: Mapping[BundleKind, frozenset[str]] = {
    "xls-r-base": frozenset({"config.json", "preprocessor_config.json", "pytorch_model.bin"}),
    "emotion2vec-baseline": frozenset(
        {"config.json", "label_mapping.json", "metrics.json", "model.safetensors"}
    ),
}


class ModelBundleError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class ModelBundleSidecar(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    schema_version: Literal["1"] = "1"
    bundle_kind: BundleKind
    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    archive_bytes: int = Field(gt=0)
    tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    file_sha256: dict[str, str]
    file_bytes: dict[str, int]

    @model_validator(mode="after")
    def valid_files(self) -> ModelBundleSidecar:
        expected = _FILES[self.bundle_kind]
        if (
            set(self.file_sha256) != expected
            or set(self.file_bytes) != expected
            or any(not _sha256(value) for value in self.file_sha256.values())
            or any(value <= 0 for value in self.file_bytes.values())
        ):
            raise ValueError("invalid model bundle sidecar")
        return self


def _sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
    except OSError:
        raise ModelBundleError("invalid_model_input") from None
    return digest.hexdigest(), size


def _tree_digest(file_sha256: Mapping[str, str], file_bytes: Mapping[str, int]) -> str:
    payload = json.dumps(
        [
            {"name": name, "sha256": file_sha256[name], "bytes": file_bytes[name]}
            for name in sorted(file_sha256)
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_tree(root: Path, kind: BundleKind) -> tuple[dict[str, str], dict[str, int]]:
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ModelBundleError("invalid_model_input")
    try:
        entries = tuple(root.iterdir())
        if {entry.name for entry in entries} != _FILES[kind] or any(
            entry.is_symlink()
            or not entry.is_file()
            or entry.stat(follow_symlinks=False).st_mode & 0o077
            for entry in entries
        ):
            raise ModelBundleError("invalid_model_input")
        if kind == "xls-r-base":
            validate_wav2vec_base(root)
        else:
            validate_checkpoint(
                root,
                architecture="emotion2vec-plus",
                model_id="iic/emotion2vec_plus_large",
            )
    except ModelBundleError:
        raise
    except Exception:
        raise ModelBundleError("invalid_model_input") from None
    file_sha256: dict[str, str] = {}
    file_bytes: dict[str, int] = {}
    for name in sorted(_FILES[kind]):
        digest, size = _digest(root / name)
        file_sha256[name] = digest
        file_bytes[name] = size
    return file_sha256, file_bytes


def _write_archive(source: Path, target: Path, kind: BundleKind) -> None:
    with target.open("xb") as output:
        process = subprocess.Popen(
            ["zstd", "-3", "--threads=1", "--quiet", "--stdout"],
            stdin=subprocess.PIPE,
            stdout=output,
            stderr=subprocess.PIPE,
        )
        if process.stdin is None:
            process.kill()
            raise ModelBundleError("model_bundle_failed")
        try:
            with tarfile.open(fileobj=process.stdin, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                for name in sorted(_FILES[kind]):
                    path = source / name
                    metadata = path.stat(follow_symlinks=False)
                    info = tarfile.TarInfo(name)
                    info.size = metadata.st_size
                    info.mode = 0o600
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    with path.open("rb") as stream:
                        tar.addfile(info, stream)
            process.stdin.close()
            stderr = process.stderr.read() if process.stderr is not None else b""
            if process.wait() != 0 or stderr:
                raise ModelBundleError("model_bundle_failed")
        except Exception:
            if process.poll() is None:
                process.kill()
                process.wait()
            raise
    target.chmod(0o600)


def build_model_bundle(source: Path, output: Path, kind: BundleKind) -> Path:
    """Publish a private model-input archive and strict sidecar beneath a new directory."""

    source = source.resolve()
    output = output.resolve()
    staging = output.with_name(f".{output.name}.staging")
    if output.exists() or output.is_symlink() or staging.exists() or staging.is_symlink():
        raise ModelBundleError("model_bundle_exists")
    file_sha256, file_bytes = _validate_tree(source, kind)
    archive_name = f"{kind}.tar.zst"
    sidecar_name = f"{kind}.sidecar.json"
    try:
        staging.mkdir(mode=0o700, parents=True)
        archive = staging / archive_name
        _write_archive(source, archive, kind)
        archive_sha256, archive_bytes = _digest(archive)
        sidecar = ModelBundleSidecar(
            bundle_kind=kind,
            archive_sha256=archive_sha256,
            archive_bytes=archive_bytes,
            tree_sha256=_tree_digest(file_sha256, file_bytes),
            file_sha256=file_sha256,
            file_bytes=file_bytes,
        )
        sidecar_path = staging / sidecar_name
        with sidecar_path.open("xb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write((sidecar.model_dump_json() + "\n").encode())
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def _copy_member(stream: IO[bytes], target: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with target.open("xb") as output:
        os.fchmod(output.fileno(), 0o600)
        while chunk := stream.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def extract_model_bundle(
    archive: Path,
    sidecar_path: Path,
    target: Path,
    *,
    expected_kind: BundleKind,
) -> Path:
    """Verify and atomically extract one model-input bundle, idempotently."""

    archive = archive.resolve()
    sidecar_path = sidecar_path.resolve()
    target = target.resolve()
    try:
        sidecar = ModelBundleSidecar.model_validate_json(read_trusted_regular_file(sidecar_path))
        if sidecar.bundle_kind != expected_kind or _digest(archive) != (
            sidecar.archive_sha256,
            sidecar.archive_bytes,
        ):
            raise ModelBundleError("model_bundle_digest_mismatch")
        if target.exists():
            file_sha256, file_bytes = _validate_tree(target, expected_kind)
            if (
                file_sha256 != sidecar.file_sha256
                or file_bytes != sidecar.file_bytes
                or _tree_digest(file_sha256, file_bytes) != sidecar.tree_sha256
            ):
                raise ModelBundleError("model_bundle_target_mismatch")
            return target
    except ModelBundleError:
        raise
    except Exception:
        raise ModelBundleError("invalid_model_bundle") from None
    staging = target.with_name(f".{target.name}.staging")
    if target.is_symlink() or staging.exists() or staging.is_symlink():
        raise ModelBundleError("invalid_model_bundle_target")
    try:
        staging.mkdir(mode=0o700, parents=True)
        process = subprocess.Popen(
            ["zstd", "--quiet", "--decompress", "--stdout", str(archive)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if process.stdout is None:
            process.kill()
            raise ModelBundleError("model_bundle_extract_failed")
        seen: set[str] = set()
        with tarfile.open(fileobj=process.stdout, mode="r|") as tar:
            for member in tar:
                if (
                    not member.isfile()
                    or member.name not in _FILES[expected_kind]
                    or member.name in seen
                    or member.size != sidecar.file_bytes[member.name]
                ):
                    raise ModelBundleError("invalid_model_bundle_members")
                extracted = tar.extractfile(member)
                if extracted is None:
                    raise ModelBundleError("invalid_model_bundle_members")
                digest, size = _copy_member(extracted, staging / member.name)
                if (
                    digest != sidecar.file_sha256[member.name]
                    or size != sidecar.file_bytes[member.name]
                ):
                    raise ModelBundleError("model_bundle_digest_mismatch")
                seen.add(member.name)
        stderr = process.stderr.read() if process.stderr is not None else b""
        if process.wait() != 0 or stderr or seen != _FILES[expected_kind]:
            raise ModelBundleError("model_bundle_extract_failed")
        file_sha256, file_bytes = _validate_tree(staging, expected_kind)
        if _tree_digest(file_sha256, file_bytes) != sidecar.tree_sha256:
            raise ModelBundleError("model_bundle_digest_mismatch")
        os.replace(staging, target)
    except Exception:
        if "process" in locals() and process.poll() is None:
            process.kill()
            process.wait()
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


__all__ = [
    "BundleKind",
    "ModelBundleError",
    "ModelBundleSidecar",
    "build_model_bundle",
    "extract_model_bundle",
]
