"""Pinned, private local artifacts for the XLS-R 300M emotion candidate."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

WAV2VEC_MODEL_ID = "facebook/wav2vec2-xls-r-300m"
WAV2VEC_MODEL_REVISION = "1a640f32ac3e39899438a2931f9924c02f080a54"
WAV2VEC_WEIGHTS_SHA256 = "d5e490574712ad0a6736923b9ed11d4cd51c78609c36205f704fc4e87b11d2e0"


@dataclass(frozen=True, slots=True)
class Wav2VecArtifact:
    name: str
    size: int
    sha256: str


WAV2VEC_ARTIFACTS: tuple[Wav2VecArtifact, ...] = (
    Wav2VecArtifact(
        "config.json",
        1_568,
        "0bffa0d0e98153e883b828d86491f3c6062cb563dc9d7a9cfd1790da30c286ac",
    ),
    Wav2VecArtifact(
        "preprocessor_config.json",
        212,
        "a2254a5b58f72cd4de3632f8eee64f3f098b7c1402128d2f419e7d00ae13e335",
    ),
    Wav2VecArtifact("pytorch_model.bin", 1_269_737_156, WAV2VEC_WEIGHTS_SHA256),
)


@dataclass(frozen=True, slots=True)
class PreparedWav2VecBase:
    path: Path
    model_id: str = WAV2VEC_MODEL_ID
    revision: str = WAV2VEC_MODEL_REVISION
    weights_sha256: str = WAV2VEC_WEIGHTS_SHA256


Downloader = Callable[[str, Path], None]


def _artifact(name: str) -> Wav2VecArtifact:
    for artifact in WAV2VEC_ARTIFACTS:
        if artifact.name == name:
            return artifact
    raise ValueError("invalid_wav2vec_base")


def _revision_url(name: str) -> str:
    _artifact(name)
    return f"https://huggingface.co/{WAV2VEC_MODEL_ID}/resolve/{WAV2VEC_MODEL_REVISION}/{name}"


def _reject_untrusted_path(path: str | Path, *, must_exist: bool) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("invalid_wav2vec_base")
    absolute = Path(os.path.abspath(candidate))
    current = Path(absolute.anchor)
    parts = absolute.parts[1:] if absolute.is_absolute() else absolute.parts
    for index, part in enumerate(parts):
        current /= part
        if current.is_symlink():
            raise ValueError("invalid_wav2vec_base")
        if not current.exists() and (must_exist or index < len(parts) - 1):
            continue
    if must_exist and not absolute.exists():
        raise ValueError("invalid_wav2vec_base")
    return absolute


def _private(metadata: os.stat_result, mode: int, *, directory: bool) -> bool:
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    return (
        expected_type(metadata.st_mode)
        and (not hasattr(os, "geteuid") or metadata.st_uid == os.geteuid())
        and stat.S_IMODE(metadata.st_mode) == mode
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, output: Path) -> None:
    artifact = _artifact(output.name)
    written = 0
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
            while chunk := response.read(1024 * 1024):
                written += len(chunk)
                if written > artifact.size:
                    raise ValueError("invalid_wav2vec_base")
                os.write(descriptor, chunk)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_wav2vec_base(path: str | Path) -> PreparedWav2VecBase:
    """Validate exact pinned files without following symlinks."""

    try:
        candidate = _reject_untrusted_path(path, must_exist=True)
        metadata = candidate.stat(follow_symlinks=False)
        if candidate.is_symlink() or not _private(metadata, 0o700, directory=True):
            raise ValueError
        expected = {artifact.name for artifact in WAV2VEC_ARTIFACTS}
        if {entry.name for entry in candidate.iterdir()} != expected:
            raise ValueError
        for artifact in WAV2VEC_ARTIFACTS:
            file = candidate / artifact.name
            file_metadata = file.stat(follow_symlinks=False)
            if file.is_symlink() or not _private(file_metadata, 0o600, directory=False):
                raise ValueError
            if file_metadata.st_size != artifact.size or _sha256(file) != artifact.sha256:
                raise ValueError
        return PreparedWav2VecBase(candidate.resolve(strict=True))
    except Exception:
        raise ValueError("invalid_wav2vec_base") from None


def prepare_wav2vec_base(
    output: str | Path,
    *,
    downloader: Downloader = _download,
) -> PreparedWav2VecBase:
    """Download, validate, and atomically publish the pinned base directory."""

    staging: Path | None = None
    try:
        target = _reject_untrusted_path(output, must_exist=False)
        if target.exists():
            raise ValueError
        parent = target.parent
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        _reject_untrusted_path(parent, must_exist=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=parent))
        os.chmod(staging, 0o700)
        for artifact in WAV2VEC_ARTIFACTS:
            destination = staging / artifact.name
            downloader(_revision_url(artifact.name), destination)
            os.chmod(destination, 0o600)
        validate_wav2vec_base(staging)
        if target.exists():
            raise ValueError
        os.replace(staging, target)
        staging = None
        if os.name == "posix":
            descriptor = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return validate_wav2vec_base(target)
    except Exception:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        raise ValueError("wav2vec_base_preparation_failed") from None
