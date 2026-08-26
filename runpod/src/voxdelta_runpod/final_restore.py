"""Build the minimal authenticated state needed to evaluate final on a fresh Pod."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tarfile
from pathlib import Path

from voxdelta.evaluation.manifest import read_trusted_regular_file
from voxdelta.providers._emotion_runtime import validate_checkpoint

from voxdelta_runpod.ledger import load_ledger
from voxdelta_runpod.workflow import (
    load_aggregate_report,
    load_run_identity,
    verify_result_checksums,
)


class FinalRestoreError(ValueError):
    pass


_RESULT_FILES = frozenset({"SHA256SUMS", "report.json"})
_CHECKPOINT_FILES = frozenset(
    {"config.json", "label_mapping.json", "metrics.json", "model.safetensors"}
)
_LEDGER_RECORD = re.compile(r"^record-[0-9]{6}\.json$")


def _private_directory(path: Path) -> tuple[Path, ...]:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise FinalRestoreError("invalid_final_restore_source")
    entries = tuple(path.iterdir())
    if path.stat(follow_symlinks=False).st_mode & 0o077:
        raise FinalRestoreError("invalid_final_restore_source")
    return entries


def _private_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file() or path.stat(follow_symlinks=False).st_mode & 0o077:
        raise FinalRestoreError("invalid_final_restore_source")
    read_trusted_regular_file(path)


def _sources(
    full_result_root: Path,
    ledger_root: Path,
    run_identity: Path,
) -> tuple[tuple[str, Path], ...]:
    full_result_root = full_result_root.resolve()
    ledger_root = ledger_root.resolve()
    run_identity = run_identity.resolve()

    result_entries = _private_directory(full_result_root)
    result_files = {entry.name for entry in result_entries if entry.is_file()}
    result_directories = {entry.name for entry in result_entries if entry.is_dir()}
    if result_files != _RESULT_FILES or result_directories != {"checkpoint"}:
        raise FinalRestoreError("invalid_final_restore_source")

    checkpoint_root = full_result_root / "checkpoint"
    checkpoint_entries = _private_directory(checkpoint_root)
    if {entry.name for entry in checkpoint_entries} != _CHECKPOINT_FILES:
        raise FinalRestoreError("invalid_final_restore_source")
    for entry in checkpoint_entries:
        _private_file(entry)
    for name in _RESULT_FILES:
        _private_file(full_result_root / name)

    ledger_entries = _private_directory(ledger_root)
    if not ledger_entries or any(
        not _LEDGER_RECORD.fullmatch(entry.name) or not entry.is_file() for entry in ledger_entries
    ):
        raise FinalRestoreError("invalid_final_restore_source")
    for entry in ledger_entries:
        _private_file(entry)
    _private_file(run_identity)

    verify_result_checksums(full_result_root)
    report = load_aggregate_report(full_result_root / "report.json")
    checkpoint = validate_checkpoint(
        checkpoint_root,
        architecture="wav2vec-xls-r",
        model_id="facebook/wav2vec2-xls-r-300m",
    )
    records = load_ledger(ledger_root)
    identity = load_run_identity(run_identity)
    if (
        checkpoint.digest != report.checkpoint_sha256
        or records[-1].to_stage != "full"
        or records[-1].identity != identity
    ):
        raise FinalRestoreError("invalid_final_restore_source")

    files = [(f"results/full/{name}", full_result_root / name) for name in sorted(_RESULT_FILES)]
    files.extend(
        (f"results/full/checkpoint/{name}", checkpoint_root / name)
        for name in sorted(_CHECKPOINT_FILES)
    )
    files.extend((f"ledger/{entry.name}", entry) for entry in sorted(ledger_entries))
    files.append(("state/run-identity.json", run_identity))
    return tuple(sorted(files))


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_final_restore_bundle(
    full_result_root: Path,
    ledger_root: Path,
    run_identity: Path,
    output: Path,
) -> tuple[Path, Path]:
    """Publish a deterministic private restore archive plus its checksum."""

    output = output.resolve()
    archive = output / "full-state.tar.zst"
    checksum = output / "full-state.tar.zst.sha256"
    staging = output / ".full-state.tar.zst.staging"
    if any(path.exists() or path.is_symlink() for path in (archive, checksum, staging)):
        raise FinalRestoreError("final_restore_exists")
    files = _sources(full_result_root, ledger_root, run_identity)
    try:
        with staging.open("xb") as target:
            os.fchmod(target.fileno(), 0o600)
            process = subprocess.Popen(
                ["zstd", "-3", "--threads=1", "--quiet", "--stdout"],
                stdin=subprocess.PIPE,
                stdout=target,
                stderr=subprocess.PIPE,
            )
            if process.stdin is None:
                process.kill()
                raise FinalRestoreError("final_restore_failed")
            with tarfile.open(fileobj=process.stdin, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                for relative, source in files:
                    metadata = source.stat(follow_symlinks=False)
                    info = tarfile.TarInfo(relative)
                    info.size = metadata.st_size
                    info.mode = 0o600
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    with source.open("rb") as stream:
                        tar.addfile(info, stream)
            process.stdin.close()
            stderr = process.stderr.read() if process.stderr is not None else b""
            if process.wait() != 0 or stderr:
                raise FinalRestoreError("final_restore_failed")
        os.replace(staging, archive)
        with checksum.open("xb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(f"{_digest(archive)}  {archive.name}\n".encode())
    except Exception:
        staging.unlink(missing_ok=True)
        archive.unlink(missing_ok=True)
        checksum.unlink(missing_ok=True)
        raise
    return archive, checksum


__all__ = ["FinalRestoreError", "build_final_restore_bundle"]
