"""Privacy-minimized RunPod data packaging and metadata-only holdout sealing."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import stat
import subprocess
import tarfile
import wave
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator
from voxdelta.domain.models import EmotionLabel
from voxdelta.evaluation.manifest import DatasetItem, load_manifest, read_trusted_regular_file

from voxdelta_runpod.config import CANONICAL_LABELS, ExperimentConfig

AuditOutcome = Literal["agree", "ambiguous", "disagree", "audio_defect"]
PackageKind = Literal["train-validation", "final-holdout"]
AUDIT_OUTCOMES: tuple[AuditOutcome, ...] = (
    "agree",
    "ambiguous",
    "disagree",
    "audio_defect",
)


class PackageError(ValueError):
    """Stable package failure that never embeds a private path or item identity."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class SanitizedRecord(_FrozenModel):
    item_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    audio_path: str = Field(pattern=r"^audio/[0-9a-f]{64}\.wav$")
    split: Literal["train", "validation", "test"]
    emotion: EmotionLabel
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def key_matches_path(self) -> SanitizedRecord:
        if self.audio_path != f"audio/{self.item_key}.wav":
            raise ValueError("invalid relative audio path")
        return self


class PackageSidecar(_FrozenModel):
    schema_version: Literal["1"] = "1"
    package_kind: PackageKind
    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    archive_bytes: int = Field(gt=0)
    audio_file_count: int = Field(gt=0)
    archive_member_count: int = Field(gt=0)
    split_counts: dict[str, int]
    label_counts: dict[EmotionLabel, int]
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def consistent_counts(self) -> PackageSidecar:
        if (
            self.archive_member_count != self.audio_file_count + 1
            or sum(self.split_counts.values()) != self.audio_file_count
            or sum(self.label_counts.values()) != self.audio_file_count
            or any(value <= 0 for value in self.split_counts.values())
            or any(value <= 0 for value in self.label_counts.values())
        ):
            raise ValueError("invalid package counts")
        return self


class HoldoutIdentity(_FrozenModel):
    schema_version: Literal["1"] = "1"
    nominal_test_count: int = Field(gt=0)
    exposed_count: int = Field(gt=0)
    final_holdout_count: int = Field(gt=0)
    exposed_label_counts: dict[EmotionLabel, int]
    holdout_label_counts: dict[EmotionLabel, int]
    identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def consistent_counts(self) -> HoldoutIdentity:
        if (
            self.nominal_test_count - self.exposed_count != self.final_holdout_count
            or sum(self.exposed_label_counts.values()) != self.exposed_count
            or sum(self.holdout_label_counts.values()) != self.final_holdout_count
        ):
            raise ValueError("invalid holdout counts")
        return self


class AuditQueueRecord(_FrozenModel):
    item_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: Literal["train", "validation"]
    emotion: EmotionLabel
    audio_path: str
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class AuditDecision(_FrozenModel):
    item_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    outcome: AuditOutcome
    note: str | None = Field(default=None, max_length=2_000)


class AuditSummary(_FrozenModel):
    schema_version: Literal["1"] = "1"
    total_count: int = Field(gt=0)
    label_outcome_counts: dict[EmotionLabel, dict[AuditOutcome, int]]

    @model_validator(mode="after")
    def consistent_total(self) -> AuditSummary:
        if sum(sum(outcomes.values()) for outcomes in self.label_outcome_counts.values()) != (
            self.total_count
        ):
            raise ValueError("invalid audit summary")
        return self


@dataclass(frozen=True, slots=True)
class _PackageItem:
    source: DatasetItem
    record: SanitizedRecord


def semantic_fingerprint(item: DatasetItem) -> str:
    """Return the immutable audio-content fingerprint used for exclusion and deduplication."""

    return item.sha256


def opaque_item_key(item: DatasetItem) -> str:
    if item.emotion not in CANONICAL_LABELS:
        raise PackageError("invalid_package_manifest")
    payload = (
        f"voxdelta-runpod-item-v1\0{semantic_fingerprint(item)}\0{item.split}\0{item.emotion}"
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _emotion_items(path: Path) -> list[DatasetItem]:
    if not path.is_absolute():
        raise PackageError("invalid_package_manifest")
    try:
        items = load_manifest(path)
    except Exception:
        raise PackageError("invalid_package_manifest") from None
    if any(item.source != "emotion" or item.emotion not in CANONICAL_LABELS for item in items):
        raise PackageError("invalid_package_manifest")
    fingerprints = [semantic_fingerprint(item) for item in items]
    if len(set(fingerprints)) != len(fingerprints):
        raise PackageError("duplicate_semantic_fingerprint")
    return items


def _identity_payload(items: Iterable[DatasetItem]) -> bytes:
    rows = sorted((semantic_fingerprint(item), cast(EmotionLabel, item.emotion)) for item in items)
    return json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode()


def derive_holdout_identity(
    source_manifest: Path,
    exposed_manifest: Path,
    config: ExperimentConfig,
) -> HoldoutIdentity:
    """Seal final-holdout identity from manifest metadata without opening any test audio."""

    source = _emotion_items(source_manifest)
    exposed_source = _emotion_items(exposed_manifest)
    test = [item for item in source if item.split == "test"]
    exposed = [item for item in exposed_source if item.split == "test"]
    exposed_fingerprints = {semantic_fingerprint(item) for item in exposed}
    if (
        len(test) != config.data.nominal_test_count
        or len(exposed) != config.data.exposed_test_count
    ):
        raise PackageError("invalid_holdout_counts")
    if len(exposed_fingerprints) != config.data.exposed_test_count:
        raise PackageError("invalid_exposed_holdout")
    test_by_fingerprint = {semantic_fingerprint(item): item for item in test}
    if not exposed_fingerprints <= set(test_by_fingerprint):
        raise PackageError("invalid_exposed_holdout")
    exposed_counts = Counter(
        cast(EmotionLabel, test_by_fingerprint[fingerprint].emotion)
        for fingerprint in exposed_fingerprints
    )
    if exposed_counts != Counter({label: 5 for label in CANONICAL_LABELS}):
        raise PackageError("invalid_exposed_holdout")
    holdout = [item for item in test if semantic_fingerprint(item) not in exposed_fingerprints]
    expected_holdout_counts: Counter[EmotionLabel] = Counter(
        {
            "anger": 695,
            "disgust": 212,
            "fear": 216,
            "happiness": 330,
            "neutral": 578,
            "sadness": 1_485,
            "surprise": 69,
        }
    )
    holdout_counts = Counter(cast(EmotionLabel, item.emotion) for item in holdout)
    if len(holdout) != config.data.final_holdout_count or holdout_counts != expected_holdout_counts:
        raise PackageError("invalid_holdout_counts")
    return HoldoutIdentity(
        nominal_test_count=len(test),
        exposed_count=len(exposed),
        final_holdout_count=len(holdout),
        exposed_label_counts=dict(exposed_counts),
        holdout_label_counts=dict(holdout_counts),
        identity_sha256=hashlib.sha256(_identity_payload(holdout)).hexdigest(),
    )


def _audit_rank(item: DatasetItem, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}\0{semantic_fingerprint(item)}".encode()).hexdigest()
    return digest, semantic_fingerprint(item)


def select_audit_queue(
    source_manifest: Path,
    validation_losses: Mapping[str, float],
    *,
    seed: int = 622,
    per_label: int = 15,
) -> tuple[AuditQueueRecord, ...]:
    """Select random train and highest-loss validation audit rows without publishing them."""

    if isinstance(seed, bool) or seed <= 0 or isinstance(per_label, bool) or per_label <= 0:
        raise PackageError("invalid_audit_inputs")
    items = _emotion_items(source_manifest)
    validation_fingerprints = {
        semantic_fingerprint(item) for item in items if item.split == "validation"
    }
    if set(validation_losses) != validation_fingerprints or any(
        not math.isfinite(value) or value < 0 for value in validation_losses.values()
    ):
        raise PackageError("invalid_audit_losses")
    selected: list[DatasetItem] = []
    for label in CANONICAL_LABELS:
        train = sorted(
            (item for item in items if item.split == "train" and item.emotion == label),
            key=lambda item: _audit_rank(item, seed),
        )
        validation = sorted(
            (item for item in items if item.split == "validation" and item.emotion == label),
            key=lambda item: (
                -validation_losses[semantic_fingerprint(item)],
                *_audit_rank(item, seed),
            ),
        )
        if len(train) < per_label or len(validation) < per_label:
            raise PackageError("invalid_audit_counts")
        selected.extend(train[:per_label])
        selected.extend(validation[:per_label])
    queue = tuple(
        AuditQueueRecord(
            item_key=opaque_item_key(item),
            split=cast(Literal["train", "validation"], item.split),
            emotion=cast(EmotionLabel, item.emotion),
            audio_path=item.audio_path,
            audio_sha256=item.sha256,
        )
        for item in selected
    )
    if len({item.item_key for item in queue}) != len(queue):
        raise PackageError("invalid_audit_counts")
    return queue


def summarize_audit(
    queue: Sequence[AuditQueueRecord], decisions: Sequence[AuditDecision]
) -> AuditSummary:
    if len(queue) != len(decisions) or len({item.item_key for item in queue}) != len(queue):
        raise PackageError("invalid_audit_decisions")
    decision_by_key = {decision.item_key: decision for decision in decisions}
    if len(decision_by_key) != len(decisions) or set(decision_by_key) != {
        item.item_key for item in queue
    }:
        raise PackageError("invalid_audit_decisions")
    counts: dict[EmotionLabel, dict[AuditOutcome, int]] = {
        label: {outcome: 0 for outcome in AUDIT_OUTCOMES} for label in CANONICAL_LABELS
    }
    for item in queue:
        counts[item.emotion][decision_by_key[item.item_key].outcome] += 1
    return AuditSummary(total_count=len(queue), label_outcome_counts=counts)


def _sanitized_items(items: Sequence[DatasetItem]) -> tuple[_PackageItem, ...]:
    packaged: list[_PackageItem] = []
    paths: set[str] = set()
    keys: set[str] = set()
    for item in items:
        if item.emotion not in CANONICAL_LABELS:
            raise PackageError("invalid_package_manifest")
        key = opaque_item_key(item)
        record = SanitizedRecord(
            item_key=key,
            audio_path=f"audio/{key}.wav",
            split=item.split,
            emotion=item.emotion,
            audio_sha256=item.sha256,
        )
        if key in keys or record.audio_path in paths:
            raise PackageError("duplicate_package_item")
        keys.add(key)
        paths.add(record.audio_path)
        packaged.append(_PackageItem(source=item, record=record))
    return tuple(sorted(packaged, key=lambda item: item.record.item_key))


def _validate_wav(item: _PackageItem) -> bytes:
    path = Path(item.source.audio_path)
    if not path.is_absolute() or path.suffix.lower() != ".wav":
        raise PackageError("invalid_package_audio")
    try:
        payload = read_trusted_regular_file(path)
        if hashlib.sha256(payload).hexdigest() != item.record.audio_sha256:
            raise ValueError
        with wave.open(io.BytesIO(payload), "rb") as stream:
            valid = (
                stream.getnchannels() == 1
                and stream.getsampwidth() == 2
                and stream.getframerate() == 16_000
                and stream.getcomptype() == "NONE"
                and stream.getnframes() > 0
            )
        if not valid:
            raise ValueError
    except Exception:
        raise PackageError("invalid_package_audio") from None
    return payload


def _manifest_payload(items: Sequence[_PackageItem]) -> bytes:
    return b"".join(
        record.record.model_dump_json(
            by_alias=False,
            exclude_none=False,
        ).encode()
        + b"\n"
        for record in items
    )


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o600
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def _write_archive(path: Path, items: Sequence[_PackageItem], manifest: bytes) -> None:
    with path.open("xb") as output:
        process = subprocess.Popen(
            ["zstd", "-19", "--threads=1", "--quiet", "--stdout"],
            stdin=subprocess.PIPE,
            stdout=output,
            stderr=subprocess.PIPE,
        )
        if process.stdin is None:
            raise PackageError("package_archive_failed")
        try:
            with tarfile.open(fileobj=process.stdin, mode="w|") as stream:
                stream.addfile(_tar_info("manifest.jsonl", len(manifest)), io.BytesIO(manifest))
                for item in items:
                    audio = _validate_wav(item)
                    stream.addfile(
                        _tar_info(item.record.audio_path, len(audio)),
                        io.BytesIO(audio),
                    )
            process.stdin.close()
            stderr = process.stderr.read() if process.stderr is not None else b""
            return_code = process.wait()
        except Exception:
            process.kill()
            process.wait()
            raise
        output.flush()
        os.fsync(output.fileno())
    if return_code != 0 or stderr:
        raise PackageError("package_archive_failed")
    path.chmod(0o600)


def _trusted_digest(path: Path) -> tuple[str, int]:
    if not path.is_absolute() or path.is_symlink():
        raise PackageError("invalid_package_file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise PackageError("invalid_package_file") from None
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PackageError("invalid_package_file")
        while payload := os.read(descriptor, 1024 * 1024):
            digest.update(payload)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise PackageError("invalid_package_file")
    return digest.hexdigest(), before.st_size


def _private_write(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _reject_symlink_components(path: Path, code: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise PackageError(code)


def build_train_validation_package(
    source_manifest: Path,
    output_root: Path,
    config: ExperimentConfig,
) -> Path:
    """Publish the deterministic 33,045-audio training packet without private fields."""

    items = _emotion_items(source_manifest)
    selected = [item for item in items if item.split in ("train", "validation")]
    counts = Counter(item.split for item in selected)
    if counts != Counter(
        {"train": config.data.train_count, "validation": config.data.validation_count}
    ):
        raise PackageError("invalid_training_counts")
    return _publish_package(
        selected,
        output_root,
        package_kind="train-validation",
        archive_name="train-validation.tar.zst",
        sidecar_name="train-validation.sidecar.json",
    )


def _publish_package(
    selected: Sequence[DatasetItem],
    output_root: Path,
    *,
    package_kind: PackageKind,
    archive_name: str,
    sidecar_name: str,
) -> Path:
    if not output_root.is_absolute():
        raise PackageError("invalid_package_target")
    _reject_symlink_components(output_root, "invalid_package_target")
    output_root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _reject_symlink_components(output_root.parent, "invalid_package_target")
    if output_root.exists():
        raise PackageError("package_exists")
    staging = output_root.with_name(f".{output_root.name}.staging")
    if staging.exists():
        raise PackageError("package_exists")
    staging.mkdir(mode=0o700)
    try:
        packaged = _sanitized_items(selected)
        manifest = _manifest_payload(packaged)
        archive = staging / archive_name
        _write_archive(archive, packaged, manifest)
        archive_sha256, archive_bytes = _trusted_digest(archive.resolve())
        sidecar = PackageSidecar(
            package_kind=package_kind,
            archive_sha256=archive_sha256,
            archive_bytes=archive_bytes,
            audio_file_count=len(packaged),
            archive_member_count=len(packaged) + 1,
            split_counts=dict(Counter(item.record.split for item in packaged)),
            label_counts=dict(Counter(item.record.emotion for item in packaged)),
            manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        )
        _private_write(
            staging / sidecar_name,
            (sidecar.model_dump_json() + "\n").encode(),
        )
        os.replace(staging, output_root)
        parent = os.open(output_root.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_root


def write_holdout_identity(path: Path, identity: HoldoutIdentity) -> None:
    if not path.is_absolute():
        raise PackageError("holdout_identity_exists")
    _reject_symlink_components(path, "holdout_identity_exists")
    if path.exists():
        raise PackageError("holdout_identity_exists")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _reject_symlink_components(path.parent, "holdout_identity_exists")
    _private_write(path, (identity.model_dump_json() + "\n").encode())


def write_audit_queue(path: Path, queue: Sequence[AuditQueueRecord]) -> None:
    if not path.is_absolute():
        raise PackageError("audit_queue_exists")
    _reject_symlink_components(path, "audit_queue_exists")
    if path.exists():
        raise PackageError("audit_queue_exists")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _reject_symlink_components(path.parent, "audit_queue_exists")
    payload = b"".join((item.model_dump_json() + "\n").encode() for item in queue)
    _private_write(path, payload)


def archive_members_are_safe(names: Iterable[str]) -> bool:
    for name in names:
        path = PurePosixPath(name)
        if not name or path.is_absolute() or ".." in path.parts:
            return False
        if name != "manifest.jsonl" and not re_full_audio_path(name):
            return False
    return True


def re_full_audio_path(value: str) -> bool:
    if not value.startswith("audio/") or not value.endswith(".wav"):
        return False
    key = value.removeprefix("audio/").removesuffix(".wav")
    return len(key) == 64 and all(character in "0123456789abcdef" for character in key)


__all__ = [
    "AuditDecision",
    "AuditQueueRecord",
    "AuditSummary",
    "HoldoutIdentity",
    "PackageError",
    "PackageSidecar",
    "SanitizedRecord",
    "archive_members_are_safe",
    "build_train_validation_package",
    "derive_holdout_identity",
    "opaque_item_key",
    "select_audit_queue",
    "semantic_fingerprint",
    "summarize_audit",
    "write_audit_queue",
    "write_holdout_identity",
]
