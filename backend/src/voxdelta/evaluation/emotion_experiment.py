"""Deterministic, privacy-minimized real-data emotion experiments."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict, Field

from voxdelta.domain.models import EmotionLabel
from voxdelta.evaluation.manifest import (
    DatasetItem,
    DatasetSplit,
    load_manifest,
    validate_disjoint_splits,
)

CANONICAL_LABELS: tuple[EmotionLabel, ...] = (
    "happiness",
    "anger",
    "disgust",
    "fear",
    "neutral",
    "sadness",
    "surprise",
)


class SmokeManifestSummary(BaseModel):
    """Aggregate identity of one deterministic smoke manifest."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    total_count: int = Field(gt=0)
    split_counts: dict[DatasetSplit, int]
    label_counts: dict[EmotionLabel, int]
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _rank(item: DatasetItem, seed: int) -> tuple[str, str]:
    payload = f"{seed}\0{item.id}\0{item.sha256}".encode()
    return hashlib.sha256(payload).hexdigest(), item.id


def _reject_symlink_components(path: Path, error_code: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(error_code)


def _publish_private_file(path: Path, payload: bytes, error_code: str) -> None:
    if not path.is_absolute():
        raise ValueError(error_code)
    target = Path(os.path.abspath(path.expanduser()))
    _reject_symlink_components(target, error_code)
    if target.exists():
        raise ValueError(error_code)

    temporary: Path | None = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        _reject_symlink_components(target.parent, error_code)
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            os.fchmod(handle.fileno(), 0o600)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists():
            raise OSError
        os.replace(temporary, target)
        temporary = None
        if os.name == "posix":
            descriptor = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise ValueError(error_code) from None


def _smoke_payload(items: list[DatasetItem]) -> bytes:
    return b"".join(
        json.dumps(
            item.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        + b"\n"
        for item in sorted(items, key=lambda item: item.id)
    )


def build_stratified_smoke_manifest(
    source: Path,
    output: Path,
    *,
    train_per_label: int = 20,
    validation_per_label: int = 5,
    test_per_label: int = 5,
    seed: int = 622,
) -> SmokeManifestSummary:
    """Publish a balanced deterministic subset of a trusted emotion manifest."""

    requested: dict[DatasetSplit, int] = {
        "train": train_per_label,
        "validation": validation_per_label,
        "test": test_per_label,
    }
    if (
        not source.is_absolute()
        or any(isinstance(value, bool) or value <= 0 for value in requested.values())
        or isinstance(seed, bool)
        or seed <= 0
    ):
        raise ValueError("invalid_smoke_manifest")

    try:
        items = load_manifest(source)
        validate_disjoint_splits(items)
        cells: dict[tuple[DatasetSplit, EmotionLabel], list[DatasetItem]] = {}
        for item in items:
            if item.source != "emotion" or item.emotion not in CANONICAL_LABELS:
                raise ValueError
            label = item.emotion
            cells.setdefault((item.split, label), []).append(item)

        selected: list[DatasetItem] = []
        for split, count in requested.items():
            for label in CANONICAL_LABELS:
                candidates = sorted(
                    cells.get((split, label), ()),
                    key=lambda item: _rank(item, seed),
                )
                if len(candidates) < count:
                    raise ValueError
                selected.extend(
                    item.model_copy(update={"transcript": ""}) for item in candidates[:count]
                )
    except Exception:
        raise ValueError("invalid_smoke_manifest") from None

    payload = _smoke_payload(selected)
    _publish_private_file(output, payload, "smoke_manifest_publication_failed")
    return SmokeManifestSummary(
        total_count=len(selected),
        split_counts=dict(Counter(item.split for item in selected)),
        label_counts=dict(Counter(cast(EmotionLabel, item.emotion) for item in selected)),
        manifest_sha256=hashlib.sha256(payload).hexdigest(),
    )
