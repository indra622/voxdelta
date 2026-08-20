from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from voxdelta.domain.models import EmotionLabel
from voxdelta.evaluation.emotion_experiment import build_stratified_smoke_manifest
from voxdelta.evaluation.manifest import DatasetItem, DatasetSplit, load_manifest

LABELS: tuple[EmotionLabel, ...] = (
    "happiness",
    "anger",
    "disgust",
    "fear",
    "neutral",
    "sadness",
    "surprise",
)


def _write_source_manifest(
    root: Path,
    *,
    train_count: int = 3,
    validation_count: int = 2,
    test_count: int = 2,
    surprise_test_count: int | None = None,
) -> Path:
    root.mkdir(parents=True)
    records: list[DatasetItem] = []
    requested: dict[DatasetSplit, int] = {
        "train": train_count,
        "validation": validation_count,
        "test": test_count,
    }
    for split, count in requested.items():
        for label in LABELS:
            cell_count = (
                surprise_test_count
                if split == "test" and label == "surprise" and surprise_test_count is not None
                else count
            )
            for index in range(cell_count):
                identifier = f"{split}-{label}-{index}"
                audio = root / f"{identifier}.wav"
                payload = identifier.encode()
                audio.write_bytes(payload)
                records.append(
                    DatasetItem(
                        id=identifier,
                        call_id=f"call-{identifier}",
                        speaker_id=f"speaker-{identifier}",
                        audio_path=str(audio.resolve()),
                        transcript="must-not-survive",
                        split=split,
                        source="emotion",
                        emotion=label,
                        sha256=hashlib.sha256(payload).hexdigest(),
                    )
                )
    manifest = root / "emotion.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(item.model_dump(mode="json"), sort_keys=True, separators=(",", ":")) + "\n"
            for item in records
        ),
        encoding="utf-8",
    )
    return manifest


def _expected_cells(
    train: int, validation: int, test: int
) -> Counter[tuple[DatasetSplit, EmotionLabel]]:
    counts: Counter[tuple[DatasetSplit, EmotionLabel]] = Counter()
    for split, count in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        for label in LABELS:
            counts[(split, label)] = count
    return counts


def test_build_smoke_manifest_is_balanced_deterministic_and_transcript_free(
    tmp_path: Path,
) -> None:
    source = _write_source_manifest(tmp_path / "source")
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    summary = build_stratified_smoke_manifest(
        source,
        first,
        train_per_label=2,
        validation_per_label=1,
        test_per_label=1,
        seed=622,
    )
    build_stratified_smoke_manifest(
        source,
        second,
        train_per_label=2,
        validation_per_label=1,
        test_per_label=1,
        seed=622,
    )

    assert summary.total_count == 28
    assert summary.split_counts == {"train": 14, "validation": 7, "test": 7}
    assert summary.label_counts == {label: 4 for label in LABELS}
    assert summary.manifest_sha256 == hashlib.sha256(first.read_bytes()).hexdigest()
    assert first.read_bytes() == second.read_bytes()
    selected = load_manifest(first)
    assert Counter((item.split, item.emotion) for item in selected) == _expected_cells(2, 1, 1)
    assert {item.transcript for item in selected} == {""}
    assert "must-not-survive" not in first.read_text(encoding="utf-8")


def test_build_smoke_manifest_rejects_scarce_cells(tmp_path: Path) -> None:
    source = _write_source_manifest(tmp_path / "source", surprise_test_count=0)

    with pytest.raises(ValueError, match="invalid_smoke_manifest"):
        build_stratified_smoke_manifest(
            source,
            tmp_path / "smoke.jsonl",
            train_per_label=2,
            validation_per_label=1,
            test_per_label=1,
        )


def test_build_smoke_manifest_refuses_existing_output(tmp_path: Path) -> None:
    source = _write_source_manifest(tmp_path / "source")
    output = tmp_path / "existing.jsonl"
    output.write_text("owner data", encoding="utf-8")

    with pytest.raises(ValueError, match="smoke_manifest_publication_failed"):
        build_stratified_smoke_manifest(
            source,
            output,
            train_per_label=2,
            validation_per_label=1,
            test_per_label=1,
        )

    assert output.read_text(encoding="utf-8") == "owner data"
