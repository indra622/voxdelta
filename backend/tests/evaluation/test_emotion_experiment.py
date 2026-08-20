from __future__ import annotations

import hashlib
import json
import stat
from collections import Counter
from pathlib import Path
from typing import cast

import pytest

from voxdelta.domain.models import (
    EmotionLabel,
    EmotionResult,
    ProviderProvenance,
    ProviderUsage,
)
from voxdelta.evaluation.emotion_experiment import (
    EmotionExperimentReport,
    build_stratified_smoke_manifest,
    evaluate_emotion_checkpoint,
    write_experiment_report,
)
from voxdelta.evaluation.manifest import DatasetItem, DatasetSplit, load_manifest
from voxdelta.providers.base import ProviderError

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


class FakeProvider:
    def __init__(self, *, fail_all: bool = False) -> None:
        self.provenance = ProviderProvenance(
            name="emotion2vec-plus",
            model="emotion2vec-plus-large-seven-emotion@v2.0.4",
            remote=False,
            revision="a" * 64,
        )
        self.fail_all = fail_all
        self.unloaded = False

    def analyze(self, utterance_id: str, audio_path: Path, transcript: str) -> EmotionResult:
        del audio_path, transcript
        if self.fail_all:
            raise ProviderError("provider_unavailable")
        expected = cast(EmotionLabel, utterance_id.split("-")[1])
        probabilities = {label: 0.01 for label in LABELS}
        probabilities[expected] = 0.94
        return EmotionResult(
            utterance_id=utterance_id,
            probabilities=probabilities,
            operational_state="stable",
            negative_intensity=sum(
                probabilities[label] for label in ("anger", "disgust", "fear", "sadness")
            ),
            confidence=0.94,
            provider=self.provenance,
            usage=ProviderUsage(latency_ms=12.0, peak_rss_mb=321.0),
        )

    def unload(self) -> None:
        self.unloaded = True


def _smoke_evaluation_manifest(root: Path) -> Path:
    source = _write_source_manifest(root / "source")
    smoke = root / "smoke.jsonl"
    build_stratified_smoke_manifest(
        source,
        smoke,
        train_per_label=1,
        validation_per_label=1,
        test_per_label=1,
    )
    return smoke


def _report_fixture() -> EmotionExperimentReport:
    return EmotionExperimentReport(
        architecture="emotion2vec-plus",
        model_id="emotion2vec-plus-large-seven-emotion@v2.0.4",
        checkpoint_digest="a" * 64,
        manifest_digest="b" * 64,
        test_count=7,
        completed_count=7,
        completion_rate=1.0,
        macro_f1=1.0,
        per_label_f1={label: 1.0 for label in LABELS},
        confusion_matrix=tuple(
            tuple(1 if row == column else 0 for column in range(7)) for row in range(7)
        ),
        expected_calibration_error=0.06,
        median_latency_ms=12.0,
        peak_rss_mb=321.0,
        elapsed_seconds=1.0,
        requested_device="auto",
    )


def test_evaluation_report_has_aggregate_metrics_without_item_content(tmp_path: Path) -> None:
    manifest = _smoke_evaluation_manifest(tmp_path)
    provider = FakeProvider()

    report = evaluate_emotion_checkpoint(
        manifest,
        tmp_path / "checkpoint",
        architecture="emotion2vec-plus",
        provider_factory=lambda _checkpoint, _device: provider,
    )

    assert provider.unloaded is True
    assert report.test_count == 7
    assert report.completed_count == 7
    assert report.completion_rate == 1.0
    assert report.macro_f1 == 1.0
    assert report.per_label_f1 == {label: 1.0 for label in LABELS}
    assert report.confusion_matrix == tuple(
        tuple(1 if row == column else 0 for column in range(7)) for row in range(7)
    )
    assert report.expected_calibration_error == pytest.approx(0.06)
    assert report.median_latency_ms == 12.0
    assert report.peak_rss_mb == 321.0
    serialized = report.model_dump_json()
    for forbidden in ("transcript", "audio_path", "train-happiness-", "test-surprise-"):
        assert forbidden not in serialized


def test_report_writer_is_atomic_private_and_refuses_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    write_experiment_report(output, _report_fixture())

    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    original = output.read_bytes()
    assert original.endswith(b"\n")
    with pytest.raises(ValueError, match="experiment_report_publication_failed"):
        write_experiment_report(output, _report_fixture())
    assert output.read_bytes() == original


def test_evaluation_rejects_hash_mismatch(tmp_path: Path) -> None:
    manifest = _smoke_evaluation_manifest(tmp_path)
    test_item = next(item for item in load_manifest(manifest) if item.split == "test")
    Path(test_item.audio_path).write_bytes(b"changed")

    with pytest.raises(ValueError, match="invalid_experiment_manifest"):
        evaluate_emotion_checkpoint(
            manifest,
            tmp_path / "checkpoint",
            architecture="emotion2vec-plus",
            provider_factory=lambda _checkpoint, _device: FakeProvider(),
        )


def test_evaluation_rejects_zero_completed_items_and_unloads(tmp_path: Path) -> None:
    manifest = _smoke_evaluation_manifest(tmp_path)
    provider = FakeProvider(fail_all=True)

    with pytest.raises(ValueError, match="invalid_experiment_report"):
        evaluate_emotion_checkpoint(
            manifest,
            tmp_path / "checkpoint",
            architecture="emotion2vec-plus",
            provider_factory=lambda _checkpoint, _device: provider,
        )

    assert provider.unloaded is True
