from __future__ import annotations

import hashlib
import json
import stat
import subprocess
import sys
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
    build_partial_last4_development_manifest,
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
BACKEND = Path(__file__).resolve().parents[2]


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


def test_build_development_manifest_is_balanced_deterministic_and_test_free(
    tmp_path: Path,
) -> None:
    source = _write_source_manifest(
        tmp_path / "source",
        train_count=101,
        validation_count=26,
        test_count=2,
    )
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    summary = build_partial_last4_development_manifest(source, first)
    build_partial_last4_development_manifest(source, second)

    assert summary.total_count == 875
    assert summary.split_counts == {"train": 700, "validation": 175}
    selected = load_manifest(first)
    assert Counter((item.split, item.emotion) for item in selected) == Counter(
        {
            **{("train", label): 100 for label in LABELS},
            **{("validation", label): 25 for label in LABELS},
        }
    )
    assert all(item.split != "test" for item in selected)
    assert {item.transcript for item in selected} == {""}
    assert first.read_bytes() == second.read_bytes()
    assert stat.S_IMODE(first.stat().st_mode) == 0o600


def test_build_development_manifest_refuses_existing_output(tmp_path: Path) -> None:
    source = _write_source_manifest(
        tmp_path / "source",
        train_count=101,
        validation_count=26,
        test_count=2,
    )
    output = tmp_path / "existing.jsonl"
    output.write_text("owner-data", encoding="utf-8")

    with pytest.raises(ValueError, match="development_manifest_publication_failed"):
        build_partial_last4_development_manifest(source, output)
    assert output.read_text(encoding="utf-8") == "owner-data"


class FakeProvider:
    def __init__(self, *, fail_all: bool = False) -> None:
        self.provenance = ProviderProvenance(
            name="emotion2vec-plus",
            model="emotion2vec-plus-large-seven-emotion@v2.0.5",
            remote=False,
            revision="a" * 64,
        )
        self.fail_all = fail_all
        self.unloaded = False
        self.calls: list[str] = []

    def analyze(self, utterance_id: str, audio_path: Path, transcript: str) -> EmotionResult:
        del audio_path, transcript
        self.calls.append(utterance_id)
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
        model_id="emotion2vec-plus-large-seven-emotion@v2.0.5",
        checkpoint_digest="a" * 64,
        manifest_digest="b" * 64,
        split="test",
        item_count=7,
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
    assert report.split == "test"
    assert report.item_count == 7
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


def test_evaluation_can_select_validation_without_opening_test(tmp_path: Path) -> None:
    manifest = _smoke_evaluation_manifest(tmp_path)
    provider = FakeProvider()

    report = evaluate_emotion_checkpoint(
        manifest,
        tmp_path / "checkpoint",
        architecture="emotion2vec-plus",
        split="validation",
        provider_factory=lambda _checkpoint, _device: provider,
    )

    assert report.split == "validation"
    assert report.item_count == 7
    assert provider.calls
    assert all(identifier.startswith("validation-") for identifier in provider.calls)
    assert not any(identifier.startswith("test-") for identifier in provider.calls)


def test_evaluation_cli_accepts_only_explicit_validation_or_test_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(BACKEND))
    from scripts.evaluate_emotion_checkpoint import _parser

    arguments = _parser().parse_args(
        [
            "--manifest",
            "/manifest.jsonl",
            "--checkpoint",
            "/checkpoint",
            "--output",
            "/report.json",
            "--architecture",
            "emotion2vec-plus",
            "--split",
            "validation",
        ]
    )

    assert arguments.split == "validation"
    with pytest.raises(ValueError, match="invalid arguments"):
        _parser().parse_args(
            [
                "--manifest",
                "/manifest.jsonl",
                "--checkpoint",
                "/checkpoint",
                "--output",
                "/report.json",
                "--architecture",
                "emotion2vec-plus",
                "--split",
                "private-secret",
            ]
        )


def test_default_xls_r_evaluation_forwards_explicit_base_model_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import voxdelta.evaluation.emotion_experiment as module

    manifest = _smoke_evaluation_manifest(tmp_path)
    provider = FakeProvider()
    base = tmp_path / "base"
    observed: list[tuple[Path, Path, str]] = []

    def wav2vec_provider(checkpoint: Path, *, base_model_path: Path, device: str) -> FakeProvider:
        observed.append((checkpoint, base_model_path, device))
        return provider

    monkeypatch.setattr(module, "Wav2VecEmotionProvider", wav2vec_provider)

    report = evaluate_emotion_checkpoint(
        manifest,
        tmp_path / "checkpoint",
        architecture="wav2vec-xls-r",
        base_model_path=base,
        device="mps",
        split="validation",
    )

    assert report.completed_count == 7
    assert observed == [(tmp_path / "checkpoint", base, "mps")]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--architecture", "wav2vec-xls-r"],
        ["--architecture", "emotion2vec-plus", "--base-model-path", "/base"],
    ],
)
def test_evaluation_cli_rejects_missing_or_incompatible_base_path(
    arguments: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(BACKEND))
    import scripts.evaluate_emotion_checkpoint as cli

    called = False

    def reject_call(*_args: object, **_kwargs: object) -> EmotionExperimentReport:
        nonlocal called
        called = True
        return _report_fixture()

    monkeypatch.setattr(cli, "evaluate_emotion_checkpoint", reject_call)
    result = cli.main(
        [
            "--manifest",
            "/manifest.jsonl",
            "--checkpoint",
            "/checkpoint",
            "--output",
            "/report.json",
            *arguments,
        ]
    )

    assert result == 2
    assert called is False


def test_report_writer_is_atomic_private_and_refuses_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    write_experiment_report(output, _report_fixture())

    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    original = output.read_bytes()
    assert original.endswith(b"\n")
    with pytest.raises(ValueError, match="experiment_report_publication_failed"):
        write_experiment_report(output, _report_fixture())
    assert output.read_bytes() == original


def test_report_writer_never_uses_an_overwriting_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_overwrite(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("overwriting rename is forbidden")

    monkeypatch.setattr("voxdelta.evaluation.emotion_experiment.os.replace", reject_overwrite)
    output = tmp_path / "report.json"

    write_experiment_report(output, _report_fixture())

    assert output.is_file()
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


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


def test_build_cli_writes_summary_without_private_content(tmp_path: Path) -> None:
    source = _write_source_manifest(tmp_path / "source")
    output = tmp_path / "smoke.jsonl"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_emotion_smoke_manifest.py",
            "--manifest",
            str(source),
            "--output",
            str(output),
            "--train-per-label",
            "2",
            "--validation-per-label",
            "1",
            "--test-per-label",
            "1",
            "--seed",
            "622",
        ],
        cwd=BACKEND,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert completed.stdout == "smoke manifest: 28 items\n"
    assert output.is_file()
    assert str(source) not in completed.stdout


def test_development_manifest_cli_failure_is_sanitized(tmp_path: Path) -> None:
    private_manifest = tmp_path / "private-source-item-id.jsonl"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_emotion_development_manifest.py",
            "--manifest",
            str(private_manifest),
            "--output",
            str(tmp_path / "development.jsonl"),
        ],
        cwd=BACKEND,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == "development manifest failed\n"
    assert "private" not in completed.stderr


def test_evaluation_cli_failure_is_sanitized(tmp_path: Path) -> None:
    private_manifest = tmp_path / "private-item-id.jsonl"
    private_checkpoint = tmp_path / "private-checkpoint"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_emotion_checkpoint.py",
            "--manifest",
            str(private_manifest),
            "--checkpoint",
            str(private_checkpoint),
            "--output",
            str(tmp_path / "report.json"),
            "--architecture",
            "emotion2vec-plus",
        ],
        cwd=BACKEND,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == "emotion evaluation failed\n"
    assert "private" not in completed.stderr
