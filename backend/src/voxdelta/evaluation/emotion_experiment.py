"""Deterministic, privacy-minimized real-data emotion experiments."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from voxdelta.domain.models import EmotionLabel
from voxdelta.evaluation.manifest import (
    DatasetItem,
    DatasetSplit,
    load_manifest,
    read_trusted_regular_file,
    validate_disjoint_splits,
)
from voxdelta.providers._emotion_runtime import Device
from voxdelta.providers.base import EmotionProvider, ProviderError
from voxdelta.providers.emotion2vec_emotion import Emotion2VecEmotionProvider
from voxdelta.providers.wav2vec_emotion import Wav2VecEmotionProvider

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


class EmotionExperimentReport(BaseModel):
    """Aggregate-only held-out metrics for one local emotion checkpoint."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    schema_version: Literal["1"] = "1"
    architecture: Literal["emotion2vec-plus", "wav2vec-xls-r"]
    model_id: str = Field(min_length=1)
    checkpoint_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    test_count: int = Field(gt=0)
    completed_count: int = Field(gt=0)
    completion_rate: float = Field(ge=0, le=1)
    macro_f1: float = Field(ge=0, le=1)
    per_label_f1: dict[EmotionLabel, float]
    confusion_matrix: tuple[tuple[int, ...], ...]
    expected_calibration_error: float = Field(ge=0, le=1)
    median_latency_ms: float = Field(ge=0)
    peak_rss_mb: float | None = Field(default=None, ge=0)
    elapsed_seconds: float = Field(ge=0)
    requested_device: Device

    @model_validator(mode="after")
    def valid_aggregate_contract(self) -> EmotionExperimentReport:
        if (
            self.completed_count > self.test_count
            or abs(self.completion_rate - self.completed_count / self.test_count) > 1e-12
            or set(self.per_label_f1) != set(CANONICAL_LABELS)
            or any(not math.isfinite(value) for value in self.per_label_f1.values())
            or any(not 0 <= value <= 1 for value in self.per_label_f1.values())
            or len(self.confusion_matrix) != len(CANONICAL_LABELS)
            or any(len(row) != len(CANONICAL_LABELS) for row in self.confusion_matrix)
            or any(value < 0 for row in self.confusion_matrix for value in row)
            or sum(value for row in self.confusion_matrix for value in row) != self.completed_count
        ):
            raise ValueError("invalid experiment report")
        return self


ProviderFactory = Callable[[Path, Device], EmotionProvider]


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
        os.link(temporary, target)
        try:
            temporary.unlink()
        except OSError:
            pass
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


def _aggregate_metrics(
    expected: Sequence[EmotionLabel],
    predicted: Sequence[EmotionLabel],
    confidence: Sequence[float],
) -> tuple[
    float,
    dict[EmotionLabel, float],
    tuple[tuple[int, ...], ...],
    float,
]:
    if not expected or len(expected) != len(predicted) or len(expected) != len(confidence):
        raise ValueError("invalid_experiment_report")
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in confidence):
        raise ValueError("invalid_experiment_report")

    matrix = [[0 for _ in CANONICAL_LABELS] for _ in CANONICAL_LABELS]
    for truth, guess in zip(expected, predicted, strict=True):
        matrix[CANONICAL_LABELS.index(truth)][CANONICAL_LABELS.index(guess)] += 1

    scores: dict[EmotionLabel, float] = {}
    for index, label in enumerate(CANONICAL_LABELS):
        true_positive = matrix[index][index]
        false_positive = sum(row[index] for row in matrix) - true_positive
        false_negative = sum(matrix[index]) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        scores[label] = 0.0 if denominator == 0 else 2 * true_positive / denominator
    macro_f1 = math.fsum(scores.values()) / len(CANONICAL_LABELS)

    bins: list[list[tuple[bool, float]]] = [[] for _ in range(10)]
    for truth, guess, score in zip(expected, predicted, confidence, strict=True):
        bins[min(int(score * 10), 9)].append((truth == guess, score))
    expected_calibration_error = math.fsum(
        len(bucket)
        / len(expected)
        * abs(
            math.fsum(float(correct) for correct, _ in bucket) / len(bucket)
            - math.fsum(score for _, score in bucket) / len(bucket)
        )
        for bucket in bins
        if bucket
    )
    return (
        macro_f1,
        scores,
        tuple(tuple(row) for row in matrix),
        expected_calibration_error,
    )


def _default_provider(
    checkpoint: Path,
    architecture: Literal["emotion2vec-plus", "wav2vec-xls-r"],
    device: Device,
) -> EmotionProvider:
    if architecture == "emotion2vec-plus":
        return Emotion2VecEmotionProvider(checkpoint, device=device)
    return Wav2VecEmotionProvider(checkpoint, device=device)


def evaluate_emotion_checkpoint(
    manifest: Path,
    checkpoint: Path,
    *,
    architecture: Literal["emotion2vec-plus", "wav2vec-xls-r"],
    device: Device = "auto",
    provider_factory: ProviderFactory | None = None,
) -> EmotionExperimentReport:
    """Evaluate a checkpoint on held-out items and retain aggregate metrics only."""

    if not manifest.is_absolute() or not checkpoint.is_absolute():
        raise ValueError("invalid_experiment_manifest")
    try:
        manifest_payload = read_trusted_regular_file(manifest)
        items = load_manifest(manifest)
        validate_disjoint_splits(items)
        if any(item.source != "emotion" or item.emotion is None for item in items):
            raise ValueError
        test_items = tuple(item for item in items if item.split == "test")
        if not test_items:
            raise ValueError
        for item in test_items:
            audio = read_trusted_regular_file(item.audio_path)
            if hashlib.sha256(audio).hexdigest() != item.sha256:
                raise ValueError
    except Exception:
        raise ValueError("invalid_experiment_manifest") from None

    try:
        provider = (
            provider_factory(checkpoint, device)
            if provider_factory is not None
            else _default_provider(checkpoint, architecture, device)
        )
    except Exception:
        raise ValueError("invalid_experiment_report") from None

    expected: list[EmotionLabel] = []
    predicted: list[EmotionLabel] = []
    confidence: list[float] = []
    latency: list[float] = []
    rss: list[float] = []
    started = time.perf_counter()
    try:
        for item in test_items:
            try:
                result = provider.analyze(item.id, Path(item.audio_path), "")
                if result.usage is None:
                    continue
                guess = max(result.probabilities, key=result.probabilities.__getitem__)
                expected.append(cast(EmotionLabel, item.emotion))
                predicted.append(guess)
                confidence.append(result.confidence)
                latency.append(result.usage.latency_ms)
                if result.usage.peak_rss_mb is not None:
                    rss.append(result.usage.peak_rss_mb)
            except ProviderError:
                continue
    finally:
        unload = getattr(provider, "unload", None)
        if callable(unload):
            unload()

    if not expected or not latency:
        raise ValueError("invalid_experiment_report")
    macro_f1, per_label_f1, matrix, ece = _aggregate_metrics(expected, predicted, confidence)
    revision = provider.provenance.revision
    if revision is None:
        raise ValueError("invalid_experiment_report")
    return EmotionExperimentReport(
        architecture=architecture,
        model_id=provider.provenance.model,
        checkpoint_digest=revision,
        manifest_digest=hashlib.sha256(manifest_payload).hexdigest(),
        test_count=len(test_items),
        completed_count=len(expected),
        completion_rate=len(expected) / len(test_items),
        macro_f1=macro_f1,
        per_label_f1=per_label_f1,
        confusion_matrix=matrix,
        expected_calibration_error=ece,
        median_latency_ms=statistics.median(latency),
        peak_rss_mb=max(rss, default=None),
        elapsed_seconds=max(0.0, time.perf_counter() - started),
        requested_device=device,
    )


def write_experiment_report(path: Path, report: EmotionExperimentReport) -> None:
    """Atomically publish one aggregate-only experiment report."""

    payload = (
        json.dumps(
            report.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        + b"\n"
    )
    _publish_private_file(path, payload, "experiment_report_publication_failed")
