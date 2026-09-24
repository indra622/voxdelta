"""Deterministic post-hoc calibration and abstention for seven-emotion providers.

Predicted softmax scores from a fine-tuned classifier are typically overconfident.
This module fits one temperature by minimizing negative log-likelihood and derives
one abstain threshold at a requested coverage, returning aggregates only: no item
identity, logit, or per-item score ever leaves a call.

The fit must run where the per-item distributions exist — inside the evaluation
process on the training host. A fit cannot be reconstructed later from an aggregate
report, because aggregate reports deliberately retain no per-item scores.

Temperature scaling is applied to `log(p)` rather than to raw logits. That is exact
rather than approximate: for `p = softmax(z)`, `log(p) = z - logsumexp(z)`, and
softmax is invariant to an additive constant, so `softmax(log(p) / T) ==
softmax(z / T)` for every `T`.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from voxdelta.domain.models import EmotionLabel
from voxdelta.evaluation.emotion_training import CANONICAL_LABELS

CALIBRATION_SCHEMA_VERSION = "1"
CALIBRATION_BINS = 10
TEMPERATURE_BOUNDS = (0.05, 10.0)
_SEARCH_ITERATIONS = 96
_GOLDEN_RATIO = (math.sqrt(5.0) - 1.0) / 2.0

# The floor exists solely to keep `log(0)` finite when a softmax underflows to exactly
# zero. It is the smallest positive normal double, not a tuning constant: any larger
# value would silently cap the loss of a confidently-wrong item and bias the fitted
# temperature downwards on a model sharper than the floor.
_PROBABILITY_FLOOR = sys.float_info.min


class CalibrationError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class CalibrationSummary(BaseModel):
    """Aggregate-only description of one fitted temperature and abstain threshold."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    schema_version: Literal["1"] = "1"
    method: Literal["temperature-scaling"] = "temperature-scaling"
    fitted_item_count: int = Field(gt=0)
    ece_bin_count: int = Field(gt=0)
    temperature: float = Field(gt=0)
    target_coverage: float = Field(gt=0, le=1)
    achieved_coverage: float = Field(gt=0, le=1)
    abstain_threshold: float = Field(ge=0, le=1)
    accuracy_at_coverage: float = Field(ge=0, le=1)
    pre_temperature_ece: float = Field(ge=0, le=1)
    post_temperature_ece: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def finite_summary(self) -> CalibrationSummary:
        values = (
            self.temperature,
            self.target_coverage,
            self.achieved_coverage,
            self.abstain_threshold,
            self.accuracy_at_coverage,
            self.pre_temperature_ece,
            self.post_temperature_ece,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("invalid calibration summary")
        if (
            self.achieved_coverage < self.target_coverage
            or not TEMPERATURE_BOUNDS[0] <= self.temperature <= TEMPERATURE_BOUNDS[1]
        ):
            raise ValueError("invalid calibration summary")
        return self


def expected_calibration_error(
    correct: Sequence[bool],
    confidence: Sequence[float],
    *,
    bins: int = CALIBRATION_BINS,
) -> float:
    """Equal-width binned ECE over predicted-class confidence."""

    if not correct or len(correct) != len(confidence) or bins < 1:
        raise CalibrationError("invalid_calibration_input")
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in confidence):
        raise CalibrationError("invalid_calibration_input")
    buckets: list[list[tuple[bool, float]]] = [[] for _ in range(bins)]
    for hit, score in zip(correct, confidence, strict=True):
        buckets[min(int(score * bins), bins - 1)].append((hit, score))
    return math.fsum(
        len(bucket)
        / len(correct)
        * abs(
            math.fsum(float(hit) for hit, _ in bucket) / len(bucket)
            - math.fsum(score for _, score in bucket) / len(bucket)
        )
        for bucket in buckets
        if bucket
    )


def _validated_distribution(distribution: Mapping[EmotionLabel, float]) -> tuple[float, ...]:
    if set(distribution) != set(CANONICAL_LABELS):
        raise CalibrationError("invalid_calibration_input")
    values = tuple(float(distribution[label]) for label in CANONICAL_LABELS)
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise CalibrationError("invalid_calibration_input")
    if abs(math.fsum(values) - 1.0) > 1e-6:
        raise CalibrationError("invalid_calibration_input")
    return values


def _log_distribution(values: Sequence[float]) -> tuple[float, ...]:
    """Recover logits up to an additive constant; softmax is invariant to that constant."""

    return tuple(math.log(max(value, _PROBABILITY_FLOOR)) for value in values)


def _scaled_from_logs(logs: Sequence[float], temperature: float) -> tuple[float, ...]:
    scaled = [value / temperature for value in logs]
    highest = max(scaled)
    exponentiated = [math.exp(value - highest) for value in scaled]
    total = math.fsum(exponentiated)
    return tuple(value / total for value in exponentiated)


def _scaled(values: Sequence[float], temperature: float) -> tuple[float, ...]:
    return _scaled_from_logs(_log_distribution(values), temperature)


def _truth_negative_log_likelihood(
    logs: Sequence[float],
    truth_index: int,
    temperature: float,
) -> float:
    """NLL of one item without materialising the whole rescaled distribution."""

    scaled = [value / temperature for value in logs]
    highest = max(scaled)
    return math.log(math.fsum(math.exp(value - highest) for value in scaled)) - (
        scaled[truth_index] - highest
    )


def apply_temperature(
    distribution: Mapping[EmotionLabel, float],
    temperature: float,
) -> dict[EmotionLabel, float]:
    """Rescale one seven-label distribution by a fitted temperature."""

    if not math.isfinite(temperature) or temperature <= 0:
        raise CalibrationError("invalid_calibration_temperature")
    scaled = _scaled(_validated_distribution(distribution), temperature)
    return dict(zip(CANONICAL_LABELS, scaled, strict=True))


def _mean_negative_log_likelihood(
    logs: Sequence[tuple[float, ...]],
    truth_indices: Sequence[int],
    temperature: float,
) -> float:
    total = math.fsum(
        _truth_negative_log_likelihood(row, index, temperature)
        for row, index in zip(logs, truth_indices, strict=True)
    )
    return total / len(logs)


def _fit_temperature(
    logs: Sequence[tuple[float, ...]],
    truth_indices: Sequence[int],
) -> float:
    """Golden-section search over log-temperature; deterministic and stdlib-only."""

    low, high = (math.log(bound) for bound in TEMPERATURE_BOUNDS)
    left = high - _GOLDEN_RATIO * (high - low)
    right = low + _GOLDEN_RATIO * (high - low)
    left_value = _mean_negative_log_likelihood(logs, truth_indices, math.exp(left))
    right_value = _mean_negative_log_likelihood(logs, truth_indices, math.exp(right))
    for _ in range(_SEARCH_ITERATIONS):
        if left_value <= right_value:
            high, right, right_value = right, left, left_value
            left = high - _GOLDEN_RATIO * (high - low)
            left_value = _mean_negative_log_likelihood(logs, truth_indices, math.exp(left))
        else:
            low, left, left_value = left, right, right_value
            right = low + _GOLDEN_RATIO * (high - low)
            right_value = _mean_negative_log_likelihood(logs, truth_indices, math.exp(right))
    temperature = math.exp((low + high) / 2)
    return min(max(temperature, TEMPERATURE_BOUNDS[0]), TEMPERATURE_BOUNDS[1])


def fit_temperature_scaling(
    distributions: Sequence[Mapping[EmotionLabel, float]],
    expected: Sequence[EmotionLabel],
    *,
    target_coverage: float,
    bins: int = CALIBRATION_BINS,
) -> CalibrationSummary:
    """Fit one temperature and abstain threshold; return aggregates only."""

    if (
        not distributions
        or len(distributions) != len(expected)
        or not math.isfinite(target_coverage)
        or not 0 < target_coverage <= 1
    ):
        raise CalibrationError("invalid_calibration_input")
    if any(label not in CANONICAL_LABELS for label in expected):
        raise CalibrationError("invalid_calibration_input")

    rows = tuple(_validated_distribution(distribution) for distribution in distributions)
    truth_indices = tuple(CANONICAL_LABELS.index(label) for label in expected)
    raw_confidence = tuple(max(row) for row in rows)
    raw_correct = tuple(
        row.index(max(row)) == index for row, index in zip(rows, truth_indices, strict=True)
    )

    logs = tuple(_log_distribution(row) for row in rows)
    temperature = _fit_temperature(logs, truth_indices)
    scaled = tuple(_scaled_from_logs(row, temperature) for row in logs)
    confidence = tuple(max(row) for row in scaled)
    correct = tuple(
        row.index(max(row)) == index for row, index in zip(scaled, truth_indices, strict=True)
    )

    # The threshold is the confidence of the `retained`-th most confident item, and the
    # rule is `>=`. Ties are therefore all kept together: a run of equal confidences
    # straddling the cut-off can only push achieved coverage above the target, never
    # below it, and never depends on the order the items arrived in.
    count = len(rows)
    retained = max(1, math.ceil(target_coverage * count))
    ordered = sorted(confidence, reverse=True)
    threshold = ordered[retained - 1]
    kept = tuple(hit for hit, score in zip(correct, confidence, strict=True) if score >= threshold)
    return CalibrationSummary(
        fitted_item_count=count,
        ece_bin_count=bins,
        temperature=temperature,
        target_coverage=target_coverage,
        achieved_coverage=len(kept) / count,
        abstain_threshold=threshold,
        accuracy_at_coverage=math.fsum(float(hit) for hit in kept) / len(kept),
        pre_temperature_ece=expected_calibration_error(raw_correct, raw_confidence, bins=bins),
        post_temperature_ece=expected_calibration_error(correct, confidence, bins=bins),
    )


__all__ = [
    "CALIBRATION_BINS",
    "CALIBRATION_SCHEMA_VERSION",
    "TEMPERATURE_BOUNDS",
    "CalibrationError",
    "CalibrationSummary",
    "apply_temperature",
    "expected_calibration_error",
    "fit_temperature_scaling",
]
