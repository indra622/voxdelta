from __future__ import annotations

import math

import pytest

from voxdelta.domain.models import EmotionLabel
from voxdelta.evaluation.calibration import (
    CALIBRATION_BINS,
    TEMPERATURE_BOUNDS,
    CalibrationError,
    CalibrationSummary,
    apply_temperature,
    expected_calibration_error,
    fit_temperature_scaling,
)
from voxdelta.evaluation.emotion_training import CANONICAL_LABELS


def _softmax(logits: tuple[float, ...]) -> dict[EmotionLabel, float]:
    highest = max(logits)
    exponentiated = [math.exp(value - highest) for value in logits]
    total = math.fsum(exponentiated)
    return dict(zip(CANONICAL_LABELS, (value / total for value in exponentiated), strict=True))


def _overconfident_corpus(
    count: int, sharpness: float
) -> tuple[list[dict[EmotionLabel, float]], list[EmotionLabel]]:
    """A deterministic corpus whose scores are far sharper than its accuracy.

    Peak sharpness rises with the item's level and correctness rises with it too,
    so confidence is informative — abstaining on the low-confidence tail should
    raise accuracy — while every score is far above the realised accuracy.
    """

    distributions: list[dict[EmotionLabel, float]] = []
    expected: list[EmotionLabel] = []
    for index in range(count):
        truth_index = index % len(CANONICAL_LABELS)
        level = index % 10
        correct = level >= 4
        argmax_index = truth_index if correct else (truth_index + 1) % len(CANONICAL_LABELS)
        peak = sharpness * (0.3 + 0.1 * level)
        logits = tuple(
            peak if position == argmax_index else 0.0 for position in range(len(CANONICAL_LABELS))
        )
        distributions.append(_softmax(logits))
        expected.append(CANONICAL_LABELS[truth_index])
    return distributions, expected


_LOW_DISCREPANCY = 0.6180339887498949


def _miscalibrated_corpus(
    count: int, *, true_temperature: float
) -> tuple[list[dict[EmotionLabel, float]], list[EmotionLabel]]:
    """Scores drawn at temperature one whose real accuracy follows `true_temperature`.

    This is the case temperature scaling is built for: the ranking is right and only
    the sharpness is wrong, so a correct fit should recover `true_temperature`. A
    low-discrepancy sequence decides correctness, keeping the corpus deterministic
    while matching the intended accuracy within each confidence band.
    """

    distributions: list[dict[EmotionLabel, float]] = []
    expected: list[EmotionLabel] = []
    for index in range(count):
        margin = 0.5 + 4.5 * ((index % 12) / 11)
        argmax_index = index % len(CANONICAL_LABELS)
        logits = tuple(
            margin if position == argmax_index else 0.0 for position in range(len(CANONICAL_LABELS))
        )
        true_accuracy = max(_softmax(tuple(value / true_temperature for value in logits)).values())
        correct = ((index * _LOW_DISCREPANCY) % 1.0) < true_accuracy
        truth_index = argmax_index if correct else (argmax_index + 1) % len(CANONICAL_LABELS)
        distributions.append(_softmax(logits))
        expected.append(CANONICAL_LABELS[truth_index])
    return distributions, expected


def test_expected_calibration_error_matches_the_previous_inline_definition() -> None:
    correct = [True, False, True, True, False, True, False, True]
    confidence = [0.95, 0.93, 0.55, 0.51, 0.34, 0.88, 0.12, 0.71]

    bins: list[list[tuple[bool, float]]] = [[] for _ in range(10)]
    for hit, score in zip(correct, confidence, strict=True):
        bins[min(int(score * 10), 9)].append((hit, score))
    reference = math.fsum(
        len(bucket)
        / len(correct)
        * abs(
            math.fsum(float(hit) for hit, _ in bucket) / len(bucket)
            - math.fsum(score for _, score in bucket) / len(bucket)
        )
        for bucket in bins
        if bucket
    )

    assert expected_calibration_error(correct, confidence) == pytest.approx(reference)
    assert expected_calibration_error(correct, confidence, bins=CALIBRATION_BINS) == pytest.approx(
        reference
    )


def test_expected_calibration_error_rejects_malformed_input() -> None:
    for correct, confidence in (
        ([], []),
        ([True], [0.5, 0.5]),
        ([True], [1.5]),
        ([True], [float("nan")]),
    ):
        with pytest.raises(CalibrationError, match="^invalid_calibration_input$"):
            expected_calibration_error(correct, confidence)
    with pytest.raises(CalibrationError, match="^invalid_calibration_input$"):
        expected_calibration_error([True], [0.5], bins=0)


def test_temperature_is_exact_on_log_probabilities_and_order_preserving() -> None:
    logits = (3.0, 1.0, 0.5, 0.0, -1.0, -2.0, -3.0)
    distribution = _softmax(logits)

    for temperature in (0.5, 1.0, 2.0, 4.0):
        scaled = apply_temperature(distribution, temperature)
        direct = _softmax(tuple(value / temperature for value in logits))
        assert all(
            scaled[label] == pytest.approx(direct[label], abs=1e-12) for label in CANONICAL_LABELS
        )
        assert math.fsum(scaled.values()) == pytest.approx(1.0)
        assert max(scaled, key=scaled.__getitem__) == max(
            distribution, key=distribution.__getitem__
        )

    assert apply_temperature(distribution, 1.0) == pytest.approx(distribution)
    # A higher temperature always softens the peak.
    assert apply_temperature(distribution, 4.0)["happiness"] < distribution["happiness"]


def test_temperature_rejects_invalid_values_and_distributions() -> None:
    distribution = _softmax((1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    for temperature in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(CalibrationError, match="^invalid_calibration_temperature$"):
            apply_temperature(distribution, temperature)
    for broken in (
        {label: 1 / 7 for label in CANONICAL_LABELS[:6]},
        {**distribution, "happiness": 0.9},
    ):
        with pytest.raises(CalibrationError, match="^invalid_calibration_input$"):
            apply_temperature(broken, 1.0)


@pytest.mark.parametrize("true_temperature", [1.0, 2.5, 4.0])
def test_fit_recovers_a_known_temperature_and_collapses_calibration_error(
    true_temperature: float,
) -> None:
    distributions, expected = _miscalibrated_corpus(1200, true_temperature=true_temperature)

    summary = fit_temperature_scaling(distributions, expected, target_coverage=0.9)

    assert isinstance(summary, CalibrationSummary)
    assert summary.method == "temperature-scaling"
    assert summary.fitted_item_count == 1200
    assert summary.ece_bin_count == CALIBRATION_BINS
    assert TEMPERATURE_BOUNDS[0] <= summary.temperature <= TEMPERATURE_BOUNDS[1]
    assert summary.temperature == pytest.approx(true_temperature, rel=0.05)
    assert summary.post_temperature_ece < 0.05
    if true_temperature > 1.0:
        # An overconfident corpus is softened, and the residual error is far smaller.
        assert summary.temperature > 1.0
        assert summary.pre_temperature_ece > 0.2
        assert summary.post_temperature_ece < summary.pre_temperature_ece / 5


def test_fit_reports_the_residual_error_a_single_temperature_cannot_remove() -> None:
    # Accuracy here is a step function of confidence, which no one-parameter family
    # can match. The summary must report the residual honestly, not hide it.
    distributions, expected = _overconfident_corpus(210, sharpness=8.0)

    summary = fit_temperature_scaling(distributions, expected, target_coverage=0.9)

    assert summary.temperature > 1.0
    assert summary.pre_temperature_ece > 0.2
    assert summary.post_temperature_ece < summary.pre_temperature_ece
    assert summary.post_temperature_ece > 0.05


def test_fit_is_deterministic_and_carries_no_item_level_content() -> None:
    distributions, expected = _overconfident_corpus(147, sharpness=6.0)

    first = fit_temperature_scaling(distributions, expected, target_coverage=0.8)
    second = fit_temperature_scaling(list(distributions), list(expected), target_coverage=0.8)

    assert first == second
    assert first.model_dump_json() == second.model_dump_json()
    assert set(first.model_dump()) == {
        "schema_version",
        "method",
        "fitted_item_count",
        "ece_bin_count",
        "temperature",
        "target_coverage",
        "achieved_coverage",
        "abstain_threshold",
        "accuracy_at_coverage",
        "pre_temperature_ece",
        "post_temperature_ece",
    }


def test_abstain_threshold_realises_at_least_the_requested_coverage() -> None:
    distributions, expected = _overconfident_corpus(200, sharpness=5.0)

    for target in (0.5, 0.75, 0.9, 1.0):
        summary = fit_temperature_scaling(distributions, expected, target_coverage=target)
        retained = [
            index
            for index, distribution in enumerate(distributions)
            if max(apply_temperature(distribution, summary.temperature).values())
            >= summary.abstain_threshold
        ]
        assert summary.achieved_coverage >= target
        assert len(retained) / len(distributions) == pytest.approx(summary.achieved_coverage)
        hits = sum(
            max(
                apply_temperature(distributions[index], summary.temperature),
                key=apply_temperature(distributions[index], summary.temperature).__getitem__,
            )
            == expected[index]
            for index in retained
        )
        assert hits / len(retained) == pytest.approx(summary.accuracy_at_coverage)

    full = fit_temperature_scaling(distributions, expected, target_coverage=1.0)
    assert full.achieved_coverage == 1.0


def test_abstaining_is_more_accurate_than_answering_everything() -> None:
    distributions, expected = _overconfident_corpus(300, sharpness=4.0)

    strict = fit_temperature_scaling(distributions, expected, target_coverage=0.5)
    everything = fit_temperature_scaling(distributions, expected, target_coverage=1.0)

    assert strict.achieved_coverage < everything.achieved_coverage
    assert strict.accuracy_at_coverage >= everything.accuracy_at_coverage
    assert strict.abstain_threshold > everything.abstain_threshold


def test_fit_rejects_malformed_corpora_and_coverages() -> None:
    distributions, expected = _overconfident_corpus(21, sharpness=3.0)

    for coverage in (0.0, -0.1, 1.5, float("nan")):
        with pytest.raises(CalibrationError, match="^invalid_calibration_input$"):
            fit_temperature_scaling(distributions, expected, target_coverage=coverage)
    with pytest.raises(CalibrationError, match="^invalid_calibration_input$"):
        fit_temperature_scaling([], [], target_coverage=0.9)
    with pytest.raises(CalibrationError, match="^invalid_calibration_input$"):
        fit_temperature_scaling(distributions, expected[:-1], target_coverage=0.9)
    with pytest.raises(CalibrationError, match="^invalid_calibration_input$"):
        fit_temperature_scaling(distributions, ["not-a-label", *expected[1:]], target_coverage=0.9)


def test_summary_rejects_a_temperature_or_coverage_outside_its_contract() -> None:
    payload = fit_temperature_scaling(
        *_overconfident_corpus(70, sharpness=5.0), target_coverage=0.9
    ).model_dump()

    with pytest.raises(ValueError, match="invalid calibration summary"):
        CalibrationSummary.model_validate({**payload, "temperature": 50.0})
    with pytest.raises(ValueError, match="invalid calibration summary"):
        CalibrationSummary.model_validate({**payload, "achieved_coverage": 0.1})
    with pytest.raises(ValueError):
        CalibrationSummary.model_validate({**payload, "unexpected": 1})
