from __future__ import annotations

import math

import pytest

from voxdelta.analysis.emotions import map_operational_state, median_smooth
from voxdelta.domain.models import EmotionLabel, EmotionResult, ProviderProvenance

PROVIDER = ProviderProvenance(name="test", model="test", remote=False)


def _probabilities(**overrides: float) -> dict[EmotionLabel, float]:
    values: dict[EmotionLabel, float] = {
        "happiness": 0.10,
        "anger": 0.10,
        "disgust": 0.10,
        "fear": 0.10,
        "neutral": 0.10,
        "sadness": 0.40,
        "surprise": 0.10,
    }
    values.update(overrides)  # type: ignore[arg-type]
    return values


def _emotion(
    utterance_id: str,
    intensity: float,
    *,
    smoothed: float | None = None,
) -> EmotionResult:
    return EmotionResult(
        utterance_id=utterance_id,
        probabilities=_probabilities(),
        operational_state="dissatisfied",
        negative_intensity=intensity,
        smoothed_negative_intensity=smoothed,
        confidence=0.9,
        provider=PROVIDER,
    )


@pytest.mark.parametrize(
    ("probabilities", "expected"),
    [
        (
            _probabilities(
                happiness=0.60,
                anger=0.05,
                disgust=0.05,
                fear=0.05,
                neutral=0.10,
                sadness=0.05,
                surprise=0.10,
            ),
            "satisfied",
        ),
        (
            _probabilities(
                happiness=0.10,
                anger=0.05,
                disgust=0.05,
                fear=0.05,
                neutral=0.60,
                sadness=0.05,
                surprise=0.10,
            ),
            "stable",
        ),
        (
            _probabilities(
                happiness=0.05,
                anger=0.60,
                disgust=0.05,
                fear=0.05,
                neutral=0.10,
                sadness=0.05,
                surprise=0.10,
            ),
            "escalated",
        ),
        (
            _probabilities(
                happiness=0.15,
                anger=0.05,
                disgust=0.21,
                fear=0.20,
                neutral=0.05,
                sadness=0.19,
                surprise=0.15,
            ),
            "dissatisfied",
        ),
    ],
)
def test_map_operational_state_uses_canonical_grouped_scores(
    probabilities: dict[EmotionLabel, float], expected: str
) -> None:
    assert map_operational_state(probabilities, confidence=0.9) == expected


@pytest.mark.parametrize(
    "probabilities",
    [
        _probabilities(
            happiness=0.20,
            anger=0.02,
            disgust=0.18,
            fear=0.18,
            neutral=0.03,
            sadness=0.19,
            surprise=0.20,
        ),
        _probabilities(
            happiness=0.19,
            anger=0.02,
            disgust=0.18,
            fear=0.18,
            neutral=0.03,
            sadness=0.19,
            surprise=0.21,
        ),
    ],
)
def test_surprise_tie_or_dominance_becomes_uncertain(
    probabilities: dict[EmotionLabel, float],
) -> None:
    assert map_operational_state(probabilities, confidence=0.9) == "uncertain"


def test_confidence_threshold_is_independent_and_inclusive() -> None:
    probabilities = _probabilities()

    assert map_operational_state(probabilities, confidence=0.549999) == "uncertain"
    assert map_operational_state(probabilities, confidence=0.55) == "dissatisfied"


@pytest.mark.parametrize(
    ("probabilities", "expected"),
    [
        (
            _probabilities(
                happiness=0.30,
                anger=0.10,
                disgust=0.05,
                fear=0.05,
                neutral=0.30,
                sadness=0.05,
                surprise=0.15,
            ),
            "satisfied",
        ),
        (
            _probabilities(
                happiness=0.10,
                anger=0.10,
                disgust=0.10,
                fear=0.10,
                neutral=0.30,
                sadness=0.10,
                surprise=0.20,
            ),
            "stable",
        ),
    ],
)
def test_grouped_score_ties_use_fixed_specification_order(
    probabilities: dict[EmotionLabel, float], expected: str
) -> None:
    assert map_operational_state(probabilities, confidence=0.9) == expected


@pytest.mark.parametrize("confidence", [-0.01, 1.01, math.inf, -math.inf, math.nan])
def test_map_operational_state_rejects_invalid_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        map_operational_state(_probabilities(), confidence)


def test_map_operational_state_requires_exactly_seven_labels() -> None:
    missing = dict(_probabilities())
    missing.pop("surprise")
    extra = {**_probabilities(), "contempt": 0.0}

    with pytest.raises(ValueError, match="seven emotion labels"):
        map_operational_state(missing, 0.9)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seven emotion labels"):
        map_operational_state(extra, 0.9)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -0.01, 1.01])
def test_map_operational_state_rejects_invalid_probability_values(invalid: float) -> None:
    probabilities = dict(_probabilities())
    probabilities["sadness"] = invalid

    with pytest.raises(ValueError, match="probabilities"):
        map_operational_state(probabilities, 0.9)  # type: ignore[arg-type]


def test_map_operational_state_requires_a_normalized_distribution() -> None:
    probabilities = _probabilities(sadness=0.39)

    with pytest.raises(ValueError, match="sum to one"):
        map_operational_state(probabilities, 0.9)


def test_median_smooth_uses_supplied_customer_result_order() -> None:
    results = [_emotion("z-last-id", 0.9), _emotion("a-first-id", 0.1), _emotion("m", 0.6)]

    smoothed = median_smooth(results)

    assert [item.utterance_id for item in smoothed] == ["z-last-id", "a-first-id", "m"]
    assert [item.smoothed_negative_intensity for item in smoothed] == [0.5, 0.6, 0.35]


def test_median_smooth_returns_copies_and_preserves_raw_values_and_provenance() -> None:
    results = [_emotion("u1", 0.2, smoothed=0.99), _emotion("u2", 0.8, smoothed=0.01)]
    snapshots = [item.model_dump() for item in results]

    smoothed = median_smooth(results)

    assert smoothed is not results
    assert all(output is not source for output, source in zip(smoothed, results, strict=True))
    assert [item.negative_intensity for item in smoothed] == [0.2, 0.8]
    assert all(item.provider == PROVIDER for item in smoothed)
    assert [item.smoothed_negative_intensity for item in smoothed] == [0.5, 0.5]
    assert [item.model_dump() for item in results] == snapshots


def test_median_smooth_handles_empty_and_single_result() -> None:
    assert median_smooth([]) == []
    result = _emotion("u1", 0.7)

    smoothed = median_smooth([result])

    assert smoothed[0] is not result
    assert smoothed[0].smoothed_negative_intensity == 0.7


def test_median_smooth_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="duplicate emotion utterance_id"):
        median_smooth([_emotion("u1", 0.1), _emotion("u1", 0.2)])


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -0.1, 1.1])
def test_median_smooth_rejects_invalid_raw_intensity(invalid: float) -> None:
    invalid_result = _emotion("u1", 0.5).model_copy(update={"negative_intensity": invalid})

    with pytest.raises(ValueError, match="negative_intensity"):
        median_smooth([invalid_result])
