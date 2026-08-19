from __future__ import annotations

import math

import pytest

from voxdelta.analysis.summary import InsufficientEmotionCoverage, build_call_summary
from voxdelta.domain.models import (
    EmotionLabel,
    EmotionResult,
    EmotionTransition,
    ProviderProvenance,
    Role,
    Utterance,
)

PROVIDER = ProviderProvenance(name="test", model="test", remote=False)
PROBABILITIES: dict[EmotionLabel, float] = {
    "happiness": 0.1,
    "anger": 0.1,
    "disgust": 0.1,
    "fear": 0.1,
    "neutral": 0.1,
    "sadness": 0.4,
    "surprise": 0.1,
}


def _customer(identifier: str, start: float) -> Utterance:
    return Utterance(
        id=identifier,
        start=start,
        end=start + 0.5,
        speaker_id="customer",
        role=Role.CUSTOMER,
        transcript=identifier,
        confidence=1.0,
    )


def _emotion(
    identifier: str,
    smoothed: float | None,
    state: str = "dissatisfied",
) -> EmotionResult:
    return EmotionResult(
        utterance_id=identifier,
        probabilities=PROBABILITIES,
        operational_state=state,  # type: ignore[arg-type]
        negative_intensity=0.5,
        smoothed_negative_intensity=smoothed,
        confidence=0.9,
        provider=PROVIDER,
    )


def _transition(identifier: str, classification: str) -> EmotionTransition:
    delta = {"recovery": -0.2, "stable": 0.0, "worsening": 0.2}[classification]
    return EmotionTransition(
        previous_customer_id=f"{identifier}-before",
        agent_id=f"{identifier}-agent",
        next_customer_id=f"{identifier}-after",
        delta=delta,
        classification=classification,  # type: ignore[arg-type]
    )


def test_build_call_summary_aligns_then_orders_emotions_by_customer_chronology() -> None:
    customers = [_customer("c3", 3.0), _customer("c1", 1.0), _customer("c2", 2.0)]
    emotions = [
        _emotion("c2", 0.9, "escalated"),
        _emotion("c3", 0.4, "stable"),
        _emotion("c1", 0.2, "dissatisfied"),
        _emotion("not-a-customer", 1.0, "satisfied"),
    ]
    transitions = [
        _transition("r1", "recovery"),
        _transition("r2", "recovery"),
        _transition("w1", "worsening"),
        _transition("s1", "stable"),
    ]

    summary = build_call_summary(customers, emotions, transitions)

    assert summary.start_state == "dissatisfied"
    assert summary.end_state == "stable"
    assert summary.peak_customer_utterance_id == "c2"
    assert summary.overall_delta == pytest.approx(0.2)
    assert summary.valid_coverage == 1.0
    assert summary.recovery_count == 2
    assert summary.worsening_count == 1
    assert summary.narrative is None


def test_build_call_summary_uses_earliest_chronological_peak_on_tie() -> None:
    customers = [_customer("later", 2.0), _customer("first", 0.0), _customer("middle", 1.0)]
    emotions = [
        _emotion("later", 0.8),
        _emotion("middle", 0.8),
        _emotion("first", 0.2),
    ]

    summary = build_call_summary(customers, emotions, [])

    assert summary.peak_customer_utterance_id == "middle"
    assert summary.overall_delta == pytest.approx(0.6)


def test_build_call_summary_accepts_exactly_half_coverage_with_three_valid_results() -> None:
    customers = [_customer(f"c{index}", float(index)) for index in range(6)]
    emotions = [_emotion("c0", 0.1), _emotion("c2", 0.5), _emotion("c5", 0.3)]

    summary = build_call_summary(customers, emotions, [])

    assert summary.valid_coverage == 0.5
    assert summary.start_state == "dissatisfied"
    assert summary.end_state == "dissatisfied"


def test_build_call_summary_excludes_missing_smoothed_values_from_valid_coverage() -> None:
    customers = [_customer(f"c{index}", float(index)) for index in range(6)]
    emotions = [
        _emotion("c0", 0.1),
        _emotion("c1", None),
        _emotion("c2", 0.2),
        _emotion("c5", 0.3),
    ]

    summary = build_call_summary(customers, emotions, [])

    assert summary.valid_coverage == 0.5
    assert summary.overall_delta == pytest.approx(0.2)


def test_build_call_summary_requires_three_valid_results_even_with_full_coverage() -> None:
    customers = [_customer("c1", 0.0), _customer("c2", 1.0)]
    emotions = [_emotion("c1", 0.1), _emotion("c2", 0.2)]

    with pytest.raises(InsufficientEmotionCoverage, match="at least three"):
        build_call_summary(customers, emotions, [])


def test_build_call_summary_rejects_coverage_below_half() -> None:
    customers = [_customer(f"c{index}", float(index)) for index in range(7)]
    emotions = [_emotion("c0", 0.1), _emotion("c2", 0.2), _emotion("c5", 0.3)]

    with pytest.raises(InsufficientEmotionCoverage, match="coverage"):
        build_call_summary(customers, emotions, [])


def test_build_call_summary_handles_zero_customers_without_division() -> None:
    with pytest.raises(InsufficientEmotionCoverage, match="customer utterances"):
        build_call_summary([], [], [])


def test_build_call_summary_rejects_duplicate_customer_or_emotion_ids() -> None:
    customers = [_customer("c1", 0.0), _customer("c1", 1.0), _customer("c2", 2.0)]
    emotions = [_emotion("c1", 0.1), _emotion("c2", 0.2), _emotion("c3", 0.3)]

    with pytest.raises(ValueError, match="duplicate customer utterance id"):
        build_call_summary(customers, emotions, [])

    unique_customers = [_customer("c1", 0.0), _customer("c2", 1.0), _customer("c3", 2.0)]
    duplicate_emotions = [
        _emotion("c1", 0.1),
        _emotion("c2", 0.2),
        _emotion("unused", 0.3),
        _emotion("unused", 0.4),
    ]
    with pytest.raises(ValueError, match="duplicate emotion utterance_id"):
        build_call_summary(unique_customers, duplicate_emotions, [])


def test_build_call_summary_rejects_non_finite_aligned_smoothed_value() -> None:
    customers = [_customer("c1", 0.0), _customer("c2", 1.0), _customer("c3", 2.0)]
    invalid = _emotion("c2", 0.2).model_copy(update={"smoothed_negative_intensity": math.nan})

    with pytest.raises(ValueError, match="smoothed_negative_intensity"):
        build_call_summary(
            customers,
            [_emotion("c1", 0.1), invalid, _emotion("c3", 0.3)],
            [],
        )


def test_build_call_summary_does_not_mutate_inputs() -> None:
    customers = [_customer("c3", 2.0), _customer("c1", 0.0), _customer("c2", 1.0)]
    emotions = [_emotion("c3", 0.3), _emotion("c1", 0.1), _emotion("c2", 0.2)]
    customer_snapshot = [item.model_dump() for item in customers]
    emotion_snapshot = [item.model_dump() for item in emotions]

    build_call_summary(customers, emotions, [])

    assert [item.model_dump() for item in customers] == customer_snapshot
    assert [item.model_dump() for item in emotions] == emotion_snapshot
