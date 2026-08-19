from __future__ import annotations

import math

import pytest

from voxdelta.analysis.transitions import build_transitions, classify_delta
from voxdelta.domain.models import EmotionLabel, EmotionResult, ProviderProvenance, Role, Utterance

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


def _utterance(identifier: str, start: float, role: Role) -> Utterance:
    return Utterance(
        id=identifier,
        start=start,
        end=start + 0.5,
        speaker_id="customer" if role is Role.CUSTOMER else "agent",
        role=role,
        transcript=identifier,
        confidence=1.0,
    )


def _emotion(identifier: str, raw: float, smoothed: float | None) -> EmotionResult:
    return EmotionResult(
        utterance_id=identifier,
        probabilities=PROBABILITIES,
        operational_state="dissatisfied",
        negative_intensity=raw,
        smoothed_negative_intensity=smoothed,
        confidence=0.9,
        provider=PROVIDER,
    )


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        (-0.20, "recovery"),
        (0.20, "worsening"),
        (0.19, "stable"),
        (-0.19996, "recovery"),
        (0.19996, "worsening"),
        (-0.19994, "stable"),
        (0.19994, "stable"),
    ],
)
def test_classify_delta_rounds_to_four_decimals_before_inclusive_thresholds(
    delta: float, expected: str
) -> None:
    assert classify_delta(delta) == expected


@pytest.mark.parametrize("delta", [math.nan, math.inf, -math.inf])
def test_classify_delta_rejects_non_finite_values(delta: float) -> None:
    with pytest.raises(ValueError, match="delta must be finite"):
        classify_delta(delta)


def test_build_transitions_sorts_then_uses_only_adjacent_confirmed_triplets() -> None:
    utterances = [
        _utterance("c3", 4.0, Role.CUSTOMER),
        _utterance("a2", 3.0, Role.AGENT),
        _utterance("c1", 0.0, Role.CUSTOMER),
        _utterance("a1", 1.0, Role.AGENT),
        _utterance("c2", 2.0, Role.CUSTOMER),
    ]
    emotions = [
        _emotion("c3", 0.1, 0.1),
        _emotion("c1", 0.9, 0.9),
        _emotion("c2", 0.5, 0.5),
    ]

    transitions = build_transitions(utterances, emotions)

    assert [item.model_dump() for item in transitions] == [
        {
            "previous_customer_id": "c1",
            "agent_id": "a1",
            "next_customer_id": "c2",
            "delta": -0.4,
            "classification": "recovery",
        },
        {
            "previous_customer_id": "c2",
            "agent_id": "a2",
            "next_customer_id": "c3",
            "delta": -0.4,
            "classification": "recovery",
        },
    ]


def test_build_transitions_uses_deterministic_timeline_tie_breaks() -> None:
    utterances = [
        _utterance("c-next", 0.0, Role.CUSTOMER),
        _utterance("b-agent", 0.0, Role.AGENT),
        _utterance("a-customer", 0.0, Role.CUSTOMER),
    ]
    emotions = [_emotion("a-customer", 0.1, 0.1), _emotion("c-next", 0.3, 0.3)]

    transitions = build_transitions(utterances, emotions)

    assert len(transitions) == 1
    assert transitions[0].previous_customer_id == "a-customer"
    assert transitions[0].agent_id == "b-agent"
    assert transitions[0].next_customer_id == "c-next"
    assert transitions[0].delta == 0.2
    assert transitions[0].classification == "worsening"


def test_build_transitions_does_not_skip_non_adjacent_turns_or_infer_roles() -> None:
    utterances = [
        _utterance("c1", 0.0, Role.CUSTOMER),
        _utterance("unknown", 1.0, Role.UNKNOWN),
        _utterance("a1", 2.0, Role.AGENT),
        _utterance("c2", 3.0, Role.CUSTOMER),
    ]
    emotions = [_emotion("c1", 0.8, 0.8), _emotion("c2", 0.1, 0.1)]

    assert build_transitions(utterances, emotions) == []


def test_build_transitions_skips_missing_emotions_or_missing_smoothed_values() -> None:
    utterances = [
        _utterance("c1", 0.0, Role.CUSTOMER),
        _utterance("a1", 1.0, Role.AGENT),
        _utterance("c2", 2.0, Role.CUSTOMER),
    ]

    assert build_transitions(utterances, [_emotion("c1", 0.8, 0.8)]) == []
    assert (
        build_transitions(
            utterances,
            [_emotion("c1", 0.8, 0.8), _emotion("c2", 0.1, None)],
        )
        == []
    )


def test_build_transitions_rejects_duplicate_emotion_ids_even_when_unused() -> None:
    utterances = [
        _utterance("c1", 0.0, Role.CUSTOMER),
        _utterance("a1", 1.0, Role.AGENT),
        _utterance("c2", 2.0, Role.CUSTOMER),
    ]
    emotions = [
        _emotion("unused", 0.2, 0.2),
        _emotion("unused", 0.3, 0.3),
        _emotion("c1", 0.8, 0.8),
        _emotion("c2", 0.1, 0.1),
    ]

    with pytest.raises(ValueError, match="duplicate emotion utterance_id"):
        build_transitions(utterances, emotions)


@pytest.mark.parametrize("duplicate_role", [Role.CUSTOMER, Role.AGENT, Role.UNKNOWN])
def test_build_transitions_rejects_duplicate_utterance_ids_for_every_role(
    duplicate_role: Role,
) -> None:
    utterances = [
        _utterance("duplicate", 0.0, duplicate_role),
        _utterance("middle", 1.0, Role.AGENT),
        _utterance("duplicate", 2.0, duplicate_role),
    ]

    with pytest.raises(ValueError, match="duplicate utterance id: duplicate"):
        build_transitions(utterances, [])


def test_build_transitions_rejects_non_finite_smoothed_intensity() -> None:
    utterances = [
        _utterance("c1", 0.0, Role.CUSTOMER),
        _utterance("a1", 1.0, Role.AGENT),
        _utterance("c2", 2.0, Role.CUSTOMER),
    ]
    invalid = _emotion("c2", 0.1, 0.1).model_copy(update={"smoothed_negative_intensity": math.nan})

    with pytest.raises(ValueError, match="smoothed_negative_intensity"):
        build_transitions(utterances, [_emotion("c1", 0.8, 0.8), invalid])


def test_build_transitions_does_not_mutate_inputs() -> None:
    utterances = [
        _utterance("c2", 2.0, Role.CUSTOMER),
        _utterance("a1", 1.0, Role.AGENT),
        _utterance("c1", 0.0, Role.CUSTOMER),
    ]
    emotions = [_emotion("c1", 0.8, 0.8), _emotion("c2", 0.1, 0.1)]
    utterance_snapshot = [item.model_dump() for item in utterances]
    emotion_snapshot = [item.model_dump() for item in emotions]

    build_transitions(utterances, emotions)

    assert [item.model_dump() for item in utterances] == utterance_snapshot
    assert [item.model_dump() for item in emotions] == emotion_snapshot
