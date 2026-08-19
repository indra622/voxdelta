"""Deterministic customer-agent-customer transition construction."""

from __future__ import annotations

import math

from voxdelta.domain.models import (
    EmotionResult,
    EmotionTransition,
    Role,
    TransitionClass,
    Utterance,
)


def classify_delta(delta: float) -> TransitionClass:
    """Classify a finite delta after reproducible four-decimal rounding."""

    if not math.isfinite(delta):
        raise ValueError("delta must be finite")
    rounded = round(delta, 4)
    if rounded <= -0.20:
        return "recovery"
    if rounded >= 0.20:
        return "worsening"
    return "stable"


def build_transitions(
    utterances: list[Utterance],
    emotions: list[EmotionResult],
) -> list[EmotionTransition]:
    """Build transitions from adjacent, explicitly confirmed timeline triples.

    A missing smoothed intensity makes its triple ineligible rather than falling back
    to the raw intensity.
    """

    utterance_ids: set[str] = set()
    for utterance in utterances:
        if utterance.id in utterance_ids:
            raise ValueError(f"duplicate utterance id: {utterance.id}")
        utterance_ids.add(utterance.id)

    emotion_by_id: dict[str, EmotionResult] = {}
    for emotion in emotions:
        if emotion.utterance_id in emotion_by_id:
            raise ValueError(f"duplicate emotion utterance_id: {emotion.utterance_id}")
        smoothed = emotion.smoothed_negative_intensity
        if smoothed is not None and (not math.isfinite(smoothed) or not 0 <= smoothed <= 1):
            raise ValueError("smoothed_negative_intensity must be finite and between zero and one")
        emotion_by_id[emotion.utterance_id] = emotion

    ordered = sorted(
        utterances,
        key=lambda utterance: (
            utterance.start,
            utterance.end,
            utterance.id,
            utterance.speaker_id,
        ),
    )
    transitions: list[EmotionTransition] = []
    for index in range(len(ordered) - 2):
        previous, agent, following = ordered[index : index + 3]
        if (
            previous.role != Role.CUSTOMER
            or agent.role != Role.AGENT
            or following.role != Role.CUSTOMER
        ):
            continue
        previous_emotion = emotion_by_id.get(previous.id)
        following_emotion = emotion_by_id.get(following.id)
        if previous_emotion is None or following_emotion is None:
            continue
        previous_intensity = previous_emotion.smoothed_negative_intensity
        following_intensity = following_emotion.smoothed_negative_intensity
        if previous_intensity is None or following_intensity is None:
            continue
        delta = round(following_intensity - previous_intensity, 4)
        transitions.append(
            EmotionTransition(
                previous_customer_id=previous.id,
                agent_id=agent.id,
                next_customer_id=following.id,
                delta=delta,
                classification=classify_delta(delta),
            )
        )
    return transitions
