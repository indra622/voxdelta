"""Call-level summaries from aligned customer emotion results."""

from __future__ import annotations

import math

from voxdelta.domain.models import CallSummary, EmotionResult, EmotionTransition, Utterance


class InsufficientEmotionCoverage(ValueError):
    """Raised when too few customer turns have usable smoothed emotion results."""


def build_call_summary(
    customer_utterances: list[Utterance],
    emotions: list[EmotionResult],
    transitions: list[EmotionTransition],
) -> CallSummary:
    """Build a deterministic summary from matched, smoothed customer emotions."""

    if not customer_utterances:
        raise InsufficientEmotionCoverage("no customer utterances are available")

    customer_ids: set[str] = set()
    for utterance in customer_utterances:
        if utterance.id in customer_ids:
            raise ValueError(f"duplicate customer utterance id: {utterance.id}")
        customer_ids.add(utterance.id)

    emotion_by_id: dict[str, EmotionResult] = {}
    for emotion in emotions:
        if emotion.utterance_id in emotion_by_id:
            raise ValueError(f"duplicate emotion utterance_id: {emotion.utterance_id}")
        emotion_by_id[emotion.utterance_id] = emotion

    ordered_customers = sorted(
        customer_utterances,
        key=lambda utterance: (
            utterance.start,
            utterance.end,
            utterance.id,
            utterance.speaker_id,
        ),
    )
    valid: list[tuple[Utterance, EmotionResult, float]] = []
    for utterance in ordered_customers:
        matched_emotion = emotion_by_id.get(utterance.id)
        if matched_emotion is None or matched_emotion.smoothed_negative_intensity is None:
            continue
        smoothed = matched_emotion.smoothed_negative_intensity
        if not math.isfinite(smoothed) or not 0 <= smoothed <= 1:
            raise ValueError("smoothed_negative_intensity must be finite and between zero and one")
        valid.append((utterance, matched_emotion, smoothed))

    if len(valid) < 3:
        raise InsufficientEmotionCoverage("at least three valid emotion results are required")
    coverage = len(valid) / len(customer_utterances)
    if coverage < 0.50:
        raise InsufficientEmotionCoverage("valid emotion coverage must be at least 0.50")

    first = valid[0]
    last = valid[-1]
    peak = max(valid, key=lambda item: item[2])
    return CallSummary(
        start_state=first[1].operational_state,
        end_state=last[1].operational_state,
        peak_customer_utterance_id=peak[0].id,
        overall_delta=last[2] - first[2],
        valid_coverage=coverage,
        recovery_count=sum(transition.classification == "recovery" for transition in transitions),
        worsening_count=sum(transition.classification == "worsening" for transition in transitions),
    )
