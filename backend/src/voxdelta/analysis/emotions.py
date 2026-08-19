"""Operational emotion mapping and customer-turn smoothing."""

from __future__ import annotations

import math
import statistics

from voxdelta.domain.models import EmotionLabel, EmotionResult, OperationalState

_CONFIDENCE_THRESHOLD = 0.55
_EMOTION_LABELS: tuple[EmotionLabel, ...] = (
    "happiness",
    "anger",
    "disgust",
    "fear",
    "neutral",
    "sadness",
    "surprise",
)
_DISSATISFIED_LABELS: tuple[EmotionLabel, ...] = ("sadness", "disgust", "fear")


def map_operational_state(
    probabilities: dict[EmotionLabel, float],
    confidence: float,
) -> OperationalState:
    """Map one valid seven-emotion distribution to a business-facing state."""

    if set(probabilities) != set(_EMOTION_LABELS):
        raise ValueError("all seven emotion labels are required")
    values = tuple(probabilities[label] for label in _EMOTION_LABELS)
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("emotion probabilities must be finite and between zero and one")
    if abs(math.fsum(values) - 1.0) > 1e-6:
        raise ValueError("emotion probabilities must sum to one")
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("confidence must be finite and between zero and one")

    non_surprise_peak = max(
        probabilities[label] for label in _EMOTION_LABELS if label != "surprise"
    )
    if probabilities["surprise"] >= non_surprise_peak or confidence < _CONFIDENCE_THRESHOLD:
        return "uncertain"

    scores: tuple[tuple[OperationalState, float], ...] = (
        ("satisfied", probabilities["happiness"]),
        ("stable", probabilities["neutral"]),
        (
            "dissatisfied",
            math.fsum(probabilities[label] for label in _DISSATISFIED_LABELS),
        ),
        ("escalated", probabilities["anger"]),
    )
    maximum = max(score for _, score in scores)
    # Treat representation-only differences as ties and preserve specification order.
    return next(
        state for state, score in scores if math.isclose(score, maximum, rel_tol=0.0, abs_tol=1e-12)
    )


def median_smooth(results: list[EmotionResult]) -> list[EmotionResult]:
    """Copy and smooth results in the caller-supplied customer chronology."""

    seen_ids: set[str] = set()
    for result in results:
        if result.utterance_id in seen_ids:
            raise ValueError(f"duplicate emotion utterance_id: {result.utterance_id}")
        seen_ids.add(result.utterance_id)
        intensity = result.negative_intensity
        if not math.isfinite(intensity) or not 0 <= intensity <= 1:
            raise ValueError("negative_intensity must be finite and between zero and one")

    smoothed: list[EmotionResult] = []
    for index, result in enumerate(results):
        start = max(0, index - 1)
        stop = min(len(results), index + 2)
        window = [item.negative_intensity for item in results[start:stop]]
        smoothed.append(
            result.model_copy(
                update={"smoothed_negative_intensity": statistics.median(window)},
                deep=True,
            )
        )
    return smoothed
