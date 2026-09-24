"""Apply a verified post-hoc calibration to a local emotion provider's results.

The wrapped provider keeps producing raw softmax scores; this adapter rescales them by
the fitted temperature, re-reads the top-class probability as the confidence, and marks
the result abstained when that confidence falls below the fitted threshold. An abstained
result reports the pre-existing ``uncertain`` operational state, so every consumer that
already handles low-confidence emotion degrades correctly without knowing this adapter
exists.

The adapter is only ever constructed from a ``VerifiedCalibration``, which can only be
obtained by verifying the artifact against the release it belongs to. There is no path
here that falls back to uncalibrated scoring once calibration has been enabled.
"""

from __future__ import annotations

import math
from pathlib import Path

from voxdelta.analysis.emotions import map_operational_state
from voxdelta.domain.models import EmotionCalibration, EmotionLabel, EmotionResult
from voxdelta.evaluation.calibration import CalibrationError, apply_temperature
from voxdelta.providers.base import EmotionProvider, ProviderError
from voxdelta.providers.calibration_artifact import VerifiedCalibration

_NEGATIVE_LABELS: tuple[EmotionLabel, ...] = ("anger", "disgust", "fear", "sadness")


def calibrate_result(result: EmotionResult, calibration: VerifiedCalibration) -> EmotionResult:
    """Rescale one result's distribution and decide whether it is answered or abstained."""

    try:
        probabilities = apply_temperature(result.probabilities, calibration.temperature)
    except CalibrationError:
        raise ProviderError("invalid_provider_output") from None
    confidence = max(probabilities.values())
    abstained = confidence < calibration.abstain_threshold
    return result.model_copy(
        update={
            "probabilities": probabilities,
            "confidence": confidence,
            "negative_intensity": min(
                1.0, math.fsum(probabilities[label] for label in _NEGATIVE_LABELS)
            ),
            "operational_state": (
                "uncertain" if abstained else map_operational_state(probabilities, confidence)
            ),
            "calibration": EmotionCalibration(
                calibration_id=calibration.calibration_id,
                method=calibration.summary.method,
                temperature=calibration.temperature,
                abstain_threshold=calibration.abstain_threshold,
                abstained=abstained,
                raw_confidence=result.confidence,
            ),
        },
        deep=True,
    )


class CalibratedEmotionProvider:
    """One emotion provider whose every result carries a verified calibration."""

    def __init__(self, inner: EmotionProvider, calibration: VerifiedCalibration) -> None:
        self._inner = inner
        self._calibration = calibration
        self.provenance = inner.provenance

    @property
    def calibration(self) -> VerifiedCalibration:
        return self._calibration

    @property
    def inner(self) -> EmotionProvider:
        """The wrapped uncalibrated provider, for readiness tooling that must compare."""

        return self._inner

    def analyze(self, utterance_id: str, audio_path: Path, transcript: str) -> EmotionResult:
        return calibrate_result(
            self._inner.analyze(utterance_id, audio_path, transcript), self._calibration
        )

    def unload(self) -> None:
        unload = getattr(self._inner, "unload", None)
        if callable(unload):
            unload()


__all__ = ["CalibratedEmotionProvider", "calibrate_result"]
