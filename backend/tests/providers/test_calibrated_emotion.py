from __future__ import annotations

import math
from pathlib import Path

import pytest

from voxdelta.domain.models import EmotionResult, ProviderProvenance, ProviderUsage
from voxdelta.evaluation.calibration import CalibrationError, CalibrationSummary, apply_temperature
from voxdelta.providers.base import ProviderError
from voxdelta.providers.calibrated_emotion import CalibratedEmotionProvider, calibrate_result
from voxdelta.providers.calibration_artifact import VerifiedCalibration


def _calibration(*, temperature: float = 2.0, threshold: float = 0.4) -> VerifiedCalibration:
    return VerifiedCalibration(
        path=Path("/calibration"),
        calibration_id="xls-r-emotion-7class-v1-calibration-v1",
        binding_sha256="a" * 64,
        release_id="xls-r-emotion-7class-v1",
        bundle_tree_sha256="b" * 64,
        candidate_checkpoint_sha256="c" * 64,
        validation_items_sha256="d" * 64,
        validation_item_count=3569,
        summary=CalibrationSummary(
            fitted_item_count=3569,
            ece_bin_count=10,
            temperature=temperature,
            target_coverage=0.9,
            achieved_coverage=0.9002521714766041,
            abstain_threshold=threshold,
            accuracy_at_coverage=0.8123249299719888,
            pre_temperature_ece=0.08974033133605751,
            post_temperature_ece=0.01881642949260545,
        ),
    )


def _result(*, peak: float = 0.6) -> EmotionResult:
    remaining = (1.0 - peak) / 6
    return EmotionResult(
        utterance_id="sample-1",
        probabilities={
            "happiness": remaining,
            "anger": remaining,
            "disgust": remaining,
            "fear": remaining,
            "neutral": peak,
            "sadness": remaining,
            "surprise": remaining,
        },
        operational_state="stable",
        negative_intensity=remaining * 4,
        confidence=peak,
        provider=ProviderProvenance(
            name="wav2vec-xls-r",
            model="wav2vec2-xls-r-300m-seven-emotion",
            remote=False,
            revision="c" * 64,
        ),
        usage=ProviderUsage(latency_ms=10.0),
    )


def test_calibration_rescales_confidence_and_marks_an_answered_result() -> None:
    raw = _result(peak=0.9)
    calibration = _calibration(threshold=0.4)

    result = calibrate_result(raw, calibration)

    expected = apply_temperature(raw.probabilities, calibration.temperature)
    assert result.probabilities == expected
    assert max(result.probabilities, key=result.probabilities.__getitem__) == "neutral"
    assert result.confidence == max(expected.values())
    assert result.calibration is not None
    assert result.calibration.raw_confidence == 0.9
    assert result.calibration.abstained is False
    assert result.operational_state != "uncertain"
    assert math.isclose(
        result.negative_intensity,
        sum(expected[label] for label in ("anger", "disgust", "fear", "sadness")),
    )


def test_calibration_marks_a_below_threshold_result_uncertain_without_changing_label() -> None:
    raw = _result(peak=0.6)
    calibration = _calibration(temperature=2.0, threshold=0.4)

    result = calibrate_result(raw, calibration)

    assert max(raw.probabilities, key=raw.probabilities.__getitem__) == "neutral"
    assert max(result.probabilities, key=result.probabilities.__getitem__) == "neutral"
    assert result.confidence < calibration.abstain_threshold
    assert result.operational_state == "uncertain"
    assert result.calibration is not None and result.calibration.abstained is True


def test_calibrated_provider_delegates_and_unloads_the_inner_provider() -> None:
    class Inner:
        provenance = _result().provider
        unloaded = False

        def analyze(self, utterance_id: str, audio_path: Path, transcript: str) -> EmotionResult:
            assert (utterance_id, audio_path, transcript) == ("sample-1", Path("/clip.wav"), "")
            return _result()

        def unload(self) -> None:
            self.unloaded = True

    inner = Inner()
    provider = CalibratedEmotionProvider(inner, _calibration())

    result = provider.analyze("sample-1", Path("/clip.wav"), "")
    provider.unload()

    assert result.calibration is not None
    assert provider.provenance == inner.provenance
    assert inner.unloaded is True


def test_invalid_raw_distribution_fails_with_the_stable_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "voxdelta.providers.calibrated_emotion.apply_temperature",
        lambda probabilities, temperature: (_ for _ in ()).throw(
            CalibrationError("invalid_calibration_input")
        ),
    )

    with pytest.raises(ProviderError) as raised:
        calibrate_result(_result(), _calibration())

    assert raised.value.code == "invalid_provider_output"
