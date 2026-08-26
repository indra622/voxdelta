"""The shadow contract: the primary result is untouchable and the candidate cannot fail it."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from voxdelta.domain.models import (
    EmotionCalibration,
    EmotionResult,
    ProviderProvenance,
    ProviderUsage,
)
from voxdelta.evaluation.emotion_training import CANONICAL_LABELS
from voxdelta.providers.base import ProviderError
from voxdelta.providers.shadow_emotion import (
    ShadowEmotionProvider,
    ShadowObservation,
)


def _provenance(name: str) -> ProviderProvenance:
    return ProviderProvenance(name=name, model=f"{name}-model", remote=False, revision="r" * 8)


def _result(
    peak: float = 0.8,
    *,
    label: str = "neutral",
    name: str = "primary",
    calibration: EmotionCalibration | None = None,
    latency_ms: float = 11.0,
) -> EmotionResult:
    remaining = (1.0 - peak) / 6
    probabilities = {key: remaining for key in CANONICAL_LABELS}
    probabilities[label] = peak  # type: ignore[index]
    return EmotionResult(
        utterance_id="utterance-1",
        probabilities=probabilities,  # type: ignore[arg-type]
        dominant_emotion=label,  # type: ignore[arg-type]
        negative_intensity=0.0,
        operational_state="uncertain" if calibration and calibration.abstained else "stable",
        confidence=peak,
        provider=_provenance(name),
        usage=ProviderUsage(latency_ms=latency_ms, peak_rss_mb=64.0),
        calibration=calibration,
    )


class _Recorder:
    def __init__(self) -> None:
        self.seen: list[ShadowObservation] = []

    def __call__(self, observation: ShadowObservation) -> None:
        self.seen.append(observation)


class _Provider:
    def __init__(self, result: EmotionResult | None = None, error: Exception | None = None) -> None:
        self._result = result
        self._error = error
        self.provenance = _provenance("stub")
        self.calls = 0

    def analyze(self, utterance_id: str, audio_path: Path, transcript: str) -> EmotionResult:
        del audio_path, transcript
        self.calls += 1
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result.model_copy(update={"utterance_id": utterance_id})


def _shadow(primary: object, candidate: object, recorder: _Recorder, **kwargs: object) -> object:
    return ShadowEmotionProvider(
        primary,  # type: ignore[arg-type]
        candidate,  # type: ignore[arg-type]
        observer=recorder,
        **kwargs,  # type: ignore[arg-type]
    )


# --- primary invariance ---


def test_primary_result_is_model_dump_identical_to_running_the_primary_alone(
    tmp_path: Path,
) -> None:
    expected = _Provider(_result()).analyze("utterance-1", tmp_path / "a.wav", "")
    primary = _Provider(_result())
    recorder = _Recorder()

    observed = _shadow(primary, _Provider(_result(name="candidate")), recorder).analyze(  # type: ignore[attr-defined]
        "utterance-1", tmp_path / "a.wav", ""
    )

    assert observed.model_dump() == expected.model_dump()
    assert observed.provider.name == "primary"


def test_shadow_returns_the_exact_object_the_primary_produced(tmp_path: Path) -> None:
    """Identity, not equality: the primary must be called once and its object returned."""

    class Remembering:
        provenance = _provenance("primary")

        def __init__(self) -> None:
            self.last: EmotionResult | None = None
            self.calls = 0

        def analyze(self, utterance_id: str, audio_path: Path, transcript: str) -> EmotionResult:
            del audio_path, transcript
            self.calls += 1
            self.last = _result().model_copy(update={"utterance_id": utterance_id})
            return self.last

    primary = Remembering()
    shadow = _shadow(primary, _Provider(_result(name="candidate")), _Recorder())

    observed = shadow.analyze("utterance-1", tmp_path / "a.wav", "")  # type: ignore[attr-defined]

    assert observed is primary.last
    assert primary.calls == 1


@pytest.mark.parametrize(
    "failure",
    [
        ProviderError("provider_unavailable"),
        ProviderError("invalid_audio_asset"),
        RuntimeError("boom"),
        MemoryError(),
    ],
)
def test_a_failing_candidate_never_turns_a_primary_success_into_a_product_failure(
    tmp_path: Path, failure: Exception
) -> None:
    primary = _Provider(_result())
    recorder = _Recorder()

    observed = _shadow(primary, _Provider(error=failure), recorder).analyze(  # type: ignore[attr-defined]
        "utterance-1", tmp_path / "a.wav", ""
    )

    assert observed.confidence == pytest.approx(0.8)
    assert len(recorder.seen) == 1
    assert recorder.seen[0].candidate_completed is False
    assert recorder.seen[0].error_category in {"provider_error", "unexpected_error"}


def test_a_primary_failure_still_propagates_and_the_candidate_is_never_consulted(
    tmp_path: Path,
) -> None:
    candidate = _Provider(_result())
    recorder = _Recorder()
    shadow = _shadow(_Provider(error=ProviderError("provider_unavailable")), candidate, recorder)

    with pytest.raises(ProviderError) as error:
        shadow.analyze("utterance-1", tmp_path / "a.wav", "")  # type: ignore[attr-defined]

    assert error.value.code == "provider_unavailable"
    assert candidate.calls == 0
    assert recorder.seen == []


def test_a_candidate_returning_a_malformed_object_is_categorised_not_raised(
    tmp_path: Path,
) -> None:
    class Malformed:
        provenance = _provenance("candidate")

        def analyze(self, utterance_id: str, audio_path: Path, transcript: str) -> object:
            del utterance_id, audio_path, transcript
            return {"not": "an EmotionResult"}

    primary = _Provider(_result())
    recorder = _Recorder()

    observed = _shadow(primary, Malformed(), recorder).analyze(  # type: ignore[attr-defined]
        "utterance-1", tmp_path / "a.wav", ""
    )

    assert observed.confidence == pytest.approx(0.8)
    assert recorder.seen[0].error_category == "invalid_output"
    assert recorder.seen[0].candidate_completed is False


def test_a_slow_candidate_times_out_without_delaying_or_changing_the_primary(
    tmp_path: Path,
) -> None:
    class Slow:
        provenance = _provenance("candidate")

        def analyze(self, utterance_id: str, audio_path: Path, transcript: str) -> EmotionResult:
            del utterance_id, audio_path, transcript
            time.sleep(2.0)
            return _result(name="candidate")

    primary = _Provider(_result())
    recorder = _Recorder()

    began = time.perf_counter()
    observed = _shadow(primary, Slow(), recorder, timeout_seconds=0.05).analyze(  # type: ignore[attr-defined]
        "utterance-1", tmp_path / "a.wav", ""
    )
    elapsed = time.perf_counter() - began

    assert observed.confidence == pytest.approx(0.8)
    assert elapsed < 1.5
    assert recorder.seen[0].error_category == "timeout"
    assert recorder.seen[0].candidate_completed is False


def test_an_observer_that_raises_cannot_break_the_product_path(tmp_path: Path) -> None:
    def hostile(observation: ShadowObservation) -> None:
        raise RuntimeError("observer exploded")

    shadow = ShadowEmotionProvider(
        _Provider(_result()),  # type: ignore[arg-type]
        _Provider(_result(name="candidate")),  # type: ignore[arg-type]
        observer=hostile,
    )

    observed = shadow.analyze("utterance-1", tmp_path / "a.wav", "")

    assert observed.confidence == pytest.approx(0.8)


# --- observation content ---


def test_a_successful_candidate_is_observed_with_agreement_and_calibration_fields(
    tmp_path: Path,
) -> None:
    calibration = EmotionCalibration(
        calibration_id="xls-r-emotion-7class-v1-calibration-v2",
        method="temperature-scaling",
        temperature=1.5,
        abstain_threshold=0.4,
        abstained=False,
        raw_confidence=0.91,
    )
    primary = _Provider(_result(0.8, label="neutral"))
    candidate = _Provider(_result(0.7, label="neutral", name="c", calibration=calibration))
    recorder = _Recorder()

    _shadow(primary, candidate, recorder).analyze("utterance-1", tmp_path / "a.wav", "")  # type: ignore[attr-defined]

    seen = recorder.seen[0]
    assert seen.candidate_completed is True
    assert seen.error_category == "none"
    assert seen.top_labels_agree is True
    assert seen.candidate_abstained is False
    assert seen.candidate_uncertain is False
    assert seen.candidate_raw_confidence == pytest.approx(0.91)
    assert seen.candidate_valid is True


def test_disagreement_and_abstention_are_both_observable(tmp_path: Path) -> None:
    calibration = EmotionCalibration(
        calibration_id="c-v2",
        method="temperature-scaling",
        temperature=1.5,
        abstain_threshold=0.9,
        abstained=True,
        raw_confidence=0.55,
    )
    primary = _Provider(_result(0.8, label="anger"))
    candidate = _Provider(_result(0.5, label="sadness", name="c", calibration=calibration))
    recorder = _Recorder()

    _shadow(primary, candidate, recorder).analyze("utterance-1", tmp_path / "a.wav", "")  # type: ignore[attr-defined]

    seen = recorder.seen[0]
    assert seen.top_labels_agree is False
    assert seen.candidate_abstained is True
    assert seen.candidate_uncertain is True


def test_an_observation_never_carries_an_utterance_id_or_a_path(tmp_path: Path) -> None:
    primary = _Provider(_result())
    recorder = _Recorder()

    _shadow(primary, _Provider(_result(name="c")), recorder).analyze(  # type: ignore[attr-defined]
        "secret-utterance-id", tmp_path / "secret-audio.wav", "secret transcript"
    )

    rendered = repr(recorder.seen[0])
    assert "secret-utterance-id" not in rendered
    assert "secret-audio" not in rendered
    assert "secret transcript" not in rendered
    assert not hasattr(recorder.seen[0], "utterance_id")
    assert not hasattr(recorder.seen[0], "audio_path")
    assert not hasattr(recorder.seen[0], "probabilities")


def test_unload_releases_both_sides(tmp_path: Path) -> None:
    unloaded: list[str] = []

    class Unloadable(_Provider):
        def __init__(self, tag: str) -> None:
            super().__init__(_result())
            self.tag = tag

        def unload(self) -> None:
            unloaded.append(self.tag)

    shadow = _shadow(Unloadable("primary"), Unloadable("candidate"), _Recorder())
    shadow.unload()  # type: ignore[attr-defined]

    assert sorted(unloaded) == ["candidate", "primary"]


def test_shadow_reports_the_primary_provenance_so_downstream_sees_no_change() -> None:
    primary = _Provider(_result())
    shadow = _shadow(primary, _Provider(_result(name="c")), _Recorder())

    assert shadow.provenance == primary.provenance  # type: ignore[attr-defined]


# --- review fixes: process control, timeout latching, candidate RSS ---


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), SystemExit(1)])
def test_process_control_exceptions_are_never_swallowed(
    tmp_path: Path, failure: BaseException
) -> None:
    """A shadow may contain provider faults; it may not contain Ctrl-C or exit."""

    class Hostile:
        provenance = _provenance("candidate")

        def analyze(self, utterance_id: str, audio_path: Path, transcript: str) -> EmotionResult:
            del utterance_id, audio_path, transcript
            raise failure

    shadow = _shadow(_Provider(_result()), Hostile(), _Recorder())

    with pytest.raises(type(failure)):
        shadow.analyze("utterance-1", tmp_path / "a.wav", "")  # type: ignore[attr-defined]


def test_memory_error_from_the_candidate_is_still_contained(tmp_path: Path) -> None:
    recorder = _Recorder()

    observed = _shadow(_Provider(_result()), _Provider(error=MemoryError()), recorder).analyze(  # type: ignore[attr-defined]
        "utterance-1", tmp_path / "a.wav", ""
    )

    assert observed.confidence == pytest.approx(0.8)
    assert recorder.seen[0].error_category == "unexpected_error"


def test_after_a_timeout_the_candidate_is_disabled_and_never_re_entered(
    tmp_path: Path,
) -> None:
    """The abandoned worker may still hold the shared model, so do not overlap on it."""

    class SlowOnce:
        provenance = _provenance("candidate")

        def __init__(self) -> None:
            self.calls = 0

        def analyze(self, utterance_id: str, audio_path: Path, transcript: str) -> EmotionResult:
            del utterance_id, audio_path, transcript
            self.calls += 1
            time.sleep(2.0)
            return _result(name="candidate")

    candidate = SlowOnce()
    recorder = _Recorder()
    shadow = _shadow(_Provider(_result()), candidate, recorder, timeout_seconds=0.05)

    for _ in range(3):
        shadow.analyze("utterance-1", tmp_path / "a.wav", "")  # type: ignore[attr-defined]

    assert shadow.candidate_timed_out is True  # type: ignore[attr-defined]
    assert candidate.calls == 1
    assert [seen.error_category for seen in recorder.seen] == ["timeout"] * 3
    assert [seen.candidate_attempted for seen in recorder.seen] == [True, False, False]


def test_candidate_peak_rss_is_observed_for_the_memory_gate(tmp_path: Path) -> None:
    candidate = _Provider(
        _result(name="candidate").model_copy(
            update={"usage": ProviderUsage(latency_ms=9.0, peak_rss_mb=4096.0)}
        )
    )
    recorder = _Recorder()

    _shadow(_Provider(_result()), candidate, recorder).analyze(  # type: ignore[attr-defined]
        "utterance-1", tmp_path / "a.wav", ""
    )

    assert recorder.seen[0].candidate_peak_rss_mb == pytest.approx(4096.0)


def test_a_candidate_without_usage_reports_no_rss_rather_than_zero(tmp_path: Path) -> None:
    candidate = _Provider(_result(name="candidate").model_copy(update={"usage": None}))
    recorder = _Recorder()

    _shadow(_Provider(_result()), candidate, recorder).analyze(  # type: ignore[attr-defined]
        "utterance-1", tmp_path / "a.wav", ""
    )

    assert recorder.seen[0].candidate_peak_rss_mb is None
    assert recorder.seen[0].candidate_completed is True


def test_a_non_finite_confidence_is_treated_as_an_invalid_candidate_output() -> None:
    from voxdelta.providers.shadow_emotion import valid_emotion_result

    assert valid_emotion_result(_result()) is True
    assert valid_emotion_result(object()) is False


def test_an_observation_cannot_claim_success_with_a_failure_category() -> None:
    with pytest.raises(ValueError, match="candidate_completed"):
        ShadowObservation(
            candidate_attempted=True,
            candidate_completed=True,
            error_category="timeout",
            candidate_valid=True,
        )

    with pytest.raises(ValueError, match="valid result"):
        ShadowObservation(
            candidate_attempted=True,
            candidate_completed=True,
            error_category="none",
            candidate_valid=False,
        )
