"""Tests for the bounded ASR fallback.

The wrapper's value is entirely in where it refuses to help. A fallback that triggers on
any failure would turn every fail-closed refusal — a malformed decode, a word timeline
that failed timestamp sanitation, an alignment that would cross a speaker boundary — into
a quietly different answer from a different model. So the tests are organised around that
line: what may be retried, what must not be, and what the report says afterwards so a
reader is never told the wrong model produced the transcript.

Neither provider here is real; both are stubs. Nothing loads a model.
"""

from __future__ import annotations

from typing import Any

import pytest

from voxdelta.domain.models import AudioAsset, ProviderProvenance, SpeakerSegment, Utterance
from voxdelta.providers.base import ProviderError
from voxdelta.providers.fallback_asr import (
    FALLBACK_ELIGIBLE_CODES,
    FallbackTranscriptionProvider,
)

CONTRACT_CODES = (
    "invalid_provider_output",
    "invalid_audio_asset",
    "unsupported_speaker_count",
)


def _utterance(text: str) -> Utterance:
    return Utterance(
        id="utt-0001", start=0.0, end=1.0, speaker_id="SPEAKER_00", confidence=1.0, transcript=text
    )


class StubProvider:
    def __init__(
        self,
        name: str,
        *,
        raises: str | None = None,
        coverage: object | None = None,
    ) -> None:
        self.provenance = ProviderProvenance(name=name, model=f"{name}-model", remote=False)
        self._raises = raises
        self.calls = 0
        self.unloaded = 0
        self.last_warning_count = 0
        if coverage is not None:
            self.last_timestamp_coverage = coverage

    def transcribe(self, asset: AudioAsset, segments: list[SpeakerSegment]) -> list[Utterance]:
        self.calls += 1
        if self._raises:
            raise ProviderError(self._raises)  # type: ignore[arg-type]
        return [_utterance(self.provenance.name)]

    def unload(self) -> None:
        self.unloaded += 1


def _asset() -> AudioAsset:
    return AudioAsset(
        source_name="call.wav",
        source_path="private/call.wav",
        normalized_paths=("normalized/mixed.wav",),
        channel_mode="mixed",
        duration_seconds=10.0,
        channels=1,
        sha256="0" * 64,
    )


def _segments() -> list[SpeakerSegment]:
    return [
        SpeakerSegment(start=0.0, end=5.0, speaker_id="SPEAKER_00", confidence=1.0),
        SpeakerSegment(start=5.0, end=10.0, speaker_id="SPEAKER_01", confidence=1.0),
    ]


def _run(primary: StubProvider, fallback: StubProvider) -> Any:
    wrapper = FallbackTranscriptionProvider(primary, fallback)
    return wrapper, wrapper.transcribe(_asset(), _segments())


# ------------------------------------------------------------------------- the happy path


def test_the_primary_runs_alone_when_it_succeeds() -> None:
    primary, fallback = StubProvider("qwen3-asr"), StubProvider("faster-whisper")

    wrapper, utterances = _run(primary, fallback)

    assert [item.transcript for item in utterances] == ["qwen3-asr"]
    assert fallback.calls == 0
    assert wrapper.last_fallback is None
    assert wrapper.provenance.name == "qwen3-asr"


# --------------------------------------------------------------------- fallback activation


@pytest.mark.parametrize("code", sorted(FALLBACK_ELIGIBLE_CODES))
def test_a_provider_that_could_not_run_falls_back(code: str) -> None:
    """These codes say nothing about the audio, so a second local attempt is reasonable."""

    primary = StubProvider("qwen3-asr", raises=code)
    fallback = StubProvider("faster-whisper")

    wrapper, utterances = _run(primary, fallback)

    assert [item.transcript for item in utterances] == ["faster-whisper"]
    assert wrapper.last_fallback is not None
    assert wrapper.last_fallback.code == code
    assert wrapper.last_fallback.primary == "qwen3-asr"
    assert wrapper.last_fallback.fallback == "faster-whisper"


def test_the_reported_provenance_names_the_model_that_actually_ran() -> None:
    """A report must never attribute one model's transcript to the other."""

    primary = StubProvider("qwen3-asr", raises="provider_unavailable")
    wrapper, _utterances = _run(primary, StubProvider("faster-whisper"))

    assert wrapper.provenance.name == "faster-whisper"
    assert wrapper.provenance.model == "faster-whisper-model"


def test_the_failed_primary_is_unloaded_so_it_cannot_hold_the_device() -> None:
    primary = StubProvider("qwen3-asr", raises="provider_runtime_unsupported")

    _run(primary, StubProvider("faster-whisper"))

    assert primary.unloaded == 1


def test_a_fallback_does_not_inherit_the_primarys_timestamp_coverage() -> None:
    """Coverage describes the run that produced the transcript, not the one that failed."""

    primary = StubProvider("qwen3-asr", raises="provider_timeout", coverage={"omitted_words": 9})
    wrapper, _utterances = _run(primary, StubProvider("faster-whisper"))

    assert wrapper.last_timestamp_coverage is None


def test_coverage_is_reported_when_the_primary_succeeds() -> None:
    primary = StubProvider("qwen3-asr", coverage={"omitted_words": 3})

    wrapper, _utterances = _run(primary, StubProvider("faster-whisper"))

    assert wrapper.last_timestamp_coverage == {"omitted_words": 3}


# ------------------------------------------------------- no fallback on contract failures


@pytest.mark.parametrize("code", CONTRACT_CODES)
def test_a_data_contract_failure_is_never_retried_on_the_other_model(code: str) -> None:
    """The point of failing closed is that a second opinion cannot overrule it."""

    primary = StubProvider("qwen3-asr", raises=code)
    fallback = StubProvider("faster-whisper")
    wrapper = FallbackTranscriptionProvider(primary, fallback)

    with pytest.raises(ProviderError) as raised:
        wrapper.transcribe(_asset(), _segments())

    assert raised.value.code == code
    assert fallback.calls == 0
    assert wrapper.last_fallback is None


def test_a_timestamp_sanitation_refusal_reaches_the_caller_unchanged() -> None:
    """Sanitation and unsafe-alignment refusals both surface as invalid_provider_output."""

    wrapper = FallbackTranscriptionProvider(
        StubProvider("qwen3-asr", raises="invalid_provider_output"),
        StubProvider("faster-whisper"),
    )

    with pytest.raises(ProviderError) as raised:
        wrapper.transcribe(_asset(), _segments())

    assert raised.value.code == "invalid_provider_output"


def test_the_eligible_set_is_an_allowlist_that_excludes_contract_codes() -> None:
    """A failure mode nobody has classified must not fall back by default."""

    assert FALLBACK_ELIGIBLE_CODES.isdisjoint(CONTRACT_CODES)
    assert FALLBACK_ELIGIBLE_CODES == {
        "provider_unavailable",
        "provider_runtime_unsupported",
        "provider_timeout",
    }


# ------------------------------------------------------------------------------ lifecycle


def test_a_later_success_clears_the_earlier_fallback_record() -> None:
    """Stale state would make a healthy run look like it had fallen back."""

    primary = StubProvider("qwen3-asr", raises="provider_unavailable")
    fallback = StubProvider("faster-whisper")
    wrapper = FallbackTranscriptionProvider(primary, fallback)
    wrapper.transcribe(_asset(), _segments())
    assert wrapper.last_fallback is not None

    primary._raises = None
    wrapper.transcribe(_asset(), _segments())

    assert wrapper.last_fallback is None
    assert wrapper.provenance.name == "qwen3-asr"


def test_unload_releases_both_providers() -> None:
    primary, fallback = StubProvider("qwen3-asr"), StubProvider("faster-whisper")

    FallbackTranscriptionProvider(primary, fallback).unload()

    assert primary.unloaded == 1
    assert fallback.unloaded == 1
