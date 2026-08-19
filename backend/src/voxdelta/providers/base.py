"""Provider boundaries for model-backed pipeline stages."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from voxdelta.domain.models import (
    AnalysisReport,
    AudioAsset,
    EmotionResult,
    ProviderProvenance,
    ResponseStrategyResult,
    SpeakerSegment,
    Utterance,
)

ProviderErrorCode = Literal[
    "invalid_audio_asset",
    "invalid_provider_output",
    "missing_huggingface_token",
    "missing_pyannote_api_key",
    "provider_timeout",
    "provider_unavailable",
    "unsupported_speaker_count",
]

_PROVIDER_ERROR_MESSAGES: dict[ProviderErrorCode, str] = {
    "invalid_audio_asset": "The normalized audio asset is invalid.",
    "invalid_provider_output": "The diarization provider returned invalid output.",
    "missing_huggingface_token": "A Hugging Face token is required for this model.",
    "missing_pyannote_api_key": "A pyannoteAI API key is required for this provider.",
    "provider_timeout": "The diarization provider timed out.",
    "provider_unavailable": "The diarization provider is unavailable.",
    "unsupported_speaker_count": "Exactly two observed speakers are required.",
}


class ProviderError(RuntimeError):
    """Typed provider failure whose public text never includes provider payloads."""

    def __init__(self, code: ProviderErrorCode) -> None:
        self.code = code
        super().__init__(_PROVIDER_ERROR_MESSAGES[code])


@runtime_checkable
class DiarizationProvider(Protocol):
    provenance: ProviderProvenance

    def diarize(self, asset: AudioAsset) -> list[SpeakerSegment]: ...


@runtime_checkable
class TranscriptionProvider(Protocol):
    provenance: ProviderProvenance

    def transcribe(self, asset: AudioAsset, segments: list[SpeakerSegment]) -> list[Utterance]: ...


@runtime_checkable
class EmotionProvider(Protocol):
    provenance: ProviderProvenance

    def analyze(self, utterance_id: str, audio_path: Path, transcript: str) -> EmotionResult: ...


@runtime_checkable
class ResponseStrategyProvider(Protocol):
    provenance: ProviderProvenance

    def classify(
        self, utterance: Utterance, context: list[Utterance]
    ) -> ResponseStrategyResult: ...


@runtime_checkable
class ReportSummaryProvider(Protocol):
    provenance: ProviderProvenance

    def summarize(self, report: AnalysisReport) -> str: ...
