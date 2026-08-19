"""Provider boundaries for model-backed pipeline stages."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from voxdelta.domain.models import (
    AnalysisReport,
    AudioAsset,
    EmotionResult,
    ProviderProvenance,
    ResponseStrategyResult,
    SpeakerSegment,
    Utterance,
)


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
