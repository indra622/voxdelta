"""Deterministic local providers for pipeline and contract testing."""

from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path
from typing import Literal

from voxdelta.domain.models import (
    AnalysisReport,
    AudioAsset,
    EmotionLabel,
    EmotionResult,
    OperationalState,
    ProviderProvenance,
    ResponseStrategyResult,
    SpeakerSegment,
    Utterance,
)
from voxdelta.providers.base import (
    DiarizationProvider,
    EmotionProvider,
    ReportSummaryProvider,
    ResponseStrategyProvider,
    TranscriptionProvider,
)

_MODEL_VERSION = "deterministic-v1"
_SCHEMA_VERSION = "1"
_FALLBACK_DURATION_SECONDS = 2.0
_SEGMENT_TARGET_SECONDS = 30.0
_EMOTION_LABELS: tuple[EmotionLabel, ...] = (
    "happiness",
    "anger",
    "disgust",
    "fear",
    "neutral",
    "sadness",
    "surprise",
)
_NEGATIVE_EMOTION_LABELS: tuple[EmotionLabel, ...] = (
    "anger",
    "disgust",
    "fear",
    "sadness",
)
_TRANSCRIPT_LINES = (
    "안녕하세요. 문의 내용을 말씀해 주세요.",
    "확인 후 안내해 드리겠습니다.",
    "요청하신 내용을 다시 확인하겠습니다.",
    "처리 결과를 안내해 드렸습니다.",
)
_FakeStrategy = Literal["apology", "solution", "policy_refusal", "information"]


def _local_provenance(name: str) -> ProviderProvenance:
    return ProviderProvenance(
        name=name,
        model=_MODEL_VERSION,
        remote=False,
        transmits=(),
        retention_policy_url=None,
        schema_version=_SCHEMA_VERSION,
    )


class FakeDiarizationProvider(DiarizationProvider):
    """Produce a bounded alternating two-speaker timeline."""

    provenance = _local_provenance("fake-diarization")

    def diarize(self, asset: AudioAsset) -> list[SpeakerSegment]:
        duration = asset.duration_seconds
        if duration is None or not math.isfinite(duration) or duration / 2 <= 0:
            duration = _FALLBACK_DURATION_SECONDS

        segment_count = max(2, math.ceil(duration / _SEGMENT_TARGET_SECONDS))
        segment_seconds = duration / segment_count
        segments: list[SpeakerSegment] = []
        for index in range(segment_count):
            start = index * segment_seconds
            end = duration if index == segment_count - 1 else (index + 1) * segment_seconds
            segments.append(
                SpeakerSegment(
                    start=start,
                    end=end,
                    speaker_id=f"SPEAKER_{index % 2:02d}",
                    confidence=1.0,
                )
            )
        return segments


class FakeTranscriptionProvider(TranscriptionProvider):
    """Attach stable Korean sample text to each diarization segment."""

    provenance = _local_provenance("fake-transcription")

    def transcribe(self, asset: AudioAsset, segments: list[SpeakerSegment]) -> list[Utterance]:
        del asset
        ordered_segments = sorted(
            segments,
            key=lambda segment: (segment.start, segment.end, segment.speaker_id),
        )
        return [
            Utterance(
                id=f"utterance-{index + 1:04d}",
                start=segment.start,
                end=segment.end,
                speaker_id=segment.speaker_id,
                overlap=segment.overlap,
                confidence=segment.confidence,
                transcript=_TRANSCRIPT_LINES[index % len(_TRANSCRIPT_LINES)],
            )
            for index, segment in enumerate(ordered_segments)
        ]


class FakeEmotionProvider(EmotionProvider):
    """Produce repeatable normalized emotion scores without external calls."""

    provenance = _local_provenance("fake-emotion")

    def analyze(self, utterance_id: str, audio_path: Path, transcript: str) -> EmotionResult:
        del audio_path
        digest = hashlib.sha256((utterance_id + transcript).encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
        generator = random.Random(seed)
        weights = [generator.random() + 1.0 for _ in _EMOTION_LABELS]
        total = math.fsum(weights)
        probabilities: dict[EmotionLabel, float] = {
            label: weight / total for label, weight in zip(_EMOTION_LABELS, weights, strict=True)
        }

        negative_intensity = math.fsum(probabilities[label] for label in _NEGATIVE_EMOTION_LABELS)
        dominant = max(_EMOTION_LABELS, key=probabilities.__getitem__)
        state_by_emotion: dict[EmotionLabel, OperationalState] = {
            "happiness": "satisfied",
            "anger": "escalated",
            "disgust": "dissatisfied",
            "fear": "dissatisfied",
            "neutral": "stable",
            "sadness": "dissatisfied",
            "surprise": "uncertain",
        }
        return EmotionResult(
            utterance_id=utterance_id,
            probabilities=probabilities,
            operational_state=state_by_emotion[dominant],
            negative_intensity=negative_intensity,
            confidence=0.85,
            provider=self.provenance,
        )


class FakeResponseStrategyProvider(ResponseStrategyProvider):
    """Classify a small deterministic subset of the response taxonomy."""

    provenance = _local_provenance("fake-response-strategy")
    _KEYWORDS: tuple[tuple[_FakeStrategy, tuple[str, ...]], ...] = (
        ("apology", ("죄송", "사과")),
        ("solution", ("해결", "조치", "도와드리", "교환해", "환불해")),
        ("policy_refusal", ("불가", "어렵", "규정", "정책", "안 됩니다")),
    )

    def classify(self, utterance: Utterance, context: list[Utterance]) -> ResponseStrategyResult:
        del context
        normalized = utterance.transcript.casefold()
        primary: _FakeStrategy = "information"
        for candidate, keywords in self._KEYWORDS:
            if any(keyword in normalized for keyword in keywords):
                primary = candidate
                break
        return ResponseStrategyResult(
            utterance_id=utterance.id,
            primary=primary,
            confidence=0.9 if primary != "information" else 0.75,
            provider=self.provenance,
        )


class FakeReportSummaryProvider(ReportSummaryProvider):
    """Summarize only aggregate fields that cannot expose provider payloads."""

    provenance = _local_provenance("fake-report-summary")

    def summarize(self, report: AnalysisReport) -> str:
        summary = report.summary
        return (
            f"통화 감정 상태는 {summary.start_state}에서 {summary.end_state}로 변화했습니다. "
            f"시간 순서상 회복 {summary.recovery_count}건과 "
            f"악화 {summary.worsening_count}건이 관찰되었습니다."
        )
