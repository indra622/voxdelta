from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from voxdelta.domain.models import AnalysisReport, AudioAsset, Role, Utterance
from voxdelta.providers.base import (
    DiarizationProvider,
    EmotionProvider,
    ReportSummaryProvider,
    ResponseStrategyProvider,
    TranscriptionProvider,
)
from voxdelta.providers.fake import (
    FakeDiarizationProvider,
    FakeEmotionProvider,
    FakeReportSummaryProvider,
    FakeResponseStrategyProvider,
    FakeTranscriptionProvider,
)


def _asset(duration_seconds: float | None = 65.0) -> AudioAsset:
    return AudioAsset(
        source_name="call.wav",
        source_path="call.wav",
        normalized_paths=("call.wav",),
        channel_mode="mixed",
        duration_seconds=duration_seconds,
        channels=1,
        sha256="0" * 64,
    )


def _utterance(transcript: str) -> Utterance:
    return Utterance(
        id="u1",
        start=0.0,
        end=1.0,
        speaker_id="SPEAKER_01",
        role=Role.AGENT,
        transcript=transcript,
        confidence=1.0,
    )


def test_fake_providers_conform_to_runtime_protocols_and_local_provenance() -> None:
    providers = (
        (FakeDiarizationProvider(), DiarizationProvider),
        (FakeTranscriptionProvider(), TranscriptionProvider),
        (FakeEmotionProvider(), EmotionProvider),
        (FakeResponseStrategyProvider(), ResponseStrategyProvider),
        (FakeReportSummaryProvider(), ReportSummaryProvider),
    )

    for provider, protocol in providers:
        assert isinstance(provider, protocol)
        assert provider.provenance.name.startswith("fake-")
        assert provider.provenance.model == "deterministic-v1"
        assert provider.provenance.remote is False
        assert provider.provenance.transmits == ()
        assert provider.provenance.retention_policy_url is None
        assert provider.provenance.schema_version == "1"


@pytest.mark.parametrize("duration_seconds", [None, 0.0, 0.001, 0.5, 65.0])
def test_fake_diarizer_returns_two_ordered_speakers_for_edge_durations(
    duration_seconds: float | None,
) -> None:
    provider = FakeDiarizationProvider()
    asset = _asset(duration_seconds)
    segments = provider.diarize(asset)

    assert segments == provider.diarize(asset)
    assert {segment.speaker_id for segment in segments} == {"SPEAKER_00", "SPEAKER_01"}
    assert all(segment.start >= 0 for segment in segments)
    assert all(segment.end > segment.start for segment in segments)
    assert all(segment.confidence == 1.0 for segment in segments)
    assert all(
        current.end == following.start
        for current, following in zip(segments, segments[1:], strict=False)
    )

    if duration_seconds is not None and duration_seconds > 0:
        assert segments[-1].end == duration_seconds


def test_fake_transcript_is_ordered_and_bound_to_diarization_segments() -> None:
    asset = _asset()
    segments = FakeDiarizationProvider().diarize(asset)

    provider = FakeTranscriptionProvider()
    utterances = provider.transcribe(asset, segments)

    assert utterances == provider.transcribe(asset, segments)
    assert len(utterances) == len(segments)
    assert [utterance.start for utterance in utterances] == sorted(
        utterance.start for utterance in utterances
    )
    assert len({utterance.id for utterance in utterances}) == len(utterances)
    for utterance, segment in zip(utterances, segments, strict=True):
        assert (utterance.start, utterance.end) == (segment.start, segment.end)
        assert utterance.speaker_id == segment.speaker_id
        assert utterance.role is Role.UNKNOWN
        assert utterance.transcript.strip()


def test_fake_emotion_is_deterministic_without_mutating_global_random_state() -> None:
    provider = FakeEmotionProvider()
    random.seed(1729)
    state_before = random.getstate()

    first = provider.analyze("u1", Path("slice.wav"), "정말 화가 납니다")
    second = provider.analyze("u1", Path("another-slice.wav"), "정말 화가 납니다")

    assert first == second
    assert random.getstate() == state_before
    assert set(first.probabilities) == {
        "happiness",
        "anger",
        "disgust",
        "fear",
        "neutral",
        "sadness",
        "surprise",
    }
    assert all(math.isfinite(value) and value >= 0 for value in first.probabilities.values())
    assert sum(first.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    assert first.provider == provider.provenance
    assert first.provider.schema_version == "1"


def test_fake_emotion_is_stable_across_python_hash_seeds() -> None:
    script = """
import json
from pathlib import Path
from voxdelta.providers.fake import FakeEmotionProvider
result = FakeEmotionProvider().analyze("u1", Path("slice.wav"), "정말 화가 납니다")
print(json.dumps(result.model_dump(mode="json"), sort_keys=True, ensure_ascii=False))
"""
    outputs = []
    for hash_seed in ("1", "987654"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = hash_seed
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        outputs.append(json.loads(completed.stdout))

    assert outputs[0] == outputs[1]


@pytest.mark.parametrize(
    ("transcript", "expected"),
    [
        ("죄송하지만 바로 해결해 드릴 수 없는 규정입니다", "apology"),
        ("바로 해결하고 처리해 드리겠습니다", "solution"),
        ("정책상 처리가 불가합니다", "policy_refusal"),
        ("요청하신 내용을 확인했습니다", "information"),
    ],
)
def test_fake_strategy_uses_deterministic_keyword_precedence(
    transcript: str, expected: str
) -> None:
    provider = FakeResponseStrategyProvider()

    result = provider.classify(_utterance(transcript), [_utterance("이전 문맥")])

    assert result == provider.classify(_utterance(transcript), [_utterance("이전 문맥")])
    assert result.primary == expected
    assert result.provider == provider.provenance
    assert result.utterance_id == "u1"


def test_fake_report_summary_uses_only_safe_aggregate_fields() -> None:
    report = AnalysisReport(
        job_id="secret-job-id",
        summary={
            "start_state": "dissatisfied",
            "end_state": "stable",
            "peak_customer_utterance_id": "secret-utterance-id",
            "overall_delta": -0.25,
            "valid_coverage": 1.0,
            "recovery_count": 2,
            "worsening_count": 1,
            "narrative": "raw-provider-payload-secret",
        },
        utterances=[_utterance("customer-secret-token")],
        emotions=[],
        strategies=[],
        transitions=[],
        warnings=["authorization-secret-value"],
    )
    provider = FakeReportSummaryProvider()

    first = provider.summarize(report)
    second = provider.summarize(report)

    assert first == second
    assert "dissatisfied" in first
    assert "stable" in first
    assert "2" in first
    assert "1" in first
    for secret in (
        "secret-job-id",
        "secret-utterance-id",
        "raw-provider-payload-secret",
        "customer-secret-token",
        "authorization-secret-value",
    ):
        assert secret not in first
