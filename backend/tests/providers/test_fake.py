from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from voxdelta.domain.models import AnalysisReport, AudioAsset, EmotionLabel, Role, Utterance
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
    _derive_operational_state,
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


def _probabilities(
    *,
    happiness: float,
    anger: float,
    disgust: float,
    fear: float,
    neutral: float,
    sadness: float,
    surprise: float,
) -> dict[EmotionLabel, float]:
    return {
        "happiness": happiness,
        "anger": anger,
        "disgust": disgust,
        "fear": fear,
        "neutral": neutral,
        "sadness": sadness,
        "surprise": surprise,
    }


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


@pytest.mark.parametrize("duration_seconds", [None, 0.0, -1.0, math.inf, math.nan])
def test_fake_diarizer_rejects_missing_non_finite_or_non_positive_duration(
    duration_seconds: float | None,
) -> None:
    provider = FakeDiarizationProvider()
    asset = _asset().model_copy(update={"duration_seconds": duration_seconds})

    with pytest.raises(ValueError, match="duration_seconds must be finite and positive"):
        provider.diarize(asset)


def test_fake_diarizer_rejects_duration_too_small_for_six_positive_segments() -> None:
    duration_seconds = math.nextafter(0.0, 1.0)
    asset = _asset(duration_seconds)

    with pytest.raises(ValueError, match="too small for six positive segments"):
        FakeDiarizationProvider().diarize(asset)


@pytest.mark.parametrize("duration_seconds", [1e-300, 0.001, 0.5, 65.0])
def test_fake_diarizer_returns_six_exactly_bounded_alternating_segments(
    duration_seconds: float,
) -> None:
    provider = FakeDiarizationProvider()
    asset = _asset(duration_seconds)
    segments = provider.diarize(asset)

    assert segments == provider.diarize(asset)
    assert len(segments) == 6
    assert [segment.speaker_id for segment in segments] == [
        "SPEAKER_00",
        "SPEAKER_01",
        "SPEAKER_00",
        "SPEAKER_01",
        "SPEAKER_00",
        "SPEAKER_01",
    ]
    assert segments[0].start == 0.0
    assert segments[-1].end == duration_seconds
    assert all(0.0 <= segment.start < segment.end <= duration_seconds for segment in segments)
    assert all(segment.confidence == 1.0 for segment in segments)
    assert all(
        current.end == following.start
        for current, following in zip(segments, segments[1:], strict=False)
    )


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


def test_fake_emotion_preserves_provider_confidence_and_maps_grouped_state() -> None:
    result = FakeEmotionProvider().analyze("u6", Path("slice.wav"), "x")

    assert result.confidence == 0.85
    assert result.operational_state == "dissatisfied"


@pytest.mark.parametrize(
    ("probabilities", "expected_state"),
    [
        (
            _probabilities(
                happiness=0.6,
                anger=0.05,
                disgust=0.05,
                fear=0.05,
                neutral=0.1,
                sadness=0.05,
                surprise=0.1,
            ),
            "satisfied",
        ),
        (
            _probabilities(
                happiness=0.1,
                anger=0.05,
                disgust=0.05,
                fear=0.05,
                neutral=0.6,
                sadness=0.05,
                surprise=0.1,
            ),
            "stable",
        ),
        (
            _probabilities(
                happiness=0.05,
                anger=0.6,
                disgust=0.05,
                fear=0.05,
                neutral=0.1,
                sadness=0.05,
                surprise=0.1,
            ),
            "escalated",
        ),
        (
            _probabilities(
                happiness=0.15,
                anger=0.05,
                disgust=0.21,
                fear=0.2,
                neutral=0.05,
                sadness=0.19,
                surprise=0.15,
            ),
            "dissatisfied",
        ),
    ],
)
def test_operational_state_compares_four_canonical_scores(
    probabilities: dict[EmotionLabel, float], expected_state: str
) -> None:
    state = _derive_operational_state(probabilities, confidence=0.85)

    assert state == expected_state


def test_operational_state_varies_confidence_independently_at_threshold() -> None:
    probabilities = _probabilities(
        happiness=0.15,
        anger=0.1,
        disgust=0.2,
        fear=0.15,
        neutral=0.1,
        sadness=0.2,
        surprise=0.1,
    )

    below_threshold = _derive_operational_state(probabilities, confidence=0.549999)
    at_threshold = _derive_operational_state(probabilities, confidence=0.55)

    assert below_threshold == "uncertain"
    assert at_threshold == "dissatisfied"


@pytest.mark.parametrize("confidence", [-0.001, 1.001, math.inf, math.nan])
def test_operational_state_rejects_invalid_provider_confidence(confidence: float) -> None:
    probabilities = _probabilities(
        happiness=0.6,
        anger=0.05,
        disgust=0.05,
        fear=0.05,
        neutral=0.1,
        sadness=0.05,
        surprise=0.1,
    )

    with pytest.raises(ValueError, match="confidence must be finite and between zero and one"):
        _derive_operational_state(probabilities, confidence=confidence)


@pytest.mark.parametrize("surprise", [0.2, 0.21])
def test_operational_state_treats_surprise_tie_or_dominance_as_uncertain(
    surprise: float,
) -> None:
    probabilities = _probabilities(
        happiness=0.2 if surprise == 0.2 else 0.19,
        anger=0.02,
        disgust=0.18,
        fear=0.18,
        neutral=0.03,
        sadness=0.19,
        surprise=surprise,
    )

    state = _derive_operational_state(probabilities, confidence=0.85)

    assert state == "uncertain"


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
