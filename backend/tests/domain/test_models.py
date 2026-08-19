from __future__ import annotations

import pytest
from pydantic import ValidationError

from voxdelta.domain.models import (
    AnalysisReport,
    AudioAsset,
    CallSummary,
    EmotionResult,
    EmotionTransition,
    ProviderProvenance,
    ProviderUsage,
    ResponseStrategyResult,
    Role,
    SpeakerSegment,
    StageName,
    StageStatus,
    Utterance,
)


def local_provider() -> ProviderProvenance:
    return ProviderProvenance(name="fake", model="v1", remote=False)


def valid_probabilities() -> dict[str, float]:
    return {
        "happiness": 0.1,
        "anger": 0.1,
        "disgust": 0.1,
        "fear": 0.1,
        "neutral": 0.4,
        "sadness": 0.1,
        "surprise": 0.1,
    }


def test_role_and_stage_enums_have_canonical_values() -> None:
    assert [role.value for role in Role] == ["customer", "agent", "unknown"]
    assert [stage.value for stage in StageName] == [
        "normalize",
        "diarize",
        "transcribe",
        "confirm_roles",
        "emotion",
        "response_strategy",
        "transitions",
        "report",
    ]
    assert [status.value for status in StageStatus] == [
        "pending",
        "running",
        "paused",
        "completed",
        "failed",
        "skipped",
    ]


def test_provider_contracts_reject_unknown_transmission_and_negative_usage() -> None:
    with pytest.raises(ValidationError):
        ProviderProvenance(name="fake", model="v1", remote=True, transmits=("credentials",))

    with pytest.raises(ValidationError):
        ProviderUsage(latency_ms=-1)


def test_audio_asset_accepts_normalized_local_metadata() -> None:
    asset = AudioAsset(
        source_name="call.wav",
        source_path="incoming/call.wav",
        normalized_paths=("jobs/j1/audio/mixed.wav",),
        channel_mode="mixed",
        duration_seconds=65.0,
        channels=1,
        sha256="0" * 64,
    )

    assert asset.normalized_paths == ("jobs/j1/audio/mixed.wav",)


def test_speaker_segment_rejects_non_positive_interval() -> None:
    with pytest.raises(ValidationError):
        SpeakerSegment(
            start=2.0,
            end=2.0,
            speaker_id="SPEAKER_00",
            confidence=0.8,
        )


def test_utterance_rejects_non_positive_interval() -> None:
    with pytest.raises(ValidationError):
        Utterance(
            id="u1",
            start=2.0,
            end=1.0,
            speaker_id="SPEAKER_00",
            role=Role.CUSTOMER,
            transcript="안녕하세요",
            confidence=0.8,
        )


def test_emotion_probabilities_must_include_all_seven_labels() -> None:
    probabilities = valid_probabilities()
    probabilities.pop("surprise")

    with pytest.raises(ValidationError):
        EmotionResult(
            utterance_id="u1",
            probabilities=probabilities,
            operational_state="stable",
            negative_intensity=0.2,
            confidence=0.8,
            provider=local_provider(),
        )


def test_emotion_probabilities_must_sum_to_one() -> None:
    with pytest.raises(ValidationError):
        EmotionResult(
            utterance_id="u1",
            probabilities={
                "happiness": 0.1,
                "anger": 0.1,
                "disgust": 0.1,
                "fear": 0.1,
                "neutral": 0.1,
                "sadness": 0.1,
                "surprise": 0.1,
            },
            operational_state="stable",
            negative_intensity=0.2,
            confidence=0.8,
            provider=local_provider(),
        )


def test_emotion_probabilities_must_be_bounded() -> None:
    with pytest.raises(ValidationError):
        EmotionResult(
            utterance_id="u1",
            probabilities={
                "happiness": 1.1,
                "anger": -0.1,
                "disgust": 0.0,
                "fear": 0.0,
                "neutral": 0.0,
                "sadness": 0.0,
                "surprise": 0.0,
            },
            operational_state="stable",
            negative_intensity=0.2,
            confidence=0.8,
            provider=local_provider(),
        )


def test_transition_and_summary_reject_out_of_range_values() -> None:
    with pytest.raises(ValidationError):
        EmotionTransition(
            previous_customer_id="u1",
            agent_id="u2",
            next_customer_id="u3",
            delta=1.1,
            classification="recovery",
        )

    with pytest.raises(ValidationError):
        CallSummary(
            start_state="dissatisfied",
            end_state="stable",
            peak_customer_utterance_id="u1",
            overall_delta=0.2,
            valid_coverage=1.1,
            recovery_count=1,
            worsening_count=0,
        )


def test_analysis_report_composes_canonical_results() -> None:
    utterance = Utterance(
        id="u1",
        start=0.0,
        end=1.0,
        speaker_id="SPEAKER_00",
        role=Role.CUSTOMER,
        transcript="안녕하세요",
        confidence=0.9,
    )
    emotion = EmotionResult(
        utterance_id="u1",
        probabilities=valid_probabilities(),
        operational_state="stable",
        negative_intensity=0.2,
        confidence=0.8,
        provider=local_provider(),
    )
    strategy = ResponseStrategyResult(
        utterance_id="u1",
        primary="greeting_closing",
        confidence=0.9,
        provider=local_provider(),
    )
    summary = CallSummary(
        start_state="stable",
        end_state="stable",
        peak_customer_utterance_id="u1",
        overall_delta=0.0,
        valid_coverage=1.0,
        recovery_count=0,
        worsening_count=0,
    )

    report = AnalysisReport(
        job_id="j1",
        summary=summary,
        utterances=[utterance],
        emotions=[emotion],
        strategies=[strategy],
        transitions=[],
    )

    assert report.schema_version == "1"
    assert report.warnings == []
