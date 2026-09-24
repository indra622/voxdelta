from __future__ import annotations

import pytest

from voxdelta.domain.models import (
    AnalysisReport,
    CallSummary,
    EmotionResult,
    ProviderProvenance,
    Role,
    Utterance,
)
from voxdelta.expert_mode import (
    ExpertGuidance,
    ExpertHandoffStore,
    ExpertHypothesis,
    ExpertObservation,
    build_expert_request,
)


def _report() -> AnalysisReport:
    provider = ProviderProvenance(name="local", model="test", remote=False)
    return AnalysisReport(
        job_id="0123456789abcdef0123456789abcdef",
        summary=CallSummary(
            start_state="stable",
            end_state="dissatisfied",
            peak_customer_utterance_id="customer-2",
            overall_delta=0.3,
            valid_coverage=1.0,
            recovery_count=0,
            worsening_count=1,
        ),
        utterances=[
            Utterance(
                id="customer-1",
                start=0,
                end=1,
                speaker_id="speaker-1",
                role=Role.CUSTOMER,
                transcript="처리가 언제 되는지 걱정돼요.",
                confidence=1,
            ),
            Utterance(
                id="agent-1",
                start=1,
                end=2,
                speaker_id="speaker-2",
                role=Role.AGENT,
                transcript="절차를 안내드리겠습니다.",
                confidence=1,
            ),
            Utterance(
                id="customer-2",
                start=2,
                end=3,
                speaker_id="speaker-1",
                role=Role.CUSTOMER,
                transcript="결과가 늦어질까 봐 불안합니다.",
                confidence=1,
            ),
        ],
        emotions=[
            EmotionResult(
                utterance_id="customer-1",
                probabilities={
                    "happiness": 0.05,
                    "anger": 0.05,
                    "disgust": 0.05,
                    "fear": 0.55,
                    "neutral": 0.1,
                    "sadness": 0.1,
                    "surprise": 0.1,
                },
                operational_state="dissatisfied",
                negative_intensity=0.7,
                confidence=0.6,
                provider=provider,
            ),
            EmotionResult(
                utterance_id="customer-2",
                probabilities={
                    "happiness": 0.05,
                    "anger": 0.05,
                    "disgust": 0.05,
                    "fear": 0.65,
                    "neutral": 0.05,
                    "sadness": 0.05,
                    "surprise": 0.1,
                },
                operational_state="escalated",
                negative_intensity=0.85,
                confidence=0.8,
                provider=provider,
            ),
        ],
        strategies=[],
        transitions=[],
    )


def _guidance() -> ExpertGuidance:
    return ExpertGuidance(
        observations=[
            ExpertObservation(
                evidence_turn_ids=["customer-2"],
                statement="고객이 결과 지연에 대한 걱정을 표현했습니다.",
            )
        ],
        hypotheses=[
            ExpertHypothesis(
                statement="가설: 처리 일정의 불확실성이 불안에 영향을 주었을 수 있습니다.",
                confidence="low",
            )
        ],
        suggested_message="걱정되시는 부분을 함께 확인해 보겠습니다.",
        next_question="가장 걱정되는 일정이 무엇인지 알려주실 수 있을까요?",
        safety_level="watch",
    )


def test_build_request_is_minimal_and_marks_low_confidence_evidence_uncertain() -> None:
    request = build_expert_request(_report(), target="claude")

    assert request.transport == "acp"
    assert request.target == "claude"
    assert request.transcription_uncertain is True
    assert {item.utterance_id for item in request.evidence} == {
        "customer-1",
        "customer-2",
        "agent-1",
    }
    assert len(request.evidence) <= 6


def test_store_rejects_guidance_that_cites_text_not_in_queued_packet(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = ExpertHandoffStore(tmp_path)
    request = build_expert_request(_report(), target="claude")
    queued = store.queue(request)
    invalid = _guidance().model_copy(
        update={
            "observations": [
                ExpertObservation(
                    evidence_turn_ids=["not-in-packet"], statement="unsupported"
                )
            ]
        }
    )

    with pytest.raises(ValueError, match="outside the evidence packet"):
        store.submit(
            request.job_id,
            request_sha256=queued.request_sha256,
            guidance=invalid,
        )


def test_store_persists_only_validated_guidance(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = ExpertHandoffStore(tmp_path)
    request = build_expert_request(_report(), target="codex")
    queued = store.queue(request)

    ready = store.submit(
        request.job_id,
        request_sha256=queued.request_sha256,
        guidance=_guidance(),
    )

    assert ready.status == "ready"
    assert ready.guidance == _guidance()
    assert store.read(request.job_id) == ready
