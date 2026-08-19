from __future__ import annotations

import math

import pytest
from pydantic import BaseModel, ValidationError

from voxdelta.evaluation.selection import (
    CandidateMetrics,
    select_asr_candidate,
)


def _candidate(
    candidate_id: str,
    provider: str,
    *,
    cer: float | None,
    latency: float | None = 100.0,
    memory: float | None = 1000.0,
    completion: float = 1.0,
    task: str = "asr",
) -> CandidateMetrics:
    return CandidateMetrics(
        candidate_id=candidate_id,
        task=task,  # type: ignore[arg-type]
        provider=provider,
        model=candidate_id,
        completion_rate=completion,
        median_latency_ms=latency,
        peak_rss_mb=memory,
        cer=cer,
    )


def test_qwen_wins_at_exact_absolute_cer_improvement() -> None:
    decision = select_asr_candidate(
        [
            _candidate("stable", "faster-whisper", cer=0.20),
            _candidate("modern", "qwen3-asr", cer=0.19, latency=200.0),
        ]
    )

    assert decision.selected_candidate_id == "modern"
    assert decision.candidate_status == {"stable": "rejected", "modern": "eligible"}
    assert set(decision.reasons) == {"stable", "modern"}


def test_qwen_loses_near_tie_when_slower_than_two_times() -> None:
    decision = select_asr_candidate(
        [
            _candidate("stable", "faster-whisper", cer=0.20, latency=100.0),
            _candidate("modern", "qwen3-asr", cer=0.195, latency=200.01),
        ]
    )

    assert decision.selected_candidate_id == "stable"


@pytest.mark.parametrize(
    "candidate",
    [
        _candidate("memory", "qwen3-asr", cer=0.1, memory=18432.01),
        _candidate("completion", "qwen3-asr", cer=0.1, completion=0.949999),
        _candidate("missing", "qwen3-asr", cer=None),
    ],
)
def test_unavailable_gate_records_reason(candidate: CandidateMetrics) -> None:
    decision = select_asr_candidate([candidate])

    assert decision.selected_candidate_id is None
    assert decision.candidate_status[candidate.candidate_id] == "unavailable"
    assert decision.reasons[candidate.candidate_id]


def test_explicit_unavailable_reason_is_preserved_safely() -> None:
    candidate = _candidate("qwen", "qwen3-asr", cer=0.1).model_copy(
        update={"unavailable_reason": "provider_runtime_unsupported"}
    )
    decision = select_asr_candidate([candidate])
    assert decision.candidate_status == {"qwen": "unavailable"}
    assert decision.reasons == {"qwen": "provider_runtime_unsupported"}


def test_arbitrary_unavailable_reason_is_never_persisted() -> None:
    sentinel = "/private/call.wav transcript-secret hf_token provider-payload"
    candidate = _candidate("qwen", "qwen3-asr", cer=0.1).model_copy(
        update={"unavailable_reason": sentinel}
    )

    decision = select_asr_candidate([candidate])

    assert decision.reasons == {"qwen": "provider_unavailable"}
    serialized = decision.model_dump_json().lower()
    for fragment in ("private", "transcript", "secret", "token", "payload"):
        assert fragment not in serialized


def test_candidate_contract_sanitizes_arbitrary_unavailable_reason() -> None:
    candidate = CandidateMetrics(
        candidate_id="qwen",
        task="asr",
        provider="qwen3-asr",
        model="Qwen3-ASR-1.7B",
        completion_rate=0.0,
        unavailable_reason="/private/audio.wav transcript-secret token payload",
    )

    assert candidate.unavailable_reason == "provider_unavailable"
    assert "private" not in candidate.model_dump_json().lower()


@pytest.mark.parametrize("bypass", ["copy", "construct"])
def test_unavailable_reason_serialization_fails_closed_after_validation_bypass(
    bypass: str,
) -> None:
    sentinel = "/private/audio.wav transcript-secret hf_token provider-payload"
    if bypass == "copy":
        candidate = _candidate("qwen", "qwen3-asr", cer=0.1).model_copy(
            update={"unavailable_reason": sentinel}
        )
    else:
        candidate = CandidateMetrics.model_construct(
            candidate_id="qwen",
            task="asr",
            provider="qwen3-asr",
            model="Qwen3-ASR-1.7B",
            completion_rate=0.0,
            unavailable_reason=sentinel,
        )

    assert candidate.model_dump()["unavailable_reason"] == "provider_unavailable"
    assert sentinel not in candidate.model_dump_json()
    assert sentinel not in repr(candidate)


def test_nested_candidate_serialization_fails_closed_after_model_construct() -> None:
    class Envelope(BaseModel):
        candidate: CandidateMetrics

    sentinel = "/private/audio.wav transcript-secret hf_token provider-payload"
    candidate = CandidateMetrics.model_construct(
        candidate_id="qwen",
        task="asr",
        provider="qwen3-asr",
        model="Qwen3-ASR-1.7B",
        completion_rate=0.0,
        unavailable_reason=sentinel,
    )
    envelope = Envelope(candidate=candidate)

    assert envelope.model_dump()["candidate"]["unavailable_reason"] == "provider_unavailable"
    assert sentinel not in envelope.model_dump_json()


@pytest.mark.parametrize(
    "unsafe_reason",
    [
        {"path": "/private/audio.wav", "token": "secret"},
        ["transcript-secret", "provider-payload"],
    ],
)
@pytest.mark.parametrize("bypass", ["copy", "construct"])
def test_unhashable_reason_bypasses_fail_closed_everywhere(
    unsafe_reason: object,
    bypass: str,
) -> None:
    class Envelope(BaseModel):
        candidate: CandidateMetrics

    if bypass == "copy":
        candidate = _candidate("qwen", "qwen3-asr", cer=0.1).model_copy(
            update={"unavailable_reason": unsafe_reason}
        )
    else:
        candidate = CandidateMetrics.model_construct(
            candidate_id="qwen",
            task="asr",
            provider="qwen3-asr",
            model="Qwen3-ASR-1.7B",
            completion_rate=0.0,
            median_latency_ms=100.0,
            peak_rss_mb=1000.0,
            cer=0.1,
            unavailable_reason=unsafe_reason,
        )

    assert candidate.model_dump()["unavailable_reason"] == "provider_unavailable"
    assert Envelope(candidate=candidate).model_dump()["candidate"]["unavailable_reason"] == (
        "provider_unavailable"
    )
    decision = select_asr_candidate([candidate])
    assert decision.reasons == {"qwen": "provider_unavailable"}
    serialized = candidate.model_dump_json() + Envelope(candidate=candidate).model_dump_json()
    serialized += decision.model_dump_json()
    for fragment in ("private", "transcript", "secret", "token", "payload"):
        assert fragment not in serialized.lower()


@pytest.mark.parametrize("unsafe_reason", [{"secret": "token"}, ["payload"]])
def test_normal_validation_remains_strict_for_non_string_reasons(
    unsafe_reason: object,
) -> None:
    with pytest.raises(ValidationError):
        CandidateMetrics.model_validate(
            {
                "candidate_id": "qwen",
                "task": "asr",
                "provider": "qwen3-asr",
                "model": "Qwen3-ASR-1.7B",
                "completion_rate": 0.0,
                "unavailable_reason": unsafe_reason,
            }
        )


def test_multiple_candidates_per_provider_select_best_and_reject_every_other() -> None:
    decision = select_asr_candidate(
        [
            _candidate("stable-a", "faster-whisper", cer=0.30, latency=50),
            _candidate("stable-z", "faster-whisper", cer=0.20, latency=100),
            _candidate("qwen-a", "qwen3-asr", cer=0.25, latency=50),
            _candidate("qwen-z", "qwen3-asr", cer=0.18, latency=190),
        ]
    )

    assert decision.selected_candidate_id == "qwen-z"
    assert decision.candidate_status == {
        "stable-a": "rejected",
        "stable-z": "rejected",
        "qwen-a": "rejected",
        "qwen-z": "eligible",
    }
    assert set(decision.reasons) == set(decision.candidate_status)


def test_duplicate_ids_and_non_asr_tasks_are_rejected() -> None:
    same = _candidate("same", "faster-whisper", cer=0.1)
    with pytest.raises(ValueError, match="duplicate"):
        select_asr_candidate([same, same])
    with pytest.raises(ValueError, match="asr"):
        select_asr_candidate([_candidate("emotion", "x", cer=0.1, task="emotion")])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("completion_rate", "1"),
        ("completion_rate", True),
        ("cer", math.nan),
        ("cer", math.inf),
        ("median_latency_ms", math.nan),
        ("peak_rss_mb", math.inf),
    ],
)
def test_candidate_contract_is_strict_and_finite(field: str, value: object) -> None:
    payload = {
        "candidate_id": "candidate",
        "task": "asr",
        "provider": "provider",
        "model": "model",
        "completion_rate": 1.0,
        "cer": 0.1,
    }
    payload[field] = value
    with pytest.raises(ValidationError):
        CandidateMetrics.model_validate(payload)
