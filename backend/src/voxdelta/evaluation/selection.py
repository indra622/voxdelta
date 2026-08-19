"""Strict, deterministic gates for model candidate evaluation."""

from __future__ import annotations

from math import isfinite
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CandidateStatus = Literal["eligible", "unavailable", "rejected"]
_MAX_RSS_MB = 18_432.0
_MIN_COMPLETION = 0.95
_SAFE_UNAVAILABLE_REASONS = frozenset(
    {
        "invalid_audio_asset",
        "invalid_local_checkpoint",
        "invalid_provider_output",
        "missing_huggingface_token",
        "missing_pyannote_api_key",
        "provider_runtime_unsupported",
        "provider_timeout",
        "provider_unavailable",
        "unsupported_speaker_count",
    }
)


class CandidateMetrics(BaseModel):
    candidate_id: str
    task: Literal["diarization", "asr", "emotion"]
    provider: str
    model: str
    completion_rate: float = Field(ge=0, le=1)
    median_latency_ms: float | None = Field(default=None, ge=0)
    peak_rss_mb: float | None = Field(default=None, ge=0)
    der: float | None = Field(default=None, ge=0)
    cer: float | None = Field(default=None, ge=0)
    macro_f1: float | None = Field(default=None, ge=0, le=1)
    expected_calibration_error: float | None = Field(default=None, ge=0, le=1)
    unavailable_reason: str | None = None

    model_config = ConfigDict(strict=True, extra="forbid")

    @field_validator("unavailable_reason")
    @classmethod
    def safe_unavailable_reason(cls, value: str | None) -> str | None:
        if value is None or value in _SAFE_UNAVAILABLE_REASONS:
            return value
        return "provider_unavailable"

    @model_validator(mode="after")
    def finite_metrics(self) -> CandidateMetrics:
        values = (
            self.completion_rate,
            self.median_latency_ms,
            self.peak_rss_mb,
            self.der,
            self.cer,
            self.macro_f1,
            self.expected_calibration_error,
        )
        if any(value is not None and not isfinite(value) for value in values):
            raise ValueError("candidate metrics must be finite")
        if not self.candidate_id or not self.provider or not self.model:
            raise ValueError("candidate identity fields must not be empty")
        return self


class SelectionDecision(BaseModel):
    selected_candidate_id: str | None
    selected_provider: str | None
    selected_model: str | None
    candidate_status: dict[str, CandidateStatus]
    reasons: dict[str, str]

    model_config = ConfigDict(strict=True, extra="forbid")


def _base_status(
    candidates: list[CandidateMetrics], task: Literal["asr", "emotion"]
) -> tuple[dict[str, CandidateStatus], dict[str, str], list[CandidateMetrics]]:
    ids = [candidate.candidate_id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate candidate_id")
    if any(candidate.task != task for candidate in candidates):
        raise ValueError(f"all candidates must use task={task}")
    status: dict[str, CandidateStatus] = {}
    reasons: dict[str, str] = {}
    eligible: list[CandidateMetrics] = []
    for candidate in candidates:
        reason: str | None = None
        if candidate.unavailable_reason is not None:
            reason = (
                candidate.unavailable_reason
                if candidate.unavailable_reason in _SAFE_UNAVAILABLE_REASONS
                else "provider_unavailable"
            )
        elif candidate.completion_rate < _MIN_COMPLETION:
            reason = "completion_rate_below_0.95"
        elif candidate.peak_rss_mb is None:
            reason = "peak_rss_mb_missing"
        elif candidate.peak_rss_mb > _MAX_RSS_MB:
            reason = "peak_rss_mb_above_18432"
        elif candidate.median_latency_ms is None:
            reason = "median_latency_ms_missing"
        elif task == "asr" and candidate.cer is None:
            reason = "cer_missing"
        elif task == "emotion" and (
            candidate.macro_f1 is None or candidate.expected_calibration_error is None
        ):
            reason = "emotion_metric_missing"
        if reason is None:
            status[candidate.candidate_id] = "eligible"
            reasons[candidate.candidate_id] = "candidate_passed_required_gates"
            eligible.append(candidate)
        else:
            status[candidate.candidate_id] = "unavailable"
            reasons[candidate.candidate_id] = reason
    return status, reasons, eligible


def _decision(
    selected: CandidateMetrics | None,
    status: dict[str, CandidateStatus],
    reasons: dict[str, str],
) -> SelectionDecision:
    return SelectionDecision(
        selected_candidate_id=None if selected is None else selected.candidate_id,
        selected_provider=None if selected is None else selected.provider,
        selected_model=None if selected is None else selected.model,
        candidate_status=status,
        reasons=reasons,
    )


def select_asr_candidate(candidates: list[CandidateMetrics]) -> SelectionDecision:
    status, reasons, eligible = _base_status(candidates, "asr")
    if not eligible:
        return _decision(None, status, reasons)

    def ranking(item: CandidateMetrics) -> tuple[float, float, float, str]:
        assert item.cer is not None
        assert item.median_latency_ms is not None
        assert item.peak_rss_mb is not None
        return item.cer, item.median_latency_ms, item.peak_rss_mb, item.candidate_id

    stable = sorted(
        (candidate for candidate in eligible if candidate.provider == "faster-whisper"),
        key=ranking,
    )
    modern = sorted(
        (candidate for candidate in eligible if candidate.provider == "qwen3-asr"),
        key=ranking,
    )
    supported_ids = {candidate.candidate_id for candidate in (*stable, *modern)}
    for candidate in eligible:
        if candidate.candidate_id not in supported_ids:
            status[candidate.candidate_id] = "rejected"
            reasons[candidate.candidate_id] = "unsupported_asr_candidate_provider"
    for group in (stable, modern):
        for candidate in group[1:]:
            status[candidate.candidate_id] = "rejected"
            reasons[candidate.candidate_id] = "superseded_within_provider_group"

    baseline = stable[0] if stable else None
    qwen = modern[0] if modern else None
    if baseline is None and qwen is None:
        return _decision(None, status, reasons)
    if baseline is None:
        assert qwen is not None
        reasons[qwen.candidate_id] = "selected_as_only_eligible_provider"
        return _decision(qwen, status, reasons)
    if qwen is None:
        reasons[baseline.candidate_id] = "selected_as_only_eligible_provider"
        return _decision(baseline, status, reasons)

    assert baseline.cer is not None and qwen.cer is not None
    assert baseline.median_latency_ms is not None and qwen.median_latency_ms is not None
    improvement = baseline.cer - qwen.cer
    if improvement + 1e-12 >= 0.01 or (
        improvement >= 0 and qwen.median_latency_ms <= baseline.median_latency_ms * 2.0
    ):
        selected = qwen
        reasons[qwen.candidate_id] = "selected_for_cer_and_latency_gate"
        status[baseline.candidate_id] = "rejected"
        reasons[baseline.candidate_id] = "qwen_passed_selection_gate"
    else:
        selected = baseline
        reasons[baseline.candidate_id] = "selected_as_stable_baseline"
        status[qwen.candidate_id] = "rejected"
        reasons[qwen.candidate_id] = "qwen_failed_cer_or_latency_gate"
    return _decision(selected, status, reasons)


def select_emotion_candidate(candidates: list[CandidateMetrics]) -> SelectionDecision:
    """Apply the later emotion benchmark gate using the shared strict contracts."""

    status, reasons, eligible = _base_status(candidates, "emotion")
    if not eligible:
        return _decision(None, status, reasons)
    ordered = sorted(
        eligible,
        key=lambda item: (
            -(item.macro_f1 if item.macro_f1 is not None else -1.0),
            item.expected_calibration_error
            if item.expected_calibration_error is not None
            else float("inf"),
            item.median_latency_ms if item.median_latency_ms is not None else float("inf"),
            item.candidate_id,
        ),
    )
    best = ordered[0]
    if len(ordered) > 1:
        alternative = ordered[1]
        assert best.expected_calibration_error is not None
        assert alternative.expected_calibration_error is not None
        if best.expected_calibration_error > alternative.expected_calibration_error + 0.02:
            best = alternative
    for candidate in eligible:
        if candidate.candidate_id != best.candidate_id:
            status[candidate.candidate_id] = "rejected"
            reasons[candidate.candidate_id] = "another_candidate_passed_emotion_gate"
    reasons[best.candidate_id] = "selected_by_macro_f1_calibration_and_latency"
    return _decision(best, status, reasons)
