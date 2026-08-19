"""Strict, deterministic gates for model candidate evaluation."""

from __future__ import annotations

from math import isfinite
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

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


def _safe_unavailable_reason(value: object) -> str | None:
    if value is None:
        return value
    if isinstance(value, str) and value in _SAFE_UNAVAILABLE_REASONS:
        return value
    return "provider_unavailable"


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
    unavailable_reason: str | None = Field(default=None, repr=False)

    model_config = ConfigDict(strict=True, extra="forbid")

    @field_validator("unavailable_reason")
    @classmethod
    def safe_unavailable_reason(cls, value: str | None) -> str | None:
        return _safe_unavailable_reason(value)

    @field_serializer("unavailable_reason")
    def serialize_unavailable_reason(self, value: object) -> str | None:
        return _safe_unavailable_reason(value)

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
            reason = _safe_unavailable_reason(candidate.unavailable_reason)
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
    """Compare one stable and one modern local seven-emotion candidate."""
    status, reasons, eligible = _base_status(candidates, "emotion")
    if not eligible:
        return _decision(None, status, reasons)

    def ranking(item: CandidateMetrics) -> tuple[float, float, float, str]:
        assert item.macro_f1 is not None
        assert item.expected_calibration_error is not None
        assert item.median_latency_ms is not None
        return (
            -item.macro_f1,
            item.expected_calibration_error,
            item.median_latency_ms,
            item.candidate_id,
        )

    stable = sorted(
        (candidate for candidate in eligible if candidate.provider == "wav2vec-xls-r"),
        key=ranking,
    )
    modern = sorted(
        (candidate for candidate in eligible if candidate.provider == "emotion2vec-plus"),
        key=ranking,
    )
    supported_ids = {candidate.candidate_id for candidate in (*stable, *modern)}
    for candidate in eligible:
        if candidate.candidate_id not in supported_ids:
            status[candidate.candidate_id] = "rejected"
            reasons[candidate.candidate_id] = "unsupported_emotion_candidate_provider"
    for group in (stable, modern):
        for candidate in group[1:]:
            status[candidate.candidate_id] = "rejected"
            reasons[candidate.candidate_id] = "superseded_within_provider_group"

    baseline = stable[0] if stable else None
    emotion2vec = modern[0] if modern else None
    if baseline is None and emotion2vec is None:
        return _decision(None, status, reasons)
    if baseline is None:
        assert emotion2vec is not None
        reasons[emotion2vec.candidate_id] = "selected_as_only_eligible_provider"
        return _decision(emotion2vec, status, reasons)
    if emotion2vec is None:
        reasons[baseline.candidate_id] = "selected_as_only_eligible_provider"
        return _decision(baseline, status, reasons)

    assert baseline.macro_f1 is not None and emotion2vec.macro_f1 is not None
    assert baseline.expected_calibration_error is not None
    assert emotion2vec.expected_calibration_error is not None
    assert baseline.median_latency_ms is not None and emotion2vec.median_latency_ms is not None

    if emotion2vec.expected_calibration_error > baseline.expected_calibration_error + 0.02:
        selected = baseline
        selected_reason = "selected_after_emotion2vec_calibration_gate"
        rejected_reason = "emotion2vec_ece_regressed_above_0.02"
    elif baseline.expected_calibration_error > emotion2vec.expected_calibration_error + 0.02:
        selected = emotion2vec
        selected_reason = "selected_after_baseline_calibration_gate"
        rejected_reason = "baseline_ece_regressed_above_0.02"
    else:
        improvement = emotion2vec.macro_f1 - baseline.macro_f1
        if improvement + 1e-12 >= 0.01:
            selected = emotion2vec
            selected_reason = "selected_for_macro_f1_gain_at_least_0.01"
            rejected_reason = "emotion2vec_passed_macro_f1_gate"
        elif improvement < -0.01 - 1e-12:
            selected = baseline
            selected_reason = "selected_for_higher_macro_f1"
            rejected_reason = "emotion2vec_macro_f1_below_near_tie_band"
        else:
            tie_ranking = lambda item: (  # noqa: E731
                item.expected_calibration_error,
                item.median_latency_ms,
                0 if item.provider == "wav2vec-xls-r" else 1,
                item.candidate_id,
            )
            selected = min((baseline, emotion2vec), key=tie_ranking)
            selected_reason = "selected_by_near_tie_calibration_then_latency"
            rejected_reason = "another_candidate_won_near_tie_gate"

    rejected = emotion2vec if selected is baseline else baseline
    status[rejected.candidate_id] = "rejected"
    reasons[rejected.candidate_id] = rejected_reason
    reasons[selected.candidate_id] = selected_reason
    return _decision(selected, status, reasons)
