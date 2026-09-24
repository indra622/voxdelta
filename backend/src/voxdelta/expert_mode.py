"""ACP handoff contract for evidence-bound expert guidance.

This module deliberately does *not* call a model SDK.  VoxDelta writes a small,
auditable request after the deterministic report exists; an OpenClaw ACP worker may
then hand that request to a connected Claude or Codex session.  The returned JSON is
validated locally before it can become visible to a reviewer.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Literal

import orjson
from pydantic import BaseModel, ConfigDict, Field, model_validator

from voxdelta.domain.models import AnalysisReport, Role
from voxdelta.jobs._ids import validate_job_id


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExpertEvidence(_StrictModel):
    utterance_id: str = Field(min_length=1, max_length=128)
    role: Role
    transcript: str = Field(min_length=1, max_length=1_200)
    emotion_state: str | None = Field(default=None, max_length=32)
    emotion_confidence: float | None = Field(default=None, ge=0, le=1)
    strategy: str | None = Field(default=None, max_length=64)


class ExpertRequest(_StrictModel):
    """Minimal text-only context an ACP worker is allowed to receive."""

    schema_version: Literal["1"] = "1"
    job_id: str = Field(min_length=1, max_length=64)
    target: Literal["claude", "codex"]
    transport: Literal["acp"] = "acp"
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    transcription_uncertain: bool
    evidence: list[ExpertEvidence] = Field(min_length=1, max_length=6)
    instruction: str = Field(min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def evidence_ids_are_unique(self) -> ExpertRequest:
        validate_job_id(self.job_id)
        ids = [item.utterance_id for item in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("expert evidence utterance IDs must be unique")
        return self


class ExpertObservation(_StrictModel):
    evidence_turn_ids: list[str] = Field(min_length=1, max_length=6)
    statement: str = Field(min_length=1, max_length=800)


class ExpertHypothesis(_StrictModel):
    statement: str = Field(min_length=1, max_length=800)
    confidence: Literal["low", "medium"]


class ExpertGuidance(_StrictModel):
    """The only response shape an ACP worker can submit back to VoxDelta."""

    observations: list[ExpertObservation] = Field(min_length=1, max_length=5)
    hypotheses: list[ExpertHypothesis] = Field(default_factory=list, max_length=3)
    suggested_message: str = Field(min_length=1, max_length=1_200)
    next_question: str = Field(min_length=1, max_length=800)
    safety_level: Literal["normal", "watch", "urgent"]


class ExpertHandoffRecord(_StrictModel):
    request: ExpertRequest
    status: Literal["queued", "ready", "rejected"]
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    guidance: ExpertGuidance | None = None
    rejection_code: str | None = Field(default=None, max_length=80)


def _sha256_model(value: BaseModel) -> str:
    payload = orjson.dumps(value.model_dump(mode="json"), option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(payload).hexdigest()


def _report_sha256(report: AnalysisReport) -> str:
    return _sha256_model(report)


def build_expert_request(
    report: AnalysisReport,
    *,
    target: Literal["claude", "codex"],
) -> ExpertRequest:
    """Select at most six relevant turns, never the whole transcript by default."""

    utterances = {utterance.id: utterance for utterance in report.utterances}
    emotions = {item.utterance_id: item for item in report.emotions}
    strategies = {item.utterance_id: item.primary for item in report.strategies}

    customer_candidates = [
        item
        for item in report.emotions
        if utterances.get(item.utterance_id) is not None
        and utterances[item.utterance_id].role == Role.CUSTOMER
    ]
    customer_candidates.sort(
        key=lambda item: (item.negative_intensity, 1.0 - item.confidence), reverse=True
    )
    selected_ids: list[str] = [item.utterance_id for item in customer_candidates[:3]]
    if report.summary.peak_customer_utterance_id not in selected_ids:
        selected_ids.append(report.summary.peak_customer_utterance_id)

    # Include one agent utterance immediately before/after each selected customer turn
    # so an ACP agent can judge whether the response addresses the concern.
    ordered = report.utterances
    positions = {item.id: position for position, item in enumerate(ordered)}
    for identifier in tuple(selected_ids):
        position = positions.get(identifier)
        if position is None:
            continue
        for adjacent in (position - 1, position + 1):
            if 0 <= adjacent < len(ordered) and ordered[adjacent].role == Role.AGENT:
                selected_ids.append(ordered[adjacent].id)
                break

    unique_ids: list[str] = []
    for identifier in selected_ids:
        if identifier in utterances and identifier not in unique_ids:
            unique_ids.append(identifier)
        if len(unique_ids) == 6:
            break
    if not unique_ids:
        raise ValueError("report has no eligible evidence for expert guidance")

    evidence: list[ExpertEvidence] = []
    for identifier in unique_ids:
        emotion = emotions.get(identifier)
        evidence.append(
            ExpertEvidence(
                utterance_id=identifier,
                role=utterances[identifier].role,
                transcript=utterances[identifier].transcript,
                emotion_state=emotion.operational_state if emotion is not None else None,
                emotion_confidence=emotion.confidence if emotion is not None else None,
                strategy=strategies.get(identifier),
            )
        )
    coverage_uncertain = bool(
        report.transcription_coverage and report.transcription_coverage.uncertain
    )
    uncertain = coverage_uncertain or any(
        item.emotion_confidence is not None and item.emotion_confidence < 0.70 for item in evidence
    )
    return ExpertRequest(
        job_id=report.job_id,
        target=target,
        report_sha256=_report_sha256(report),
        transcription_uncertain=uncertain,
        evidence=evidence,
        instruction=(
            "근거 turn ID만 인용해 관찰과 가능한 원인 가설, 공감적 대응 문구, 확인 질문을 "
            "한국어 JSON으로 작성하세요. "
            "원인을 단정하거나 진단하지 말고, 전사/감정 불확실성이 있으면 확인 질문을 우선하세요. "
            "위기·자해·학대 등 긴급 위험이 명시될 때만 safety_level을 urgent로 설정하세요."
        ),
    )


def validate_expert_guidance(request: ExpertRequest, guidance: ExpertGuidance) -> None:
    """Fail closed when an ACP response cites unavailable evidence or over-claims."""

    allowed_ids = {item.utterance_id for item in request.evidence}
    for observation in guidance.observations:
        if not set(observation.evidence_turn_ids).issubset(allowed_ids):
            raise ValueError("expert guidance cited an utterance outside the evidence packet")
    forbidden = ("확정", "반드시", "진단", "원인이다")
    combined = " ".join(item.statement for item in guidance.hypotheses)
    if any(token in combined for token in forbidden):
        raise ValueError("expert hypotheses must not state a diagnosis or certainty")
    if any(
        "가설" not in item.statement and "가능" not in item.statement
        for item in guidance.hypotheses
    ):
        raise ValueError("expert hypotheses must be explicitly framed as hypotheses")
    if request.transcription_uncertain and not guidance.next_question.strip():
        raise ValueError("uncertain evidence requires a confirmation question")


class ExpertHandoffStore:
    """Private per-job persistence for a queued ACP request and validated response."""

    _FILENAME = "expert-handoff.v1.json"

    def __init__(self, jobs_root: Path) -> None:
        self._root = Path(jobs_root).expanduser().resolve(strict=False)

    def _path(self, job_id: str) -> Path:
        validate_job_id(job_id)
        candidate = self._root / job_id / self._FILENAME
        if candidate.parent.resolve(strict=False).parent != self._root:
            raise ValueError("expert handoff path is outside the jobs root")
        return candidate

    def _write(self, path: Path, record: ExpertHandoffRecord) -> None:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        payload = orjson.dumps(record.model_dump(mode="json"), option=orjson.OPT_INDENT_2)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def queue(self, request: ExpertRequest) -> ExpertHandoffRecord:
        record = ExpertHandoffRecord(
            request=request,
            status="queued",
            request_sha256=_sha256_model(request),
        )
        self._write(self._path(request.job_id), record)
        return record

    def read(self, job_id: str) -> ExpertHandoffRecord | None:
        path = self._path(job_id)
        if not path.exists():
            return None
        return ExpertHandoffRecord.model_validate_json(path.read_bytes())

    def submit(
        self,
        job_id: str,
        *,
        request_sha256: str,
        guidance: ExpertGuidance,
    ) -> ExpertHandoffRecord:
        current = self.read(job_id)
        if current is None or current.status != "queued":
            raise ValueError("no queued expert handoff exists")
        if current.request_sha256 != request_sha256:
            raise ValueError("expert response does not match the queued request")
        validate_expert_guidance(current.request, guidance)
        record = ExpertHandoffRecord(
            request=current.request,
            status="ready",
            request_sha256=current.request_sha256,
            guidance=guidance,
        )
        self._write(self._path(job_id), record)
        return record
