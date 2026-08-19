"""Canonical domain models shared by the VoxDelta pipeline."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

EmotionLabel = Literal["happiness", "anger", "disgust", "fear", "neutral", "sadness", "surprise"]
OperationalState = Literal["satisfied", "stable", "dissatisfied", "escalated", "uncertain"]
TransitionClass = Literal["recovery", "stable", "worsening"]


class Role(StrEnum):
    CUSTOMER = "customer"
    AGENT = "agent"
    UNKNOWN = "unknown"


class StageName(StrEnum):
    NORMALIZE = "normalize"
    DIARIZE = "diarize"
    TRANSCRIBE = "transcribe"
    CONFIRM_ROLES = "confirm_roles"
    EMOTION = "emotion"
    RESPONSE_STRATEGY = "response_strategy"
    TRANSITIONS = "transitions"
    REPORT = "report"


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ProviderProvenance(BaseModel):
    name: str
    model: str
    remote: bool
    transmits: tuple[Literal["audio", "text", "features"], ...] = ()
    retention_policy_url: str | None = None
    schema_version: str = "1"


class ProviderUsage(BaseModel):
    latency_ms: float = Field(ge=0)
    peak_rss_mb: float | None = Field(default=None, ge=0)
    input_units: int | None = Field(default=None, ge=0)
    output_units: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    remote_file_deleted: bool | None = None


class AudioAsset(BaseModel):
    source_name: str
    source_path: str
    normalized_paths: tuple[str, ...] = ()
    channel_mode: Literal["mixed", "separate"] | None = None
    duration_seconds: float | None = None
    channels: int | None = None
    sha256: str


class SpeakerSegment(BaseModel):
    start: float = Field(ge=0)
    end: float
    speaker_id: str
    overlap: bool = False
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def valid_interval(self) -> SpeakerSegment:
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self


class Utterance(SpeakerSegment):
    id: str
    role: Role = Role.UNKNOWN
    transcript: str


class EmotionResult(BaseModel):
    utterance_id: str
    probabilities: dict[EmotionLabel, float]
    operational_state: OperationalState
    negative_intensity: float = Field(ge=0, le=1)
    smoothed_negative_intensity: float | None = Field(default=None, ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    provider: ProviderProvenance
    usage: ProviderUsage | None = None

    @model_validator(mode="after")
    def valid_distribution(self) -> EmotionResult:
        expected = {"happiness", "anger", "disgust", "fear", "neutral", "sadness", "surprise"}
        if set(self.probabilities) != expected:
            raise ValueError("all seven emotion labels are required")
        if any(value < 0 or value > 1 for value in self.probabilities.values()):
            raise ValueError("emotion probabilities must be between zero and one")
        if abs(sum(self.probabilities.values()) - 1.0) > 1e-6:
            raise ValueError("emotion probabilities must sum to one")
        return self


class ResponseStrategyResult(BaseModel):
    utterance_id: str
    primary: Literal[
        "apology",
        "empathy",
        "clarification",
        "information",
        "solution",
        "policy_refusal",
        "greeting_closing",
        "other",
    ]
    secondary: tuple[str, ...] = ()
    confidence: float = Field(ge=0, le=1)
    provider: ProviderProvenance


class EmotionTransition(BaseModel):
    previous_customer_id: str
    agent_id: str
    next_customer_id: str
    delta: float = Field(ge=-1, le=1)
    classification: TransitionClass


class CallSummary(BaseModel):
    start_state: OperationalState
    end_state: OperationalState
    peak_customer_utterance_id: str
    overall_delta: float = Field(ge=-1, le=1)
    valid_coverage: float = Field(ge=0, le=1)
    recovery_count: int = Field(ge=0)
    worsening_count: int = Field(ge=0)
    narrative: str | None = None


class AnalysisReport(BaseModel):
    job_id: str
    summary: CallSummary
    utterances: list[Utterance]
    emotions: list[EmotionResult]
    strategies: list[ResponseStrategyResult]
    transitions: list[EmotionTransition]
    warnings: list[str] = Field(default_factory=list)
    schema_version: str = "1"
