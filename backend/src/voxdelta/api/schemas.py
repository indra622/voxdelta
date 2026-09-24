"""Strict public request and response contracts for the local API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from voxdelta.domain.models import Role, StageName
from voxdelta.expert_mode import ExpertGuidance


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RoleConfirmation(StrictSchema):
    mapping: dict[str, Role]

    @model_validator(mode="after")
    def one_customer_one_agent(self) -> RoleConfirmation:
        if (
            len(self.mapping) != 2
            or any(not key for key in self.mapping)
            or sorted(self.mapping.values()) != [Role.AGENT, Role.CUSTOMER]
        ):
            raise ValueError("mapping must contain two speakers, one customer and one agent")
        return self


class RetryRequest(StrictSchema):
    stage: StageName


class ExpertGuidanceRequest(StrictSchema):
    """Explicit acknowledgement before transcript evidence leaves the local app."""

    target: str = Field(pattern=r"^(claude|codex)$")
    acknowledge_text_transfer: bool

    @model_validator(mode="after")
    def acknowledgement_is_required(self) -> ExpertGuidanceRequest:
        if not self.acknowledge_text_transfer:
            raise ValueError("text-transfer acknowledgement is required")
        return self


class ExpertGuidanceSubmission(StrictSchema):
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    guidance: ExpertGuidance


class ExpertGuidanceStatus(StrictSchema):
    status: str
    target: str | None = None
    transport: str | None = None
    request_sha256: str | None = None
    transcription_uncertain: bool | None = None
    evidence_turn_count: int = Field(ge=0)
    guidance: ExpertGuidance | None = None


class JobCreated(StrictSchema):
    job_id: str
    status_url: str


class PublicError(StrictSchema):
    code: str
    message: str


class PublicErrorEnvelope(StrictSchema):
    detail: PublicError


class PublicStage(StrictSchema):
    status: str
    error: PublicError | None = None


class RoleCandidate(StrictSchema):
    speakers: list[str]
    suggested_mapping: dict[str, Role] | None = None
    samples: dict[str, list[RoleSample]] = Field(default_factory=dict)


class RoleSample(StrictSchema):
    """A short, already-transcribed excerpt used only to verify a diarized voice."""

    speaker_id: str
    index: int
    start: float
    end: float
    transcript: str
    clip_truncated: bool = False


class PublicJob(StrictSchema):
    job_id: str
    status: str
    diagnostic_capture: bool
    created_at: str
    updated_at: str
    stages: dict[str, PublicStage]
    role_candidate: RoleCandidate | None = None


class PublicProvenance(StrictSchema):
    name: str
    model: str
    remote: bool
    schema_version: str
    revision: str | None = None


class ProviderDisclosure(StrictSchema):
    stage: StageName
    provenance: PublicProvenance | None
    transmits: tuple[str, ...]
    retention_policy_url: str | None
    retention_window_hours: int | None


class ProviderConfiguration(StrictSchema):
    stages: list[ProviderDisclosure]


class GeminiDisclosure(StrictSchema):
    """The Gemini transfer, disclosed on its own terms.

    Kept out of ``ProviderConfiguration.stages`` deliberately. Those describe the analysis
    pipeline; this describes an optional annotation workflow that sends audio to a
    different company. Consenting to one has never been consent to the other, and a client
    that renders the stage list will not accidentally render this as already agreed.
    """

    enabled: bool
    consent_granted: bool
    provider: str
    model: str
    remote: bool
    transmits: tuple[str, ...]
    retention_policy_url: str | None
    produces: str
    requires_human_review: bool


class AnnotationReviewState(StrictSchema):
    """What a reviewer is being asked to look at, in counts only."""

    conversation_id: str
    review_state: str
    promotable: bool
    created_at: str
    model: str
    input_sha256: str
    content_sha256: str
    remote_file_deleted: bool
    turn_count: int
    speaker_count: int
    uncertain_turns: int
    mean_confidence: float | None
    gold_present: bool
    emotion_candidate_count: int = 0


class AnnotationReviewIndex(StrictSchema):
    """Every draft this machine holds. Counts only, so choosing one reveals no speech."""

    annotations: list[AnnotationReviewState]
    unreadable_count: int


class ReviewTurn(StrictSchema):
    """One turn as the reviewer sees and returns it. Carries transcript in both directions.

    Bounds are declared here so an oversized submission is refused before it is read into
    a domain object; the meaning of each field is checked in
    :mod:`voxdelta.annotation.review`, which answers with a code the reviewer can act on.
    """

    start: float
    end: float
    speaker: str = Field(max_length=64)
    transcript: str = Field(max_length=4_000)
    emotion: str = Field(max_length=32)
    emotion_rationale: str = Field(default="", max_length=1_000)
    confidence: float


class ReviewWarning(StrictSchema):
    """Something that makes a draft provisional, stated rather than left to be inferred."""

    code: str
    detail: str
    count: int | None = None
    by_rule: dict[str, int] | None = None


class SilverReviewDraft(StrictSchema):
    """The full reviewable draft. Served only over the local capability-fenced API."""

    state: AnnotationReviewState
    speakers: list[str]
    notes: str
    turns: list[ReviewTurn]
    emotion_labels: list[str]
    warnings: list[ReviewWarning]


class TurnClip(StrictSchema):
    """The playable range for one draft turn, clamped to audio that exists."""

    position: int
    start: float
    end: float


class ReviewGap(StrictSchema):
    """A stretch of the recording no valid draft turn covers.

    Not described as a dropped turn anywhere in this contract. The silver artifact records
    how many turns validation rejected but discards where they were, so what is known is
    only that the draft says nothing here.
    """

    start: float
    end: float


class AnnotationAudioOverview(StrictSchema):
    """What a reviewer can listen to for one draft. Numbers about audio, never speech."""

    conversation_id: str
    duration_seconds: float
    sample_rate: int
    max_clip_seconds: float
    min_gap_seconds: float
    turn_clips: list[TurnClip]
    gaps: list[ReviewGap]


class AlignmentProposalRow(StrictSchema):
    position: int
    start: float
    end: float
    confidence: float


class AlignmentProposal(StrictSchema):
    source_silver_content_sha256: str
    target_start_position: int
    target_end_position: int
    rows: list[AlignmentProposalRow]
    dropped_row_count: int
    dropped_rows_by_rule: dict[str, int]


class AlignmentProposalCollection(StrictSchema):
    """Independent timing-only suggestions for one immutable Silver draft."""

    proposals: list[AlignmentProposal]


class ReferenceResegmentationCandidate(StrictSchema):
    """A read-only KCSC reference candidate; applying it never writes Silver or Gold."""

    source_silver_content_sha256: str
    source_reference_sha256: str
    reference_turn_count: int
    speakers: list[str]
    notes: str
    turns: list[ReviewTurn]


class EmotionOverlayCandidate(StrictSchema):
    """A local emotion-only hypothesis aligned to one immutable Silver draft."""

    source_silver_content_sha256: str
    model: str
    remote_audio_transmitted: bool
    emotion_histogram: dict[str, int]
    uncertain_turns: int
    turns: list[ReviewTurn]


class GeminiEmotionOverlayCandidate(StrictSchema):
    """A remote Gemini emotion-only hypothesis aligned to one immutable Silver draft.

    Distinct from :class:`EmotionOverlayCandidate` on purpose: producing it transmitted
    the recording to Google, and a reviewer deciding whether to trust it needs to be told
    that rather than have it inferred from the model name.
    """

    source_silver_content_sha256: str
    model: str
    remote_audio_transmitted: bool
    review_required: bool
    promotable: bool
    emotion_histogram: dict[str, int]
    uncertain_turns: int
    mean_confidence: float | None
    turns: list[ReviewTurn]


class ReviewChangeReason(StrictSchema):
    """A compact audit label for one human-corrected, one-based turn position."""

    position: int = Field(ge=1, le=5_000)
    reason: str = Field(max_length=64)


class GoldPromotionRequest(StrictSchema):
    """A reviewer's sign-off. Every field is theirs; none of it defaults to the model's."""

    reviewer: str = Field(max_length=128)
    acknowledged: bool
    turns: list[ReviewTurn] = Field(max_length=5_000)
    review_note: str = Field(default="", max_length=2_000)
    change_reasons: list[ReviewChangeReason] = Field(default_factory=list, max_length=5_000)


class GoldPromotionResult(StrictSchema):
    """The receipt for an irreversible write, including the silver-untouched check."""

    conversation_id: str
    reviewer: str
    reviewed_at: str
    review_note: str
    content_sha256: str
    parent_silver_sha256: str
    unchanged_from_silver: bool
    turn_count: int
    speaker_count: int
    uncertain_turns: int
    mean_confidence: float | None
    change_reason_count: int
    silver_unmodified: bool


__all__ = [
    "AnnotationAudioOverview",
    "AlignmentProposal",
    "AlignmentProposalCollection",
    "AlignmentProposalRow",
    "ReferenceResegmentationCandidate",
    "EmotionOverlayCandidate",
    "GeminiEmotionOverlayCandidate",
    "AnnotationReviewIndex",
    "AnnotationReviewState",
    "GeminiDisclosure",
    "GoldPromotionRequest",
    "GoldPromotionResult",
    "JobCreated",
    "ProviderConfiguration",
    "ProviderDisclosure",
    "PublicError",
    "PublicErrorEnvelope",
    "PublicJob",
    "PublicProvenance",
    "PublicStage",
    "RetryRequest",
    "ReviewTurn",
    "ReviewChangeReason",
    "ReviewWarning",
    "RoleCandidate",
    "ReviewGap",
    "RoleConfirmation",
    "SilverReviewDraft",
    "TurnClip",
]
