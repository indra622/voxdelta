export const STAGE_NAMES = [
  "normalize",
  "diarize",
  "transcribe",
  "confirm_roles",
  "emotion",
  "response_strategy",
  "transitions",
  "report",
] as const;

export type StageName = (typeof STAGE_NAMES)[number];
export type StageStatus = "pending" | "running" | "paused" | "completed" | "failed" | "skipped";
export type Role = "customer" | "agent" | "unknown";
export type EmotionLabel =
  | "happiness"
  | "anger"
  | "disgust"
  | "fear"
  | "neutral"
  | "sadness"
  | "surprise";
export type OperationalState = "satisfied" | "stable" | "dissatisfied" | "escalated" | "uncertain";

export interface PublicError {
  code: string;
  message: string;
}

export interface PublicStage {
  status: StageStatus;
  error: PublicError | null;
}

export interface RoleCandidate {
  speakers: string[];
  suggested_mapping: Record<string, Role> | null;
  /** Absent only when a browser is temporarily paired with an older backend. */
  samples?: Record<string, RoleSample[]>;
}

export interface RoleSample {
  speaker_id: string;
  index: number;
  start: number;
  end: number;
  transcript: string;
  clip_truncated: boolean;
}

export interface PublicJob {
  job_id: string;
  status: StageStatus;
  diagnostic_capture: boolean;
  created_at: string;
  updated_at: string;
  stages: Record<StageName, PublicStage>;
  role_candidate: RoleCandidate | null;
}

export interface JobCreated {
  job_id: string;
  status_url: string;
}

export interface ProviderProvenance {
  name: string;
  model: string;
  remote: boolean;
  transmits: Array<"audio" | "text" | "features">;
  retention_policy_url: string | null;
  schema_version: string;
  revision: string | null;
}

export interface EmotionCalibration {
  calibration_id: string;
  method: "temperature-scaling";
  temperature: number;
  abstain_threshold: number;
  abstained: boolean;
  raw_confidence: number;
}

export interface Utterance {
  id: string;
  start: number;
  end: number;
  speaker_id: string;
  role: Role;
  transcript: string;
  overlap: boolean;
  confidence: number;
}

export interface EmotionResult {
  utterance_id: string;
  probabilities: Record<EmotionLabel, number>;
  operational_state: OperationalState;
  negative_intensity: number;
  smoothed_negative_intensity: number | null;
  confidence: number;
  provider: ProviderProvenance;
  calibration?: EmotionCalibration | null;
}

/**
 * How much recognized speech reached the transcript. Present only when the recognizer's
 * word timestamps had to be sanitized before speaker attribution; null means no
 * sanitation applied, which is not the same as a guarantee that nothing was lost.
 */
export interface TranscriptionCoverage {
  policy: string;
  attributed_words: number;
  attributed_ratio: number;
  omitted_words: number;
  uncertain: boolean;
}

export interface AnalysisReport {
  job_id: string;
  summary: {
    start_state: OperationalState;
    end_state: OperationalState;
    peak_customer_utterance_id: string;
    overall_delta: number;
    valid_coverage: number;
    recovery_count: number;
    worsening_count: number;
    narrative: string | null;
  };
  utterances: Utterance[];
  emotions: EmotionResult[];
  strategies: unknown[];
  transitions: unknown[];
  warnings: string[];
  transcription_coverage: TranscriptionCoverage | null;
  schema_version: string;
}

export interface ExpertObservation {
  evidence_turn_ids: string[];
  statement: string;
}

export interface ExpertHypothesis {
  statement: string;
  confidence: "low" | "medium";
}

export interface ExpertGuidance {
  observations: ExpertObservation[];
  hypotheses: ExpertHypothesis[];
  suggested_message: string;
  next_question: string;
  safety_level: "normal" | "watch" | "urgent";
}

export interface ExpertGuidanceStatus {
  status: "not_requested" | "queued" | "completed";
  target: "claude" | "codex" | null;
  transport: string | null;
  request_sha256: string | null;
  transcription_uncertain: boolean | null;
  evidence_turn_count: number;
  guidance: ExpertGuidance | null;
}

export interface PublicProvenance {
  name: string;
  model: string;
  remote: boolean;
  schema_version: string;
  revision: string | null;
}

export interface ProviderDisclosure {
  stage: StageName;
  provenance: PublicProvenance | null;
  transmits: string[];
  retention_policy_url: string | null;
  /** Longest window the provider states it may hold transmitted input for. */
  retention_window_hours: number | null;
}

export interface ProviderConfiguration {
  stages: ProviderDisclosure[];
}

/** Counts and digests for one silver draft. Deliberately carries no transcript, so a
 *  reviewer choosing what to open has not yet been shown any speech. */
export interface AnnotationReviewState {
  conversation_id: string;
  review_state: string;
  promotable: boolean;
  created_at: string;
  model: string;
  input_sha256: string;
  content_sha256: string;
  remote_file_deleted: boolean;
  turn_count: number;
  speaker_count: number;
  uncertain_turns: number;
  mean_confidence: number | null;
  gold_present: boolean;
  emotion_candidate_count?: number;
}

export interface AnnotationReviewIndex {
  annotations: AnnotationReviewState[];
  /** Drafts this machine holds but could not parse. Counted rather than hidden. */
  unreadable_count: number;
}

export interface ReviewTurn {
  start: number;
  end: number;
  speaker: string;
  transcript: string;
  emotion: string;
  emotion_rationale: string;
  confidence: number;
}

export interface ReviewWarning {
  code: string;
  detail: string;
  count: number | null;
  by_rule: Record<string, number> | null;
}

export interface SilverReviewDraft {
  state: AnnotationReviewState;
  speakers: string[];
  notes: string;
  turns: ReviewTurn[];
  /** The emotion labels the backend will accept, so this screen never invents one. */
  emotion_labels: string[];
  warnings: ReviewWarning[];
}

/** A timing-only Gemini suggestion for an existing Silver draft.
 *
 * It deliberately contains no transcript or emotion values. Applying it is always an
 * explicit reviewer action and only changes the matching turns' start/end fields. */
export interface AlignmentProposalRow {
  position: number;
  start: number;
  end: number;
  confidence: number;
}

export interface AlignmentProposal {
  source_silver_content_sha256: string;
  target_start_position: number;
  target_end_position: number;
  rows: AlignmentProposalRow[];
  dropped_row_count: number;
  dropped_rows_by_rule: Record<string, number>;
}

export interface AlignmentProposalCollection {
  proposals: AlignmentProposal[];
}

/** Local KCSC human-reference candidate. It is not Silver or Gold and is only applied
 * to the browser's working copy after an explicit reviewer action. */
export interface ReferenceResegmentationCandidate {
  source_silver_content_sha256: string;
  source_reference_sha256: string;
  reference_turn_count: number;
  speakers: string[];
  notes: string;
  turns: ReviewTurn[];
}

/** Local calibrated XLS-R emotion hypothesis. Its timing, speaker, and transcript are
 * copied from the immutable clean Silver; applying it changes emotions only in the
 * reviewer's browser working copy. */
export interface EmotionOverlayCandidate {
  source_silver_content_sha256: string;
  model: string;
  remote_audio_transmitted: boolean;
  emotion_histogram: Record<string, number>;
  uncertain_turns: number;
  turns: ReviewTurn[];
}

/** Remote Gemini emotion hypothesis. Producing it transmitted the recording to Google,
 * which is why it is a separate type from EmotionOverlayCandidate rather than a flag on
 * it: the reviewer has to be told, not left to infer it from the model name. Timing,
 * speaker, and transcript are copied from the immutable clean Silver; applying it changes
 * emotions only in the reviewer's browser working copy, and it is never promotable. */
export interface GeminiEmotionOverlayCandidate {
  source_silver_content_sha256: string;
  model: string;
  remote_audio_transmitted: boolean;
  review_required: boolean;
  promotable: boolean;
  emotion_histogram: Record<string, number>;
  uncertain_turns: number;
  mean_confidence: number | null;
  turns: ReviewTurn[];
}

/** The playable range for one draft turn, clamped to audio that exists. */
export interface TurnClip {
  position: number;
  start: number;
  end: number;
}

/** A stretch of the recording no valid draft turn covers.
 *
 * Not a dropped turn. The silver artifact records how many turns validation rejected but
 * discards where they were, so all that is known is that the draft says nothing here. */
export interface ReviewGap {
  start: number;
  end: number;
}

export interface AnnotationAudioOverview {
  conversation_id: string;
  duration_seconds: number;
  sample_rate: number;
  max_clip_seconds: number;
  min_gap_seconds: number;
  turn_clips: TurnClip[];
  gaps: ReviewGap[];
}

export interface GoldPromotionRequest {
  reviewer: string;
  acknowledged: boolean;
  turns: ReviewTurn[];
  review_note: string;
  /** Human-selected audit labels for corrected turns; transcript stays in the turn data. */
  change_reasons: ReviewChangeReason[];
}

export type ReviewChangeReasonCode =
  | "speaker_mismatch"
  | "transcript_mismatch"
  | "timing_mismatch"
  | "emotion_mismatch"
  | "model_candidate"
  | "other";

export interface ReviewChangeReason {
  /** One-based turn position in the promoted annotation. */
  position: number;
  reason: ReviewChangeReasonCode;
}

export interface GoldPromotionResult {
  conversation_id: string;
  reviewer: string;
  reviewed_at: string;
  review_note: string;
  content_sha256: string;
  parent_silver_sha256: string;
  unchanged_from_silver: boolean;
  turn_count: number;
  speaker_count: number;
  uncertain_turns: number;
  mean_confidence: number | null;
  change_reason_count: number;
  /** The backend re-read silver after writing gold and found it byte-identical. */
  silver_unmodified: boolean;
}

export interface ApiClient {
  getProviderConfiguration(): Promise<ProviderConfiguration>;
  createJob(file: File): Promise<JobCreated>;
  getJob(jobId: string): Promise<PublicJob>;
  /** A short diarization excerpt selected by the backend for role confirmation. */
  getRoleSampleClip?(jobId: string, speakerId: string, sampleIndex: number): Promise<Blob>;
  confirmRoles(jobId: string, mapping: Record<string, "customer" | "agent">): Promise<PublicJob>;
  retryStage(jobId: string, stage: StageName): Promise<PublicJob>;
  getReport(jobId: string): Promise<AnalysisReport>;
  /** Optional while ACP Expert mode rolls out; the core analysis must work without it. */
  getExpertGuidance?(jobId: string): Promise<ExpertGuidanceStatus>;
  requestExpertGuidance?(
    jobId: string,
    request: { target: "claude" | "codex"; acknowledge_text_transfer: boolean },
  ): Promise<ExpertGuidanceStatus>;
  deleteJob(jobId: string): Promise<void>;
  listSilverAnnotations(): Promise<AnnotationReviewIndex>;
  getSilverAnnotation(conversationId: string): Promise<SilverReviewDraft>;
  getAlignmentProposal(conversationId: string): Promise<AlignmentProposal | null>;
  getAlignmentProposals(conversationId: string): Promise<AlignmentProposalCollection>;
  getReferenceResegmentationCandidate(conversationId: string): Promise<ReferenceResegmentationCandidate | null>;
  /** Optional during the staged review-UI rollout; normal review must work without it. */
  getEmotionOverlayCandidate?(conversationId: string): Promise<EmotionOverlayCandidate | null>;
  /** Optional during the staged review-UI rollout; normal review must work without it. */
  getGeminiEmotionOverlayCandidate?(
    conversationId: string,
  ): Promise<GeminiEmotionOverlayCandidate | null>;
  getAnnotationAudio(conversationId: string): Promise<AnnotationAudioOverview>;
  /** One clipped range as a WAV blob. Fetched rather than pointed at with an <audio src>
   *  so the capability header stays a header and never becomes part of a URL. */
  getAnnotationClip(conversationId: string, start: number, end: number): Promise<Blob>;
  promoteSilverToGold(
    conversationId: string,
    submission: GoldPromotionRequest,
  ): Promise<GoldPromotionResult>;
}
