import type { ReviewTurn, ReviewWarning } from "../../api/types";

/** The reviewer's working copy of one turn.
 *
 * Numbers are held as the text the reviewer typed rather than as parsed numbers. A field
 * that is mid-edit ("1.", "", "-") has no number yet, and coercing one would either
 * silently rewrite what they typed or quietly submit a zero they never entered.
 */
export interface EditableTurn {
  start: string;
  end: string;
  speaker: string;
  transcript: string;
  emotion: string;
  emotion_rationale: string;
  confidence: string;
}

export type TurnField = keyof EditableTurn;

export interface TurnIssue {
  index: number;
  field: TurnField;
  message: string;
}

/** A browser-local checkpoint. It is deliberately separate from both Silver and Gold:
 * Silver stays the model's immutable input, and Gold remains an explicit one-time
 * sign-off. The reviewer may overwrite this checkpoint as often as needed. */
export interface ReviewCheckpoint {
  version: 1;
  silverContentSha256: string;
  savedAt: string;
  turns: EditableTurn[];
  reviewer: string;
  reviewNote: string;
  acknowledged: boolean;
  /** Per-turn human audit labels. These are metadata, not part of the model output. */
  changeReasons?: Record<string, ReviewChangeReason>;
}

/** A deliberately small taxonomy: it is useful for later error analysis without making
 * a reviewer write sensitive free-form notes for every correction. */
export const REVIEW_CHANGE_REASONS = [
  "speaker_mismatch",
  "transcript_mismatch",
  "timing_mismatch",
  "emotion_mismatch",
  "model_candidate",
  "other",
] as const;

export type ReviewChangeReason = (typeof REVIEW_CHANGE_REASONS)[number];

export const reviewChangeReasonLabel: Record<ReviewChangeReason, string> = {
  speaker_mismatch: "화자 혼동",
  transcript_mismatch: "전사 불일치",
  timing_mismatch: "시간·경계 불일치",
  emotion_mismatch: "감정 과대·오인식",
  model_candidate: "모델 제안 비교·반영",
  other: "기타 검수 판단",
};

export interface ReviewQuality {
  meanConfidence: number | null;
  uncertainPositions: number[];
  shortPositions: number[];
  overlapPositions: number[];
  priorityPositions: number[];
}

function checkpointKey(conversationId: string, silverContentSha256: string): string {
  return `voxdelta.review-checkpoint.v1:${conversationId}:${silverContentSha256}`;
}

function isEditableTurn(value: unknown): value is EditableTurn {
  if (!value || typeof value !== "object") return false;
  const row = value as Record<string, unknown>;
  return (
    typeof row.start === "string" &&
    typeof row.end === "string" &&
    typeof row.speaker === "string" &&
    typeof row.transcript === "string" &&
    typeof row.emotion === "string" &&
    typeof row.emotion_rationale === "string" &&
    typeof row.confidence === "string"
  );
}

function isReviewChangeReason(value: unknown): value is ReviewChangeReason {
  return typeof value === "string" && REVIEW_CHANGE_REASONS.includes(value as ReviewChangeReason);
}

function isChangeReasons(value: unknown): value is Record<string, ReviewChangeReason> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  return Object.entries(value).every(([position, reason]) => /^\d+$/.test(position) && isReviewChangeReason(reason));
}

/** Read only a checkpoint bound to this exact Silver digest. Stale or malformed browser
 * data is ignored rather than being applied to a different draft. */
export function loadCheckpoint(
  conversationId: string,
  silverContentSha256: string,
  expectedTurns: number,
): ReviewCheckpoint | null {
  try {
    const raw = window.localStorage.getItem(checkpointKey(conversationId, silverContentSha256));
    if (!raw) return null;
    const value: unknown = JSON.parse(raw);
    if (!value || typeof value !== "object") return null;
    const checkpoint = value as Record<string, unknown>;
    if (
      checkpoint.version !== 1 ||
      checkpoint.silverContentSha256 !== silverContentSha256 ||
      typeof checkpoint.savedAt !== "string" ||
      typeof checkpoint.reviewer !== "string" ||
      typeof checkpoint.reviewNote !== "string" ||
      typeof checkpoint.acknowledged !== "boolean" ||
      !Array.isArray(checkpoint.turns) ||
      checkpoint.turns.length !== expectedTurns ||
      !checkpoint.turns.every(isEditableTurn)
    ) {
      return null;
    }
    if (checkpoint.changeReasons !== undefined && !isChangeReasons(checkpoint.changeReasons)) return null;
    return checkpoint as unknown as ReviewCheckpoint;
  } catch {
    return null;
  }
}

export function saveCheckpoint(
  conversationId: string,
  silverContentSha256: string,
  checkpoint: Omit<ReviewCheckpoint, "version" | "silverContentSha256" | "savedAt">,
): ReviewCheckpoint {
  const saved: ReviewCheckpoint = {
    version: 1,
    silverContentSha256,
    savedAt: new Date().toISOString(),
    ...checkpoint,
  };
  window.localStorage.setItem(
    checkpointKey(conversationId, silverContentSha256),
    JSON.stringify(saved),
  );
  return saved;
}

export function removeCheckpoint(conversationId: string, silverContentSha256: string): void {
  window.localStorage.removeItem(checkpointKey(conversationId, silverContentSha256));
}

/** Mirrors the bounds the backend enforces, so a submission is not sent to be refused. */
export const MAX_SPEAKER_CHARACTERS = 64;
export const MAX_TRANSCRIPT_CHARACTERS = 4000;
export const MAX_REVIEWER_CHARACTERS = 128;
export const MAX_REVIEW_NOTE_CHARACTERS = 2000;

export function toEditable(turn: ReviewTurn): EditableTurn {
  return {
    start: formatSeconds(turn.start),
    end: formatSeconds(turn.end),
    speaker: turn.speaker,
    transcript: turn.transcript,
    emotion: turn.emotion,
    emotion_rationale: turn.emotion_rationale,
    confidence: formatConfidence(turn.confidence),
  };
}

export function formatSeconds(value: number): string {
  return Number.isFinite(value) ? String(Number(value.toFixed(3))) : "";
}

export function formatConfidence(value: number): string {
  return Number.isFinite(value) ? String(Number(value.toFixed(3))) : "";
}

/** Seconds as mm:ss for reading, never for editing. The editable field stays in seconds
 *  because that is the unit the artifact stores and the reviewer compares against. */
export function clockTime(value: number): string {
  if (!Number.isFinite(value) || value < 0) return "--:--";
  const total = Math.floor(value);
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function parsed(raw: string): number | null {
  const text = raw.trim();
  if (!text) return null;
  const value = Number(text);
  return Number.isFinite(value) ? value : null;
}

/** Every reason this submission would not be accepted, in the order a reviewer reads. */
export function turnIssues(rows: EditableTurn[], allowedEmotions: readonly string[]): TurnIssue[] {
  const issues: TurnIssue[] = [];
  rows.forEach((row, index) => {
    const speaker = row.speaker.trim();
    if (!speaker) {
      issues.push({ index, field: "speaker", message: "화자를 입력해 주세요." });
    } else if (speaker.length > MAX_SPEAKER_CHARACTERS) {
      issues.push({
        index,
        field: "speaker",
        message: `화자는 ${MAX_SPEAKER_CHARACTERS}자 이내로 입력해 주세요.`,
      });
    }

    const transcript = row.transcript.trim();
    if (!transcript) {
      issues.push({ index, field: "transcript", message: "전사문을 입력해 주세요." });
    } else if (row.transcript.length > MAX_TRANSCRIPT_CHARACTERS) {
      issues.push({
        index,
        field: "transcript",
        message: `전사문은 ${MAX_TRANSCRIPT_CHARACTERS}자 이내로 입력해 주세요.`,
      });
    }

    if (!allowedEmotions.includes(row.emotion)) {
      issues.push({ index, field: "emotion", message: "허용된 감정 라벨을 선택해 주세요." });
    }

    const start = parsed(row.start);
    const end = parsed(row.end);
    if (start === null || start < 0) {
      issues.push({ index, field: "start", message: "시작 시각은 0 이상의 숫자여야 합니다." });
    }
    if (end === null) {
      issues.push({ index, field: "end", message: "종료 시각은 숫자여야 합니다." });
    } else if (start !== null && end <= start) {
      issues.push({ index, field: "end", message: "종료 시각은 시작 시각보다 커야 합니다." });
    }

    const confidence = parsed(row.confidence);
    if (confidence === null || confidence < 0 || confidence > 1) {
      issues.push({ index, field: "confidence", message: "확신도는 0에서 1 사이여야 합니다." });
    }
  });
  return issues;
}

/** Build the submission payload. Returns null when anything is still invalid, so there is
 *  no shape in which an unchecked turn reaches the network. */
export function toSubmission(
  rows: EditableTurn[],
  allowedEmotions: readonly string[],
): ReviewTurn[] | null {
  if (turnIssues(rows, allowedEmotions).length > 0) return null;
  return rows.map((row) => ({
    start: Number(row.start.trim()),
    end: Number(row.end.trim()),
    speaker: row.speaker.trim(),
    transcript: row.transcript,
    emotion: row.emotion,
    emotion_rationale: row.emotion_rationale,
    confidence: Number(row.confidence.trim()),
  }));
}

/** How many turns the reviewer actually changed. Shown rather than assumed: a promotion
 *  with zero edits is allowed, but the reviewer should see that is what they are signing. */
export function changedTurnCount(original: ReviewTurn[], edited: EditableTurn[]): number {
  if (original.length !== edited.length) return edited.length;
  let changed = 0;
  original.forEach((turn, index) => {
    const row = edited[index];
    if (!row) return;
    const baseline = toEditable(turn);
    const differs = (Object.keys(baseline) as TurnField[]).some(
      (field) => baseline[field] !== row[field],
    );
    if (differs) changed += 1;
  });
  return changed;
}

export function changedTurnPositions(original: ReviewTurn[], edited: EditableTurn[]): number[] {
  if (original.length !== edited.length) return edited.map((_turn, index) => index);
  return original.flatMap((turn, index) => {
    const row = edited[index];
    if (!row) return [];
    const baseline = toEditable(turn);
    return (Object.keys(baseline) as TurnField[]).some((field) => baseline[field] !== row[field])
      ? [index]
      : [];
  });
}

/** Signals that warrant listening before a reviewer spends time on ordinary turns.
 * They are triage hints, never claims that a turn is wrong. */
export function reviewQuality(rows: EditableTurn[]): ReviewQuality {
  const confidences = rows
    .map((row) => parsed(row.confidence))
    .filter((value): value is number => value !== null && value >= 0 && value <= 1);
  const uncertain = new Set<number>();
  const short = new Set<number>();
  const overlap = new Set<number>();

  rows.forEach((row, index) => {
    const start = parsed(row.start);
    const end = parsed(row.end);
    const confidence = parsed(row.confidence);
    if (row.emotion === "uncertain" || confidence === null || confidence < 0.6) uncertain.add(index);
    if (start !== null && end !== null && end > start && end - start < 1) short.add(index);
  });
  rows.forEach((row, index) => {
    const start = parsed(row.start);
    if (start === null) return;
    rows.slice(0, index).forEach((previous, previousIndex) => {
      const previousStart = parsed(previous.start);
      const previousEnd = parsed(previous.end);
      if (previousStart !== null && previousEnd !== null && previousStart < start && previousEnd > start) {
        overlap.add(previousIndex);
        overlap.add(index);
      }
    });
  });
  const priority = new Set([...uncertain, ...short, ...overlap]);
  return {
    meanConfidence: confidences.length
      ? Math.round((confidences.reduce((total, value) => total + value, 0) / confidences.length) * 100) / 100
      : null,
    uncertainPositions: [...uncertain].sort((left, right) => left - right),
    shortPositions: [...short].sort((left, right) => left - right),
    overlapPositions: [...overlap].sort((left, right) => left - right),
    priorityPositions: [...priority].sort((left, right) => left - right),
  };
}

/** Korean copy for each backend warning code, with its own count woven in.
 *
 * The backend sends sanitized English detail for every warning. That text is kept as the
 * fallback rather than dropped: an unmapped code has to stay visible, because the codes
 * that matter most here are the ones nobody anticipated. */
export function warningCopy(warning: ReviewWarning): { title: string; detail: string } {
  switch (warning.code) {
    case "review_required":
      return {
        title: "이 초안은 아직 정답이 아닙니다",
        detail:
          "모델이 만든 잠정 결과입니다. 검수자가 음성과 대조해 확정하기 전까지는 어떤 항목도 정답으로 쓸 수 없습니다.",
      };
    case "salvaged_dropped_turns":
      return {
        title: `검증에서 탈락한 발화 ${warning.count ?? 0}개가 이 초안에 없습니다`,
        detail:
          "모델이 반환했지만 검증 규칙에 걸려 제외된 발화입니다. 복구되지 않았으므로 그 구간의 말은 여기에 없습니다. 음성을 직접 들어 확인해 주세요." +
          formatRules(warning.by_rule),
      };
    case "remote_file_not_deleted":
      return {
        title: "업로드된 음성 파일의 삭제를 확인하지 못했습니다",
        detail: "원격 주석 서비스에서 파일이 지워졌다는 확인을 받지 못했습니다.",
      };
    case "remote_audio_transmitted":
      return {
        title: "이 초안의 음성은 원격 서비스로 전송됐습니다",
        detail: "초안을 만들 때 음성이 외부 주석 서비스로 전송됐습니다.",
      };
    case "gold_already_exists":
      return {
        title: "이 대화에는 이미 확정본이 있습니다",
        detail: "확정본은 덮어쓸 수 없습니다. 이 화면은 읽기 확인용으로만 사용해 주세요.",
      };
    default:
      return { title: warning.code, detail: warning.detail };
  }
}

function formatRules(byRule: Record<string, number> | null): string {
  if (!byRule) return "";
  const entries = Object.entries(byRule);
  if (entries.length === 0) return "";
  return ` (규칙별: ${entries.map(([rule, count]) => `${rule} ${count}건`).join(", ")})`;
}

export function isBlockingWarning(code: string): boolean {
  return code === "gold_already_exists";
}
