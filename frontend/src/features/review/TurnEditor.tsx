import { annotationEmotionLabel } from "../../labels";
import { ClipButton } from "./ClipButton";
import type { Playback } from "./playback";
import {
  clockTime,
  REVIEW_CHANGE_REASONS,
  reviewChangeReasonLabel,
  type EditableTurn,
  type ReviewChangeReason,
  type TurnField,
  type TurnIssue,
} from "./review";

interface TurnEditorProps {
  index: number;
  turn: EditableTurn;
  original: EditableTurn;
  emotions: string[];
  issues: TurnIssue[];
  disabled: boolean;
  /** Absent when this machine holds the draft but not its recording. The control is then
   *  not rendered at all rather than shown disabled: there is nothing to wait for. */
  playback: Playback | null;
  /** The current editable range, bounded to the source recording by the parent. */
  clip: { start: number; end: number } | null;
  priorityHints: string[];
  changeReason: ReviewChangeReason | "";
  onChange(index: number, field: TurnField, value: string): void;
  onReasonChange(index: number, value: ReviewChangeReason | ""): void;
}

function issueFor(issues: TurnIssue[], field: TurnField): string | undefined {
  return issues.find((issue) => issue.field === field)?.message;
}

export function TurnEditor({
  index,
  turn,
  original,
  emotions,
  issues,
  disabled,
  playback,
  clip,
  priorityHints,
  changeReason,
  onChange,
  onReasonChange,
}: TurnEditorProps) {
  const position = index + 1;
  const edited = (Object.keys(original) as TurnField[]).some(
    (field) => original[field] !== turn[field],
  );
  const startSeconds = Number(turn.start);
  const endSeconds = Number(turn.end);
  const field = (name: TurnField) => `turn-${index}-${name}`;

  return (
    <li className={edited ? "turn-card edited" : "turn-card"}>
      <div className="turn-head">
        <span className="turn-index">{position}</span>
        <span className="turn-clock mono">
          {clockTime(startSeconds)} - {clockTime(endSeconds)}
        </span>
        {edited && <span className="turn-flag">수정함</span>}
        {priorityHints.length > 0 && <span className="turn-priority">확인 우선</span>}
        {playback && clip && (
          <ClipButton
            clip={{ kind: "turn", index }}
            start={clip.start}
            end={clip.end}
            playback={playback}
            label="이 구간 듣기"
            accessibleLabel={`${position}번 발화 ${clockTime(clip.start)}부터 ${clockTime(
              clip.end,
            )}까지 듣기`}
          />
        )}
      </div>
      {playback && clip && playback.error && playback.errorClip?.kind === "turn" &&
        playback.errorClip.index === index && (
          <p className="clip-error" role="alert">
            {playback.error.title}. {playback.error.detail}
          </p>
        )}

      <div className="turn-grid">
        <div className="turn-field">
          <label className="field-label" htmlFor={field("speaker")}>
            화자
          </label>
          <input
            id={field("speaker")}
            type="text"
            value={turn.speaker}
            disabled={disabled}
            onChange={(event) => onChange(index, "speaker", event.target.value)}
          />
          {issueFor(issues, "speaker") && (
            <p className="field-error">{issueFor(issues, "speaker")}</p>
          )}
        </div>

        <div className="turn-field">
          <label className="field-label" htmlFor={field("start")}>
            시작 (초)
          </label>
          <input
            id={field("start")}
            type="text"
            inputMode="decimal"
            value={turn.start}
            disabled={disabled}
            onChange={(event) => onChange(index, "start", event.target.value)}
          />
          {issueFor(issues, "start") && <p className="field-error">{issueFor(issues, "start")}</p>}
        </div>

        <div className="turn-field">
          <label className="field-label" htmlFor={field("end")}>
            종료 (초)
          </label>
          <input
            id={field("end")}
            type="text"
            inputMode="decimal"
            value={turn.end}
            disabled={disabled}
            onChange={(event) => onChange(index, "end", event.target.value)}
          />
          {issueFor(issues, "end") && <p className="field-error">{issueFor(issues, "end")}</p>}
        </div>

        <div className="turn-field">
          <label className="field-label" htmlFor={field("emotion")}>
            감정
          </label>
          <select
            id={field("emotion")}
            value={turn.emotion}
            disabled={disabled}
            onChange={(event) => onChange(index, "emotion", event.target.value)}
          >
            {/* A label the artifact holds but the backend no longer accepts stays visible
                rather than being silently rewritten to the first option. */}
            {!emotions.includes(turn.emotion) && (
              <option value={turn.emotion}>{annotationEmotionLabel(turn.emotion)}</option>
            )}
            {emotions.map((emotion) => (
              <option key={emotion} value={emotion}>
                {annotationEmotionLabel(emotion)}
              </option>
            ))}
          </select>
          {issueFor(issues, "emotion") && (
            <p className="field-error">{issueFor(issues, "emotion")}</p>
          )}
        </div>

        <div className="turn-field">
          <label className="field-label" htmlFor={field("confidence")}>
            확신도
          </label>
          <input
            id={field("confidence")}
            type="text"
            inputMode="decimal"
            value={turn.confidence}
            disabled={disabled}
            onChange={(event) => onChange(index, "confidence", event.target.value)}
          />
          {issueFor(issues, "confidence") && (
            <p className="field-error">{issueFor(issues, "confidence")}</p>
          )}
        </div>
      </div>

      <div className="turn-field wide">
        <label className="field-label" htmlFor={field("transcript")}>
          전사문
        </label>
        <textarea
          id={field("transcript")}
          rows={2}
          value={turn.transcript}
          disabled={disabled}
          onChange={(event) => onChange(index, "transcript", event.target.value)}
        />
        {issueFor(issues, "transcript") && (
          <p className="field-error">{issueFor(issues, "transcript")}</p>
        )}
      </div>

      {turn.emotion_rationale && (
        <p className="turn-rationale">
          <span className="field-label">모델이 적은 근거</span>
          {turn.emotion_rationale}
        </p>
      )}

      {edited && (
        <div className="turn-field wide review-reason">
          <label className="field-label" htmlFor={`turn-${index}-review-reason`}>
            수정 사유 <em>필수</em>
          </label>
          <select
            id={`turn-${index}-review-reason`}
            value={changeReason}
            disabled={disabled}
            onChange={(event) => onReasonChange(index, event.target.value as ReviewChangeReason | "")}
          >
            <option value="">사유를 선택해 주세요</option>
            {REVIEW_CHANGE_REASONS.map((reason) => (
              <option key={reason} value={reason}>
                {reviewChangeReasonLabel[reason]}
              </option>
            ))}
          </select>
          <p className="turn-reason-help">
            Gold 확정본에 이 수정 사유만 함께 남습니다. 전사문을 별도로 복사하지 않습니다.
          </p>
        </div>
      )}
    </li>
  );
}
