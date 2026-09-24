import {
  AudioLines,
  CheckCircle2,
  FileSearch,
  Loader2,
  Save,
  ShieldCheck,
  Trash2,
  TriangleAlert,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { describeError, type ErrorGuidance } from "../../api/errors";
import type {
  AnnotationAudioOverview,
  AlignmentProposal,
  AnnotationReviewIndex,
  AnnotationReviewState,
  ApiClient,
  GoldPromotionResult,
  EmotionOverlayCandidate,
  GeminiEmotionOverlayCandidate,
  ReferenceResegmentationCandidate,
  SilverReviewDraft,
} from "../../api/types";
import { ClipButton } from "./ClipButton";
import { useClipPlayback } from "./playback";
import { TurnEditor } from "./TurnEditor";
import {
  changedTurnCount,
  changedTurnPositions,
  clockTime,
  isBlockingWarning,
  loadCheckpoint,
  MAX_REVIEWER_CHARACTERS,
  MAX_REVIEW_NOTE_CHARACTERS,
  removeCheckpoint,
  reviewQuality,
  saveCheckpoint,
  toEditable,
  toSubmission,
  turnIssues,
  warningCopy,
  type EditableTurn,
  type ReviewChangeReason,
  type TurnField,
} from "./review";

interface SilverReviewProps {
  client: ApiClient;
}

type Phase = "loading" | "ready" | "failed";

/** The Silver review mode.
 *
 * The screen's whole job is to make the crossing from a machine's guess to reviewed truth
 * something a person did on purpose. So the promote control stays closed until the
 * reviewer has named themselves, said they checked the draft, and left no field the
 * backend would refuse. Nothing here writes silver, and nothing promotes on its own.
 */
export function SilverReview({ client }: SilverReviewProps) {
  const [phase, setPhase] = useState<Phase>("loading");
  const [index, setIndex] = useState<AnnotationReviewIndex | null>(null);
  const [indexError, setIndexError] = useState<ErrorGuidance | null>(null);

  const [selected, setSelected] = useState<string | null>(null);
  const [draft, setDraft] = useState<SilverReviewDraft | null>(null);
  const [draftLoading, setDraftLoading] = useState(false);
  const [draftError, setDraftError] = useState<ErrorGuidance | null>(null);
  const [alignmentProposals, setAlignmentProposals] = useState<AlignmentProposal[]>([]);
  const [alignmentNotice, setAlignmentNotice] = useState<string | null>(null);
  const [referenceCandidate, setReferenceCandidate] = useState<ReferenceResegmentationCandidate | null>(null);
  const [referenceNotice, setReferenceNotice] = useState<string | null>(null);
  const [emotionCandidate, setEmotionCandidate] = useState<EmotionOverlayCandidate | null>(null);
  const [emotionCandidateNotice, setEmotionCandidateNotice] = useState<string | null>(null);
  const [geminiEmotionCandidate, setGeminiEmotionCandidate] =
    useState<GeminiEmotionOverlayCandidate | null>(null);
  const [geminiEmotionNotice, setGeminiEmotionNotice] = useState<string | null>(null);

  const [turns, setTurns] = useState<EditableTurn[]>([]);
  const [reviewer, setReviewer] = useState("");
  const [reviewNote, setReviewNote] = useState("");
  const [changeReasons, setChangeReasons] = useState<Record<string, ReviewChangeReason>>({});
  const [acknowledged, setAcknowledged] = useState(false);
  const [attempted, setAttempted] = useState(false);
  const [checkpointSavedAt, setCheckpointSavedAt] = useState<string | null>(null);
  const [checkpointError, setCheckpointError] = useState<string | null>(null);

  const [audio, setAudio] = useState<AnnotationAudioOverview | null>(null);
  const [audioError, setAudioError] = useState<ErrorGuidance | null>(null);
  const [audioLoading, setAudioLoading] = useState(false);
  const playback = useClipPlayback(client, selected);

  const [promoting, setPromoting] = useState(false);
  const [promotionError, setPromotionError] = useState<ErrorGuidance | null>(null);
  const [promoted, setPromoted] = useState<GoldPromotionResult | null>(null);

  const loadIndex = useCallback(() => {
    setPhase("loading");
    setIndexError(null);
    return client
      .listSilverAnnotations()
      .then((next) => {
        setIndex(next);
        setPhase("ready");
      })
      .catch((reason: unknown) => {
        setIndexError(describeError(reason));
        setPhase("failed");
      });
  }, [client]);

  useEffect(() => {
    void loadIndex();
  }, [loadIndex]);

  const openDraft = useCallback(
    (conversationId: string) => {
      setSelected(conversationId);
      setDraft(null);
      setDraftError(null);
      setAlignmentProposals([]);
      setAlignmentNotice(null);
      setReferenceCandidate(null);
      setReferenceNotice(null);
      setEmotionCandidate(null);
      setEmotionCandidateNotice(null);
      setGeminiEmotionCandidate(null);
      setGeminiEmotionNotice(null);
      setDraftLoading(true);
      setPromotionError(null);
      setPromoted(null);
      setAttempted(false);
      setCheckpointSavedAt(null);
      setCheckpointError(null);
      // Identity and acknowledgement are per draft: they say this person checked this
      // material, so carrying them across a switch would be a claim nobody made.
      setReviewer("");
      setReviewNote("");
      setChangeReasons({});
      setAcknowledged(false);
      setAudio(null);
      setAudioError(null);
      setAudioLoading(true);
      // Fetched alongside the draft rather than inside it: a machine can hold the
      // annotation without holding the recording, and that must not fail the draft.
      void client
        .getAnnotationAudio(conversationId)
        .then(setAudio)
        .catch((reason: unknown) => setAudioError(describeError(reason)))
        .finally(() => setAudioLoading(false));
      void client
        .getSilverAnnotation(conversationId)
        .then((next) => {
          setDraft(next);
          const checkpoint = loadCheckpoint(
            next.state.conversation_id,
            next.state.content_sha256,
            next.turns.length,
          );
          if (checkpoint) {
            setTurns(checkpoint.turns);
            setReviewer(checkpoint.reviewer);
            setReviewNote(checkpoint.reviewNote);
            setChangeReasons(checkpoint.changeReasons ?? {});
            setAcknowledged(checkpoint.acknowledged);
            setCheckpointSavedAt(checkpoint.savedAt);
          } else {
            setTurns(next.turns.map(toEditable));
          }
          // A proposal is timing-only and tied to this exact immutable Silver digest.
          // Do not let a stale proposal affect the reviewer's browser-local working copy.
          void client
            .getAlignmentProposals(conversationId)
            .then(({ proposals }) => {
              setAlignmentProposals(
                proposals.filter(
                  (proposal) => proposal.source_silver_content_sha256 === next.state.content_sha256,
                ),
              );
            })
            .catch(() => {
              // A missing or unreadable optional proposal must never block normal review.
            });
          if (next.state.model !== "kcsc-human-reference") {
            void client
              .getReferenceResegmentationCandidate(conversationId)
              .then((candidate) => {
                if (candidate?.source_silver_content_sha256 === next.state.content_sha256) {
                  setReferenceCandidate(candidate);
                }
              })
              .catch(() => {
                // KCSC reference material is optional and must never block normal review.
              });
          }
          // Older test/dev clients may not implement this optional review enhancement.
          // Never let that prevent the clean Silver itself from opening.
          if (typeof client.getEmotionOverlayCandidate === "function") {
            void client
              .getEmotionOverlayCandidate(conversationId)
              .then((candidate) => {
                if (candidate?.source_silver_content_sha256 === next.state.content_sha256) {
                  setEmotionCandidate(candidate);
                }
              })
              .catch(() => {
                // An optional local hypothesis must never prevent regular review.
              });
          }
          // The remote overlay is fetched, never generated, from here. Producing one
          // transmits audio to Google and that is a deliberate operator action taken
          // outside the browser; the review screen only ever reads what already exists.
          if (typeof client.getGeminiEmotionOverlayCandidate === "function") {
            void client
              .getGeminiEmotionOverlayCandidate(conversationId)
              .then((candidate) => {
                if (
                  candidate?.source_silver_content_sha256 === next.state.content_sha256 &&
                  candidate.review_required &&
                  !candidate.promotable
                ) {
                  setGeminiEmotionCandidate(candidate);
                }
              })
              .catch(() => {
                // An optional remote hypothesis must never prevent regular review.
              });
          }
        })
        .catch((reason: unknown) => setDraftError(describeError(reason)))
        .finally(() => setDraftLoading(false));
    },
    [client],
  );

  function editTurn(position: number, field: TurnField, value: string) {
    setTurns((current) =>
      current.map((row, at) => (at === position ? { ...row, [field]: value } : row)),
    );
  }

  function editChangeReason(position: number, reason: ReviewChangeReason | "") {
    setChangeReasons((current) => {
      const next = { ...current };
      if (reason) next[String(position)] = reason;
      else delete next[String(position)];
      return next;
    });
  }

  function applyAlignmentProposal(alignmentProposal: AlignmentProposal) {
    if (!draft || goldExists || promoting) return;
    const byPosition = new Map(alignmentProposal.rows.map((row) => [row.position, row]));
    const applied = alignmentProposal.rows.filter(
      (row) => row.position >= alignmentProposal.target_start_position && row.position <= alignmentProposal.target_end_position,
    ).length;
    setTurns((current) =>
      current.map((turn, index) => {
        const row = byPosition.get(index + 1);
        if (
          !row ||
          index + 1 < alignmentProposal.target_start_position ||
          index + 1 > alignmentProposal.target_end_position
        ) return turn;
        return { ...turn, start: String(row.start), end: String(row.end) };
      }),
    );
    setChangeReasons((current) => {
      const next = { ...current };
      alignmentProposal.rows.forEach((row) => {
        if (
          row.position >= alignmentProposal.target_start_position &&
          row.position <= alignmentProposal.target_end_position
        ) next[String(row.position - 1)] = "timing_mismatch";
      });
      return next;
    });
    setAlignmentNotice(
      `Gemini 시간 제안 ${applied}개를 편집 화면에만 반영했습니다. 확인한 뒤 임시 저장하세요.`,
    );
  }

  function applyReferenceCandidate(candidate: ReferenceResegmentationCandidate) {
    if (!draft || goldExists || promoting) return;
    if (candidate.source_silver_content_sha256 !== draft.state.content_sha256) return;
    // Preserve everything the reviewer currently has before replacing the in-memory
    // working copy. The candidate itself is not auto-saved: refresh restores this
    // checkpoint unless the reviewer explicitly chooses to save the candidate.
    try {
      const checkpoint = saveCheckpoint(draft.state.conversation_id, draft.state.content_sha256, {
        turns,
        reviewer,
        reviewNote,
        acknowledged,
        changeReasons,
      });
      setCheckpointSavedAt(checkpoint.savedAt);
      setCheckpointError(null);
    } catch {
      setCheckpointError(
        "현재 작업본을 보호하지 못해 KCSC 재분할 후보를 적용하지 않았습니다. 브라우저 저장 공간을 확인해 주세요.",
      );
      return;
    }
    setTurns(candidate.turns.map(toEditable));
    setChangeReasons(
      Object.fromEntries(candidate.turns.map((_turn, position) => [String(position), "model_candidate"])),
    );
    setReferenceNotice(
      `현재 작업본을 임시 저장한 뒤 KCSC 기준 재분할 후보 ${candidate.reference_turn_count}개를 편집 화면에만 반영했습니다. 감정은 모두 판단 불확실로 남아 있습니다.`,
    );
  }

  function applyEmotionCandidate(candidate: EmotionOverlayCandidate) {
    if (!draft || goldExists || promoting) return;
    if (candidate.source_silver_content_sha256 !== draft.state.content_sha256) return;
    if (candidate.turns.length !== turns.length) return;
    try {
      const checkpoint = saveCheckpoint(draft.state.conversation_id, draft.state.content_sha256, {
        turns,
        reviewer,
        reviewNote,
        acknowledged,
        changeReasons,
      });
      setCheckpointSavedAt(checkpoint.savedAt);
      setCheckpointError(null);
    } catch {
      setCheckpointError(
        "현재 작업본을 보호하지 못해 로컬 감정 후보를 적용하지 않았습니다. 브라우저 저장 공간을 확인해 주세요.",
      );
      return;
    }
    setTurns((current) =>
      current.map((turn, position) => ({
        ...turn,
        emotion: candidate.turns[position].emotion,
        emotion_rationale: candidate.turns[position].emotion_rationale,
        confidence: String(candidate.turns[position].confidence),
      })),
    );
    setChangeReasons((current) =>
      Object.fromEntries(turns.map((_turn, position) => [String(position), current[String(position)] ?? "emotion_mismatch"])),
    );
    setEmotionCandidateNotice(
      `현재 작업본을 임시 저장한 뒤 로컬 XLS-R 감정 후보 ${candidate.turns.length}개를 편집 화면에만 반영했습니다. 시간·화자·전사는 바꾸지 않았습니다.`,
    );
  }

  function applyGeminiEmotionCandidate(candidate: GeminiEmotionOverlayCandidate) {
    if (!draft || goldExists || promoting) return;
    if (candidate.source_silver_content_sha256 !== draft.state.content_sha256) return;
    // The overlay is a positional mapping onto turns it does not restate, so a length
    // mismatch is not a partial application to fall back on -- it means this candidate
    // is not about the working copy on screen.
    if (candidate.turns.length !== turns.length) return;
    try {
      const checkpoint = saveCheckpoint(draft.state.conversation_id, draft.state.content_sha256, {
        turns,
        reviewer,
        reviewNote,
        acknowledged,
        changeReasons,
      });
      setCheckpointSavedAt(checkpoint.savedAt);
      setCheckpointError(null);
    } catch {
      setCheckpointError(
        "현재 작업본을 보호하지 못해 Gemini 원격 감정 후보를 적용하지 않았습니다. 브라우저 저장 공간을 확인해 주세요.",
      );
      return;
    }
    setTurns((current) =>
      current.map((turn, position) => ({
        ...turn,
        emotion: candidate.turns[position].emotion,
        emotion_rationale: candidate.turns[position].emotion_rationale,
        confidence: String(candidate.turns[position].confidence),
      })),
    );
    setChangeReasons((current) =>
      Object.fromEntries(turns.map((_turn, position) => [String(position), current[String(position)] ?? "model_candidate"])),
    );
    setGeminiEmotionNotice(
      `현재 작업본을 임시 저장한 뒤 Gemini 원격 감정 후보 ${candidate.turns.length}개를 편집 화면에만 반영했습니다. 시간·화자·전사는 바꾸지 않았으며, Gold 승격은 여전히 검수자 확인이 필요합니다.`,
    );
  }

  const ninthReferenceSubsegments = useMemo(() => {
    if (
      !draft ||
      draft.state.model === "kcsc-human-reference" ||
      !referenceCandidate ||
      draft.turns.length < 9
    ) {
      return [];
    }
    const source = draft.turns[8];
    return referenceCandidate.turns.filter(
      (turn) => turn.start < source.end && turn.end > source.start,
    );
  }, [draft, referenceCandidate]);

  function applyNinthReferenceSubsegments() {
    if (!draft || !referenceCandidate || goldExists || promoting) return;
    if (referenceCandidate.source_silver_content_sha256 !== draft.state.content_sha256) return;
    if (ninthReferenceSubsegments.length < 2) return;
    try {
      const checkpoint = saveCheckpoint(draft.state.conversation_id, draft.state.content_sha256, {
        turns,
        reviewer,
        reviewNote,
        acknowledged,
        changeReasons,
      });
      setCheckpointSavedAt(checkpoint.savedAt);
      setCheckpointError(null);
    } catch {
      setCheckpointError(
        "현재 작업본을 보호하지 못해 9번 세분화 후보를 적용하지 않았습니다. 브라우저 저장 공간을 확인해 주세요.",
      );
      return;
    }
    const splitTurns = turns.flatMap((turn, index) =>
      index === 8 ? ninthReferenceSubsegments.map(toEditable) : [turn],
    );
    setTurns(splitTurns);
    setChangeReasons(
      Object.fromEntries(
        splitTurns.map((_turn, position) => [
          String(position),
          "model_candidate",
        ]),
      ),
    );
    // Timing proposals are bound to the immutable Silver's original positions. After
    // replacing one turn with many, they must not be applied to shifted positions.
    setAlignmentProposals([]);
    setReferenceNotice(
      `현재 작업본을 임시 저장한 뒤 9번 긴 구간을 KCSC 기준 ${ninthReferenceSubsegments.length}개 발화로 세분화했습니다. 감정은 모두 판단 불확실로 남아 있으며, 원래 번호 기준 시간 제안은 숨겼습니다.`,
    );
  }

  function currentTurnClip(turn: EditableTurn): { start: number; end: number } | null {
    if (!audio) return null;
    const start = Number(turn.start);
    const requestedEnd = Number(turn.end);
    if (
      !Number.isFinite(start) ||
      !Number.isFinite(requestedEnd) ||
      start < 0 ||
      requestedEnd <= start ||
      start >= audio.duration_seconds
    ) {
      return null;
    }
    const end = Math.min(requestedEnd, audio.duration_seconds);
    return end > start ? { start, end } : null;
  }

  const emotions = useMemo(() => draft?.emotion_labels ?? [], [draft]);
  const originals = useMemo(() => draft?.turns.map(toEditable) ?? [], [draft]);
  const issues = useMemo(() => turnIssues(turns, emotions), [turns, emotions]);
  const quality = useMemo(() => reviewQuality(turns), [turns]);
  const changedPositions = useMemo(() => changedTurnPositions(draft?.turns ?? [], turns), [draft, turns]);
  const missingReasonPositions = useMemo(
    () => changedPositions.filter((position) => !changeReasons[String(position)]),
    [changeReasons, changedPositions],
  );
  const prioritySet = useMemo(() => new Set(quality.priorityPositions), [quality.priorityPositions]);
  const changed = useMemo(
    () => (draft ? changedTurnCount(draft.turns, turns) : 0),
    [draft, turns],
  );
  const goldExists = Boolean(draft?.state.gold_present) || Boolean(promoted);
  const namedReviewer = reviewer.trim();
  const promotable =
    Boolean(draft) &&
    !goldExists &&
    !promoting &&
    namedReviewer.length > 0 &&
    acknowledged &&
    turns.length > 0 &&
    issues.length === 0 &&
    missingReasonPositions.length === 0;

  function saveWorkingCopy() {
    if (!draft || goldExists) return;
    try {
      const checkpoint = saveCheckpoint(draft.state.conversation_id, draft.state.content_sha256, {
        turns,
        reviewer,
        reviewNote,
        acknowledged,
        changeReasons,
      });
      setCheckpointSavedAt(checkpoint.savedAt);
      setCheckpointError(null);
    } catch {
      setCheckpointError(
        "이 브라우저에 임시 저장하지 못했습니다. 저장 공간 또는 브라우저 설정을 확인해 주세요.",
      );
    }
  }

  function discardWorkingCopy() {
    if (!draft) return;
    try {
      removeCheckpoint(draft.state.conversation_id, draft.state.content_sha256);
      setCheckpointSavedAt(null);
      setCheckpointError(null);
    } catch {
      setCheckpointError("이 브라우저의 임시 저장본을 지우지 못했습니다.");
    }
  }

  async function promote() {
    setAttempted(true);
    const submission = draft && toSubmission(turns, emotions);
    if (!draft || !submission || !promotable) return;
    setPromoting(true);
    setPromotionError(null);
    try {
      const receipt = await client.promoteSilverToGold(draft.state.conversation_id, {
        reviewer: namedReviewer,
        acknowledged: true,
        turns: submission,
        review_note: reviewNote,
        change_reasons: changedPositions.flatMap((position) => {
          const reason = changeReasons[String(position)];
          return reason ? [{ position: position + 1, reason }] : [];
        }),
      });
      setPromoted(receipt);
      removeCheckpoint(draft.state.conversation_id, draft.state.content_sha256);
      setCheckpointSavedAt(null);
      // Marked from the receipt rather than refetched: the backend has just said gold
      // exists, and a refresh here would blank this screen while the receipt is being read.
      setIndex((current) =>
        current === null
          ? current
          : {
              ...current,
              annotations: current.annotations.map((state) =>
                state.conversation_id === receipt.conversation_id
                  ? { ...state, gold_present: true }
                  : state,
              ),
            },
      );
    } catch (reason) {
      setPromotionError(describeError(reason));
    } finally {
      setPromoting(false);
    }
  }

  if (phase === "loading") {
    return (
      <section className="panel review-panel" aria-busy="true" aria-labelledby="review-heading">
        <header className="panel-head">
          <div>
            <span className="kicker">SILVER REVIEW</span>
            <h2 id="review-heading">검수할 초안을 불러오는 중입니다</h2>
          </div>
        </header>
        <p className="review-status">
          <Loader2 aria-hidden="true" size={16} className="spin" /> 로컬 백엔드에서 초안 목록을
          읽고 있습니다.
        </p>
      </section>
    );
  }

  if (phase === "failed") {
    return (
      <section className="panel review-panel" aria-labelledby="review-heading">
        <header className="panel-head">
          <div>
            <span className="kicker">SILVER REVIEW</span>
            <h2 id="review-heading">초안 목록을 불러오지 못했습니다</h2>
          </div>
        </header>
        <div className="notice alert" role="alert">
          <TriangleAlert aria-hidden="true" size={18} />
          <div>
            <strong>{indexError?.title ?? "초안 목록을 불러오지 못했어요"}</strong>
            <p>{indexError?.detail ?? "잠시 후 다시 시도해 주세요."}</p>
          </div>
          <button className="button ghost" type="button" onClick={() => void loadIndex()}>
            다시 시도
          </button>
        </div>
      </section>
    );
  }

  const annotations = index?.annotations ?? [];

  return (
    <section className="panel review-panel" aria-labelledby="review-heading">
      <header className="panel-head">
        <div>
          <span className="kicker">SILVER REVIEW</span>
          <h2 id="review-heading">Silver 초안을 검수해 확정본으로 올립니다</h2>
          <p>
            Silver는 모델이 만든 잠정 결과입니다. 검수자가 음성과 대조해 고치고 확인한 뒤에만
            확정본이 되며, 확정본은 한 번만 기록되고 다시 바꿀 수 없습니다.
          </p>
        </div>
      </header>

      {annotations.length === 0 ? (
        <div className="empty-result">
          <FileSearch aria-hidden="true" size={30} />
          <h2>검수할 초안이 없습니다</h2>
          <p>
            이 컴퓨터에 저장된 Silver 초안이 아직 없습니다. 초안을 만든 뒤 이 화면을 다시 열어
            주세요.
          </p>
          <button className="button ghost" type="button" onClick={() => void loadIndex()}>
            목록 새로 고침
          </button>
        </div>
      ) : (
        <>
          {(index?.unreadable_count ?? 0) > 0 && (
            <div className="notice alert" role="alert">
              <TriangleAlert aria-hidden="true" size={18} />
              <div>
                <strong>읽을 수 없는 초안 {index?.unreadable_count}건이 있습니다</strong>
                <p>파일이 손상됐거나 형식이 맞지 않아 이 목록에 나오지 않습니다.</p>
              </div>
            </div>
          )}

          <ul className="review-list">
            {annotations.map((state) => (
              <li key={state.conversation_id}>
                <button
                  type="button"
                  className={
                    state.conversation_id === selected
                      ? "review-item selected"
                      : "review-item"
                  }
                  aria-pressed={state.conversation_id === selected}
                  disabled={promoting}
                  onClick={() => openDraft(state.conversation_id)}
                >
                  <strong className="mono">{state.conversation_id}</strong>
                  <span>{summaryLine(state)}</span>
                  {(state.emotion_candidate_count ?? 0) > 0 && (
                    <span className="review-candidate-count">
                      기준 Silver + 감정 후보 {state.emotion_candidate_count}개
                    </span>
                  )}
                  <span className={state.gold_present ? "review-tag gold" : "review-tag"}>
                    {state.gold_present ? "확정본 있음" : "검수 필요"}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </>
      )}

      {draftLoading && (
        <p className="review-status" aria-busy="true">
          <Loader2 aria-hidden="true" size={16} className="spin" /> 초안을 불러오는 중입니다.
        </p>
      )}

      {draftError && (
        <div className="notice alert" role="alert">
          <TriangleAlert aria-hidden="true" size={18} />
          <div>
            <strong>{draftError.title}</strong>
            <p>{draftError.detail}</p>
          </div>
          {selected && (
            <button
              className="button ghost"
              type="button"
              onClick={() => openDraft(selected)}
            >
              다시 시도
            </button>
          )}
        </div>
      )}

      {promoted && (
        <div className="notice gold" role="status">
          <CheckCircle2 aria-hidden="true" size={18} />
          <div>
            <strong>확정본으로 기록했습니다</strong>
            <p>
              {promoted.conversation_id} / 검수자 {promoted.reviewer} / 발화{" "}
              {promoted.turn_count}개 / 기록 시각 {promoted.reviewed_at}
            </p>
            <p className="mono review-digest">gold sha256 {promoted.content_sha256}</p>
            <p>
              {promoted.silver_unmodified
                ? "Silver 초안은 그대로 남아 있습니다. 확정본은 다시 쓸 수 없습니다."
                : "Silver 초안이 바뀐 것으로 확인됐습니다. 이 결과는 신뢰할 수 없으니 백엔드 로그를 확인해 주세요."}
            </p>
          </div>
        </div>
      )}

      {draft && !promoted && (
        <div className="review-draft">
          {draft.warnings.map((warning) => {
            const copy = warningCopy(warning);
            return (
              <div
                key={warning.code}
                className={isBlockingWarning(warning.code) ? "notice alert" : "notice"}
                role={isBlockingWarning(warning.code) ? "alert" : undefined}
              >
                <TriangleAlert aria-hidden="true" size={18} />
                <div>
                  <strong>{copy.title}</strong>
                  <p>{copy.detail}</p>
                </div>
              </div>
            );
          })}

          {alignmentProposals.map((alignmentProposal) => (
            <div
              className="notice"
              role="status"
              key={`${alignmentProposal.target_start_position}-${alignmentProposal.target_end_position}`}
            >
              <AudioLines aria-hidden="true" size={18} />
              <div>
                <strong>
                  {alignmentProposal.target_start_position}~{alignmentProposal.target_end_position}번 시간 정렬 제안 {alignmentProposal.rows.length}개
                </strong>
                <p>
                  Gemini가 제안한 시작·끝 시각입니다. 전사문·화자·감정은 바꾸지 않으며,
                  기존 임시 저장본도 자동으로 덮어쓰지 않습니다.
                </p>
                <button
                  className="button ghost"
                  type="button"
                  disabled={promoting || goldExists}
                  onClick={() => applyAlignmentProposal(alignmentProposal)}
                >
                  {alignmentProposal.target_start_position}~{alignmentProposal.target_end_position}번 시간 제안 적용
                </button>
                {alignmentNotice && <p className="review-alignment-note">{alignmentNotice}</p>}
              </div>
            </div>
          ))}
          {emotionCandidate && !goldExists && (
            <div className="notice" role="status" aria-label="로컬 XLS-R 감정 후보">
              <div>
                <p className="eyebrow">Local calibrated XLS-R</p>
                <h3>감정 검수 후보 {emotionCandidate.turns.length}개</h3>
                <p>
                  KCSC 기준 시간·화자·전사는 그대로 두고, 로컬 모델의 감정·확신도만 덧씌운 후보야.
                  Gold는 아니며, 적용 전 현재 작업본을 임시 저장해.
                </p>
                <p className="review-reference-note">
                  외부 전송 없음 · 판단 불확실 {emotionCandidate.uncertain_turns}개 · {Object.entries(emotionCandidate.emotion_histogram).filter(([, count]) => count > 0).map(([label, count]) => `${label} ${count}`).join(" · ")}
                </p>
              </div>
              <button
                type="button"
                className="button ghost"
                onClick={() => applyEmotionCandidate(emotionCandidate)}
                disabled={promoting}
              >
                로컬 감정 후보를 작업본에 적용
              </button>
              {emotionCandidateNotice && <p className="review-alignment-note">{emotionCandidateNotice}</p>}
            </div>
          )}
          {geminiEmotionCandidate && !goldExists && (
            <div className="notice" role="status" aria-label="Gemini 원격 감정 후보">
              <div>
                <p className="eyebrow">Gemini remote · 외부 전송됨</p>
                <h3>원격 감정 검수 후보 {geminiEmotionCandidate.turns.length}개</h3>
                <p>
                  이 후보를 만들 때 녹음이 Google Gemini로 전송됐어. KCSC 기준 시간·화자·전사는
                  그대로 두고 감정·확신도·근거만 덧씌운 제안이야. Silver도 Gold도 아니며 자동
                  승격되지 않아. 적용 전 현재 작업본을 임시 저장해.
                </p>
                <p className="review-reference-note">
                  모델 {geminiEmotionCandidate.model} · 원격 전송 있음 · 검수 필요 · 판단 불확실{" "}
                  {geminiEmotionCandidate.uncertain_turns}개
                  {geminiEmotionCandidate.mean_confidence !== null
                    ? ` · 평균 확신도 ${geminiEmotionCandidate.mean_confidence.toFixed(2)}`
                    : ""}{" "}
                  ·{" "}
                  {Object.entries(geminiEmotionCandidate.emotion_histogram)
                    .filter(([, count]) => count > 0)
                    .map(([label, count]) => `${label} ${count}`)
                    .join(" · ")}
                </p>
              </div>
              <button
                type="button"
                className="button ghost"
                onClick={() => applyGeminiEmotionCandidate(geminiEmotionCandidate)}
                disabled={promoting}
              >
                Gemini 원격 감정 후보를 작업본에 적용
              </button>
              {geminiEmotionNotice && (
                <p className="review-alignment-note">{geminiEmotionNotice}</p>
              )}
            </div>
          )}
          {referenceCandidate && !goldExists && (
            <div className="notice" role="status" aria-label="KCSC 기준 재분할 후보">
              <div>
                <p className="eyebrow">KCSC source reference</p>
                <h3>전사·시간 결합 재분할 후보 {referenceCandidate.reference_turn_count}개</h3>
                <p>
                  Gemini Silver의 발화-전사 매핑이 흐트러진 경우를 위한 사람 기준 후보야.
                  화자·전사·시간은 KCSC reference를 쓰고 감정은 모두 판단 불확실로 둬.
                </p>
                <p className="review-reference-note">{referenceCandidate.notes}</p>
              </div>
              <button
                type="button"
                className="button ghost"
                onClick={() => applyReferenceCandidate(referenceCandidate)}
                disabled={promoting}
              >
                KCSC 재분할 후보를 작업본에 적용
              </button>
              {referenceNotice && <p className="review-alignment-note">{referenceNotice}</p>}
            </div>
          )}
          {referenceCandidate && ninthReferenceSubsegments.length >= 2 && !goldExists && (
            <div className="notice" role="status" aria-label="9번 세밀 재분할 후보">
              <div>
                <p className="eyebrow">KCSC source reference</p>
                <h3>9번 긴 구간 세밀 재분할 후보 {ninthReferenceSubsegments.length}개</h3>
                <p>
                  9번의 긴 발화를 사람 기준의 짧은 교대·겹침 발화로만 나눕니다. 감정 변화는
                  자동 추정하지 않고 모두 판단 불확실로 둡니다.
                </p>
              </div>
              <button
                type="button"
                className="button ghost"
                onClick={applyNinthReferenceSubsegments}
                disabled={promoting}
              >
                9번을 KCSC 기준으로 세분화
              </button>
            </div>
          )}

          <dl className="review-meta">
            <div>
              <dt>대화</dt>
              <dd className="mono">{draft.state.conversation_id}</dd>
            </div>
            <div>
              <dt>상태</dt>
              <dd>{draft.state.review_state === "review_required" ? "검수 필요" : draft.state.review_state}</dd>
            </div>
            <div>
              <dt>초안 생성</dt>
              <dd className="mono">{draft.state.created_at}</dd>
            </div>
            <div>
              <dt>모델</dt>
              <dd className="mono">{draft.state.model}</dd>
            </div>
            <div>
              <dt>발화 수</dt>
              <dd>{draft.state.turn_count}</dd>
            </div>
            <div>
              <dt>판단 불확실</dt>
              <dd>{draft.state.uncertain_turns}</dd>
            </div>
          </dl>

          <section className="review-quality" aria-labelledby="review-quality-heading">
            <div>
              <p className="eyebrow">REVIEW SIGNALS</p>
              <h3 id="review-quality-heading">신뢰도와 확인 우선순위</h3>
              <p>
                자동 판정이 아니라, 현재 작업본에서 먼저 들어볼 구간을 정하는 신호입니다.
              </p>
            </div>
            <ul>
              <li>
                <strong>감정·확신도</strong>
                <span>
                  판단 불확실 또는 낮은 확신도 {quality.uncertainPositions.length}개
                  {quality.meanConfidence === null ? "" : ` · 평균 ${quality.meanConfidence.toFixed(2)}`}
                </span>
              </li>
              <li>
                <strong>화자·경계</strong>
                <span>겹침 {quality.overlapPositions.length}개 · 1초 미만 {quality.shortPositions.length}개</span>
              </li>
              <li>
                <strong>우선 검수</strong>
                <span>음성과 먼저 대조할 발화 {quality.priorityPositions.length}개</span>
              </li>
            </ul>
          </section>

          {draft.notes && (
            <p className="review-notes">
              <span className="field-label">모델이 먼저 확인해 달라고 적은 것</span>
              {draft.notes}
            </p>
          )}

          {audioLoading && (
            <p className="review-status" aria-busy="true">
              <Loader2 aria-hidden="true" size={16} className="spin" /> 원본 음성을 확인하는
              중입니다.
            </p>
          )}

          {audioError && (
            <div className="notice" role="status">
              <AudioLines aria-hidden="true" size={18} />
              <div>
                <strong>{audioError.title}</strong>
                <p>{audioError.detail} 음성 없이 글로만 검수할 수 있습니다.</p>
              </div>
            </div>
          )}

          {audio && (
            <section className="review-gaps" aria-labelledby="review-gaps-heading">
              <details>
                <summary id="review-gaps-heading">
                  <AudioLines aria-hidden="true" size={16} />
                  <span>초안에 포함되지 않은 음성 구간</span>
                  <span className="review-gap-count">{audio.gaps.length}곳</span>
                  <span className="review-gap-disclosure">펼쳐서 듣기</span>
                </summary>
                <div className="review-gaps-body">
                  <p className="review-gap-note">
                    초안의 발화가 덮지 않는 구간입니다. 모델이 검증에서 탈락시킨 발화의 정확한
                    위치는 초안에 남아 있지 않으므로, 이 목록은 &ldquo;탈락한 발화&rdquo;가 아니라
                    &ldquo;초안이 아무 말도 하지 않는 구간&rdquo;입니다. 말이 있는지 직접 들어
                    확인해 주세요. {audio.min_gap_seconds}초 미만은 표시하지 않습니다.
                  </p>
                  {audio.gaps.length === 0 ? (
                    <p className="review-gap-empty">
                      {audio.min_gap_seconds}초 이상 비어 있는 구간은 없습니다.
                    </p>
                  ) : (
                    <ul className="review-gap-list">
                      {audio.gaps.map((gap, position) => (
                        <li key={`${gap.start}-${gap.end}`}>
                          <span className="mono">
                            {clockTime(gap.start)} - {clockTime(gap.end)}
                          </span>
                          <span className="review-gap-duration">
                            {(gap.end - gap.start).toFixed(1)}초
                          </span>
                          <ClipButton
                            clip={{ kind: "gap", index: position }}
                            start={gap.start}
                            end={Math.min(gap.end, gap.start + audio.max_clip_seconds)}
                            playback={playback}
                            label="이 구간 듣기"
                            accessibleLabel={`포함되지 않은 구간 ${clockTime(
                              gap.start,
                            )}부터 ${clockTime(gap.end)}까지 듣기`}
                          />
                        </li>
                      ))}
                    </ul>
                  )}
                  {playback.error && playback.errorClip?.kind === "gap" && (
                    <p className="clip-error" role="alert">
                      {playback.error.title}. {playback.error.detail}
                    </p>
                  )}
                </div>
              </details>
            </section>
          )}

          {quality.priorityPositions.length > 0 && (
            <section className="review-priority" aria-labelledby="review-priority-heading">
              <div className="review-section-head">
                <div>
                  <p className="eyebrow">LISTEN FIRST</p>
                  <h3 id="review-priority-heading">확인 우선 발화 {quality.priorityPositions.length}개</h3>
                </div>
                <p>낮은 확신도·짧은 발화·겹침 신호가 있는 구간입니다.</p>
              </div>
              <ul className="turn-list">
                {quality.priorityPositions.map((position) => {
                  const turn = turns[position];
                  if (!turn) return null;
                  const hints = [
                    quality.uncertainPositions.includes(position) ? "낮은 확신도" : null,
                    quality.shortPositions.includes(position) ? "짧은 발화" : null,
                    quality.overlapPositions.includes(position) ? "겹침" : null,
                  ].filter((hint): hint is string => Boolean(hint));
                  return (
                    <TurnEditor
                      key={position}
                      index={position}
                      turn={turn}
                      original={originals[position] ?? turn}
                      emotions={emotions}
                      issues={issues.filter((issue) => issue.index === position)}
                      disabled={promoting || goldExists}
                      playback={audio ? playback : null}
                      clip={currentTurnClip(turn)}
                      priorityHints={hints}
                      changeReason={changeReasons[String(position)] ?? ""}
                      onChange={editTurn}
                      onReasonChange={editChangeReason}
                    />
                  );
                })}
              </ul>
            </section>
          )}

          <section className="review-all-turns" aria-labelledby="review-all-turns-heading">
            <div className="review-section-head">
              <div>
                <p className="eyebrow">FULL TIMELINE</p>
                <h3 id="review-all-turns-heading">
                  {quality.priorityPositions.length > 0 ? "나머지 발화" : "전체 발화"}
                </h3>
              </div>
              <p>시간순으로 유지된 원래 검수 흐름입니다.</p>
            </div>
            <ul className="turn-list">
              {turns.map((turn, position) => {
                if (prioritySet.has(position)) return null;
                return (
                  <TurnEditor
                    key={position}
                    index={position}
                    turn={turn}
                    original={originals[position] ?? turn}
                    emotions={emotions}
                    issues={issues.filter((issue) => issue.index === position)}
                    disabled={promoting || goldExists}
                    playback={audio ? playback : null}
                    clip={currentTurnClip(turn)}
                    priorityHints={[]}
                    changeReason={changeReasons[String(position)] ?? ""}
                    onChange={editTurn}
                    onReasonChange={editChangeReason}
                  />
                );
              })}
            </ul>
          </section>

          <div className="review-signoff">
            <div className="turn-field wide">
              <label className="field-label" htmlFor="review-reviewer">
                검수자 <em>필수</em>
              </label>
              <input
                id="review-reviewer"
                type="text"
                value={reviewer}
                maxLength={MAX_REVIEWER_CHARACTERS}
                disabled={promoting || goldExists}
                placeholder="확정본에 남길 본인 식별자"
                onChange={(event) => setReviewer(event.target.value)}
              />
              {attempted && !namedReviewer && (
                <p className="field-error">확정본에 남길 검수자 식별자를 입력해 주세요.</p>
              )}
            </div>

            <div className="turn-field wide">
              <label className="field-label" htmlFor="review-note">
                검수 메모
              </label>
              <textarea
                id="review-note"
                rows={2}
                value={reviewNote}
                maxLength={MAX_REVIEW_NOTE_CHARACTERS}
                disabled={promoting || goldExists}
                placeholder="무엇을 어떻게 확인했는지 남겨 주세요."
                onChange={(event) => setReviewNote(event.target.value)}
              />
            </div>

            <label className="consent-ack review-ack">
              <input
                type="checkbox"
                checked={acknowledged}
                disabled={promoting || goldExists}
                onChange={(event) => setAcknowledged(event.target.checked)}
              />
              <span>
                초안 전체를 음성과 대조해 확인했으며, 이 내용을 제 판단으로 확정본에 올립니다.
                수정한 발화는 {changed}개입니다.
              </span>
            </label>
            {attempted && !acknowledged && (
              <p className="field-error">확인했다는 표시를 체크해 주세요.</p>
            )}
            {attempted && issues.length > 0 && (
              <p className="field-error">
                아직 올릴 수 없는 항목이 {issues.length}개 있습니다. 위에서 표시된 곳을 고쳐
                주세요.
              </p>
            )}
            {attempted && missingReasonPositions.length > 0 && (
              <p className="field-error">
                수정한 발화 {missingReasonPositions.length}개에 사유를 선택해 주세요. 사유는 Gold 확정본의
                검수 이력에만 기록됩니다.
              </p>
            )}

            {promotionError && (
              <div className="notice alert" role="alert">
                <TriangleAlert aria-hidden="true" size={18} />
                <div>
                  <strong>{promotionError.title}</strong>
                  <p>{promotionError.detail}</p>
                </div>
              </div>
            )}

            <div className="review-checkpoint" role="status">
              <div>
                <strong>검수 임시 저장</strong>
                <p>
                  {checkpointSavedAt
                    ? `이 브라우저에 ${new Date(checkpointSavedAt).toLocaleString("ko-KR")} 저장됨`
                    : "중간 수정은 아직 저장되지 않았습니다."}
                </p>
              </div>
              <div className="review-checkpoint-actions">
                <button
                  className="button ghost"
                  type="button"
                  disabled={promoting || goldExists}
                  onClick={saveWorkingCopy}
                >
                  <Save aria-hidden="true" size={16} /> 임시 저장
                </button>
                {checkpointSavedAt && (
                  <button
                    className="button ghost"
                    type="button"
                    disabled={promoting || goldExists}
                    onClick={discardWorkingCopy}
                  >
                    <Trash2 aria-hidden="true" size={16} /> 저장본 삭제
                  </button>
                )}
              </div>
            </div>
            {checkpointError && <p className="field-error">{checkpointError}</p>}

            <button
              className="button primary"
              type="button"
              disabled={promoting || goldExists}
              onClick={() => void promote()}
            >
              <ShieldCheck aria-hidden="true" size={16} />{" "}
              {promoting ? "확정본으로 기록하는 중" : "검수 완료로 확정"}
            </button>
            <p className="review-footnote">
              확정본은 한 번만 기록되며 이후에는 바꿀 수 없습니다. Silver 초안은 이 작업으로
              바뀌지 않습니다. 임시 저장본은 이 브라우저에만 보관되며, 확정본이 아닙니다.
            </p>
          </div>
        </div>
      )}
    </section>
  );
}

function summaryLine(state: AnnotationReviewState): string {
  const confidence =
    state.mean_confidence === null ? "확신도 없음" : `평균 확신도 ${state.mean_confidence.toFixed(2)}`;
  return `발화 ${state.turn_count}개 / 화자 ${state.speaker_count}명 / 판단 불확실 ${state.uncertain_turns}개 / ${confidence}`;
}
