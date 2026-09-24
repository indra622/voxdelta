"""Serving a private silver draft to a reviewer, and the one gated way it becomes gold.

:mod:`voxdelta.annotation.store` already makes the silver-to-gold crossing an operation
with preconditions rather than a status field. This module is what a user interface is
allowed to call, and it adds the two things a transport boundary needs on top of that:

* **Nothing here writes silver.** A draft is read, edited in the reviewer's browser, and
  submitted back as a new artifact. :func:`promote_reviewed` re-reads the silver file after
  the gold write and compares its digest against the one it read before, so "silver was not
  touched" is a checked postcondition rather than a claim about what the code does.
* **No filesystem path leaves this module.** Conversations are addressed by an id matched
  against :data:`CONVERSATION_ID`, never by a caller-supplied path, and every refusal is a
  fixed code with sanitized English text. ``AnnotationError`` from the store carries the
  path it refused, so it is caught and re-raised as a code rather than forwarded.

Drafts carry verbatim transcript. The listing does not: it is counts and digests only, so a
reviewer choosing what to open has not yet been shown anything a log must not hold.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voxdelta.annotation.audio import (
    MAX_CLIP_SECONDS,
    MIN_GAP_SECONDS,
    ClipPlan,
    ReviewGap,
    plan_clip,
    resolve_source,
    review_gaps,
    source_sha256,
)
from voxdelta.annotation.gemini_silver import EMOTION_LABELS, SilverTurn
from voxdelta.annotation.store import (
    REVIEW_REQUIRED,
    AnnotationError,
    promote,
    read_silver,
    summarise,
)

#: Conversation ids name a directory, so the accepted set excludes every character that
#: could make one mean something other than a single name under the annotation root.
CONVERSATION_ID = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")

MAX_TURNS = 5_000
MAX_SPEAKER_CHARACTERS = 64
MAX_TRANSCRIPT_CHARACTERS = 4_000
MAX_RATIONALE_CHARACTERS = 1_000
MAX_REVIEWER_CHARACTERS = 128
MAX_REVIEW_NOTE_CHARACTERS = 2_000
REVIEW_CHANGE_REASONS = frozenset(
    {
        "speaker_mismatch",
        "transcript_mismatch",
        "timing_mismatch",
        "emotion_mismatch",
        "model_candidate",
        "other",
    }
)


class ReviewRejected(Exception):
    """A refusal a client can act on: a stable code and text that names no path."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class ReviewState:
    """What a reviewer is being asked to look at, in counts and digests only."""

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


@dataclass(frozen=True, slots=True)
class ReviewWarning:
    """One thing the reviewer has to know before trusting anything on the screen."""

    code: str
    detail: str
    count: int | None = None
    by_rule: Mapping[str, int] | None = None


@dataclass(frozen=True, slots=True)
class SilverDraft:
    """The full reviewable draft, transcript included. Never logged, never summarised out."""

    state: ReviewState
    speakers: tuple[str, ...]
    notes: str
    turns: tuple[SilverTurn, ...]
    warnings: tuple[ReviewWarning, ...]


@dataclass(frozen=True, slots=True)
class GoldPromotion:
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


@dataclass(frozen=True, slots=True)
class ReviewIndex:
    """Every draft this machine holds, plus how many could not be read at all."""

    annotations: tuple[ReviewState, ...]
    unreadable_count: int


def _rejected(code: str, message: str) -> ReviewRejected:
    return ReviewRejected(code, message)


def _conversation_directory(root: Path, conversation_id: str) -> Path:
    if not CONVERSATION_ID.fullmatch(conversation_id):
        raise _rejected(
            "invalid_conversation_id",
            "The requested annotation identifier is not a valid conversation id.",
        )
    return root / conversation_id


def _silver_record(directory: Path) -> dict[str, Any]:
    path = directory / "silver.json"
    if not path.is_file():
        raise _rejected(
            "annotation_not_found",
            "No silver annotation is available for that conversation.",
        )
    try:
        record = read_silver(path)
    except (AnnotationError, OSError, json.JSONDecodeError, UnicodeDecodeError):
        # The store names the offending file; a client never learns where it lives.
        raise _rejected(
            "annotation_unreadable",
            "The stored silver annotation could not be read as a silver artifact.",
        ) from None
    return record


def _text(record: Mapping[str, Any], *keys: str, default: str = "") -> str:
    current: Any = record
    for key in keys:
        if not isinstance(current, Mapping):
            return default
        current = current.get(key)
    return current if isinstance(current, str) else default


def _stored_turns(record: Mapping[str, Any]) -> tuple[SilverTurn, ...]:
    content = record.get("content")
    rows = content.get("turns") if isinstance(content, Mapping) else None
    if not isinstance(rows, list) or not rows:
        raise _rejected(
            "annotation_unreadable",
            "The stored silver annotation holds no reviewable turns.",
        )
    turns: list[SilverTurn] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise _rejected(
                "annotation_unreadable",
                "The stored silver annotation holds a turn that is not an object.",
            )
        try:
            turns.append(
                SilverTurn(
                    start=float(row["start"]),
                    end=float(row["end"]),
                    speaker=str(row["speaker"]),
                    transcript=str(row["transcript"]),
                    emotion=str(row["emotion"]),
                    emotion_rationale=str(row["emotion_rationale"]),
                    confidence=float(row["confidence"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            raise _rejected(
                "annotation_unreadable",
                "The stored silver annotation holds a turn with missing or invalid fields.",
            ) from None
    return tuple(turns)


def _emotion_candidate_count(
    *, root: Path, conversation_id: str, source_silver_content_sha256: str
) -> int:
    """Count usable optional overlays without making one unreadable overlay hide Silver."""

    from voxdelta.annotation.emotion_candidates import (
        EmotionCandidateError,
        load_local_candidate,
    )
    from voxdelta.annotation.gemini_emotion_overlay import (
        OverlayCandidateError,
    )
    from voxdelta.annotation.gemini_emotion_overlay import load_candidate as load_gemini_candidate

    count = 0
    for loader, error_type in (
        (load_local_candidate, EmotionCandidateError),
        (load_gemini_candidate, OverlayCandidateError),
    ):
        try:
            if (
                loader(
                    annotation_root=root,
                    conversation_id=conversation_id,
                    source_silver_content_sha256=source_silver_content_sha256,
                )
                is not None
            ):
                count += 1
        except error_type:
            continue
    return count


def _is_listed_for_review(record: Mapping[str, Any]) -> bool:
    """Respect an explicit archival flag without making unlisted drafts disappear.

    A source recording can be retained for provenance after it has been split into
    review cases.  Its original Silver remains addressable by id, but it must not
    compete with the derived cases in the review queue.  Missing flags keep the
    historical default: list the draft.
    """

    return record.get("review_listing_visible") is not False


def _state(
    record: Mapping[str, Any],
    *,
    conversation_id: str,
    gold_present: bool,
    emotion_candidate_count: int = 0,
) -> ReviewState:
    turns = _stored_turns(record)
    summary = summarise(turns)
    mean = summary["mean_confidence"]
    remote_file = record.get("remote_file")
    return ReviewState(
        conversation_id=conversation_id,
        review_state=_text(record, "review_state", default=REVIEW_REQUIRED),
        promotable=bool(record.get("promotable", False)),
        created_at=_text(record, "created_at"),
        model=_text(record, "source", "model"),
        input_sha256=_text(record, "source", "input_sha256"),
        content_sha256=_text(record, "content_sha256"),
        remote_file_deleted=bool(
            remote_file.get("deleted", False) if isinstance(remote_file, Mapping) else False
        ),
        turn_count=int(summary["turn_count"]),
        speaker_count=int(summary["speaker_count"]),
        uncertain_turns=int(summary["uncertain_turns"]),
        mean_confidence=float(mean) if isinstance(mean, (int, float)) else None,
        gold_present=gold_present,
        emotion_candidate_count=emotion_candidate_count,
    )


def _warnings(record: Mapping[str, Any], state: ReviewState) -> tuple[ReviewWarning, ...]:
    """Everything that makes this draft provisional, stated rather than implied."""

    warnings = [
        ReviewWarning(
            code="review_required",
            detail=(
                "This draft is a model's provisional guess. Nothing here is ground truth "
                "until a reviewer checks it against the audio and promotes it."
            ),
        )
    ]

    validation = record.get("validation")
    dropped = 0
    by_rule: dict[str, int] = {}
    if isinstance(validation, Mapping):
        raw_dropped = validation.get("dropped_turn_count")
        dropped = int(raw_dropped) if isinstance(raw_dropped, int) else 0
        raw_rules = validation.get("dropped_turns_by_rule")
        if isinstance(raw_rules, Mapping):
            by_rule = {
                str(rule): int(count) for rule, count in raw_rules.items() if isinstance(count, int)
            }
    if dropped:
        warnings.append(
            ReviewWarning(
                code="salvaged_dropped_turns",
                detail=(
                    "Turns the model returned were rejected by validation and are absent "
                    "from this draft. They were never recovered, so any speech they held "
                    "is missing here and has to be judged against the audio."
                ),
                count=dropped,
                by_rule=by_rule,
            )
        )

    source = record.get("source")
    if (
        isinstance(source, Mapping)
        and source.get("remote_audio_transmitted")
        and not state.remote_file_deleted
    ):
        warnings.append(
            ReviewWarning(
                code="remote_file_not_deleted",
                detail=(
                    "The uploaded audio file could not be confirmed as deleted from the "
                    "remote annotation service."
                ),
            )
        )

    if isinstance(source, Mapping) and source.get("remote_audio_transmitted"):
        warnings.append(
            ReviewWarning(
                code="remote_audio_transmitted",
                detail="The audio behind this draft was sent to a remote annotation service.",
            )
        )

    if state.gold_present:
        warnings.append(
            ReviewWarning(
                code="gold_already_exists",
                detail=(
                    "A reviewed gold annotation already exists for this conversation and "
                    "cannot be replaced."
                ),
            )
        )
    return tuple(warnings)


def available(root: Path) -> ReviewIndex:
    """List every draft under the root, in counts only, without opening any transcript."""

    if not root.is_dir():
        return ReviewIndex(annotations=(), unreadable_count=0)
    states: list[ReviewState] = []
    unreadable = 0
    for directory in sorted(root.iterdir(), key=lambda entry: entry.name):
        if not directory.is_dir() or not CONVERSATION_ID.fullmatch(directory.name):
            continue
        if not (directory / "silver.json").is_file():
            continue
        try:
            record = _silver_record(directory)
            if not _is_listed_for_review(record):
                continue
            states.append(
                _state(
                    record,
                    conversation_id=directory.name,
                    gold_present=(directory / "gold.json").is_file(),
                    emotion_candidate_count=_emotion_candidate_count(
                        root=root,
                        conversation_id=directory.name,
                        source_silver_content_sha256=_text(record, "content_sha256"),
                    ),
                )
            )
        except ReviewRejected:
            # Counted rather than dropped: a draft this machine holds but cannot parse is
            # a fact the reviewer needs, and it is one a silent skip would hide.
            unreadable += 1
    return ReviewIndex(annotations=tuple(states), unreadable_count=unreadable)


def load_draft(root: Path, conversation_id: str) -> SilverDraft:
    """Read one draft in full, transcript included, for a reviewer to correct."""

    directory = _conversation_directory(root, conversation_id)
    record = _silver_record(directory)
    state = _state(
        record,
        conversation_id=conversation_id,
        gold_present=(directory / "gold.json").is_file(),
    )
    content = record.get("content")
    speakers = content.get("speakers") if isinstance(content, Mapping) else None
    return SilverDraft(
        state=state,
        speakers=tuple(str(name) for name in speakers) if isinstance(speakers, list) else (),
        notes=_text(content if isinstance(content, Mapping) else {}, "notes"),
        turns=_stored_turns(record),
        warnings=_warnings(record, state),
    )


@dataclass(frozen=True, slots=True)
class TurnClip:
    """The playable range for one draft turn, clamped to audio that actually exists."""

    position: int
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class AudioOverview:
    """Everything a reviewer's screen needs to offer listening, and no transcript.

    Served separately from the draft rather than folded into it, because the audio may be
    absent on a machine that still holds the draft. A draft that cannot be listened to is
    still reviewable text; making the whole draft fail because a WAV is missing would be
    the wrong trade.
    """

    conversation_id: str
    duration_seconds: float
    sample_rate: int
    max_clip_seconds: float
    min_gap_seconds: float
    turn_clips: tuple[TurnClip, ...]
    gaps: tuple[ReviewGap, ...]


def audio_overview(root: Path, conversation_id: str, *, audio_root: Path) -> AudioOverview:
    """Describe what can be played for one draft: its turns, and the gaps between them."""

    directory = _conversation_directory(root, conversation_id)
    record = _silver_record(directory)
    turns = _stored_turns(record)
    source = resolve_source(
        audio_root,
        conversation_id,
        expected_sha256=source_sha256(record),
    )
    duration = source.duration_seconds
    clips = tuple(
        TurnClip(
            position=index + 1,
            start=round(min(turn.start, duration), 3),
            end=round(min(turn.end, duration), 3),
        )
        for index, turn in enumerate(turns)
        if turn.start < duration
    )
    return AudioOverview(
        conversation_id=conversation_id,
        duration_seconds=round(duration, 3),
        sample_rate=source.frame_rate,
        max_clip_seconds=MAX_CLIP_SECONDS,
        min_gap_seconds=MIN_GAP_SECONDS,
        turn_clips=clips,
        gaps=review_gaps(turns, duration_seconds=duration),
    )


def plan_audio_clip(
    root: Path,
    conversation_id: str,
    *,
    audio_root: Path,
    start: float,
    end: float,
) -> ClipPlan:
    """Validate one playback request against the draft's own recording."""

    directory = _conversation_directory(root, conversation_id)
    record = _silver_record(directory)
    source = resolve_source(
        audio_root,
        conversation_id,
        expected_sha256=source_sha256(record),
    )
    return plan_clip(source, start=start, end=end)


def _finite(value: float) -> bool:
    return math.isfinite(value)


def validate_corrections(rows: Sequence[Mapping[str, Any]]) -> tuple[SilverTurn, ...]:
    """Turn a reviewer's submission into turns, refusing anything unusable as gold.

    Rejections name the turn's position so the reviewer can find it. They never quote the
    turn, because the offending value is transcript as often as it is a label.
    """

    if not rows:
        raise _rejected(
            "corrected_turns_required",
            "Promotion requires the reviewer's corrected annotation, which was empty.",
        )
    if len(rows) > MAX_TURNS:
        raise _rejected(
            "too_many_turns",
            f"A reviewed annotation may hold at most {MAX_TURNS} turns.",
        )

    turns: list[SilverTurn] = []
    for index, row in enumerate(rows):
        position = index + 1
        speaker = str(row.get("speaker", ""))
        transcript = str(row.get("transcript", ""))
        emotion = str(row.get("emotion", ""))
        rationale = str(row.get("emotion_rationale", ""))
        try:
            start = float(row["start"])
            end = float(row["end"])
            confidence = float(row["confidence"])
        except (KeyError, TypeError, ValueError):
            raise _rejected(
                "invalid_turn_interval",
                f"Turn {position} is missing a numeric start, end, or confidence.",
            ) from None

        if not speaker.strip() or len(speaker) > MAX_SPEAKER_CHARACTERS:
            raise _rejected(
                "invalid_turn_speaker",
                f"Turn {position} needs a speaker label of 1 to "
                f"{MAX_SPEAKER_CHARACTERS} characters.",
            )
        if not transcript.strip() or len(transcript) > MAX_TRANSCRIPT_CHARACTERS:
            raise _rejected(
                "invalid_turn_transcript",
                f"Turn {position} needs a transcript of 1 to "
                f"{MAX_TRANSCRIPT_CHARACTERS} characters.",
            )
        if len(rationale) > MAX_RATIONALE_CHARACTERS:
            raise _rejected(
                "invalid_turn_rationale",
                f"Turn {position} has a rationale longer than "
                f"{MAX_RATIONALE_CHARACTERS} characters.",
            )
        if emotion not in EMOTION_LABELS:
            raise _rejected(
                "invalid_turn_emotion",
                f"Turn {position} carries an emotion label outside the allowed set.",
            )
        if not _finite(start) or not _finite(end) or start < 0 or end <= start:
            raise _rejected(
                "invalid_turn_interval",
                f"Turn {position} needs a finite time range that starts at or after zero "
                "and ends after it starts.",
            )
        if not _finite(confidence) or not 0.0 <= confidence <= 1.0:
            raise _rejected(
                "invalid_turn_confidence",
                f"Turn {position} needs a confidence between 0 and 1.",
            )
        turns.append(
            SilverTurn(
                start=start,
                end=end,
                speaker=speaker.strip(),
                transcript=transcript,
                emotion=emotion,
                emotion_rationale=rationale,
                confidence=confidence,
            )
        )
    return tuple(turns)


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_change_reasons(
    value: Sequence[Mapping[str, Any]],
    *,
    turn_count: int,
) -> tuple[dict[str, Any], ...]:
    """Keep the audit trail structural and transcript-free.

    Positions are one-based because they are displayed that way to a reviewer. The endpoint
    deliberately records only a short controlled label, not an extra copy of the edited
    transcript or a free-form per-turn comment.
    """

    reasons: list[dict[str, Any]] = []
    positions: set[int] = set()
    for row in value:
        position = row.get("position")
        reason = row.get("reason")
        if (
            not isinstance(position, int)
            or isinstance(position, bool)
            or not 1 <= position <= turn_count
        ):
            raise _rejected(
                "invalid_review_change_reason", "A change-reason position is not a valid turn."
            )
        if position in positions:
            raise _rejected(
                "invalid_review_change_reason", "A turn may have only one change reason."
            )
        if not isinstance(reason, str) or reason not in REVIEW_CHANGE_REASONS:
            raise _rejected(
                "invalid_review_change_reason",
                "A change reason is not one of the supported labels.",
            )
        positions.add(position)
        reasons.append({"position": position, "reason": reason})
    return tuple(sorted(reasons, key=lambda row: int(row["position"])))


def promote_reviewed(
    root: Path,
    conversation_id: str,
    *,
    reviewer: str,
    acknowledged: bool,
    corrected_turns: Sequence[Mapping[str, Any]],
    review_note: str = "",
    change_reasons: Sequence[Mapping[str, Any]] = (),
) -> GoldPromotion:
    """Freeze one reviewer's corrected draft as gold, or refuse and write nothing.

    The acknowledgement is required even when the reviewer changed every field. Edits alone
    would let a client that echoed the draft back with one character changed produce gold
    without anyone claiming to have listened, which is exactly the thing gold is supposed
    to mean.
    """

    directory = _conversation_directory(root, conversation_id)
    silver_path = directory / "silver.json"
    record = _silver_record(directory)

    named_reviewer = reviewer.strip()
    if not named_reviewer:
        raise _rejected(
            "reviewer_required",
            "Promotion requires the reviewer's identity, which was blank.",
        )
    if len(named_reviewer) > MAX_REVIEWER_CHARACTERS:
        raise _rejected(
            "reviewer_required",
            f"A reviewer identity may hold at most {MAX_REVIEWER_CHARACTERS} characters.",
        )
    if not acknowledged:
        raise _rejected(
            "review_acknowledgement_required",
            "Promotion requires the reviewer to state that they checked this draft.",
        )
    if len(review_note) > MAX_REVIEW_NOTE_CHARACTERS:
        raise _rejected(
            "review_note_too_long",
            f"A review note may hold at most {MAX_REVIEW_NOTE_CHARACTERS} characters.",
        )

    turns = validate_corrections(corrected_turns)
    normalized_reasons = _validate_change_reasons(change_reasons, turn_count=len(turns))

    gold_path = directory / "gold.json"
    if gold_path.exists():
        raise _rejected(
            "gold_already_exists",
            "A reviewed gold annotation already exists for this conversation and is immutable.",
        )

    silver_digest_before = _file_digest(silver_path)
    try:
        _path, content_digest, summary = promote(
            record,
            corrected_turns=turns,
            reviewer=named_reviewer,
            root=root,
            conversation_id=conversation_id,
            review_note=review_note,
            review_change_reasons=normalized_reasons,
        )
    except AnnotationError:
        # The store names the file it refused, so its text never reaches a client. A gold
        # file that appeared between the check above and the write is the one refusal a
        # caller can act on; anything else is a rejected promotion, not a race.
        if gold_path.exists():
            raise _rejected(
                "gold_already_exists",
                "A reviewed gold annotation already exists for this conversation and is immutable.",
            ) from None
        raise _rejected(
            "promotion_refused",
            "The promotion did not satisfy the gold artifact's preconditions.",
        ) from None

    silver_unmodified = _file_digest(silver_path) == silver_digest_before
    if not silver_unmodified:
        raise _rejected(
            "silver_modified",
            "The silver draft changed while gold was being written; the review is not trusted.",
        )

    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    mean = summary["mean_confidence"]
    return GoldPromotion(
        conversation_id=conversation_id,
        reviewer=named_reviewer,
        reviewed_at=_text(gold, "reviewed_at"),
        review_note=review_note,
        content_sha256=content_digest,
        parent_silver_sha256=_text(gold, "parent_silver_sha256"),
        unchanged_from_silver=bool(gold.get("unchanged_from_silver", False)),
        turn_count=int(summary["turn_count"]),
        speaker_count=int(summary["speaker_count"]),
        uncertain_turns=int(summary["uncertain_turns"]),
        mean_confidence=float(mean) if isinstance(mean, (int, float)) else None,
        change_reason_count=len(normalized_reasons),
        silver_unmodified=silver_unmodified,
    )


__all__ = [
    "CONVERSATION_ID",
    "AudioOverview",
    "TurnClip",
    "audio_overview",
    "plan_audio_clip",
    "MAX_REVIEWER_CHARACTERS",
    "MAX_REVIEW_NOTE_CHARACTERS",
    "MAX_TRANSCRIPT_CHARACTERS",
    "MAX_TURNS",
    "GoldPromotion",
    "ReviewIndex",
    "ReviewRejected",
    "ReviewState",
    "ReviewWarning",
    "SilverDraft",
    "available",
    "load_draft",
    "promote_reviewed",
    "validate_corrections",
]
