"""Shared validation and deterministic speaker alignment for local ASR providers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from threading import Lock

from voxdelta.domain.models import AudioAsset, SpeakerSegment, Utterance
from voxdelta.providers.base import ProviderError

LOCAL_ASR_INFERENCE_LOCK = Lock()


@dataclass(frozen=True, slots=True, order=True)
class AlignedWord:
    start: float
    end: float
    text: str


def validate_asset(asset: AudioAsset) -> tuple[float, tuple[str, ...]]:
    duration = asset.duration_seconds
    if duration is None or not isfinite(duration) or duration <= 0:
        raise ProviderError("invalid_audio_asset")
    paths = asset.normalized_paths
    expected = 1 if asset.channel_mode == "mixed" else 2 if asset.channel_mode == "separate" else 0
    if len(paths) != expected or any(not path for path in paths):
        raise ProviderError("invalid_audio_asset")
    return duration, paths


def _number(value: object) -> float:
    if isinstance(value, bool):
        raise ProviderError("invalid_provider_output")
    try:
        converted = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        raise ProviderError("invalid_provider_output") from None
    if not isfinite(converted):
        raise ProviderError("invalid_provider_output")
    return converted


def _validated(
    records: Iterable[tuple[object, object, object]],
    duration: float,
    *,
    allow_points: bool,
) -> list[AlignedWord]:
    """Validate recognizer word records, optionally admitting zero-duration instants.

    ``allow_points`` is opt-in per provider and relaxes exactly two rules, both only for
    spans where ``end == start``: such a span is accepted at all, and two of them may
    share an instant. An aligner that cannot separate two adjacent tokens legitimately
    reports both at the same moment, and rejecting that would discard real text. Nothing
    else is loosened — ``end < start`` still fails closed, because a span running
    backwards is a broken aligner rather than a point event, and ordering is still
    required so a caller can rely on word order without re-sorting.
    """

    words: list[AlignedWord] = []
    seen_spans: set[tuple[float, float]] = set()
    previous: AlignedWord | None = None
    try:
        for raw_start, raw_end, raw_text in records:
            start = _number(raw_start)
            end = _number(raw_end)
            point = allow_points and end == start
            if start < 0 or end > duration or (end <= start and not point):
                raise ProviderError("invalid_provider_output")
            if not isinstance(raw_text, str):
                raise ProviderError("invalid_provider_output")
            text = " ".join(raw_text.split())
            if not text:
                raise ProviderError("invalid_provider_output")
            word = AlignedWord(start=start, end=end, text=text)
            span = (start, end)
            if span in seen_spans and not point:
                raise ProviderError("invalid_provider_output")
            if previous is not None and (start, end) < (previous.start, previous.end):
                raise ProviderError("invalid_provider_output")
            if previous is not None and not point and word < previous:
                raise ProviderError("invalid_provider_output")
            seen_spans.add(span)
            words.append(word)
            previous = word
    except ProviderError:
        raise
    except TimeoutError:
        raise ProviderError("provider_timeout") from None
    except Exception:
        raise ProviderError("invalid_provider_output") from None
    return words


def validated_words(
    records: Iterable[tuple[object, object, object]], duration: float
) -> list[AlignedWord]:
    """Validate word records, refusing any span without positive duration."""

    return _validated(records, duration, allow_points=False)


def validated_point_words(
    records: Iterable[tuple[object, object, object]], duration: float
) -> list[AlignedWord]:
    """Validate word records from an aligner that also reports zero-duration instants.

    Used only by providers whose aligner is known to emit them. The instants survive
    validation here; whether any of them may be attributed to a speaker is decided later,
    by the alignment, under a rule that cannot cross a diarization boundary.
    """

    return _validated(records, duration, allow_points=True)


def point_segment_index(instant: float, timeline: list[SpeakerSegment]) -> int | None:
    """The single segment strictly containing ``instant``, or None when that is ambiguous.

    Strict containment is deliberate. A point sitting exactly on a segment edge belongs
    equally to that segment and to whatever abuts it, so attributing it would be a guess
    that can cross a speech or speaker boundary. The same applies when overlapping
    segments both contain it. In either case the caller omits the word rather than
    picking, because a misattributed word is worse than a reported gap.
    """

    matches = [
        index for index, segment in enumerate(timeline) if segment.start < instant < segment.end
    ]
    return matches[0] if len(matches) == 1 else None


def _validated_segments(segments: list[SpeakerSegment], duration: float) -> list[SpeakerSegment]:
    if not segments:
        raise ProviderError("invalid_provider_output")
    ordered = sorted(segments, key=lambda item: (item.start, item.end, item.speaker_id))
    unique = {(segment.start, segment.end, segment.speaker_id) for segment in ordered}
    if ordered != segments or len(unique) != len(ordered):
        raise ProviderError("invalid_provider_output")
    if any(
        not isfinite(segment.start)
        or not isfinite(segment.end)
        or segment.start < 0
        or segment.end > duration
        for segment in ordered
    ):
        raise ProviderError("invalid_provider_output")
    if len({segment.speaker_id for segment in ordered}) != 2:
        raise ProviderError("unsupported_speaker_count")
    return ordered


def validate_mixed_segments(segments: list[SpeakerSegment], duration: float) -> None:
    """Fail closed on a non-canonical mixed-audio alignment timeline."""

    _validated_segments(segments, duration)


def _grouped(assignments: list[tuple[AlignedWord, str]]) -> list[tuple[str, list[AlignedWord]]]:
    """Maximal runs of consecutive words assigned to the same speaker, in word order."""

    groups: list[tuple[str, list[AlignedWord]]] = []
    for word, speaker_id in assignments:
        if groups and groups[-1][0] == speaker_id:
            groups[-1][1].append(word)
        else:
            groups.append((speaker_id, [word]))
    return groups


def _has_extent(words: list[AlignedWord]) -> bool:
    """Whether a group contains a word with real duration, not only instants."""

    return any(word.end > word.start for word in words)


def _utterances(assignments: list[tuple[AlignedWord, str]]) -> tuple[list[Utterance], int]:
    """Build one utterance per speaker run, dropping runs made only of instants.

    An utterance is required to have positive extent, so a run of zero-duration words
    alone cannot become one. Widening it to manufacture a span would be inventing timing
    the aligner never reported, so those words are omitted and counted instead.
    """

    utterances: list[Utterance] = []
    omitted = 0
    for speaker_id, words in _grouped(assignments):
        if not _has_extent(words):
            omitted += len(words)
            continue
        utterances.append(
            Utterance(
                id=f"utt-{len(utterances) + 1:04d}",
                start=min(word.start for word in words),
                end=max(word.end for word in words),
                speaker_id=speaker_id,
                confidence=1.0,
                transcript=" ".join(word.text for word in words),
            )
        )
    return utterances, omitted


def _turns(
    assignments: list[tuple[AlignedWord, int]], timeline: list[SpeakerSegment]
) -> list[tuple[int, int, list[AlignedWord]]]:
    """Word runs that stay inside one speaker's uninterrupted stretch of the timeline.

    Consecutive words held by the same speaker belong to one utterance, but only while
    the timeline between their segments belongs to that speaker too. Merging across the
    other speaker's turn would produce an utterance covering time it does not own, and
    the pipeline rejects the stage when one does.
    """

    turns: list[tuple[int, int, list[AlignedWord]]] = []
    for word, index in assignments:
        if turns:
            first, last, words = turns[-1]
            low, high = min(last, index), max(last, index)
            if all(
                timeline[position].speaker_id == timeline[index].speaker_id
                for position in range(low, high + 1)
            ):
                turns[-1] = (min(first, index), max(last, index), [*words, word])
                continue
        turns.append((index, index, [word]))
    return turns


def _turn_bounds(
    timeline: list[SpeakerSegment], first: int, last: int, duration: float
) -> tuple[float, float]:
    """The interval a turn may occupy, bounded by the other speaker's nearest segments.

    Word timing is the recognizer's estimate and a word straddling a turn boundary is
    still assigned whole to whichever segment owns most of it, so the raw hull of a
    turn's words can reach across that boundary. Clamping to the neighbouring foreign
    segments removes only that reach; timing inside the turn's own gap is left alone.
    """

    speaker_id = timeline[first].speaker_id
    lower = max(
        (
            segment.end
            for segment in timeline
            if segment.speaker_id != speaker_id and segment.end <= timeline[first].start
        ),
        default=0.0,
    )
    upper = min(
        (
            segment.start
            for segment in timeline
            if segment.speaker_id != speaker_id and segment.start >= timeline[last].end
        ),
        default=duration,
    )
    return lower, upper


def align_mixed(
    words: list[AlignedWord], segments: list[SpeakerSegment], duration: float
) -> tuple[list[Utterance], int]:
    timeline = _validated_segments(segments, duration)
    assignments: list[tuple[AlignedWord, int]] = []
    omitted = 0
    for word in words:
        if word.end <= word.start:
            # A zero-duration instant overlaps nothing, so it is placed by containment
            # instead. Ambiguous instants are omitted rather than guessed at.
            index = point_segment_index(word.start, timeline)
            if index is None:
                omitted += 1
                continue
            assignments.append((word, index))
            continue
        candidates = [
            (max(0.0, min(word.end, segment.end) - max(word.start, segment.start)), index, segment)
            for index, segment in enumerate(timeline)
        ]
        positive = [item for item in candidates if item[0] > 0]
        if not positive:
            omitted += 1
            continue
        _overlap, index, _selected = min(
            positive,
            key=lambda item: (-item[0], item[2].start, item[2].end, item[2].speaker_id),
        )
        assignments.append((word, index))
    utterances: list[Utterance] = []
    for first, last, group in _turns(assignments, timeline):
        if not _has_extent(group):
            # Instants that landed in a turn of their own: attributable to a speaker, but
            # with no span to occupy. Omitted rather than widened into invented timing.
            omitted += len(group)
            continue
        lower, upper = _turn_bounds(timeline, first, last, duration)
        start = max(min(word.start for word in group), lower)
        end = min(max(word.end for word in group), upper)
        if end <= start:
            # Unreachable while segments do not overlap. Refuse rather than emit a
            # degenerate span: a silently repaired one would be a fabricated turn.
            raise ProviderError("invalid_provider_output")
        utterances.append(
            Utterance(
                id=f"utt-{len(utterances) + 1:04d}",
                start=start,
                end=end,
                speaker_id=timeline[first].speaker_id,
                confidence=1.0,
                transcript=" ".join(word.text for word in group),
            )
        )
    return utterances, omitted


def align_separate(channel_words: list[list[AlignedWord]]) -> tuple[list[Utterance], int]:
    assignments = sorted(
        (
            (word, f"SPEAKER_{channel_index:02d}")
            for channel_index, words in enumerate(channel_words)
            for word in words
        ),
        key=lambda item: (item[0].start, item[0].end, item[1], item[0].text),
    )
    return _utterances(assignments)
