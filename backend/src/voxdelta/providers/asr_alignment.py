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


def validated_words(
    records: Iterable[tuple[object, object, object]], duration: float
) -> list[AlignedWord]:
    words: list[AlignedWord] = []
    seen_spans: set[tuple[float, float]] = set()
    previous: AlignedWord | None = None
    try:
        for raw_start, raw_end, raw_text in records:
            start = _number(raw_start)
            end = _number(raw_end)
            if start < 0 or end <= start or end > duration:
                raise ProviderError("invalid_provider_output")
            if not isinstance(raw_text, str):
                raise ProviderError("invalid_provider_output")
            text = " ".join(raw_text.split())
            if not text:
                raise ProviderError("invalid_provider_output")
            word = AlignedWord(start=start, end=end, text=text)
            span = (start, end)
            if span in seen_spans or (previous is not None and word < previous):
                raise ProviderError("invalid_provider_output")
            seen_spans.add(span)
            words.append(word)
            previous = word
    except ProviderError:
        raise
    except Exception:
        raise ProviderError("invalid_provider_output") from None
    return words


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
    return ordered


def _utterances(assignments: list[tuple[AlignedWord, str]]) -> list[Utterance]:
    groups: list[tuple[str, list[AlignedWord]]] = []
    for word, speaker_id in assignments:
        if groups and groups[-1][0] == speaker_id:
            groups[-1][1].append(word)
        else:
            groups.append((speaker_id, [word]))
    return [
        Utterance(
            id=f"utt-{index:04d}",
            start=words[0].start,
            end=words[-1].end,
            speaker_id=speaker_id,
            confidence=1.0,
            transcript=" ".join(word.text for word in words),
        )
        for index, (speaker_id, words) in enumerate(groups, start=1)
    ]


def align_mixed(
    words: list[AlignedWord], segments: list[SpeakerSegment], duration: float
) -> tuple[list[Utterance], int]:
    timeline = _validated_segments(segments, duration)
    assignments: list[tuple[AlignedWord, str]] = []
    omitted = 0
    for word in words:
        candidates = [
            (max(0.0, min(word.end, segment.end) - max(word.start, segment.start)), segment)
            for segment in timeline
        ]
        positive = [(overlap, segment) for overlap, segment in candidates if overlap > 0]
        if not positive:
            omitted += 1
            continue
        _overlap, selected = min(
            positive,
            key=lambda item: (-item[0], item[1].start, item[1].end, item[1].speaker_id),
        )
        assignments.append((word, selected.speaker_id))
    return _utterances(assignments), omitted


def align_separate(channel_words: list[list[AlignedWord]]) -> tuple[list[Utterance], int]:
    assignments = sorted(
        (
            (word, f"SPEAKER_{channel_index:02d}")
            for channel_index, words in enumerate(channel_words)
            for word in words
        ),
        key=lambda item: (item[0].start, item[0].end, item[1], item[0].text),
    )
    return _utterances(assignments), 0
