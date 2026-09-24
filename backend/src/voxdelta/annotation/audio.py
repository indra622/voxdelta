"""Serving the source audio behind a private silver draft, in clips, without copying it.

A reviewer cannot check a draft they have only read. This module is what lets them listen,
and it is built so that listening adds no new way to reach a file:

* **The caller never names a file.** A conversation id is matched against
  :data:`voxdelta.annotation.review.CONVERSATION_ID` and joined to one configured root.
  Nothing here accepts a path, and the resolved file is checked to be a regular file that
  really sits under that root after symlinks are followed.
* **Only audio a draft already vouches for is served.** Resolution requires a readable
  ``silver.json`` for that conversation, and the file's digest has to equal the
  ``source.input_sha256`` the draft recorded. A source that was swapped after annotation is
  refused rather than played, because the draft would no longer describe what is heard.
* **Nothing is written.** A clip is a WAV header computed for the requested frame span
  followed by those frames read straight out of the source, so no excerpt is ever created
  on disk, and the source is opened read-only.

Refusals are the same shape as the rest of the review path: a stable code and text that
names no path. Durations and frame counts are numbers about audio, not about speech, so
they are safe to return; nothing in this module reads or returns transcript.
"""

from __future__ import annotations

import hashlib
import math
import struct
import wave
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voxdelta.annotation.gemini_silver import SilverTurn

#: One clip is a review aid, not a way to stream a whole call through a narrow door. The
#: ceiling is generous for a single turn and far below any conversation length.
MAX_CLIP_SECONDS = 120.0

#: A gap shorter than this is the ordinary space between turns, not something a reviewer
#: needs to open. Half a second is long enough to hold a short word, which is the smallest
#: thing that being missing from the draft would actually matter for.
MIN_GAP_SECONDS = 0.5

#: Streamed in frames rather than bytes so a chunk never splits a sample.
CLIP_CHUNK_FRAMES = 8192

_RIFF_HEADER_BYTES = 44
_MAX_WAV_DATA_BYTES = 0xFFFFFFFF - _RIFF_HEADER_BYTES


@dataclass(frozen=True, slots=True)
class SourceAudio:
    """One verified source recording, described in numbers only."""

    path: Path
    channels: int
    sample_width: int
    frame_rate: int
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.frame_rate if self.frame_rate else 0.0


@dataclass(frozen=True, slots=True)
class ReviewGap:
    """A stretch of the recording no valid draft turn covers.

    Deliberately not called a missing turn. The draft records how many turns validation
    dropped but not where they were, so a gap is where the draft says nothing — which may
    be a dropped turn, may be silence, and may be speech nobody transcribed. Naming it for
    what is known keeps the interface from asserting the part that is not.
    """

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def merge_intervals(
    intervals: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    """Collapse overlapping and touching spans into the fewest that cover the same time.

    Draft turns overlap each other routinely — one speaker starts before the other has
    finished — so the covered region has to be computed rather than assumed from the
    turn list, or every overlap would show up as a gap running backwards.
    """

    usable = [
        (float(start), float(end))
        for start, end in intervals
        if _finite(start) and _finite(end) and end > start
    ]
    if not usable:
        return ()
    usable.sort()
    merged: list[tuple[float, float]] = [usable[0]]
    for start, end in usable[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            if end > last_end:
                merged[-1] = (last_start, end)
        else:
            merged.append((start, end))
    return tuple(merged)


def review_gaps(
    turns: Sequence[SilverTurn],
    *,
    duration_seconds: float,
    minimum_seconds: float = MIN_GAP_SECONDS,
) -> tuple[ReviewGap, ...]:
    """The recording minus everything the draft's valid turns already cover."""

    if not _finite(duration_seconds) or duration_seconds <= 0:
        return ()
    covered = merge_intervals([(turn.start, turn.end) for turn in turns])
    gaps: list[ReviewGap] = []
    cursor = 0.0
    for start, end in covered:
        if start > cursor:
            gaps.append(ReviewGap(start=cursor, end=min(start, duration_seconds)))
        cursor = max(cursor, end)
        if cursor >= duration_seconds:
            break
    if cursor < duration_seconds:
        gaps.append(ReviewGap(start=cursor, end=duration_seconds))
    return tuple(
        ReviewGap(start=round(gap.start, 3), end=round(gap.end, 3))
        for gap in gaps
        if gap.duration >= minimum_seconds
    )


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_source(
    audio_root: Path,
    conversation_id: str,
    *,
    expected_sha256: str,
) -> SourceAudio:
    """Find the one recording a draft describes, or refuse.

    ``conversation_id`` is assumed to have already passed the review module's id check;
    it is re-checked here anyway, because this function opens a file and a guard that only
    holds when the caller remembered to call another one is not a guard.
    """

    from voxdelta.annotation.review import CONVERSATION_ID, ReviewRejected

    if not CONVERSATION_ID.fullmatch(conversation_id):
        raise ReviewRejected(
            "invalid_conversation_id",
            "The requested annotation identifier is not a valid conversation id.",
        )

    unavailable = ReviewRejected(
        "audio_source_unavailable",
        "The source recording for this conversation is not available on this machine.",
    )

    try:
        root = audio_root.resolve(strict=True)
    except OSError:
        raise unavailable from None

    # The production review root is ``data/derived``.  It deliberately exposes only the
    # named, locally-derived collections below; it is *not* a recursive search over
    # the data directory.  A custom root remains exact so tests and deployments cannot
    # accidentally widen what a draft can play.
    roots: tuple[Path, ...] = (root,)
    if root.name == "derived":
        roots = (
                root / "kcsc" / "audio",
                root / "022-finance-silver" / "audio",
                root / "user-provided" / "2026-09-04-evaluation-audio",
                root / "user-provided" / "2026-09-11-korean-evaluation-audio",
                root
                / "user-provided"
                / "2026-09-11-korean-evaluation-audio"
                / "gold-splits-v1",
            )

    resolved: Path | None = None
    for candidate_root in roots:
        try:
            allowed_root = candidate_root.resolve(strict=True)
            candidate = allowed_root / f"{conversation_id}.wav"
            possible = candidate.resolve(strict=True)
        except OSError:
            continue
        # Resolved on both sides before comparing: a symlink under an allowed root
        # pointing anywhere else is refused rather than becoming a way out of it.
        if possible.is_file() and allowed_root in possible.parents:
            resolved = possible
            break
    if resolved is None:
        raise unavailable

    if len(expected_sha256) != 64 or _file_digest(resolved) != expected_sha256:
        raise ReviewRejected(
            "audio_source_mismatch",
            "The stored recording does not match the audio this draft was made from.",
        )

    try:
        with wave.open(str(resolved), "rb") as source:
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            frame_rate = source.getframerate()
            frame_count = source.getnframes()
    except (wave.Error, OSError, EOFError):
        raise ReviewRejected(
            "audio_source_unreadable",
            "The source recording could not be read as WAV audio.",
        ) from None
    if channels < 1 or sample_width < 1 or frame_rate < 1 or frame_count < 1:
        raise ReviewRejected(
            "audio_source_unreadable",
            "The source recording could not be read as WAV audio.",
        )
    return SourceAudio(
        path=resolved,
        channels=channels,
        sample_width=sample_width,
        frame_rate=frame_rate,
        frame_count=frame_count,
    )


def source_sha256(record: Mapping[str, Any]) -> str:
    source = record.get("source")
    digest = source.get("input_sha256") if isinstance(source, Mapping) else None
    return digest if isinstance(digest, str) else ""


@dataclass(frozen=True, slots=True)
class ClipPlan:
    """A validated frame span plus the exact byte length its WAV response will have."""

    source: SourceAudio
    start_frame: int
    frame_count: int

    @property
    def data_bytes(self) -> int:
        return self.frame_count * self.source.channels * self.source.sample_width

    @property
    def total_bytes(self) -> int:
        return _RIFF_HEADER_BYTES + self.data_bytes

    @property
    def start_seconds(self) -> float:
        return self.start_frame / self.source.frame_rate

    @property
    def end_seconds(self) -> float:
        return (self.start_frame + self.frame_count) / self.source.frame_rate


def plan_clip(
    source: SourceAudio,
    *,
    start: float,
    end: float,
    max_seconds: float = MAX_CLIP_SECONDS,
) -> ClipPlan:
    """Turn a requested time range into frames, refusing anything outside the recording.

    The range is clamped to the recording rather than rejected for running past its end:
    a draft's last turn often ends a fraction of a second after the audio does, and
    refusing to play it would make the ordinary case look like an error. A range that
    starts past the end has nothing to clamp to and is refused.
    """

    from voxdelta.annotation.review import ReviewRejected

    if not _finite(start) or not _finite(end):
        raise ReviewRejected(
            "invalid_clip_range",
            "A clip needs a finite start and end in seconds.",
        )
    if start < 0 or end <= start:
        raise ReviewRejected(
            "invalid_clip_range",
            "A clip needs a start at or after zero and an end after its start.",
        )
    if end - start > max_seconds:
        raise ReviewRejected(
            "clip_too_long",
            f"A single clip may cover at most {max_seconds:g} seconds.",
        )

    start_frame = int(start * source.frame_rate)
    if start_frame >= source.frame_count:
        raise ReviewRejected(
            "invalid_clip_range",
            "The requested clip starts after the end of the recording.",
        )
    end_frame = min(int(round(end * source.frame_rate)), source.frame_count)
    frame_count = end_frame - start_frame
    if frame_count < 1:
        raise ReviewRejected(
            "invalid_clip_range",
            "The requested clip covers no audio.",
        )
    plan = ClipPlan(source=source, start_frame=start_frame, frame_count=frame_count)
    if plan.data_bytes > _MAX_WAV_DATA_BYTES:
        raise ReviewRejected(
            "clip_too_long",
            f"A single clip may cover at most {max_seconds:g} seconds.",
        )
    return plan


def wav_header(plan: ClipPlan) -> bytes:
    """A canonical 44-byte PCM header for exactly this clip's frames."""

    source = plan.source
    block_align = source.channels * source.sample_width
    byte_rate = source.frame_rate * block_align
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + plan.data_bytes,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        source.channels,
        source.frame_rate,
        byte_rate,
        block_align,
        source.sample_width * 8,
        b"data",
        plan.data_bytes,
    )


def clip_bytes(plan: ClipPlan) -> Iterator[bytes]:
    """Stream the clip as a WAV, reading frames from the source and copying nothing.

    Short reads are padded with silence rather than truncated: the header has already
    declared the clip's length, and a body that does not match it is a corrupt file. A
    source that shrank mid-read is the only way to reach that, and silence is the honest
    rendering of audio that is no longer there.
    """

    yield wav_header(plan)
    frame_bytes = plan.source.channels * plan.source.sample_width
    with wave.open(str(plan.source.path), "rb") as source:
        source.setpos(plan.start_frame)
        remaining = plan.frame_count
        while remaining > 0:
            wanted = min(CLIP_CHUNK_FRAMES, remaining)
            chunk = source.readframes(wanted)
            remaining -= wanted
            if len(chunk) < wanted * frame_bytes:
                yield chunk + b"\x00" * (wanted * frame_bytes - len(chunk))
                continue
            yield chunk


__all__ = [
    "CLIP_CHUNK_FRAMES",
    "MAX_CLIP_SECONDS",
    "MIN_GAP_SECONDS",
    "ClipPlan",
    "ReviewGap",
    "SourceAudio",
    "clip_bytes",
    "merge_intervals",
    "plan_clip",
    "resolve_source",
    "review_gaps",
    "source_sha256",
    "wav_header",
]
