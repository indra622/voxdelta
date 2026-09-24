"""Sanitation contract for Qwen3-ASR word timestamps, applied before attribution.

The Qwen forced aligner reports a substantial minority of words as *instants*: units
whose ``start_time`` equals their ``end_time``. Measured on six ten-minute Korean tracks,
between 41 and 95 words per track arrive this way. They are not errors — the text is real
and the moment is real — but they carry no span, and everything downstream of the
recognizer assumes a span: overlap with a diarization segment decides which speaker owns
a word, and an utterance is required to have positive extent.

Two responses would be wrong. Rejecting the whole result fails closed on every real
recording, which is what the shared validator does today and why this contract exists.
Widening each instant into a small span would invent timing the aligner never reported,
and near a turn boundary that invented timing decides which speaker gets the word.

So the contract keeps genuine spans untouched and applies one rule to instants:

    **Strict-interior point attribution.** An instant is attributed to a diarization
    segment only when exactly one segment strictly contains it. Otherwise the word is
    omitted from attribution and counted.

"Strictly" excludes the segment edges, so an instant landing exactly on a boundary — the
case where a guess could cross into the wrong speaker — is never placed. "Exactly one"
excludes instants inside overlapping segments. An attributed instant joins its segment's
turn carrying its own moment; it is never stretched, and it can only extend a turn to a
moment the aligner actually reported.

What is omitted is not silently lost. :class:`TimestampCoverage` reports how much of the
recognizer's text reached attribution and whether anything did not, so a caller can tell
a complete transcript from one with holes instead of assuming the first.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

from voxdelta.domain.models import SpeakerSegment
from voxdelta.providers.asr_alignment import (
    AlignedWord,
    point_segment_index,
    validated_point_words,
)

#: Stable identifier for the rule applied to zero-duration spans, so a result can state
#: which contract produced it rather than leaving the policy implicit.
SANITATION_POLICY = "strict-interior-point-attribution"


@dataclass(frozen=True, slots=True)
class TimestampCoverage:
    """How much recognizer text reached speaker attribution, and what did not.

    ``omitted_words`` is the authoritative count of text absent from the transcript: the
    instants this contract refused to place, plus any the alignment could not seat in a
    turn with real extent. ``uncertain`` is the single flag a caller should branch on.
    """

    total_words: int
    positive_spans: int
    zero_spans: int
    zero_spans_unplaceable: int
    omitted_words: int

    @property
    def attributed_words(self) -> int:
        return self.total_words - self.omitted_words

    @property
    def attributed_ratio(self) -> float:
        """Share of recognized words that reached a speaker turn; 1.0 when none were lost."""

        if self.total_words == 0:
            return 1.0
        return self.attributed_words / self.total_words

    @property
    def zero_span_ratio(self) -> float:
        if self.total_words == 0:
            return 0.0
        return self.zero_spans / self.total_words

    @property
    def uncertain(self) -> bool:
        """Whether any recognized word is missing from the attributed transcript."""

        return self.omitted_words > 0

    def with_alignment_omissions(self, omitted: int) -> TimestampCoverage:
        """Fold the alignment's own omissions into the total this contract reports."""

        if omitted < 0:
            raise ValueError("omitted must not be negative")
        return replace(self, omitted_words=self.zero_spans_unplaceable + omitted)

    def merged(self, other: TimestampCoverage) -> TimestampCoverage:
        """Combine per-channel coverage into one figure for a separate-channel asset."""

        return TimestampCoverage(
            total_words=self.total_words + other.total_words,
            positive_spans=self.positive_spans + other.positive_spans,
            zero_spans=self.zero_spans + other.zero_spans,
            zero_spans_unplaceable=self.zero_spans_unplaceable + other.zero_spans_unplaceable,
            omitted_words=self.omitted_words + other.omitted_words,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "policy": SANITATION_POLICY,
            "total_words": self.total_words,
            "positive_spans": self.positive_spans,
            "zero_spans": self.zero_spans,
            "zero_spans_unplaceable": self.zero_spans_unplaceable,
            "omitted_words": self.omitted_words,
            "attributed_words": self.attributed_words,
            "attributed_ratio": round(self.attributed_ratio, 6),
            "zero_span_ratio": round(self.zero_span_ratio, 6),
            "uncertain": self.uncertain,
        }


def sanitize_words(
    records: Iterable[tuple[object, object, object]],
    duration: float,
    segments: list[SpeakerSegment] | None = None,
) -> tuple[list[AlignedWord], TimestampCoverage]:
    """Validate recognizer words and drop instants that cannot be placed safely.

    ``segments`` is the diarization timeline for mixed audio. When it is None the asset is
    separate-channel: each channel already belongs to one speaker, so no instant can be
    attributed to the wrong one and none is dropped here. Whether an instant can be seated
    in a turn with real extent is still decided later, by the alignment.

    Word order is preserved. Nothing is re-timed, widened, merged, or reordered; the only
    edit this function makes to a recognizer's output is removal.
    """

    words = validated_point_words(records, duration)
    kept: list[AlignedWord] = []
    zero_spans = 0
    unplaceable = 0
    for word in words:
        if word.end > word.start:
            kept.append(word)
            continue
        zero_spans += 1
        if segments is not None and point_segment_index(word.start, segments) is None:
            unplaceable += 1
            continue
        kept.append(word)

    coverage = TimestampCoverage(
        total_words=len(words),
        positive_spans=len(words) - zero_spans,
        zero_spans=zero_spans,
        zero_spans_unplaceable=unplaceable,
        omitted_words=unplaceable,
    )
    return kept, coverage


__all__ = [
    "SANITATION_POLICY",
    "TimestampCoverage",
    "sanitize_words",
]
