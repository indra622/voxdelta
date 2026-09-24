"""Where G5999's deletions concentrate, in aggregate bins only.

The corrected end-to-end evaluation reported SPEAKER_00 → G5999 at CER 0.4216 with 243
deletions against 963 reference characters — deletion-dominated, and unexplained. This
module asks *where* those deletions sit, along four factors that could plausibly drive
them, and reports counts and rates per bin. It never reports text.

Two things about the method matter more than the numbers it produces.

**Per-turn scoring is not a decomposition of the pooled figure.** The pooled score comes
from one global edit path over the whole speaker; these bins come from scoring each
reference turn against the words associated with it. The two disagree by construction, and
the per-bin totals will not sum to the pooled totals. They are a diagnostic, and the
artifact says so in its own fields rather than only in prose.

**Association is by midpoint, and unassigned words are counted, not hidden.** A word is
assigned to the reference turn containing its midpoint. Midpoint containment is
unambiguous because this speaker's reference turns never overlap, and it tolerates the
timestamp jitter that whole-span containment would punish. A word whose midpoint lands in
no turn of this speaker is counted in an explicit unassigned bucket. That bucket is the
honest size of the association's own error, and it is reported beside every bin so a
reader can judge whether the bins mean anything.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from voxdelta.domain.models import SpeakerSegment
from voxdelta.evaluation.kcsc_asr_benchmark import ErrorCounts, characters, score_sequences
from voxdelta.evaluation.kcsc_attributed_asr import KcscAttributedError

SCHEMA_VERSION = "1"

#: Half-open, in seconds, so every duration lands in exactly one bin.
DURATION_BINS: tuple[tuple[str, float, float], ...] = (
    ("lt_1s", 0.0, 1.0),
    ("1_3s", 1.0, 3.0),
    ("3_8s", 3.0, 8.0),
    ("gte_8s", 8.0, float("inf")),
)

#: A turn counts as overlapped only when the intersection has positive duration. Two turns
#: that merely touch (a.end == b.start) share an instant, not speech, so they are "none".
OVERLAP_POLICY = "positive-duration intersection only; touching endpoints are not overlap"

#: Distance from a turn edge to the nearest mapped-speaker segment edge.
BOUNDARY_TOLERANCE_SECONDS = 0.250
BOUNDARY_POLICY = (
    "min over mapped-speaker segment edges of the distance to either turn edge; "
    f"<= {BOUNDARY_TOLERANCE_SECONDS}s is near_boundary, otherwise interior"
)

#: Coverage is measured as the fraction of the turn's duration intersected.
FULL_COVERAGE = 0.99
TRACE_COVERAGE = 0.01

#: Precedence for the mutually exclusive primary category. First match wins, most
#: structurally severe first: a turn the diarizer gave to the other speaker is a different
#: failure from one it merely clipped.
PRIMARY_PRECEDENCE: tuple[str, ...] = (
    "mixed_other_speaker",
    "uncovered_gap",
    "same_speaker_partial",
    "near_boundary",
    "covered_interior",
)

ASSOCIATION_POLICY = (
    "a hypothesis word is assigned to the reference turn containing its midpoint; this "
    "speaker's reference turns do not overlap, so containment is unambiguous. A word whose "
    "midpoint falls in no turn of this speaker is counted as unassigned and scored nowhere."
)


@dataclass(frozen=True, slots=True)
class Turn:
    """One reference turn, with only the fields binning needs plus its text for scoring."""

    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2.0


def duration_bin_edges() -> list[list[object]]:
    """The bin edges in a JSON-safe form.

    The open-ended bin's upper edge is ``inf`` in Python because the comparison needs it,
    but ``inf`` is not valid JSON and the artifact is written with ``allow_nan=False``.
    It is published as ``null``, which is how JSON says "no upper bound".
    """

    return [
        [name, low, None if high == float("inf") else high] for name, low, high in DURATION_BINS
    ]


def duration_bin(duration: float) -> str:
    for name, low, high in DURATION_BINS:
        if low <= duration < high:
            return name
    raise KcscAttributedError(f"duration {duration} falls in no bin")


def _intersection(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def overlap_bin(turn: Turn, other_speaker_turns: Sequence[Turn]) -> str:
    """Whether any of the other reference speaker's turns overlaps this one."""

    span = (turn.start, turn.end)
    for other in other_speaker_turns:
        if other.start >= turn.end:
            break
        if _intersection(span, (other.start, other.end)) > 0.0:
            return "any"
    return "none"


def boundary_bin(turn: Turn, mapped: Sequence[SpeakerSegment]) -> tuple[str, float]:
    """How close this turn's edges sit to the mapped diarization segment edges."""

    if not mapped:
        return "interior", float("inf")
    distance = min(
        min(abs(edge - turn.start), abs(edge - turn.end))
        for segment in mapped
        for edge in (segment.start, segment.end)
    )
    label = "near_boundary" if distance <= BOUNDARY_TOLERANCE_SECONDS else "interior"
    return label, distance


def coverage_bin(
    turn: Turn, mapped: Sequence[SpeakerSegment], other: Sequence[SpeakerSegment]
) -> tuple[str, float, float]:
    """How the diarizer's timeline covers this reference turn, as one of four categories.

    Fractions are of the turn's own duration, so a long turn and a short one are described
    on the same scale. ``TRACE_COVERAGE`` keeps a few milliseconds of bleed from promoting
    a turn into a category it does not belong in.
    """

    if turn.duration <= 0:
        return "ambiguous", 0.0, 0.0
    span = (turn.start, turn.end)
    mapped_fraction = sum(_intersection(span, (s.start, s.end)) for s in mapped) / turn.duration
    other_fraction = sum(_intersection(span, (s.start, s.end)) for s in other) / turn.duration

    if other_fraction > TRACE_COVERAGE:
        return "mixed_other_speaker", mapped_fraction, other_fraction
    if mapped_fraction <= TRACE_COVERAGE:
        return "uncovered_gap", mapped_fraction, other_fraction
    if mapped_fraction >= FULL_COVERAGE:
        return "full", mapped_fraction, other_fraction
    return "same_speaker_partial", mapped_fraction, other_fraction


def preceding_silence(turn: Turn, all_turns: Sequence[Turn]) -> float:
    """Gap between this turn's onset and the last reference speech of either speaker.

    Descriptive only. It measures silence in the *reference*, which is not the same as
    silence the recogniser heard, and it is reported as a marginal factor for that reason.
    """

    previous = [other.end for other in all_turns if other.end <= turn.start]
    return turn.start - max(previous) if previous else turn.start


def primary_category(coverage: str, boundary: str) -> str:
    """Collapse the factors into one mutually exclusive label by :data:`PRIMARY_PRECEDENCE`."""

    if coverage == "mixed_other_speaker":
        return "mixed_other_speaker"
    if coverage == "uncovered_gap":
        return "uncovered_gap"
    if coverage == "same_speaker_partial":
        return "same_speaker_partial"
    if boundary == "near_boundary":
        return "near_boundary"
    return "covered_interior"


@dataclass(frozen=True, slots=True)
class TurnDiagnosis:
    """One reference turn's bins and its score against the words associated with it."""

    duration_bin: str
    overlap_bin: str
    boundary_bin: str
    coverage_bin: str
    primary: str
    preceding_silence_bin: str
    counts: ErrorCounts


def associate(words: Sequence[Any], turns: Sequence[Turn]) -> tuple[dict[int, list[str]], int, int]:
    """Assign each word to the turn containing its midpoint; count the rest as unassigned.

    Returns the per-turn word text, the number of unassigned words, and the number of
    characters they carried, so the size of the association's own blind spot is visible.
    """

    assigned: dict[int, list[str]] = {}
    unassigned = 0
    unassigned_chars = 0
    for word in words:
        midpoint = (word.start + word.end) / 2.0
        index = next(
            (position for position, turn in enumerate(turns) if turn.start <= midpoint <= turn.end),
            None,
        )
        if index is None:
            unassigned += 1
            unassigned_chars += len(characters(word.text))
            continue
        assigned.setdefault(index, []).append(word.text)
    return assigned, unassigned, unassigned_chars


def diagnose(
    *,
    turns: Sequence[Turn],
    other_turns: Sequence[Turn],
    mapped: Sequence[SpeakerSegment],
    other_segments: Sequence[SpeakerSegment],
    words: Sequence[Any],
) -> tuple[list[TurnDiagnosis], int, int]:
    """Bin and score every reference turn of the speaker under diagnosis."""

    ordered = sorted(turns, key=lambda turn: (turn.start, turn.end))
    other_ordered = sorted(other_turns, key=lambda turn: (turn.start, turn.end))
    assigned, unassigned, unassigned_chars = associate(words, ordered)

    diagnoses: list[TurnDiagnosis] = []
    for index, turn in enumerate(ordered):
        boundary, _distance = boundary_bin(turn, mapped)
        coverage, _mapped_fraction, _other_fraction = coverage_bin(turn, mapped, other_segments)
        silence = preceding_silence(turn, [*ordered, *other_ordered])
        diagnoses.append(
            TurnDiagnosis(
                duration_bin=duration_bin(turn.duration),
                overlap_bin=overlap_bin(turn, other_ordered),
                boundary_bin=boundary,
                coverage_bin=coverage,
                primary=primary_category(coverage, boundary),
                preceding_silence_bin="lt_0.5s" if silence < 0.5 else "gte_0.5s",
                counts=score_sequences(
                    characters(turn.text), characters(" ".join(assigned.get(index, [])))
                ),
            )
        )
    return diagnoses, unassigned, unassigned_chars


def _pool(counts: Sequence[ErrorCounts]) -> dict[str, Any]:
    """Pool one bin's turns into a JSON-safe row.

    A bin whose turns all normalise to empty text has no denominator, and ErrorCounts
    rightly refuses to invent a rate for it. The row still has to serialise, so the rate
    is published as null — "not defined here" — rather than as a zero that would read as
    a perfect score, or an infinity that is not valid JSON at all.
    """

    merged = counts[0]
    for other in counts[1:]:
        merged = merged.merged(other)
    if merged.reference_length == 0:
        payload: dict[str, Any] = {
            "error_rate": None,
            "substitutions": merged.substitutions,
            "deletions": merged.deletions,
            "insertions": merged.insertions,
            "errors": merged.errors,
            "reference_length": 0,
            "hypothesis_length": merged.hypothesis_length,
        }
    else:
        payload = dict(merged.as_dict())
    payload["turn_count"] = len(counts)
    return payload


def tabulate(diagnoses: Sequence[TurnDiagnosis], factor: str) -> dict[str, Any]:
    """Group turn scores by one factor. Empty bins are reported, not omitted."""

    grouped: dict[str, list[ErrorCounts]] = {}
    for diagnosis in diagnoses:
        grouped.setdefault(getattr(diagnosis, factor), []).append(diagnosis.counts)
    table = {label: _pool(counts) for label, counts in sorted(grouped.items())}
    if factor == "duration_bin":
        for name, _low, _high in DURATION_BINS:
            table.setdefault(
                name,
                {
                    "turn_count": 0,
                    "reference_length": 0,
                    "errors": 0,
                    "deletions": 0,
                    "substitutions": 0,
                    "insertions": 0,
                    "error_rate": None,
                },
            )
    return table


__all__ = [
    "ASSOCIATION_POLICY",
    "BOUNDARY_POLICY",
    "BOUNDARY_TOLERANCE_SECONDS",
    "DURATION_BINS",
    "FULL_COVERAGE",
    "OVERLAP_POLICY",
    "PRIMARY_PRECEDENCE",
    "SCHEMA_VERSION",
    "TRACE_COVERAGE",
    "Turn",
    "TurnDiagnosis",
    "associate",
    "boundary_bin",
    "coverage_bin",
    "diagnose",
    "duration_bin",
    "duration_bin_edges",
    "overlap_bin",
    "preceding_silence",
    "primary_category",
    "tabulate",
]
