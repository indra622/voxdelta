"""Speaker-attributed Qwen ASR loss on a preserved Precision-2 timeline, offline.

The earlier per-track benchmark answered "how well does Qwen hear one speaker when the
segmentation is perfect". This answers the harder, more honest question the product
actually faces: how much of what was said survives when Qwen's words are attributed to
speakers by a *real* diarizer's timeline. The timeline is the one the retry-2 run already
obtained from Precision-2 and persisted; nothing is submitted anywhere to reproduce it.

Two figures are computed from a single decode, over one shared set of reference
characters:

* **mixed_stream** — every recognized word, in one stream, against every attributed
  reference row in *time* order. No speaker attribution, so this is the recogniser's own
  error on this conversation.
* **speaker_attributed_pooled** — each speaker's recognized words, as placed by the
  preserved timeline, against that same speaker's reference rows.

Their difference is reported as ``e2e_vs_mixed_stream_delta`` and deliberately not called
an attribution cost. Both sides score the same reference characters, so the subtraction is
sound arithmetic, but the two readings differ in more than attribution — see
:data:`DELTA_CONFOUNDS`. An earlier version of this module built the mixed-stream
reference by concatenating one speaker's rows after the other's and compared that to a
time-ordered decode; that scores even a perfect transcript as roughly half wrong, and the
artifact it produced has been retracted.

Reference and hypothesis text exist only inside the scoring functions. Every value that
reaches an artifact or a log line is a count, a rate, or a digest.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from voxdelta.domain.models import SpeakerSegment
from voxdelta.evaluation.kcsc_asr_benchmark import (
    NORMALIZATION_STEPS,
    ErrorCounts,
    KcscAsrError,
    ModelIdentity,
    characters,
    eojeol,
    score_sequences,
)
from voxdelta.evaluation.kcsc_diarization_benchmark import file_sha256
from voxdelta.providers.asr_alignment import align_mixed
from voxdelta.providers.qwen_timestamps import SANITATION_POLICY, TimestampCoverage, sanitize_words

SCHEMA_VERSION = "1"

#: Time-only, so the mapping cannot be chosen to flatter the transcript.
SPEAKER_MAPPING_RULE = "one-to-one assignment maximising total temporal overlap; timings only"

#: What the reported delta contains besides speaker attribution. Named so the figure is
#: read as an observation and not as a clean decomposition of the pipeline's error.
DELTA_CONFOUNDS: tuple[str, ...] = (
    "The mixed-stream side is scored against both speakers interleaved in time, so it is "
    "never charged for placing a word under the wrong speaker; the attributed side is. "
    "That asymmetry is the intended contrast, but it is the whole of the difference only "
    "if the confounds below are negligible, which has not been established.",
    "Words the timestamp sanitation contract declined to place are absent from the "
    "attributed side and present in the mixed stream, so they count as deletions on one "
    "side only.",
    "The decoder chunks long audio at low-energy boundaries; a word split across a chunk "
    "edge can differ between the two readings of the same decode.",
    "Turn-level grouping concatenates each speaker's rows with single spaces, which is a "
    "different spacing convention from the interleaved stream and shifts the eojeol rate "
    "more than the character rate.",
)


class KcscAttributedError(KcscAsrError):
    """Raised for any condition that would make a reported rate untrustworthy."""


@dataclass(frozen=True, slots=True)
class PinnedInput:
    """An input this evaluation refuses to run without, identified by digest."""

    path: Path
    expected_sha256: str

    def verify(self) -> str:
        if not self.path.is_file():
            raise KcscAttributedError(f"missing pinned input: {self.path}")
        found = file_sha256(self.path)
        if found != self.expected_sha256:
            raise KcscAttributedError(
                f"{self.path.name}: digest {found[:12]}… does not match the pinned "
                f"{self.expected_sha256[:12]}…"
            )
        return found


def load_timeline(record_path: Path) -> tuple[list[SpeakerSegment], Mapping[str, object]]:
    """Read the preserved Precision-2 timeline out of the immutable retry-2 record.

    The record is the only source of segmentation here. Nothing is recomputed and no
    provider is contacted; if the record lacks a usable timeline the run stops rather
    than falling back to producing one.
    """

    record = json.loads(record_path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise KcscAttributedError(f"retry-2 record has an unexpected shape: {record_path}")
    raw = record.get("hypothesis_timeline")
    if not isinstance(raw, list) or not raw:
        raise KcscAttributedError("retry-2 record carries no preserved hypothesis timeline")

    segments: list[SpeakerSegment] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise KcscAttributedError(f"timeline[{index}] is not an object")
        try:
            segments.append(
                SpeakerSegment(
                    start=float(item["start"]),
                    end=float(item["end"]),
                    speaker_id=str(item["speaker_id"]),
                    confidence=1.0,
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise KcscAttributedError(f"timeline[{index}] is unusable: {error}") from None
    segments.sort(key=lambda segment: (segment.start, segment.end, segment.speaker_id))
    speakers = {segment.speaker_id for segment in segments}
    if len(speakers) != 2:
        raise KcscAttributedError(f"expected two diarized speakers, found {len(speakers)}")
    return segments, record


@dataclass(frozen=True, slots=True)
class ReferenceRows:
    """Reference text arranged the two ways this evaluation needs to read it.

    ``by_speaker`` preserves each speaker's own chronological order, for the attributed
    scores. ``chronological`` interleaves both speakers and the unattributed rows in time
    order, which is the only ordering a single decoded stream can be compared against —
    concatenating one speaker's turns after the other's would score a correct transcript
    as almost entirely wrong.
    """

    by_speaker: dict[str, list[str]]
    events: list[str]
    turn_count: int
    chronological: list[str]
    chronological_attributed: list[str]


def reference_rows(reference_path: Path) -> ReferenceRows:
    """Group reference transcript by speaker and in time order.

    Returns text held only for scoring. Unattributed rows — laughter, coughs, ambient
    noise — cannot belong to a speaker, so they are kept apart rather than assigned to
    one: charging a speaker for a cough would inflate that speaker's error.
    """

    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if not isinstance(reference, dict):
        raise KcscAttributedError(f"reference has an unexpected shape: {reference_path}")
    turns = reference.get("turns")
    if not isinstance(turns, list) or not turns:
        raise KcscAttributedError("reference carries no turns")

    by_speaker: dict[str, list[str]] = {}
    timed: list[tuple[float, float, str]] = []
    for row in turns:
        if not isinstance(row, dict):
            raise KcscAttributedError("reference turn is not an object")
        by_speaker.setdefault(str(row["speaker"]), []).append(str(row["transcript"]))
        timed.append((float(row["start"]), float(row["end"]), str(row["transcript"])))
    attributed_timed = sorted(timed)

    events: list[str] = []
    for row in reference.get("unattributed_events", []):
        if not isinstance(row, dict):
            continue
        events.append(str(row["transcript"]))
        timed.append((float(row["start"]), float(row["end"]), str(row["transcript"])))
    timed.sort()

    return ReferenceRows(
        by_speaker=by_speaker,
        events=events,
        turn_count=len(turns),
        chronological=[text for _start, _end, text in timed],
        chronological_attributed=[text for _start, _end, text in attributed_timed],
    )


def _reference_intervals(reference_path: Path) -> dict[str, list[tuple[float, float]]]:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    intervals: dict[str, list[tuple[float, float]]] = {}
    for row in reference["turns"]:
        intervals.setdefault(str(row["speaker"]), []).append(
            (float(row["start"]), float(row["end"]))
        )
    return intervals


def _overlap(spans: Sequence[tuple[float, float]], segments: Sequence[SpeakerSegment]) -> float:
    total = 0.0
    for start, end in spans:
        for segment in segments:
            if segment.start >= end:
                break
            total += max(0.0, min(end, segment.end) - max(start, segment.start))
    return total


def map_speakers(
    segments: Sequence[SpeakerSegment], reference_path: Path
) -> tuple[dict[str, str], dict[str, float]]:
    """Assign each diarized label to a reference speaker by temporal overlap alone.

    Both assignments are scored and the larger total wins. Using time rather than text
    keeps the mapping independent of the transcript being measured; a mapping chosen by
    error rate would be choosing the answer.
    """

    intervals = _reference_intervals(reference_path)
    reference_speakers = sorted(intervals)
    hypothesis_speakers = sorted({segment.speaker_id for segment in segments})
    if len(reference_speakers) != 2 or len(hypothesis_speakers) != 2:
        raise KcscAttributedError("speaker mapping requires exactly two speakers on each side")

    by_label = {
        label: [segment for segment in segments if segment.speaker_id == label]
        for label in hypothesis_speakers
    }
    straight = _overlap(intervals[reference_speakers[0]], by_label[hypothesis_speakers[0]]) + (
        _overlap(intervals[reference_speakers[1]], by_label[hypothesis_speakers[1]])
    )
    crossed = _overlap(intervals[reference_speakers[0]], by_label[hypothesis_speakers[1]]) + (
        _overlap(intervals[reference_speakers[1]], by_label[hypothesis_speakers[0]])
    )
    if straight >= crossed:
        mapping = dict(zip(hypothesis_speakers, reference_speakers, strict=True))
    else:
        mapping = dict(zip(hypothesis_speakers, reversed(reference_speakers), strict=True))
    return mapping, {
        "straight_overlap_seconds": round(straight, 3),
        "crossed_overlap_seconds": round(crossed, 3),
        "margin_seconds": round(abs(straight - crossed), 3),
    }


def time_records(stamps: object) -> Iterable[tuple[object, object, object]]:
    """Flatten the aligner's units into the (start, end, text) triples sanitation expects."""

    for raw in cast(Iterable[object], stamps):
        start = getattr(raw, "start_time", getattr(raw, "start", None))
        end = getattr(raw, "end_time", getattr(raw, "end", start))
        yield start, end, getattr(raw, "text", "")


@dataclass(frozen=True, slots=True)
class AttributedScores:
    """Every rate this evaluation reports, and the counts behind them."""

    mixed_stream: ErrorCounts
    speaker_attributed_pooled: ErrorCounts
    per_speaker: Mapping[str, ErrorCounts]
    mixed_stream_eojeol: ErrorCounts
    speaker_attributed_eojeol: ErrorCounts

    @property
    def e2e_vs_mixed_stream_delta(self) -> float:
        """Observed difference between the two rates, over one shared denominator.

        Deliberately not called an attribution cost. The subtraction is arithmetically
        sound — both sides score the same reference characters — but the two numbers are
        not otherwise like for like, so the difference is an observation rather than a
        decomposition. See ``DELTA_CONFOUNDS`` for what it contains besides attribution.
        """

        return self.speaker_attributed_pooled.error_rate - self.mixed_stream.error_rate


def score(
    *,
    stream_text: str,
    hypothesis_by_speaker: Mapping[str, str],
    reference: ReferenceRows,
    mapping: Mapping[str, str],
) -> AttributedScores:
    """Compute every rate from one decode. Text enters here and never leaves."""

    reference_by_speaker = reference.by_speaker
    # Time order, matching the decoder's own ordering. Speaker-grouped order would score a
    # perfect chronological transcript as roughly half wrong; a test pins that.
    #
    # Unattributed rows are excluded so this shares an exact denominator with the pooled
    # per-speaker score. They are event tags that normalise to zero characters anyway, so
    # including them would change the count of rows but not the reference length.
    attributed_reference = " ".join(reference.chronological_attributed)

    per_speaker: dict[str, ErrorCounts] = {}
    for label, reference_speaker in sorted(mapping.items()):
        per_speaker[label] = score_sequences(
            characters(" ".join(reference_by_speaker.get(reference_speaker, []))),
            characters(hypothesis_by_speaker.get(label, "")),
        )
    pooled = list(per_speaker.values())[0]
    for counts in list(per_speaker.values())[1:]:
        pooled = pooled.merged(counts)

    pooled_eojeol = None
    for label, reference_speaker in sorted(mapping.items()):
        counts = score_sequences(
            eojeol(" ".join(reference_by_speaker.get(reference_speaker, []))),
            eojeol(hypothesis_by_speaker.get(label, "")),
        )
        pooled_eojeol = counts if pooled_eojeol is None else pooled_eojeol.merged(counts)
    assert pooled_eojeol is not None

    return AttributedScores(
        mixed_stream=score_sequences(characters(attributed_reference), characters(stream_text)),
        speaker_attributed_pooled=pooled,
        per_speaker=per_speaker,
        mixed_stream_eojeol=score_sequences(eojeol(attributed_reference), eojeol(stream_text)),
        speaker_attributed_eojeol=pooled_eojeol,
    )


LIMITATIONS: tuple[str, ...] = (
    "One conversation. Not a basis for any general claim about Korean ASR, diarization, "
    "or the two combined.",
    "The reported delta is an observed difference between two rates, NOT a decomposition "
    "and NOT diarization error. See delta_confounds in this artifact for what else it "
    "contains.",
    "The diarized timeline is reused from the retry-2 record and was produced by a remote "
    "provider under a limited override whose processing rights were never confirmed. This "
    "evaluation transmitted nothing, but it inherits that provenance.",
    "Speaker mapping is chosen by temporal overlap, so a diarizer that swapped the two "
    "speakers for most of the call would be scored after un-swapping them. The overlap "
    "margin is reported so a near-tie is visible.",
    "Unattributed reference rows (laughter, coughs, ambient noise) carry no speaker and "
    "are excluded from both figures, so the two share an exact denominator. They are "
    "event tags that normalise to zero characters, so excluding them changes the row "
    "count but not the reference length.",
    "The eojeol rate is secondary and conflates spacing with recognition; Qwen's spacing "
    "conventions differ from the corpus's.",
    "Edit operations come from rapidfuzz, present transitively rather than as a declared "
    "dependency.",
)


def build_report(
    *,
    scores: AttributedScores,
    coverage: TimestampCoverage,
    mapping: Mapping[str, str],
    mapping_evidence: Mapping[str, float],
    identity: ModelIdentity,
    aligner: ModelIdentity,
    inputs: Mapping[str, str],
    segment_count: int,
    reference_turn_count: int,
    reference_event_count: int,
    utterance_count: int,
    omitted_by_alignment: int,
    duration_seconds: float,
    elapsed_seconds: float,
    egress_attempts: int,
    device: str,
    dtype: str,
) -> dict[str, Any]:
    """Assemble the transcript-free artifact."""

    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "kcsc-precision-qwen-e2e-asr",
        "transcript_free": True,
        "task": "speaker-attributed-asr-on-preserved-diarization",
        "task_note": (
            "Qwen words attributed to speakers by a preserved Precision-2 timeline and "
            "scored per speaker against KCSC reference rows. The timeline was reused from "
            "an earlier record; no audio was transmitted by this run."
        ),
        "local_only": {
            "external_calls": 0,
            "network_egress_attempts": egress_attempts,
            "audio_transmitted": False,
            "diarization_recomputed": False,
            "diarization_source": "preserved retry-2 hypothesis_timeline",
            "hub_offline": True,
        },
        "inputs": dict(inputs),
        "model": {
            "name": identity.repo_id.removeprefix("Qwen/"),
            "repo_id": identity.repo_id,
            "revision": identity.revision,
            "tree_sha256": identity.tree_sha256,
            "aligner_repo_id": aligner.repo_id,
            "aligner_revision": aligner.revision,
            "aligner_tree_sha256": aligner.tree_sha256,
            "device": device,
            "dtype": dtype,
            "remote": False,
        },
        "scope": {
            "conversation_count": 1,
            "diarized_segment_count": segment_count,
            "reference_turn_count": reference_turn_count,
            "reference_unattributed_event_count": reference_event_count,
            "attributed_utterance_count": utterance_count,
            "words_omitted_by_alignment": omitted_by_alignment,
            "duration_seconds": round(duration_seconds, 3),
        },
        "speaker_mapping": {
            "rule": SPEAKER_MAPPING_RULE,
            "mapping": dict(sorted(mapping.items())),
            "evidence_seconds": dict(mapping_evidence),
        },
        "scoring": {
            "primary_metric": "korean_normalized_character_error_rate",
            "normalization_steps": list(NORMALIZATION_STEPS),
            "aggregation": "pooled edit operations over pooled reference length",
            "implementation": "rapidfuzz.distance.Levenshtein.editops",
            "timestamp_sanitation_policy": SANITATION_POLICY,
        },
        "metrics": {
            "mixed_stream": scores.mixed_stream.as_dict(),
            "mixed_stream_note": (
                "one decoded stream against the attributed reference rows in TIME order; "
                "no speaker attribution is scored"
            ),
            "speaker_attributed_pooled": scores.speaker_attributed_pooled.as_dict(),
            "per_speaker": {
                label: counts.as_dict() for label, counts in sorted(scores.per_speaker.items())
            },
            "e2e_vs_mixed_stream_delta_cer": round(scores.e2e_vs_mixed_stream_delta, 6),
            "delta_note": (
                "speaker_attributed_pooled minus mixed_stream over the same reference "
                "characters. An observed difference, NOT a decomposition and NOT "
                "diarization error."
            ),
            "delta_confounds": list(DELTA_CONFOUNDS),
            "eojeol_secondary": {
                "mixed_stream": scores.mixed_stream_eojeol.as_dict(),
                "speaker_attributed_pooled": scores.speaker_attributed_eojeol.as_dict(),
            },
        },
        "timestamp_coverage": coverage.as_dict(),
        "timing": {
            "elapsed_seconds": round(elapsed_seconds, 3),
            "audio_seconds": round(duration_seconds, 3),
            "real_time_factor": round(elapsed_seconds / duration_seconds, 6),
        },
        "limitations": list(LIMITATIONS),
    }


def attribute(
    words: Sequence[Any], segments: Sequence[SpeakerSegment], duration: float
) -> tuple[dict[str, str], int, int]:
    """Place sanitized words on the preserved timeline and group the text by speaker."""

    utterances, omitted = align_mixed(list(words), list(segments), duration)
    by_speaker: dict[str, list[str]] = {}
    for utterance in utterances:
        by_speaker.setdefault(utterance.speaker_id, []).append(utterance.transcript)
    return (
        {label: " ".join(parts) for label, parts in by_speaker.items()},
        omitted,
        len(utterances),
    )


def sanitize(
    records: Iterable[tuple[object, object, object]],
    duration: float,
    segments: Sequence[SpeakerSegment],
) -> tuple[list[Any], TimestampCoverage]:
    """Apply the shipped sanitation contract to the decoder's word timeline."""

    return sanitize_words(records, duration, list(segments))


def write_report(report: Mapping[str, object], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    path.write_text(f"{raw}\n", encoding="utf-8")
    return file_sha256(path)


def elapsed_since(started: float) -> float:
    return time.monotonic() - started


__all__ = [
    "DELTA_CONFOUNDS",
    "LIMITATIONS",
    "SCHEMA_VERSION",
    "SPEAKER_MAPPING_RULE",
    "AttributedScores",
    "ReferenceRows",
    "KcscAttributedError",
    "PinnedInput",
    "attribute",
    "build_report",
    "load_timeline",
    "map_speakers",
    "reference_rows",
    "sanitize",
    "score",
    "time_records",
    "write_report",
]
