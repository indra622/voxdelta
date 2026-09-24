"""Local-only ASR upper bound on the derived KCSC per-speaker track set.

The mixed benchmark in :mod:`voxdelta.evaluation.kcsc_asr_benchmark` decodes one stream
from the mono mix of a two-party call. That number confounds two different failures: the
recogniser mishearing speech, and the recogniser being handed two people at once. This
benchmark removes the second by decoding each speaker's own microphone track separately
and scoring it against that speaker's own reference rows. The result is the *intrinsic*
error the recogniser would still make if speaker segmentation were perfect, so the gap
between the two is the part of the mixed error that better separation could recover.

Two properties make that subtraction meaningful, and both are enforced rather than
assumed:

* **One timeline, one ground truth.** The track set is cut at the mixed set's own trim
  offsets, and its per-track reference rows were verified row for row against the mixed
  reference at derivation time. Both benchmarks therefore score the same annotated words.
* **One decoder, one configuration.** The same pinned checkpoint, device, compute type,
  and decode options are used, and the model's tree digest is recorded in the report. A
  gap measured across two different decoders would not be a segmentation gap.

The bound is an upper bound on what segmentation can buy, not a claim about clean audio:
a KCSC track is one participant's microphone in a shared room, so the other participant
is still audible at a lower level. That, and the fact that no segmentation error is being
modelled at all, are reported as limitations rather than smoothed away.

Nothing in this module's public surface carries transcript text. Reference and hypothesis
strings exist only inside the scoring functions; every value that reaches a report is a
count, a rate, or safe metadata.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from voxdelta.evaluation.kcsc_asr_benchmark import (
    CORPUS_TAGS,
    NORMALIZATION_STEPS,
    DecodeStats,
    ErrorCounts,
    KcscAsrError,
    ModelIdentity,
    WhisperModel,
    characters,
    decode_chronological,
    eojeol,
    reference_transcript,
    score_sequences,
)
from voxdelta.evaluation.kcsc_diarization_benchmark import (
    KcscBenchmarkError,
    file_sha256,
    verify_inputs,
)

SCHEMA_VERSION = "1"

#: The approved fixed selection, mirroring the mixed benchmark's so the two are subtractable.
BENCHMARK_CONVERSATIONS: tuple[str, ...] = (
    "A0051_S0001_0",
    "A0055_S0006_0",
    "A6000_S0005_0",
)

#: KCSC records one microphone per participant, and every selected call is two-party.
TRACKS_PER_CONVERSATION = 2


def load_track_manifest(derived_root: Path) -> Mapping[str, object]:
    """Read the track derivation manifest, failing closed when absent or malformed."""

    path = derived_root / "manifest.json"
    if not path.is_file():
        raise KcscAsrError(f"missing track derivation manifest: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise KcscAsrError(f"track derivation manifest is not valid JSON: {path}") from None
    if not isinstance(manifest, dict) or not isinstance(manifest.get("tracks"), list):
        raise KcscAsrError(f"track derivation manifest has an unexpected shape: {path}")
    return manifest


def select_tracks(
    manifest: Mapping[str, object], conversation_ids: Sequence[str]
) -> tuple[Mapping[str, object], ...]:
    """Return every track of the requested conversations, in manifest order.

    A conversation that is not represented by exactly both of its speakers aborts the run.
    Scoring one side of a call and pooling it with two-sided calls would weight the result
    towards whichever speaker happened to survive.
    """

    entries = manifest["tracks"]
    assert isinstance(entries, list)
    selected: list[Mapping[str, object]] = []
    for conversation_id in conversation_ids:
        tracks = [
            entry
            for entry in entries
            if isinstance(entry, dict) and entry.get("conversation_id") == conversation_id
        ]
        if len(tracks) != TRACKS_PER_CONVERSATION:
            raise KcscAsrError(
                f"{conversation_id}: expected {TRACKS_PER_CONVERSATION} speaker tracks, "
                f"found {len(tracks)}"
            )
        speakers = [entry.get("speaker") for entry in tracks]
        if len(set(speakers)) != TRACKS_PER_CONVERSATION or not all(
            isinstance(speaker, str) for speaker in speakers
        ):
            raise KcscAsrError(f"{conversation_id}: speaker tracks are not distinctly labelled")
        selected.extend(tracks)
    return tuple(selected)


def track_id(entry: Mapping[str, object]) -> str:
    return f"{entry.get('conversation_id')}_{entry.get('speaker')}"


def verify_track_inputs(entry: Mapping[str, object], *, derived_root: Path) -> tuple[Path, Path]:
    """Re-hash one track's audio and reference, naming the speaker on failure."""

    try:
        return verify_inputs(entry, derived_root=derived_root)
    except KcscBenchmarkError as error:
        raise KcscAsrError(f"track {entry.get('speaker')}: {error}") from None


def track_reference_transcript(
    reference_path: Path, entry: Mapping[str, object]
) -> tuple[str, int]:
    """Read one track's reference, refusing a file that belongs to the other speaker."""

    try:
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise KcscAsrError(f"reference is not valid JSON: {reference_path}") from None
    if not isinstance(reference, dict):
        raise KcscAsrError(f"reference has an unexpected shape: {reference_path}")
    if reference.get("speaker") != entry.get("speaker"):
        raise KcscAsrError(f"{reference_path.name}: speaker does not match manifest")
    return reference_transcript(reference_path, entry)


def track_composition(reference_path: Path) -> dict[str, object]:
    """Transcript-free characterisation of one track's reference.

    Cross-speaker overlap is absent by construction here — that is the whole point of the
    comparison — so it is not reported. ``self_overlapped_speech_seconds`` is reported
    instead: it should be zero, and a non-zero value would mean the annotation contains
    rows this metric silently double-counts.
    """

    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    turns = reference["turns"]
    events = reference.get("unattributed_events", [])

    intervals = sorted((float(turn["start"]), float(turn["end"])) for turn in turns)
    speech = sum(end - start for start, end in intervals)
    overlap = 0.0
    for index, (_start, end) in enumerate(intervals):
        for other_start, other_end in intervals[index + 1 :]:
            if other_start >= end:
                break
            overlap += min(end, other_end) - other_start

    return {
        "turn_count": len(turns),
        "unattributed_event_count": len(events),
        "reference_speech_seconds": round(speech, 3),
        "self_overlapped_speech_seconds": round(overlap, 3),
        "rows_containing_overlap_marker": sum(
            1 for row in [*turns, *events] if "+" in str(row.get("transcript", ""))
        ),
        "rows_containing_tag": {
            tag: sum(1 for row in [*turns, *events] if tag in str(row.get("transcript", "")))
            for tag in CORPUS_TAGS
        },
    }


@dataclass(frozen=True, slots=True)
class TrackAsrScore:
    """Per-track ASR result. Rates and counts only, never text."""

    conversation_id: str
    speaker: str
    duration_seconds: float
    audio_sha256: str
    reference_sha256: str
    reference_row_count: int
    decode: DecodeStats
    elapsed_seconds: float
    real_time_factor: float
    character: ErrorCounts
    eojeol: ErrorCounts
    composition: Mapping[str, object]

    @property
    def track_id(self) -> str:
        return f"{self.conversation_id}_{self.speaker}"


def _pool(counts: Sequence[ErrorCounts]) -> ErrorCounts:
    merged = counts[0]
    for other in counts[1:]:
        merged = merged.merged(other)
    return merged


@dataclass(frozen=True, slots=True)
class TrackAsrBenchmarkReport:
    """Aggregate per-speaker ASR result. Carries no transcript."""

    model: ModelIdentity
    device: str
    compute_type: str
    decode_options: Mapping[str, object]
    source_revision: str
    manifest_sha256: str
    mixed_manifest_sha256: str
    scores: tuple[TrackAsrScore, ...]
    egress_attempts: int
    total_elapsed_seconds: float

    @property
    def character(self) -> ErrorCounts:
        return _pool([score.character for score in self.scores])

    @property
    def eojeol(self) -> ErrorCounts:
        return _pool([score.eojeol for score in self.scores])

    def conversation_character(self, conversation_id: str) -> ErrorCounts:
        """Pool a conversation's tracks, so it can be set beside the mixed figure."""

        counts = [
            score.character for score in self.scores if score.conversation_id == conversation_id
        ]
        if not counts:
            raise KcscAsrError(f"no scored tracks for conversation {conversation_id}")
        return _pool(counts)

    def conversation_eojeol(self, conversation_id: str) -> ErrorCounts:
        counts = [score.eojeol for score in self.scores if score.conversation_id == conversation_id]
        if not counts:
            raise KcscAsrError(f"no scored tracks for conversation {conversation_id}")
        return _pool(counts)


def score_track(
    entry: Mapping[str, object],
    *,
    derived_root: Path,
    model: WhisperModel,
    decode_options: Mapping[str, object],
) -> TrackAsrScore:
    """Verify checksums, transcribe one speaker track once, and score it."""

    audio_path, reference_path = verify_track_inputs(entry, derived_root=derived_root)
    duration = entry.get("derived_duration_seconds")
    if not isinstance(duration, int | float) or duration <= 0:
        raise KcscAsrError(f"{track_id(entry)}: manifest duration is unusable")

    reference_text, row_count = track_reference_transcript(reference_path, entry)

    started = time.monotonic()
    hypothesis_text, decode = decode_chronological(
        model, audio_path, float(duration), decode_options
    )
    elapsed = time.monotonic() - started

    character = score_sequences(characters(reference_text), characters(hypothesis_text))
    tokens = score_sequences(eojeol(reference_text), eojeol(hypothesis_text))
    # Both strings go out of scope here; nothing below this line can reach a report.

    outputs = entry["outputs"]
    assert isinstance(outputs, dict)
    return TrackAsrScore(
        conversation_id=str(entry["conversation_id"]),
        speaker=str(entry["speaker"]),
        duration_seconds=round(float(duration), 3),
        audio_sha256=str(outputs["audio_sha256"]),
        reference_sha256=str(outputs["reference_sha256"]),
        reference_row_count=row_count,
        decode=decode,
        elapsed_seconds=round(elapsed, 3),
        real_time_factor=round(elapsed / float(duration), 6),
        character=character,
        eojeol=tokens,
        composition=track_composition(reference_path),
    )


LIMITATIONS: tuple[str, ...] = (
    "This is an intrinsic upper bound under ideal speaker segmentation, not a clean-audio "
    "result. Each KCSC track is one participant's microphone in a shared room, so the "
    "other participant remains audible at a lower level; no acoustic source separation "
    "is performed.",
    "No segmentation or diarization error is modelled. The reference speaker assignment "
    "is taken as given, so the gap against the mixed benchmark bounds what perfect "
    "separation could recover and does not predict what any real diarizer would recover.",
    "The gap against the mixed benchmark is only subtractable because both runs score the "
    "same annotated rows with the same decoder: the track set is cut at the mixed set's "
    "trim offsets and its rows were verified against the mixed reference at derivation "
    "time. Comparing against a mixed report produced from a different derivation manifest "
    "or a different checkpoint would not be a like-for-like subtraction.",
    "Errors are NOT stratified by overlap, noise, or laughter. Doing so would require "
    "aligning character-level edits to time regions, which needs an alignment between the "
    "reference turns and the recogniser's own segmentation that this benchmark does not "
    "compute.",
    "The eojeol rate is secondary and conflates spacing with recognition: Korean word "
    "spacing is applied inconsistently by both the corpus annotators and the recogniser, "
    "so it should be read as an upper bound, not as a word error rate.",
    "Each track's reference includes the unattributed rows (laughter, coughs, ambient "
    "noise) that the corpus annotated on that track. Event tags normalise away to empty "
    "and contribute no reference characters.",
    "Three conversations, six speaker tracks, one language and recording condition. Not a "
    "basis for a general Korean ASR claim.",
    "Edit operations come from rapidfuzz, which is currently present transitively rather "
    "than as a declared dependency; declare it explicitly before relying on this in CI.",
)


def run_track_asr_benchmark(
    *,
    derived_root: Path,
    model: WhisperModel,
    identity: ModelIdentity,
    device: str,
    compute_type: str,
    decode_options: Mapping[str, object],
    conversation_ids: Sequence[str] = BENCHMARK_CONVERSATIONS,
    egress_attempts: int = 0,
) -> TrackAsrBenchmarkReport:
    """Verify every checksum first, then transcribe each speaker track exactly once."""

    if tuple(conversation_ids) != BENCHMARK_CONVERSATIONS:
        raise KcscAsrError("conversation selection does not match the approved scope")
    manifest = load_track_manifest(derived_root)
    entries = select_tracks(manifest, conversation_ids)
    for entry in entries:
        verify_track_inputs(entry, derived_root=derived_root)

    started = time.monotonic()
    scores = tuple(
        score_track(
            entry,
            derived_root=derived_root,
            model=model,
            decode_options=decode_options,
        )
        for entry in entries
    )
    total_elapsed = time.monotonic() - started

    source = manifest.get("source")
    revision = source.get("revision") if isinstance(source, dict) else None
    mixed = manifest.get("mixed_set")
    mixed_digest = mixed.get("manifest_sha256") if isinstance(mixed, dict) else None
    return TrackAsrBenchmarkReport(
        model=identity,
        device=device,
        compute_type=compute_type,
        decode_options=dict(decode_options),
        source_revision=str(revision),
        manifest_sha256=file_sha256(derived_root / "manifest.json"),
        mixed_manifest_sha256=str(mixed_digest),
        scores=scores,
        egress_attempts=egress_attempts,
        total_elapsed_seconds=round(total_elapsed, 3),
    )


def write_track_report(report: TrackAsrBenchmarkReport, path: Path) -> str:
    """Write the transcript-free aggregate report and return its digest."""

    total_audio = sum(score.duration_seconds for score in report.scores)
    conversation_ids: list[str] = []
    for score in report.scores:
        if score.conversation_id not in conversation_ids:
            conversation_ids.append(score.conversation_id)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "kcsc-asr-track",
        "transcript_free": True,
        "task": "per-speaker-track-asr-upper-bound",
        "task_note": (
            "Each speaker's own microphone track decoded and scored separately against "
            "that speaker's reference rows. No diarization runs and no segmentation error "
            "is modelled: this is the intrinsic recogniser error under ideal speaker "
            "segmentation, to be subtracted from the mixed-conversation figure."
        ),
        "model": {
            "name": "faster-whisper large-v3-turbo",
            "repo_id": report.model.repo_id,
            "revision": report.model.revision,
            "tree_sha256": report.model.tree_sha256,
            "total_bytes": report.model.total_bytes,
            "files": [dict(entry) for entry in report.model.files],
            "device": report.device,
            "compute_type": report.compute_type,
            "decode_options": dict(report.decode_options),
            "remote": False,
        },
        "dataset": {
            "name": "kcsc-derived-speaker-track-set",
            "source_revision": report.source_revision,
            "derivation_manifest_sha256": report.manifest_sha256,
            "mixed_set_manifest_sha256": report.mixed_manifest_sha256,
            "conversation_count": len(conversation_ids),
            "conversation_ids": conversation_ids,
            "track_count": len(report.scores),
            "track_ids": [score.track_id for score in report.scores],
        },
        "scoring": {
            "primary_metric": "korean_normalized_character_error_rate",
            "primary_metric_note": (
                "Levenshtein over NFC-normalised Korean syllables with whitespace removed; "
                "CER = (substitutions + deletions + insertions) / reference characters."
            ),
            "secondary_metric": "eojeol_error_rate",
            "secondary_metric_note": (
                "Whitespace-delimited tokens. Secondary only: Korean spacing is applied "
                "inconsistently by annotators and recogniser alike, so this conflates "
                "spacing with recognition and reads as an upper bound."
            ),
            "normalization_steps": list(NORMALIZATION_STEPS),
            "aggregation": "pooled edit operations over pooled reference length",
            "implementation": "rapidfuzz.distance.Levenshtein.editops",
        },
        "offline": {
            "network_egress_attempts": report.egress_attempts,
            "hub_offline": True,
        },
        "timing": {
            "total_elapsed_seconds": report.total_elapsed_seconds,
            "total_audio_seconds": round(total_audio, 3),
            "real_time_factor": round(report.total_elapsed_seconds / total_audio, 6),
        },
        "aggregate": {
            "character": report.character.as_dict(),
            "eojeol": report.eojeol.as_dict(),
        },
        "conversations": [
            {
                "conversation_id": conversation_id,
                "character": report.conversation_character(conversation_id).as_dict(),
                "eojeol": report.conversation_eojeol(conversation_id).as_dict(),
            }
            for conversation_id in conversation_ids
        ],
        "tracks": [
            {
                "track_id": score.track_id,
                "conversation_id": score.conversation_id,
                "speaker": score.speaker,
                "duration_seconds": score.duration_seconds,
                "audio_sha256": score.audio_sha256,
                "reference_sha256": score.reference_sha256,
                "reference_row_count": score.reference_row_count,
                "decode": score.decode.as_dict(),
                "elapsed_seconds": score.elapsed_seconds,
                "real_time_factor": score.real_time_factor,
                "character": score.character.as_dict(),
                "eojeol": score.eojeol.as_dict(),
                "reference_composition": dict(score.composition),
            }
            for score in report.scores
        ],
        "limitations": list(LIMITATIONS),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    path.write_text(f"{raw}\n", encoding="utf-8")
    return file_sha256(path)


__all__ = [
    "BENCHMARK_CONVERSATIONS",
    "LIMITATIONS",
    "SCHEMA_VERSION",
    "TRACKS_PER_CONVERSATION",
    "TrackAsrBenchmarkReport",
    "TrackAsrScore",
    "load_track_manifest",
    "run_track_asr_benchmark",
    "score_track",
    "select_tracks",
    "track_composition",
    "track_id",
    "track_reference_transcript",
    "verify_track_inputs",
    "write_track_report",
]
