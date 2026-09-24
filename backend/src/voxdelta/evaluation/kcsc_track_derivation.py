"""Derive isolated per-speaker audio from the raw KCSC pairs, on the mixed timeline.

The mixed evaluation set answers "what does the recogniser hear when both speakers share
one channel". To separate *recogniser* error from *diarization* error we also need the
other half of that comparison: what the same recogniser achieves when a speaker's own
microphone track is handed to it directly, with no diarization in the path at all. That
is the intrinsic upper bound this module prepares the inputs for.

The derivation is deliberately the *same* derivation, not a similar one:

* **One timeline.** Each track is trimmed at the conversation-level offset the mixed set
  used, so a per-track second and a mixed second are the same second. A separate
  per-track trim would produce a cheaper-looking upper bound measured against a reference
  that had quietly moved.
* **The same vendor treatment.** Every vendor-marker interval in the conversation is
  zeroed on every track, not merely the markers annotated on that track. The two
  microphones record one room, so the other speaker's marker bleeds into this track's
  audio; silencing only the local ones would leave audible branding for the recogniser to
  transcribe against a reference that never contained it.
* **The same reference rows.** The per-track reference is re-derived from the raw
  annotations and then required to match, row for row, the subset of the already-derived
  mixed reference belonging to that speaker. A disagreement aborts the run: it would mean
  the two benchmarks were scoring different ground truth and any difference between them
  could be an artifact of the derivation rather than of diarization.

What this does **not** do is separate the speakers acoustically. A KCSC track is one
participant's microphone in a shared room, so the other participant remains audible at a
lower level. The result is therefore an upper bound on per-speaker ASR quality given
*ideal* speaker segmentation, not a measurement of clean single-speaker audio.

The raw tree is opened for reading only. Transcript text is written into the per-track
reference files because scoring needs it, and never appears in a summary object.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voxdelta.evaluation.kcsc_derivation import (
    EXPECTED_CHANNELS,
    EXPECTED_SAMPLE_RATE,
    EXPECTED_SAMPLE_WIDTH,
    AnnotationRow,
    ConversationSource,
    KcscDerivationError,
    SourceProvenance,
    discover_conversations,
    limit_to_int16,
    parse_annotations,
    read_provenance,
    read_track_audio,
    sha256_file,
    shift_rows,
    silence_intervals,
    write_json,
    write_wav,
)

SCHEMA_VERSION = "1"

#: The approved fixed selection, frozen here so a mistyped argument cannot widen the run.
BENCHMARK_CONVERSATIONS: tuple[str, ...] = (
    "A0051_S0001_0",
    "A0055_S0006_0",
    "A6000_S0005_0",
)

#: Written beside the derived tracks. The derived tree is Git-ignored, so a reader who
#: copies the directory finds the constraints only here.
USAGE_README = """# KCSC derived per-speaker track set (local only)

Derived by `backend/scripts/derive_kcsc_track_set.py` from the pinned local copy of the
MagicHub Korean Conversational Speech Corpus, on the same trimmed timeline as the mixed
conversation set in `../kcsc`. Regenerate with that script; do not edit by hand.

Each track is one participant's own microphone recording, trimmed at the conversation's
trim offset and with every vendor-marker interval in the conversation zeroed.
`reference/<conversation>_<speaker>.json` holds that speaker's own annotated turns, and
each has been verified row for row against the corresponding subset of the mixed set's
reference.

## What this is for

Measuring the ASR intrinsic upper bound: recognition quality given ideal speaker
segmentation. It is NOT clean single-speaker audio — the two microphones record one
room, so the other participant remains audible at a lower level.

## Permitted use

- Research-only, offline evaluation of VoxDelta on this machine.

## Not permitted

- Training, fine-tuning, or any other model fitting on this data.
- Use in a deployed or production workflow.
- Redistribution or republication, in original or derived form.
- Upload to pyannoteAI or any other third-party service. This stays blocked until
  third-party processing rights for this corpus are confirmed with the copyright
  holder, Beijing Magic Data Technology Co., Ltd.

## Handling

Reference files under `reference/` contain verbatim Korean transcript, kept locally
because ASR scoring needs it. Do not paste it into shared logs, issues, or tickets.
"""


@dataclass(frozen=True, slots=True)
class DerivedTrack:
    """Audit record for one derived speaker track. Carries no transcript text."""

    conversation_id: str
    speaker: str
    sample_rate: int
    source_duration_seconds: float
    derived_duration_seconds: float
    trim_offset_seconds: float
    gain: float
    turn_count: int
    unattributed_event_count: int
    silenced_intervals: tuple[tuple[float, float], ...]
    source_audio: str
    source_audio_sha256: str
    source_annotation: str
    source_annotation_sha256: str
    mixed_reference_sha256: str
    audio_path: Path
    audio_sha256: str
    reference_path: Path
    reference_sha256: str


@dataclass(frozen=True, slots=True)
class TrackDerivationSummary:
    """Aggregate audit record for a track derivation run. Carries no transcript text."""

    provenance: SourceProvenance
    manifest_path: Path
    manifest_sha256: str
    tracks: tuple[DerivedTrack, ...]

    @property
    def track_count(self) -> int:
        return len(self.tracks)

    @property
    def turn_count(self) -> int:
        return sum(track.turn_count for track in self.tracks)

    @property
    def total_duration_seconds(self) -> float:
        return sum(track.derived_duration_seconds for track in self.tracks)


def _mixed_entry(mixed_root: Path, conversation_id: str) -> Mapping[str, object]:
    """Read one conversation's entry from the mixed derivation manifest."""

    path = mixed_root / "manifest.json"
    if not path.is_file():
        raise KcscDerivationError(f"missing mixed derivation manifest: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise KcscDerivationError(f"mixed derivation manifest is not valid JSON: {path}") from None
    entries = manifest.get("conversations") if isinstance(manifest, dict) else None
    if not isinstance(entries, list):
        raise KcscDerivationError(f"mixed derivation manifest has an unexpected shape: {path}")
    for entry in entries:
        if isinstance(entry, dict) and entry.get("conversation_id") == conversation_id:
            return entry
    raise KcscDerivationError(f"conversation {conversation_id} is absent from {path}")


def _mixed_reference(
    mixed_root: Path, entry: Mapping[str, object]
) -> tuple[Mapping[str, object], str]:
    """Read and checksum-verify the mixed reference this derivation is tied to.

    Returns the reference and its verified digest, so the caller records the digest it
    actually checked rather than re-reading the claim out of the manifest.
    """

    outputs = entry.get("outputs")
    if not isinstance(outputs, dict):
        raise KcscDerivationError(f"{entry.get('conversation_id')}: manifest entry has no outputs")
    relative, expected = outputs.get("reference"), outputs.get("reference_sha256")
    if not isinstance(relative, str) or not isinstance(expected, str):
        raise KcscDerivationError(
            f"{entry.get('conversation_id')}: manifest reference is malformed"
        )
    path = mixed_root / relative
    if not path.is_file():
        raise KcscDerivationError(f"missing mixed reference: {path}")
    found = sha256_file(path)
    if found != expected:
        raise KcscDerivationError(
            f"{entry.get('conversation_id')}: mixed reference checksum mismatch "
            f"(manifest {expected[:12]}…, file {found[:12]}…)"
        )
    reference = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(reference, dict):
        raise KcscDerivationError(f"mixed reference has an unexpected shape: {path}")
    return reference, expected


def _row_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (row["start"], row["end"], row["speaker"], row["gender"], row["transcript"])


def cross_check_against_mixed(
    turns: Sequence[Mapping[str, object]],
    mixed_reference: Mapping[str, object],
    *,
    conversation_id: str,
    speaker: str,
) -> None:
    """Require the re-derived per-track turns to equal the mixed set's rows for that speaker.

    Comparison is on the full row — timings, speaker, gender, and transcript — because a
    match on timings alone would not catch a reference that had drifted in content. The
    keys are compared, never reported: a mismatch names counts and the conversation, not
    the rows that differed.
    """

    mixed_turns = mixed_reference.get("turns")
    if not isinstance(mixed_turns, list):
        raise KcscDerivationError(f"{conversation_id}: mixed reference has no turns")
    expected = sorted(
        _row_key(row)
        for row in mixed_turns
        if isinstance(row, dict) and row.get("speaker") == speaker
    )
    found = sorted(_row_key(row) for row in turns)
    if found != expected:
        raise KcscDerivationError(
            f"{conversation_id}/{speaker}: re-derived turns disagree with the mixed reference "
            f"({len(found)} rows derived, {len(expected)} rows in the mixed set)"
        )


def conversation_trim_samples(annotations: Sequence[Sequence[AnnotationRow]], length: int) -> int:
    """Recompute the mixed derivation's trim boundary from both tracks' annotations."""

    speech = [
        row for rows in annotations for row in rows if not row.vendor_marker and row.attributed
    ]
    if not speech:
        raise KcscDerivationError("conversation has no non-vendor speech rows")
    trim = min(max(0, int(np.floor(row.start * EXPECTED_SAMPLE_RATE))) for row in speech)
    return min(trim, length)


def derive_conversation_tracks(
    conversation: ConversationSource,
    *,
    mixed_root: Path,
    output_root: Path,
    provenance: SourceProvenance,
) -> tuple[DerivedTrack, ...]:
    """Derive both speaker tracks of one conversation onto the mixed timeline."""

    conversation_id = conversation.conversation_id
    entry = _mixed_entry(mixed_root, conversation_id)
    mixed_reference, mixed_reference_sha256 = _mixed_reference(mixed_root, entry)

    annotations = tuple(
        parse_annotations(track.annotation_path, expected_speaker=track.speaker)
        for track in conversation.tracks
    )
    audio = [read_track_audio(track.audio_path) for track in conversation.tracks]

    # The mixed set padded the shorter track up to the longer one before trimming, so the
    # trim boundary is a property of the conversation, not of either track.
    length = max(samples.size for samples in audio)
    trim_samples = conversation_trim_samples(annotations, length)
    trim_offset = trim_samples / EXPECTED_SAMPLE_RATE

    declared = entry.get("trim_offset_seconds")
    if not isinstance(declared, int | float) or abs(float(declared) - trim_offset) > 1e-9:
        raise KcscDerivationError(
            f"{conversation_id}: recomputed trim offset {trim_offset:.6f}s disagrees with the "
            f"mixed manifest ({declared!r})"
        )

    # Every marker in the conversation, on every track: the other microphone hears it too.
    markers = sorted(
        (row.start - trim_offset, row.end - trim_offset)
        for rows in annotations
        for row in rows
        if row.vendor_marker
    )

    audio_root = output_root / "audio"
    reference_root = output_root / "reference"
    audio_root.mkdir(parents=True, exist_ok=True)
    reference_root.mkdir(parents=True, exist_ok=True)

    derived: list[DerivedTrack] = []
    for track, rows, samples in zip(conversation.tracks, annotations, audio, strict=True):
        stem = f"{conversation_id}_{track.speaker}"
        source_duration = samples.size / EXPECTED_SAMPLE_RATE
        trimmed = samples[trim_samples:].copy()
        if trimmed.size == 0:
            raise KcscDerivationError(f"{stem}: track is empty after trimming")

        silenced = silence_intervals(trimmed, markers, sample_rate=EXPECTED_SAMPLE_RATE)
        limited, gain = limit_to_int16(trimmed)

        kept = [row for row in rows if not row.vendor_marker]
        turns = shift_rows([row for row in kept if row.attributed], trim_offset)
        events = shift_rows([row for row in kept if not row.attributed], trim_offset)
        if not turns:
            raise KcscDerivationError(f"{stem}: track has no scorable turns")
        cross_check_against_mixed(
            turns, mixed_reference, conversation_id=conversation_id, speaker=track.speaker
        )

        audio_path = audio_root / f"{stem}.wav"
        reference_path = reference_root / f"{stem}.json"
        write_wav(audio_path, limited, sample_rate=EXPECTED_SAMPLE_RATE)
        derived_duration = limited.size / EXPECTED_SAMPLE_RATE
        reference_sha256 = write_json(
            reference_path,
            {
                "schema_version": SCHEMA_VERSION,
                "conversation_id": conversation_id,
                "speaker": track.speaker,
                "source_dataset": provenance.dataset,
                "source_revision": provenance.revision,
                "sample_rate": EXPECTED_SAMPLE_RATE,
                "duration_seconds": round(derived_duration, 6),
                "trim_offset_seconds": round(trim_offset, 6),
                "vendor_marker_intervals_silenced": [list(interval) for interval in silenced],
                "turns": turns,
                "unattributed_events": events,
            },
        )

        derived.append(
            DerivedTrack(
                conversation_id=conversation_id,
                speaker=track.speaker,
                sample_rate=EXPECTED_SAMPLE_RATE,
                source_duration_seconds=round(source_duration, 6),
                derived_duration_seconds=round(derived_duration, 6),
                trim_offset_seconds=round(trim_offset, 6),
                gain=gain,
                turn_count=len(turns),
                unattributed_event_count=len(events),
                silenced_intervals=silenced,
                source_audio=track.audio_path.name,
                source_audio_sha256=sha256_file(track.audio_path),
                source_annotation=track.annotation_path.name,
                source_annotation_sha256=sha256_file(track.annotation_path),
                mixed_reference_sha256=mixed_reference_sha256,
                audio_path=audio_path,
                audio_sha256=sha256_file(audio_path),
                reference_path=reference_path,
                reference_sha256=reference_sha256,
            )
        )
    return tuple(derived)


def _manifest_entry(track: DerivedTrack, *, output_root: Path) -> dict[str, object]:
    return {
        "conversation_id": track.conversation_id,
        "speaker": track.speaker,
        "track_id": f"{track.conversation_id}_{track.speaker}",
        "sample_rate": track.sample_rate,
        "source_duration_seconds": track.source_duration_seconds,
        "derived_duration_seconds": track.derived_duration_seconds,
        "trim_offset_seconds": track.trim_offset_seconds,
        "gain": track.gain,
        "turn_count": track.turn_count,
        "unattributed_event_count": track.unattributed_event_count,
        "vendor_marker_intervals_silenced": [
            list(interval) for interval in track.silenced_intervals
        ],
        "source": {
            "audio": track.source_audio,
            "audio_sha256": track.source_audio_sha256,
            "annotation": track.source_annotation,
            "annotation_sha256": track.source_annotation_sha256,
        },
        "mixed_reference_sha256": track.mixed_reference_sha256,
        "outputs": {
            "audio": track.audio_path.relative_to(output_root).as_posix(),
            "audio_sha256": track.audio_sha256,
            "audio_bytes": track.audio_path.stat().st_size,
            "reference": track.reference_path.relative_to(output_root).as_posix(),
            "reference_sha256": track.reference_sha256,
        },
    }


def derive_track_set(
    source_root: Path,
    mixed_root: Path,
    output_root: Path,
    *,
    conversation_ids: Sequence[str] = BENCHMARK_CONVERSATIONS,
) -> TrackDerivationSummary:
    """Derive both speaker tracks for each selected conversation.

    ``source_root`` and ``mixed_root`` are only ever read from. The manifest is written
    last so a manifest on disk always describes outputs that were fully written.
    """

    if tuple(conversation_ids) != BENCHMARK_CONVERSATIONS:
        raise KcscDerivationError("conversation selection does not match the approved scope")

    provenance = read_provenance(source_root)
    available = {
        conversation.conversation_id: conversation
        for conversation in discover_conversations(source_root)
    }
    missing = [name for name in conversation_ids if name not in available]
    if missing:
        raise KcscDerivationError(f"unknown conversation ids: {', '.join(sorted(missing))}")

    output_root.mkdir(parents=True, exist_ok=True)
    tracks: list[DerivedTrack] = []
    for conversation_id in conversation_ids:
        tracks.extend(
            derive_conversation_tracks(
                available[conversation_id],
                mixed_root=mixed_root,
                output_root=output_root,
                provenance=provenance,
            )
        )

    (output_root / "README.md").write_text(USAGE_README, encoding="utf-8")
    manifest_path = output_root / "manifest.json"
    manifest_sha256 = write_json(
        manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "dataset": "kcsc-derived-speaker-track-set",
            "purpose": (
                "Isolated per-speaker audio on the mixed set's timeline, for measuring the "
                "ASR intrinsic upper bound under ideal speaker segmentation."
            ),
            "source": {
                "dataset": provenance.dataset,
                "revision": provenance.revision,
                "local_tree_sha256": provenance.local_tree_sha256,
                "provenance_sha256": provenance.provenance_sha256,
            },
            "mixed_set": {
                "root": mixed_root.as_posix(),
                "manifest_sha256": sha256_file(mixed_root / "manifest.json"),
            },
            "usage": {
                "scope": "local research evaluation only",
                "training": "prohibited",
                "deployment": "prohibited",
                "redistribution": "prohibited",
                "third_party_upload": "prohibited pending confirmed processing rights",
            },
            "acoustic_note": (
                "Each track is one participant's microphone in a shared room. The other "
                "participant remains audible at a lower level; this is not source-separated "
                "single-speaker audio."
            ),
            "sample_rate": EXPECTED_SAMPLE_RATE,
            "channels": EXPECTED_CHANNELS,
            "sample_width_bytes": EXPECTED_SAMPLE_WIDTH,
            "conversation_ids": list(conversation_ids),
            "track_count": len(tracks),
            "turn_count": sum(track.turn_count for track in tracks),
            "tracks": [_manifest_entry(track, output_root=output_root) for track in tracks],
        },
    )

    return TrackDerivationSummary(
        provenance=provenance,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        tracks=tuple(tracks),
    )


__all__ = [
    "BENCHMARK_CONVERSATIONS",
    "SCHEMA_VERSION",
    "USAGE_README",
    "DerivedTrack",
    "TrackDerivationSummary",
    "conversation_trim_samples",
    "cross_check_against_mixed",
    "derive_conversation_tracks",
    "derive_track_set",
]
