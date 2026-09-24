"""Derive a local-only, auditable evaluation set from the vendor KCSC corpus.

The upstream corpus ships one WAV and one annotation file per *speaker track*: two
tracks recorded simultaneously make up one conversation. Diarization and ASR are
evaluated on a conversation, not on a track, so the evaluation set is derived by
mixing each pair down to a single mono channel.

Three properties matter more than convenience here and are therefore enforced rather
than assumed:

* **The raw tree is read-only.** Every function in this module opens source files for
  reading and writes only under the caller-supplied output root.
* **Vendor branding is removed from the derived audio.** The corpus opens (and
  occasionally closes) each track with a spoken vendor marker. Left in place it would
  be transcribed as speech and scored against a reference that does not contain it, so
  the derived audio is trimmed to the first real utterance and every surviving marker
  interval is zeroed.
* **Failures are loud.** A missing pair member, an unexpected audio format, or an
  unparsable annotation row aborts the whole derivation instead of silently producing
  a partial evaluation set that would quietly bias later measurements.

Rows the corpus leaves unattributed (laughter, coughs, ambient noise) are preserved in
a separate list rather than promoted to a third speaker or discarded.

Transcript text is written into the per-conversation reference files because ASR
scoring needs it. It is never returned in a summary object and never printed: the
public surface of this module carries identifiers, timings, counts, and checksums only.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import wave
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

SCHEMA_VERSION = "1"
EXPECTED_SAMPLE_RATE = 16_000
EXPECTED_CHANNELS = 1
EXPECTED_SAMPLE_WIDTH = 2
TRACKS_PER_CONVERSATION = 2
INT16_PEAK = 32_767

#: Both spellings the vendor uses for its spoken corpus marker. Comparison is done on
#: an NFC-normalised, whitespace-free form so "매직데이터" matches "매직 데이터".
VENDOR_MARKERS: tuple[str, ...] = ("매직 데이터", "매직 데이타")

#: The corpus writes this literal in the speaker column, with gender ``none``, for events
#: it does not attribute to either participant — laughter, coughs, ambient noise, and
#: unintelligible fragments. Such rows are real annotations and are preserved, but they
#: are kept out of the scored turn list: a diarization reference containing them would
#: assert a third speaker that no track ever records.
UNATTRIBUTED_SPEAKER = "0"

#: Written beside the derived outputs so the constraints travel with the files rather
#: than living only in a commit message. The derived tree is Git-ignored, so this is the
#: only place a reader who copies the directory will find them.
USAGE_README = """# KCSC derived evaluation set (local only)

Derived by `backend/scripts/derive_kcsc_eval_set.py` from the pinned local copy of the
MagicHub Korean Conversational Speech Corpus. Regenerate with that script; do not edit
these files by hand. `manifest.json` records the source revision, the source file names
and digests, and a checksum for every output.

Each `reference/<conversation>.json` holds `turns` (speaker-attributed speech, already
shifted onto the trimmed timeline, vendor-marker rows removed) and `unattributed_events`
(the corpus's own unattributed laughter/noise rows, kept out of `turns` so they cannot
be scored as a third speaker).

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

_TRACK_STEM = re.compile(r"^(?P<conversation>.+)_(?P<speaker>G\d+)$")
_INTERVAL = re.compile(r"^\[\s*(?P<start>-?\d+(?:\.\d+)?)\s*,\s*(?P<end>-?\d+(?:\.\d+)?)\s*\]$")
_WHITESPACE = re.compile(r"\s+")
_REVISION = re.compile(r"\*\*Immutable revision:\*\*\s*`(?P<value>[0-9a-f]{40})`")
_TREE_DIGEST = re.compile(r"\*\*Local tree SHA-256:\*\*\s*`(?P<value>[0-9a-f]{64})`")
_SOURCE_DATASET = re.compile(r"\*\*Source:\*\*\s*`(?P<value>[^`]+)`")


class KcscDerivationError(RuntimeError):
    """Raised for any condition that would yield an untrustworthy evaluation set."""


def _canonical_marker(text: str) -> str:
    return _WHITESPACE.sub("", unicodedata.normalize("NFC", text))


_CANONICAL_VENDOR_MARKERS = frozenset(_canonical_marker(marker) for marker in VENDOR_MARKERS)


def is_vendor_marker(transcript: str) -> bool:
    """Report whether an annotation row carries the vendor's spoken corpus marker."""

    return _canonical_marker(transcript) in _CANONICAL_VENDOR_MARKERS


@dataclass(frozen=True, slots=True)
class AnnotationRow:
    """One annotated turn on a single speaker track, in source-file timing."""

    start: float
    end: float
    speaker: str
    gender: str
    transcript: str
    vendor_marker: bool
    attributed: bool


@dataclass(frozen=True, slots=True)
class TrackSource:
    """The audio and annotation pair that make up one speaker track."""

    speaker: str
    audio_path: Path
    annotation_path: Path


@dataclass(frozen=True, slots=True)
class ConversationSource:
    """The two speaker tracks that make up one conversation."""

    conversation_id: str
    tracks: tuple[TrackSource, ...]


@dataclass(frozen=True, slots=True)
class SourceProvenance:
    """The immutable identity of the raw tree the evaluation set was derived from."""

    dataset: str
    revision: str
    local_tree_sha256: str
    provenance_sha256: str


@dataclass(frozen=True, slots=True)
class DerivedConversation:
    """Audit record for one derived conversation. Carries no transcript text."""

    conversation_id: str
    speakers: tuple[str, ...]
    sample_rate: int
    source_duration_seconds: float
    derived_duration_seconds: float
    trim_offset_seconds: float
    padded_samples: tuple[int, ...]
    mix_gain: float
    turn_count: int
    unattributed_event_count: int
    vendor_marker_count: int
    silenced_intervals: tuple[tuple[float, float], ...]
    source_files: tuple[Mapping[str, object], ...]
    audio_path: Path
    audio_sha256: str
    reference_path: Path
    reference_sha256: str


@dataclass(frozen=True, slots=True)
class DerivationSummary:
    """Aggregate audit record for a full derivation run. Carries no transcript text."""

    provenance: SourceProvenance
    manifest_path: Path
    manifest_sha256: str
    conversations: tuple[DerivedConversation, ...]

    @property
    def conversation_count(self) -> int:
        return len(self.conversations)

    @property
    def turn_count(self) -> int:
        return sum(conversation.turn_count for conversation in self.conversations)

    @property
    def unattributed_event_count(self) -> int:
        return sum(conversation.unattributed_event_count for conversation in self.conversations)

    @property
    def vendor_marker_count(self) -> int:
        return sum(conversation.vendor_marker_count for conversation in self.conversations)

    @property
    def silenced_interval_count(self) -> int:
        return sum(len(conversation.silenced_intervals) for conversation in self.conversations)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_provenance(source_root: Path) -> SourceProvenance:
    """Read the pinned upstream identity, failing closed when it is absent."""

    path = source_root / "PROVENANCE.md"
    if not path.is_file():
        raise KcscDerivationError(f"missing provenance record: {path}")
    text = path.read_text(encoding="utf-8")
    revision = _REVISION.search(text)
    tree_digest = _TREE_DIGEST.search(text)
    dataset = _SOURCE_DATASET.search(text)
    if revision is None or tree_digest is None or dataset is None:
        raise KcscDerivationError(
            f"provenance record is missing source, revision, or digest: {path}"
        )
    return SourceProvenance(
        dataset=dataset.group("value"),
        revision=revision.group("value"),
        local_tree_sha256=tree_digest.group("value"),
        provenance_sha256=sha256_file(path),
    )


def discover_conversations(source_root: Path) -> tuple[ConversationSource, ...]:
    """Group speaker-track WAVs into conversations by stripping the ``_G<speaker>`` suffix.

    Every conversation must contribute exactly two tracks, each with a matching
    annotation file; anything else aborts the run rather than deriving a partial set.
    """

    audio_root = source_root / "WAV"
    annotation_root = source_root / "TXT"
    for directory in (audio_root, annotation_root):
        if not directory.is_dir():
            raise KcscDerivationError(f"missing source directory: {directory}")

    grouped: dict[str, list[TrackSource]] = {}
    for audio_path in sorted(audio_root.glob("*.wav")):
        match = _TRACK_STEM.match(audio_path.stem)
        if match is None:
            raise KcscDerivationError(f"unrecognised track filename: {audio_path.name}")
        annotation_path = annotation_root / f"{audio_path.stem}.txt"
        if not annotation_path.is_file():
            raise KcscDerivationError(f"missing annotation file for track: {audio_path.name}")
        grouped.setdefault(match.group("conversation"), []).append(
            TrackSource(
                speaker=match.group("speaker"),
                audio_path=audio_path,
                annotation_path=annotation_path,
            )
        )

    if not grouped:
        raise KcscDerivationError(f"no speaker tracks found under {audio_root}")

    conversations: list[ConversationSource] = []
    for conversation_id, tracks in sorted(grouped.items()):
        if len(tracks) != TRACKS_PER_CONVERSATION:
            raise KcscDerivationError(
                f"conversation {conversation_id} has {len(tracks)} tracks, "
                f"expected {TRACKS_PER_CONVERSATION}"
            )
        ordered = tuple(sorted(tracks, key=lambda track: track.speaker))
        if ordered[0].speaker == ordered[1].speaker:
            raise KcscDerivationError(
                f"conversation {conversation_id} has duplicate speaker {ordered[0].speaker}"
            )
        conversations.append(ConversationSource(conversation_id=conversation_id, tracks=ordered))
    return tuple(conversations)


def parse_annotations(path: Path, *, expected_speaker: str) -> tuple[AnnotationRow, ...]:
    """Parse one ``[start,end]<TAB>speaker<TAB>gender<TAB>transcript`` annotation file.

    The speaker column must hold either this track's speaker or the corpus's
    unattributed placeholder; any third identity means the file does not describe the
    track it is named for, and the run aborts.

    Diagnostics quote the file and line number but never the row's text, so a parse
    failure cannot leak transcript into a log.
    """

    rows: list[AnnotationRow] = []
    text = path.read_text(encoding="utf-8")
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 4:
            raise KcscDerivationError(
                f"{path.name}:{number}: expected 4 tab-separated fields, found {len(fields)}"
            )
        interval, speaker, gender, transcript = fields
        match = _INTERVAL.match(interval.strip())
        if match is None:
            raise KcscDerivationError(f"{path.name}:{number}: malformed interval field")
        start = float(match.group("start"))
        end = float(match.group("end"))
        if start < 0.0 or end < start:
            raise KcscDerivationError(f"{path.name}:{number}: interval is negative or inverted")
        found = speaker.strip()
        if found not in (expected_speaker, UNATTRIBUTED_SPEAKER):
            raise KcscDerivationError(
                f"{path.name}:{number}: speaker field is neither track speaker "
                f"{expected_speaker} nor the unattributed placeholder"
            )
        rows.append(
            AnnotationRow(
                start=start,
                end=end,
                speaker=found,
                gender=gender.strip(),
                transcript=transcript.strip(),
                vendor_marker=is_vendor_marker(transcript),
                attributed=found != UNATTRIBUTED_SPEAKER,
            )
        )
    if not rows:
        raise KcscDerivationError(f"{path.name}: annotation file contains no rows")
    return tuple(rows)


def read_track_audio(path: Path) -> NDArray[np.int32]:
    """Read one mono 16 kHz 16-bit PCM track, failing closed on any other format."""

    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if channels != EXPECTED_CHANNELS:
        raise KcscDerivationError(f"{path.name}: expected mono audio, found {channels} channels")
    if sample_width != EXPECTED_SAMPLE_WIDTH:
        raise KcscDerivationError(
            f"{path.name}: expected 16-bit PCM, found {sample_width * 8}-bit samples"
        )
    if sample_rate != EXPECTED_SAMPLE_RATE:
        raise KcscDerivationError(
            f"{path.name}: expected {EXPECTED_SAMPLE_RATE} Hz, found {sample_rate} Hz"
        )
    return np.frombuffer(frames, dtype="<i2").astype(np.int32)


def mix_tracks(tracks: Sequence[NDArray[np.int32]]) -> tuple[NDArray[np.int32], tuple[int, ...]]:
    """Sum zero-aligned tracks into one channel, padding shorter tracks with silence.

    Both tracks of a KCSC conversation are recordings of the same session that begin at
    the same instant, so mixing is a plain sample-wise sum at a common origin. The sum
    is kept in ``int32`` — the caller normalises it back into ``int16`` range after
    trimming and silencing, so headroom is not spent on audio that gets discarded.
    """

    if len(tracks) != TRACKS_PER_CONVERSATION:
        raise KcscDerivationError(f"expected {TRACKS_PER_CONVERSATION} tracks to mix")
    length = max(track.size for track in tracks)
    if length == 0:
        raise KcscDerivationError("cannot mix empty tracks")
    mixed = np.zeros(length, dtype=np.int32)
    padded: list[int] = []
    for track in tracks:
        mixed[: track.size] += track
        padded.append(length - track.size)
    return mixed, tuple(padded)


def limit_to_int16(mixed: NDArray[np.int32]) -> tuple[NDArray[np.int16], float]:
    """Scale a summed signal into ``int16`` range with a single uniform gain.

    A sum of two full-scale tracks can exceed ``int16``. Hard-clipping the overshoot
    would fabricate harmonics exactly where both speakers are loudest — the overlapping
    speech that diarization is most likely to be scored on. One global gain instead
    preserves the waveform shape and every relative level in the file.
    """

    peak = int(np.max(np.abs(mixed))) if mixed.size else 0
    gain = 1.0 if peak <= INT16_PEAK else INT16_PEAK / peak
    scaled = np.rint(mixed.astype(np.float64) * gain) if gain != 1.0 else mixed.astype(np.float64)
    return np.clip(scaled, -INT16_PEAK, INT16_PEAK).astype(np.int16), gain


def silence_intervals(
    audio: NDArray[np.int32],
    intervals: Iterable[tuple[float, float]],
    *,
    sample_rate: int,
) -> tuple[tuple[float, float], ...]:
    """Zero every sample covered by ``intervals``, in place, returning what was applied.

    Bounds are widened outward (floor of the start, ceil of the end) so a marker is
    never left with an audible fragment, and clamped to the buffer so a terminal marker
    that runs past the last annotated sample is still removed.
    """

    applied: list[tuple[float, float]] = []
    for start, end in intervals:
        first = max(0, int(np.floor(start * sample_rate)))
        last = min(audio.size, int(np.ceil(end * sample_rate)))
        if last <= first:
            continue
        audio[first:last] = 0
        applied.append((round(first / sample_rate, 6), round(last / sample_rate, 6)))
    return tuple(applied)


def write_wav(path: Path, samples: NDArray[np.int16], *, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(EXPECTED_CHANNELS)
        handle.setsampwidth(EXPECTED_SAMPLE_WIDTH)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.astype("<i2").tobytes())


def write_json(path: Path, payload: object) -> str:
    raw = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    path.write_text(f"{raw}\n", encoding="utf-8")
    return sha256_file(path)


def shift_rows(rows: Sequence[AnnotationRow], trim_offset: float) -> list[dict[str, object]]:
    """Rebase rows onto the trimmed timeline, dropping anything it no longer covers."""

    shifted: list[dict[str, object]] = []
    for row in rows:
        end = row.end - trim_offset
        if end <= 0.0:
            continue
        shifted.append(
            {
                "start": round(max(0.0, row.start - trim_offset), 6),
                "end": round(end, 6),
                "speaker": row.speaker,
                "gender": row.gender,
                "transcript": row.transcript,
            }
        )
    return sorted(shifted, key=lambda row: (row["start"], row["speaker"], row["end"]))


def derive_conversation(
    conversation: ConversationSource,
    *,
    output_root: Path,
    provenance: SourceProvenance,
) -> DerivedConversation:
    """Derive one mixed, trimmed, vendor-silenced conversation and its reference file."""

    annotations = tuple(
        parse_annotations(track.annotation_path, expected_speaker=track.speaker)
        for track in conversation.tracks
    )
    kept = [row for rows in annotations for row in rows if not row.vendor_marker]
    speech = [row for row in kept if row.attributed]
    events = [row for row in kept if not row.attributed]
    if not speech:
        raise KcscDerivationError(
            f"conversation {conversation.conversation_id} has no non-vendor speech rows"
        )

    tracks = [read_track_audio(track.audio_path) for track in conversation.tracks]
    mixed, padded = mix_tracks(tracks)
    source_duration = mixed.size / EXPECTED_SAMPLE_RATE

    # Floor the trim so the first real utterance keeps its onset, then express the offset
    # as the sample boundary actually applied: audio and reference timings then share one
    # origin exactly, with no sub-sample drift between them.
    trim_samples = min(max(0, int(np.floor(row.start * EXPECTED_SAMPLE_RATE))) for row in speech)
    trim_samples = min(trim_samples, mixed.size)
    trim_offset = trim_samples / EXPECTED_SAMPLE_RATE
    trimmed = mixed[trim_samples:].copy()
    if trimmed.size == 0:
        raise KcscDerivationError(
            f"conversation {conversation.conversation_id} is empty after trimming"
        )

    markers = sorted(
        (row.start - trim_offset, row.end - trim_offset)
        for rows in annotations
        for row in rows
        if row.vendor_marker
    )
    silenced = silence_intervals(trimmed, markers, sample_rate=EXPECTED_SAMPLE_RATE)
    samples, gain = limit_to_int16(trimmed)

    turns = shift_rows(speech, trim_offset)
    # Unattributed events are carried in their own list so a consumer scoring speaker
    # attribution never sees them, while a consumer inspecting the audio can still
    # explain a burst of laughter it hears.
    unattributed_events = shift_rows(events, trim_offset)

    audio_root = output_root / "audio"
    reference_root = output_root / "reference"
    audio_root.mkdir(parents=True, exist_ok=True)
    reference_root.mkdir(parents=True, exist_ok=True)
    audio_path = audio_root / f"{conversation.conversation_id}.wav"
    reference_path = reference_root / f"{conversation.conversation_id}.json"

    write_wav(audio_path, samples, sample_rate=EXPECTED_SAMPLE_RATE)
    derived_duration = samples.size / EXPECTED_SAMPLE_RATE
    reference_sha256 = write_json(
        reference_path,
        {
            "schema_version": SCHEMA_VERSION,
            "conversation_id": conversation.conversation_id,
            "source_dataset": provenance.dataset,
            "source_revision": provenance.revision,
            "sample_rate": EXPECTED_SAMPLE_RATE,
            "duration_seconds": round(derived_duration, 6),
            "trim_offset_seconds": round(trim_offset, 6),
            "speakers": [track.speaker for track in conversation.tracks],
            "vendor_marker_intervals_silenced": [list(interval) for interval in silenced],
            "turns": turns,
            "unattributed_events": unattributed_events,
        },
    )

    source_files = tuple(
        {
            "speaker": track.speaker,
            "audio": track.audio_path.name,
            "audio_sha256": sha256_file(track.audio_path),
            "annotation": track.annotation_path.name,
            "annotation_sha256": sha256_file(track.annotation_path),
        }
        for track in conversation.tracks
    )

    return DerivedConversation(
        conversation_id=conversation.conversation_id,
        speakers=tuple(track.speaker for track in conversation.tracks),
        sample_rate=EXPECTED_SAMPLE_RATE,
        source_duration_seconds=round(source_duration, 6),
        derived_duration_seconds=round(derived_duration, 6),
        trim_offset_seconds=round(trim_offset, 6),
        padded_samples=padded,
        mix_gain=gain,
        turn_count=len(turns),
        unattributed_event_count=len(unattributed_events),
        vendor_marker_count=sum(row.vendor_marker for rows in annotations for row in rows),
        silenced_intervals=silenced,
        source_files=source_files,
        audio_path=audio_path,
        audio_sha256=sha256_file(audio_path),
        reference_path=reference_path,
        reference_sha256=reference_sha256,
    )


def _manifest_entry(conversation: DerivedConversation, *, output_root: Path) -> dict[str, object]:
    return {
        "conversation_id": conversation.conversation_id,
        "speakers": list(conversation.speakers),
        "sample_rate": conversation.sample_rate,
        "source_duration_seconds": conversation.source_duration_seconds,
        "derived_duration_seconds": conversation.derived_duration_seconds,
        "trim_offset_seconds": conversation.trim_offset_seconds,
        "padded_samples": list(conversation.padded_samples),
        "mix_gain": conversation.mix_gain,
        "turn_count": conversation.turn_count,
        "unattributed_event_count": conversation.unattributed_event_count,
        "vendor_marker_count": conversation.vendor_marker_count,
        "vendor_marker_intervals_silenced": [
            list(interval) for interval in conversation.silenced_intervals
        ],
        "sources": [dict(entry) for entry in conversation.source_files],
        "outputs": {
            "audio": conversation.audio_path.relative_to(output_root).as_posix(),
            "audio_sha256": conversation.audio_sha256,
            "audio_bytes": conversation.audio_path.stat().st_size,
            "reference": conversation.reference_path.relative_to(output_root).as_posix(),
            "reference_sha256": conversation.reference_sha256,
        },
    }


def derive_evaluation_set(source_root: Path, output_root: Path) -> DerivationSummary:
    """Derive every conversation under ``source_root`` into ``output_root``.

    ``source_root`` is only ever read from. The aggregate manifest is written last so a
    manifest on disk always describes outputs that were fully written.
    """

    provenance = read_provenance(source_root)
    conversations = discover_conversations(source_root)
    output_root.mkdir(parents=True, exist_ok=True)
    derived = tuple(
        derive_conversation(conversation, output_root=output_root, provenance=provenance)
        for conversation in conversations
    )

    (output_root / "README.md").write_text(USAGE_README, encoding="utf-8")
    manifest_path = output_root / "manifest.json"
    manifest_sha256 = write_json(
        manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "dataset": "kcsc-derived-evaluation-set",
            "source": {
                "dataset": provenance.dataset,
                "revision": provenance.revision,
                "local_tree_sha256": provenance.local_tree_sha256,
                "provenance_sha256": provenance.provenance_sha256,
            },
            "usage": {
                "scope": "local research evaluation only",
                "training": "prohibited",
                "deployment": "prohibited",
                "redistribution": "prohibited",
                "third_party_upload": "prohibited pending confirmed processing rights",
            },
            "sample_rate": EXPECTED_SAMPLE_RATE,
            "conversation_count": len(derived),
            "turn_count": sum(entry.turn_count for entry in derived),
            "unattributed_event_count": sum(entry.unattributed_event_count for entry in derived),
            "vendor_marker_count": sum(entry.vendor_marker_count for entry in derived),
            "conversations": [_manifest_entry(entry, output_root=output_root) for entry in derived],
        },
    )

    return DerivationSummary(
        provenance=provenance,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        conversations=derived,
    )


__all__ = [
    "EXPECTED_CHANNELS",
    "EXPECTED_SAMPLE_RATE",
    "EXPECTED_SAMPLE_WIDTH",
    "SCHEMA_VERSION",
    "TRACKS_PER_CONVERSATION",
    "UNATTRIBUTED_SPEAKER",
    "USAGE_README",
    "VENDOR_MARKERS",
    "AnnotationRow",
    "ConversationSource",
    "DerivationSummary",
    "DerivedConversation",
    "KcscDerivationError",
    "SourceProvenance",
    "TrackSource",
    "derive_conversation",
    "derive_evaluation_set",
    "discover_conversations",
    "is_vendor_marker",
    "limit_to_int16",
    "mix_tracks",
    "parse_annotations",
    "read_provenance",
    "read_track_audio",
    "sha256_file",
    "shift_rows",
    "write_json",
    "write_wav",
    "silence_intervals",
]
