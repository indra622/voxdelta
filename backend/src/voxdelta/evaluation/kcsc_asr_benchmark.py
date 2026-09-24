"""Local-only ASR baseline on the derived KCSC evaluation set.

This measures **input-level mixed-conversation ASR**: one chronological transcript
decoded from the single mono mix of a two-party call, scored against the union of that
call's reference rows. It is deliberately *not* turn-level speaker ASR — no diarization
runs, nothing is attributed to a speaker, and no per-turn alignment is attempted. The
number it produces answers "how much of what was said does the recogniser get, given the
whole room at once", which is a strictly harder question than per-speaker WER and must
not be compared against per-speaker figures.

Two consequences of that framing are structural, not incidental, and are reported as
limitations rather than smoothed away:

* Where the two speakers overlap, the reference contains both utterances but a single
  ASR stream can only emit one. Those characters are unrecoverable deletions, and the
  overlap fraction is reported so the floor they impose is visible.
* The reference includes the corpus's unattributed rows (laughter, coughs, ambient
  noise, unintelligible fragments). They are audible in the mix, so excluding them would
  turn anything the recogniser emits there into a false insertion.

The primary metric is a Korean-normalised **character (syllable) error rate**. Korean
orthographic word spacing is not consistently realised by either the corpus annotators or
the recogniser, so a whitespace-sensitive metric would mostly measure spacing conventions.
An eojeol rate is reported as a clearly-labelled secondary figure with that caveat.

Nothing in this module's public surface carries transcript text. Reference and hypothesis
strings exist only inside the scoring functions; every value that reaches a report is a
count, a rate, or safe metadata.
"""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, cast

from voxdelta.evaluation.kcsc_diarization_benchmark import (
    KcscBenchmarkError,
    file_sha256,
    load_manifest,
    select_conversations,
    verify_inputs,
)

SCHEMA_VERSION = "1"

#: The approved fixed selection. Frozen here so a mistyped argument cannot widen the run.
BENCHMARK_CONVERSATIONS: tuple[str, ...] = (
    "A0051_S0001_0",
    "A0055_S0006_0",
    "A6000_S0005_0",
)

MODEL_REPO_ID = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
MODEL_REVISION = "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"
MODEL_FILES = (
    "config.json",
    "model.bin",
    "preprocessor_config.json",
    "tokenizer.json",
    "vocabulary.json",
)

#: The corpus's non-lexical annotation tags. These mark events, not words: the recogniser
#: is not expected to emit them, and counting them as reference characters would charge it
#: for failing to transcribe a cough.
CORPUS_TAGS = ("[*]", "[LAUGHTER]", "[SONANT]", "[MUSIC]", "[SYSTEM]", "[ENS]")

_TAG = re.compile(r"\[(?:\*|LAUGHTER|SONANT|MUSIC|SYSTEM|ENS)\]")
_WHITESPACE = re.compile(r"\s+")

#: Written into the report so the metric is reproducible from the artifact alone.
NORMALIZATION_STEPS: tuple[str, ...] = (
    "Unicode NFC normalisation",
    f"remove corpus event tags ({', '.join(CORPUS_TAGS)})",
    "remove the '+' overlapping-speech marker",
    "remove all Unicode punctuation and symbol characters",
    "lowercase (affects embedded Latin only)",
    "collapse runs of whitespace to one space, then strip",
    "for the character metric: remove all remaining whitespace",
)


class KcscAsrError(KcscBenchmarkError):
    """Raised for any condition that would make a reported rate untrustworthy."""


class WhisperModel(Protocol):
    """The narrow slice of the faster-whisper model this benchmark depends on."""

    def transcribe(self, path: str, **kwargs: object) -> tuple[object, object]: ...


@dataclass(frozen=True, slots=True)
class DecodeStats:
    """What the decoder emitted, including the quirks that were tolerated.

    VoxDelta's production ASR path routes word timestamps through an alignment validator
    that treats a zero-length span as a provider fault, because a diarization alignment
    cannot place a word that occupies no time. This benchmark does no diarization, and
    dropping those words would discard real transcribed text and inflate the deletion
    count. They are therefore kept for scoring and counted here, so the artifact shows
    how often the decoder produced them rather than hiding it.
    """

    word_count: int
    zero_length_spans: int
    words_past_declared_duration: int
    segments_without_word_timestamps: int

    def as_dict(self) -> dict[str, int]:
        return {
            "word_count": self.word_count,
            "zero_length_spans": self.zero_length_spans,
            "words_past_declared_duration": self.words_past_declared_duration,
            "segments_without_word_timestamps": self.segments_without_word_timestamps,
        }


def _strip_punctuation(text: str) -> str:
    return "".join(
        character
        for character in text
        if not unicodedata.category(character).startswith(("P", "S"))
    )


def normalize(text: str) -> str:
    """Apply the documented normalisation, returning space-separated tokens."""

    folded = unicodedata.normalize("NFC", text)
    folded = _TAG.sub(" ", folded)
    folded = folded.replace("+", " ")
    folded = _strip_punctuation(folded).lower()
    return _WHITESPACE.sub(" ", folded).strip()


def characters(text: str) -> str:
    """The character sequence the primary metric is computed over."""

    return normalize(text).replace(" ", "")


def eojeol(text: str) -> tuple[str, ...]:
    """Whitespace-delimited tokens, for the secondary rate only."""

    normalized = normalize(text)
    return tuple(normalized.split(" ")) if normalized else ()


@dataclass(frozen=True, slots=True)
class ErrorCounts:
    """Levenshtein operation counts against a reference of known length."""

    substitutions: int
    deletions: int
    insertions: int
    reference_length: int
    hypothesis_length: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def error_rate(self) -> float:
        if self.reference_length == 0:
            raise KcscAsrError("cannot compute an error rate against an empty reference")
        return self.errors / self.reference_length

    def merged(self, other: ErrorCounts) -> ErrorCounts:
        return ErrorCounts(
            substitutions=self.substitutions + other.substitutions,
            deletions=self.deletions + other.deletions,
            insertions=self.insertions + other.insertions,
            reference_length=self.reference_length + other.reference_length,
            hypothesis_length=self.hypothesis_length + other.hypothesis_length,
        )

    def as_dict(self) -> dict[str, float | int]:
        return {
            "error_rate": round(self.error_rate, 6),
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "errors": self.errors,
            "reference_length": self.reference_length,
            "hypothesis_length": self.hypothesis_length,
        }


def score_sequences(reference: Sequence[Any], hypothesis: Sequence[Any]) -> ErrorCounts:
    """Count substitutions, deletions and insertions on the minimal edit path.

    ``rapidfuzz`` supplies a C++ Levenshtein with explicit edit operations. A pure-Python
    dynamic program over sequences this long would take minutes per conversation, and the
    approximate matchers in the standard library do not compute a minimal edit distance,
    so they would silently report the wrong rate.
    """

    try:
        levenshtein = import_module("rapidfuzz.distance.Levenshtein")
    except ImportError:
        raise KcscAsrError("rapidfuzz is required to compute edit operations") from None
    operations = cast(Any, levenshtein).editops(list(reference), list(hypothesis))
    counts = {"replace": 0, "delete": 0, "insert": 0}
    for operation in operations:
        counts[operation.tag] += 1
    return ErrorCounts(
        substitutions=counts["replace"],
        deletions=counts["delete"],
        insertions=counts["insert"],
        reference_length=len(reference),
        hypothesis_length=len(hypothesis),
    )


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """The exact local weights a run used, resolved without contacting any hub."""

    repo_id: str
    revision: str
    files: tuple[Mapping[str, object], ...]
    tree_sha256: str
    total_bytes: int


def verify_model_identity(
    cache_root: Path,
    *,
    repo_id: str = MODEL_REPO_ID,
    revision: str = MODEL_REVISION,
    files: Sequence[str] = MODEL_FILES,
) -> ModelIdentity:
    """Resolve and hash the cached snapshot, failing closed if anything is absent.

    Symlinks are followed here rather than rejected: the Hugging Face cache stores every
    file as a link into a shared blob store, so refusing links would refuse every real
    cache. Each link is required to land inside the cache root, so the digest still
    describes bytes this repository controls.
    """

    root = cache_root.resolve()
    repo_dir = root / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / revision
    if not snapshot.is_dir():
        raise KcscAsrError(f"model snapshot not cached locally: {snapshot}")

    reference_file = repo_dir / "refs" / "main"
    if reference_file.is_file():
        pinned = reference_file.read_text(encoding="utf-8").strip()
        if pinned != revision:
            raise KcscAsrError(
                f"cached revision {pinned[:12]}… does not match pinned {revision[:12]}…"
            )

    resolved_files: list[Mapping[str, object]] = []
    total = 0
    for name in files:
        path = snapshot / name
        if not path.is_file():
            raise KcscAsrError(f"model file missing from cache: {name}")
        resolved = path.resolve()
        if root not in resolved.parents:
            raise KcscAsrError(f"model file resolves outside the cache root: {name}")
        size = resolved.stat().st_size
        total += size
        resolved_files.append({"name": name, "sha256": file_sha256(resolved), "bytes": size})

    tree = json.dumps(resolved_files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    import hashlib

    return ModelIdentity(
        repo_id=repo_id,
        revision=revision,
        files=tuple(resolved_files),
        tree_sha256=hashlib.sha256(tree).hexdigest(),
        total_bytes=total,
    )


def configure_offline_cache(cache_root: Path) -> None:
    """Point the hub client at the local cache and forbid it from fetching.

    Set before ``faster_whisper`` is imported. The egress guard is the real enforcement;
    this simply stops the library from treating a cache miss as a reason to try.
    """

    os.environ["HF_HUB_CACHE"] = str(cache_root.resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def reference_transcript(reference_path: Path, entry: Mapping[str, object]) -> tuple[str, int]:
    """Concatenate every reference row chronologically, returning text and row count.

    Unattributed rows are included: they are audible in the mix, so omitting them would
    charge the recogniser an insertion for correctly transcribing them.
    """

    try:
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise KcscAsrError(f"reference is not valid JSON: {reference_path}") from None
    if not isinstance(reference, dict):
        raise KcscAsrError(f"reference has an unexpected shape: {reference_path}")
    if reference.get("conversation_id") != entry.get("conversation_id"):
        raise KcscAsrError(f"{reference_path.name}: conversation id does not match manifest")

    rows: list[tuple[float, float, str]] = []
    for key in ("turns", "unattributed_events"):
        block = reference.get(key, [])
        if not isinstance(block, list):
            raise KcscAsrError(f"{reference_path.name}: {key} is not a list")
        for index, row in enumerate(block):
            if not isinstance(row, dict):
                raise KcscAsrError(f"{reference_path.name}: {key}[{index}] is not an object")
            start, end, text = row.get("start"), row.get("end"), row.get("transcript")
            if not isinstance(start, int | float) or not isinstance(end, int | float):
                raise KcscAsrError(f"{reference_path.name}: {key}[{index}] has non-numeric timing")
            if not isinstance(text, str):
                raise KcscAsrError(f"{reference_path.name}: {key}[{index}] has no transcript")
            rows.append((float(start), float(end), text))
    if not rows:
        raise KcscAsrError(f"{reference_path.name}: reference has no rows")
    rows.sort(key=lambda row: (row[0], row[1]))
    return " ".join(row[2] for row in rows), len(rows)


def reference_composition(reference_path: Path) -> dict[str, object]:
    """Aggregate, transcript-free characterisation of what the reference contains.

    This describes the *input*, not the errors. Attributing character-level edits to a
    time region would need an alignment between the reference turns and the recogniser's
    own segmentation, which this benchmark does not compute — see the report's
    ``limitations``.
    """

    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    turns = reference["turns"]
    events = reference.get("unattributed_events", [])

    intervals = sorted((float(t["start"]), float(t["end"]), str(t["speaker"])) for t in turns)
    speech = sum(end - start for start, end, _ in intervals)
    overlap = 0.0
    for index, (_start, end, speaker) in enumerate(intervals):
        for other_start, other_end, other_speaker in intervals[index + 1 :]:
            if other_start >= end:
                break
            if other_speaker != speaker:
                overlap += min(end, other_end) - other_start

    tag_counts = {
        tag: sum(1 for row in [*turns, *events] if tag in str(row.get("transcript", "")))
        for tag in CORPUS_TAGS
    }
    return {
        "turn_count": len(turns),
        "unattributed_event_count": len(events),
        "reference_speech_seconds": round(speech, 3),
        "overlapped_speech_seconds": round(overlap, 3),
        "overlapped_speech_fraction": round(overlap / speech, 6) if speech else 0.0,
        "rows_containing_overlap_marker": sum(
            1 for row in [*turns, *events] if "+" in str(row.get("transcript", ""))
        ),
        "rows_containing_tag": tag_counts,
    }


@dataclass(frozen=True, slots=True)
class ConversationAsrScore:
    """Per-conversation ASR result. Rates and counts only, never text."""

    conversation_id: str
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


@dataclass(frozen=True, slots=True)
class AsrBenchmarkReport:
    """Aggregate ASR benchmark result. Carries no transcript."""

    model: ModelIdentity
    device: str
    compute_type: str
    decode_options: Mapping[str, object]
    source_revision: str
    manifest_sha256: str
    scores: tuple[ConversationAsrScore, ...]
    egress_attempts: int
    total_elapsed_seconds: float

    @property
    def character(self) -> ErrorCounts:
        merged = self.scores[0].character
        for score in self.scores[1:]:
            merged = merged.merged(score.character)
        return merged

    @property
    def eojeol(self) -> ErrorCounts:
        merged = self.scores[0].eojeol
        for score in self.scores[1:]:
            merged = merged.merged(score.eojeol)
        return merged


def load_model(*, device: str = "cpu", compute_type: str = "int8") -> WhisperModel:
    """Construct the model the production provider would, from the local cache only.

    ``MODEL_ID`` is imported from the provider rather than restated so the benchmark
    cannot drift onto a different checkpoint than the pipeline uses.
    """

    provider = import_module("voxdelta.providers.faster_whisper_asr")
    factory = cast(Any, provider)._default_factory
    return cast(
        WhisperModel,
        factory(cast(Any, provider).MODEL_ID, device=device, compute_type=compute_type),
    )


def decode_chronological(
    model: WhisperModel,
    audio_path: Path,
    duration: float,
    options: Mapping[str, object],
) -> tuple[str, DecodeStats]:
    """Decode the whole mix into one chronological string, with no diarization.

    Words are ordered by onset with a stable sort, so words sharing a timestamp keep the
    order the decoder emitted them in. Timings are used only for ordering; every word
    carrying text is kept regardless of its span, because the metric scores characters.
    """

    segments, _info = model.transcribe(str(audio_path), **dict(options))
    entries: list[tuple[float, int, str]] = []
    zero_length = past_duration = without_words = 0
    index = 0
    for raw_segment in cast(Any, segments):
        words = getattr(raw_segment, "words", None)
        if words is None:
            text = str(getattr(raw_segment, "text", "")).strip()
            if text:
                without_words += 1
                entries.append((float(getattr(raw_segment, "start", 0.0)), index, text))
                index += 1
            continue
        for raw_word in words:
            text = str(getattr(raw_word, "word", "")).strip()
            if not text:
                continue
            word_start = float(getattr(raw_word, "start", 0.0))
            word_end = float(getattr(raw_word, "end", word_start))
            if word_end <= word_start:
                zero_length += 1
            if word_end > duration:
                past_duration += 1
            entries.append((word_start, index, text))
            index += 1
    if not entries:
        raise KcscAsrError(f"{audio_path.name}: transcription returned no words")
    entries.sort(key=lambda entry: (entry[0], entry[1]))
    return " ".join(entry[2] for entry in entries), DecodeStats(
        word_count=len(entries),
        zero_length_spans=zero_length,
        words_past_declared_duration=past_duration,
        segments_without_word_timestamps=without_words,
    )


def score_conversation(
    entry: Mapping[str, object],
    *,
    derived_root: Path,
    model: WhisperModel,
    decode_options: Mapping[str, object],
) -> ConversationAsrScore:
    """Verify checksums, transcribe the mix once, and score it."""

    audio_path, reference_path = verify_inputs(entry, derived_root=derived_root)
    duration = entry.get("derived_duration_seconds")
    if not isinstance(duration, int | float) or duration <= 0:
        raise KcscAsrError(f"{entry.get('conversation_id')}: manifest duration is unusable")

    reference_text, row_count = reference_transcript(reference_path, entry)

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
    return ConversationAsrScore(
        conversation_id=str(entry["conversation_id"]),
        duration_seconds=round(float(duration), 3),
        audio_sha256=str(outputs["audio_sha256"]),
        reference_sha256=str(outputs["reference_sha256"]),
        reference_row_count=row_count,
        decode=decode,
        elapsed_seconds=round(elapsed, 3),
        real_time_factor=round(elapsed / float(duration), 6),
        character=character,
        eojeol=tokens,
        composition=reference_composition(reference_path),
    )


LIMITATIONS: tuple[str, ...] = (
    "Input-level mixed-conversation ASR, not turn-level speaker ASR: one stream is "
    "decoded from the mono mix and no diarization, speaker attribution, or per-turn "
    "alignment is performed. These rates are not comparable to per-speaker WER/CER.",
    "Overlapping speech imposes an irreducible deletion floor: the reference contains "
    "both speakers' words in overlapped regions but a single ASR stream can emit only "
    "one. See reference_composition.overlapped_speech_fraction per conversation.",
    "Errors are NOT stratified by overlap, noise, or laughter. Doing so would require "
    "aligning character-level edits to time regions, which needs an alignment between "
    "the reference turns and the recogniser's own segmentation that this benchmark does "
    "not compute. Publishing a stratification derived from a global edit path would "
    "attribute errors to regions arbitrarily, so it is omitted.",
    "The eojeol rate is secondary and conflates spacing with recognition: Korean word "
    "spacing is applied inconsistently by both the corpus annotators and the recogniser, "
    "so it should be read as an upper bound, not as a word error rate.",
    "The reference includes the corpus's unattributed rows (laughter, coughs, ambient "
    "noise, unintelligible fragments) because they are audible in the mix. Event tags "
    "normalise away to empty and contribute no reference characters.",
    "Three conversations, six speakers, one language and recording condition. Not a "
    "basis for a general Korean ASR claim.",
    "Edit operations come from rapidfuzz, which is currently present transitively rather "
    "than as a declared dependency; declare it explicitly before relying on this in CI.",
)


def run_asr_benchmark(
    *,
    derived_root: Path,
    model: WhisperModel,
    identity: ModelIdentity,
    device: str,
    compute_type: str,
    decode_options: Mapping[str, object],
    conversation_ids: Sequence[str] = BENCHMARK_CONVERSATIONS,
    egress_attempts: int = 0,
) -> AsrBenchmarkReport:
    """Verify every checksum first, then transcribe each conversation exactly once."""

    if tuple(conversation_ids) != BENCHMARK_CONVERSATIONS:
        raise KcscAsrError("conversation selection does not match the approved scope")
    manifest = load_manifest(derived_root)
    entries = select_conversations(manifest, conversation_ids)
    for entry in entries:
        verify_inputs(entry, derived_root=derived_root)

    started = time.monotonic()
    scores = tuple(
        score_conversation(
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
    return AsrBenchmarkReport(
        model=identity,
        device=device,
        compute_type=compute_type,
        decode_options=dict(decode_options),
        source_revision=str(revision),
        manifest_sha256=file_sha256(derived_root / "manifest.json"),
        scores=scores,
        egress_attempts=egress_attempts,
        total_elapsed_seconds=round(total_elapsed, 3),
    )


def write_report(report: AsrBenchmarkReport, path: Path) -> str:
    """Write the transcript-free aggregate report and return its digest."""

    total_audio = sum(score.duration_seconds for score in report.scores)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "kcsc-asr",
        "transcript_free": True,
        "task": "input-level-mixed-conversation-asr",
        "task_note": (
            "One chronological transcript decoded from the mono mix of a two-party call. "
            "No diarization and no speaker attribution. Not turn-level speaker ASR."
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
            "name": "kcsc-derived-evaluation-set",
            "source_revision": report.source_revision,
            "derivation_manifest_sha256": report.manifest_sha256,
            "conversation_count": len(report.scores),
            "conversation_ids": [score.conversation_id for score in report.scores],
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
                "conversation_id": score.conversation_id,
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
    "CORPUS_TAGS",
    "LIMITATIONS",
    "MODEL_FILES",
    "MODEL_REPO_ID",
    "MODEL_REVISION",
    "NORMALIZATION_STEPS",
    "SCHEMA_VERSION",
    "AsrBenchmarkReport",
    "ConversationAsrScore",
    "DecodeStats",
    "ErrorCounts",
    "KcscAsrError",
    "ModelIdentity",
    "characters",
    "configure_offline_cache",
    "eojeol",
    "normalize",
    "reference_composition",
    "reference_transcript",
    "run_asr_benchmark",
    "score_conversation",
    "score_sequences",
    "decode_chronological",
    "load_model",
    "verify_model_identity",
    "write_report",
]
