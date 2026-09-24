"""Local-only Qwen3-ASR baseline on the derived KCSC per-speaker track set.

This is the faster-whisper track benchmark's question asked of a different recogniser:
same six speaker tracks, same reference rows, same Korean-normalised character error
rate, same pooling. Everything that decides a number — the normalisation steps, the edit
accounting, the track selection, the checksum verification — is imported from those
modules rather than restated, so a difference between the two reports is a difference
between the models and not between two scorers that drifted apart.

What legitimately differs is the decode contract, and it is recorded rather than hidden:

* **Chunking.** Qwen3-ASR takes long audio through the official splitter, which cuts at
  low-energy boundaries and reassembles losslessly. With timestamps requested the library
  uses 180 s chunks; a whole 900 s track in one generation call would sit under a token
  cap and silently truncate.
* **Timestamps.** The Qwen stack produces them through a separate forced-aligner
  checkpoint, so both checkpoints are hashed into the report. The word units are used for
  the alignment diagnostics only — the scored hypothesis is the recogniser's own text,
  so the aligner cannot move the error rate.
* **Device.** MPS/float16 here versus CPU/int8 for faster-whisper. Real-time factors are
  therefore reported per run and are not a like-for-like speed comparison; the error
  rates are.

Nothing in this module's public surface carries transcript text. Reference and hypothesis
strings exist only inside the scoring functions; every value that reaches a report is a
count, a rate, or safe metadata.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, cast

from voxdelta.evaluation.kcsc_asr_benchmark import (
    NORMALIZATION_STEPS,
    ErrorCounts,
    KcscAsrError,
    ModelIdentity,
    characters,
    eojeol,
    score_sequences,
    verify_model_identity,
)
from voxdelta.evaluation.kcsc_diarization_benchmark import file_sha256
from voxdelta.evaluation.kcsc_track_asr_benchmark import (
    BENCHMARK_CONVERSATIONS,
    load_track_manifest,
    select_tracks,
    track_composition,
    track_id,
    track_reference_transcript,
    verify_track_inputs,
)
from voxdelta.providers.qwen3_asr import ALIGNER_MODEL_ID, DEFAULT_MODEL_ID, LOW_MEMORY_MODEL_ID

SCHEMA_VERSION = "1"

#: Immutable revisions of the two ASR checkpoints and the aligner, pinned so a cache that
#: has moved on aborts the run instead of quietly benchmarking different weights.
MODEL_REVISIONS: Mapping[str, str] = {
    LOW_MEMORY_MODEL_ID: "5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
    DEFAULT_MODEL_ID: "7278e1e70fe206f11671096ffdd38061171dd6e5",
    ALIGNER_MODEL_ID: "c7cbfc2048c462b0d63a45797104fc9db3ad62b7",
}

_COMMON_FILES = (
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
)
#: Weight layout differs by size: 0.6B ships one safetensors file, 1.7B ships two shards
#: plus an index. Both lists are exhaustive, so an extra or missing weight file is caught.
MODEL_FILES: Mapping[str, tuple[str, ...]] = {
    LOW_MEMORY_MODEL_ID: (*_COMMON_FILES, "model.safetensors"),
    DEFAULT_MODEL_ID: (
        *_COMMON_FILES,
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
    ),
    ALIGNER_MODEL_ID: (*_COMMON_FILES, "model.safetensors"),
}

PROFILES: Mapping[str, str] = {"0.6b": LOW_MEMORY_MODEL_ID, "1.7b": DEFAULT_MODEL_ID}

LANGUAGE = "Korean"

#: 180 s of Korean speech cannot approach this many tokens, so the generation cap cannot
#: bind and truncate a chunk. Left explicit because the library default (512) can.
MAX_NEW_TOKENS = 2048


class QwenModel(Protocol):
    """The narrow slice of ``qwen_asr.Qwen3ASRModel`` this benchmark depends on."""

    def transcribe(self, audio: object, *, language: str, return_time_stamps: bool) -> object: ...


@dataclass(frozen=True, slots=True)
class DecodeStats:
    """What the decoder emitted, in counts. Never the words themselves."""

    word_count: int
    zero_length_spans: int
    words_past_declared_duration: int
    non_monotonic_starts: int
    last_word_end_seconds: float
    max_word_gap_seconds: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "word_count": self.word_count,
            "zero_length_spans": self.zero_length_spans,
            "words_past_declared_duration": self.words_past_declared_duration,
            "non_monotonic_starts": self.non_monotonic_starts,
            "last_word_end_seconds": round(self.last_word_end_seconds, 3),
            "max_word_gap_seconds": round(self.max_word_gap_seconds, 3),
        }


def verify_qwen_identity(cache_root: Path, model_id: str) -> ModelIdentity:
    """Hash one pinned Qwen snapshot out of the local cache, failing closed if absent."""

    revision = MODEL_REVISIONS.get(model_id)
    files = MODEL_FILES.get(model_id)
    if revision is None or files is None:
        raise KcscAsrError(f"no pinned revision for {model_id}")
    return verify_model_identity(cache_root, repo_id=model_id, revision=revision, files=files)


def load_qwen_model(
    model_id: str, *, device: str, dtype: str, max_new_tokens: int = MAX_NEW_TOKENS
) -> QwenModel:
    """Construct the model the production provider would, from the local cache only.

    The aligner is attached because timestamps are requested; it is the same checkpoint
    the provider pairs with, so the benchmark exercises the shipped combination.
    """

    qwen = import_module("qwen_asr")
    torch = import_module("torch")
    torch_dtype = getattr(torch, dtype)
    return cast(
        QwenModel,
        cast(Any, qwen).Qwen3ASRModel.from_pretrained(
            model_id,
            forced_aligner=ALIGNER_MODEL_ID,
            forced_aligner_kwargs={"device_map": device, "dtype": torch_dtype},
            device_map=device,
            dtype=torch_dtype,
            max_inference_batch_size=1,
            max_new_tokens=max_new_tokens,
        ),
    )


def _units(stamps: object) -> list[tuple[float, float, str]]:
    """Flatten the aligner's result into (start, end, text), failing closed on any other shape."""

    try:
        rows = []
        for raw in cast(Iterable[object], stamps):
            text = str(getattr(raw, "text", "")).strip()
            start = getattr(raw, "start_time", getattr(raw, "start", None))
            end = getattr(raw, "end_time", getattr(raw, "end", start))
            if start is None or end is None:
                raise KcscAsrError("aligner unit carries no timing")
            rows.append((float(start), float(end), text))
        return rows
    except KcscAsrError:
        raise
    except Exception:
        raise KcscAsrError("aligner returned an unreadable timestamp structure") from None


def decode_track(model: QwenModel, audio_path: Path, duration: float) -> tuple[str, DecodeStats]:
    """Transcribe one track and characterise the timeline the aligner returned.

    The scored hypothesis is the recogniser's own text. The word units are read only to
    describe the alignment, so a weak aligner cannot flatter or damage the error rate.
    """

    outputs = model.transcribe(str(audio_path), language=LANGUAGE, return_time_stamps=True)
    try:
        results = list(cast(Iterable[object], outputs))
    except Exception:
        raise KcscAsrError(f"{audio_path.name}: transcription returned no result") from None
    if len(results) != 1:
        raise KcscAsrError(f"{audio_path.name}: expected one result, got {len(results)}")
    result = results[0]

    text = getattr(result, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise KcscAsrError(f"{audio_path.name}: transcription returned no text")

    stamps = getattr(result, "time_stamps", None)
    if stamps is None:
        raise KcscAsrError(f"{audio_path.name}: timestamps were requested but none returned")
    units = _units(stamps)
    if not units:
        raise KcscAsrError(f"{audio_path.name}: aligner returned no word units")

    zero_length = sum(1 for start, end, _ in units if end <= start)
    past_duration = sum(1 for _, end, _ in units if end > duration)
    non_monotonic = sum(
        1 for before, after in zip(units, units[1:], strict=False) if after[0] < before[0]
    )
    gaps = [after[0] - before[1] for before, after in zip(units, units[1:], strict=False)]
    return text, DecodeStats(
        word_count=len(units),
        zero_length_spans=zero_length,
        words_past_declared_duration=past_duration,
        non_monotonic_starts=non_monotonic,
        last_word_end_seconds=max(end for _, end, _ in units),
        max_word_gap_seconds=max(gaps) if gaps else 0.0,
    )


@dataclass(frozen=True, slots=True)
class QwenTrackScore:
    """Per-track Qwen result. Rates and counts only, never text."""

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
class QwenTrackReport:
    """Aggregate Qwen per-speaker result. Carries no transcript."""

    model: ModelIdentity
    aligner: ModelIdentity
    device: str
    dtype: str
    max_new_tokens: int
    source_revision: str
    manifest_sha256: str
    mixed_manifest_sha256: str
    scores: tuple[QwenTrackScore, ...]
    egress_attempts: int
    total_elapsed_seconds: float

    @property
    def character(self) -> ErrorCounts:
        return _pool([score.character for score in self.scores])

    @property
    def eojeol(self) -> ErrorCounts:
        return _pool([score.eojeol for score in self.scores])

    def conversation_character(self, conversation_id: str) -> ErrorCounts:
        counts = [s.character for s in self.scores if s.conversation_id == conversation_id]
        if not counts:
            raise KcscAsrError(f"no scored tracks for conversation {conversation_id}")
        return _pool(counts)

    def conversation_eojeol(self, conversation_id: str) -> ErrorCounts:
        counts = [s.eojeol for s in self.scores if s.conversation_id == conversation_id]
        if not counts:
            raise KcscAsrError(f"no scored tracks for conversation {conversation_id}")
        return _pool(counts)


def score_track(
    entry: Mapping[str, object],
    *,
    derived_root: Path,
    model: QwenModel,
) -> QwenTrackScore:
    """Verify checksums, transcribe one speaker track once, and score it."""

    audio_path, reference_path = verify_track_inputs(entry, derived_root=derived_root)
    duration = entry.get("derived_duration_seconds")
    if not isinstance(duration, int | float) or duration <= 0:
        raise KcscAsrError(f"{track_id(entry)}: manifest duration is unusable")

    reference_text, row_count = track_reference_transcript(reference_path, entry)

    started = time.monotonic()
    hypothesis_text, stats = decode_track(model, audio_path, float(duration))
    elapsed = time.monotonic() - started

    character = score_sequences(characters(reference_text), characters(hypothesis_text))
    tokens = score_sequences(eojeol(reference_text), eojeol(hypothesis_text))
    # Both strings go out of scope here; nothing below this line can reach a report.

    outputs = entry["outputs"]
    assert isinstance(outputs, dict)
    return QwenTrackScore(
        conversation_id=str(entry["conversation_id"]),
        speaker=str(entry["speaker"]),
        duration_seconds=round(float(duration), 3),
        audio_sha256=str(outputs["audio_sha256"]),
        reference_sha256=str(outputs["reference_sha256"]),
        reference_row_count=row_count,
        decode=stats,
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
    "is performed and no diarization runs.",
    "Real-time factors are NOT comparable to the faster-whisper track report: that run "
    "was CPU/int8, this one is MPS/float16, and the two stacks chunk long audio "
    "differently. The error rates are comparable; the speed figures are not.",
    "Long audio is chunked by the official Qwen splitter at low-energy boundaries. Chunk "
    "boundaries can cut a word, and the per-chunk texts are concatenated by the library, "
    "so a boundary can cost a substitution that a single-pass decoder would not make.",
    "Timestamps come from a separate forced-aligner checkpoint, not from the recogniser. "
    "They are reported as alignment diagnostics and are deliberately not used to build "
    "the scored hypothesis, so aligner quality cannot move the error rate.",
    "Errors are NOT stratified by overlap, noise, or laughter: that needs an alignment "
    "between the reference turns and the recogniser's own segmentation, which this "
    "benchmark does not compute.",
    "The eojeol rate is secondary and conflates spacing with recognition. Qwen3-ASR emits "
    "its own spacing and punctuation conventions, which differ from the corpus's, so the "
    "eojeol gap against another model overstates the recognition gap.",
    "Three conversations, six speaker tracks, one language and recording condition. Not a "
    "basis for a general Korean ASR claim.",
    "Edit operations come from rapidfuzz, which is currently present transitively rather "
    "than as a declared dependency; declare it explicitly before relying on this in CI.",
)


def run_qwen_track_benchmark(
    *,
    derived_root: Path,
    model: QwenModel,
    identity: ModelIdentity,
    aligner: ModelIdentity,
    device: str,
    dtype: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    conversation_ids: Sequence[str] = BENCHMARK_CONVERSATIONS,
    egress_attempts: int = 0,
) -> QwenTrackReport:
    """Verify every checksum first, then transcribe each speaker track exactly once."""

    if tuple(conversation_ids) != BENCHMARK_CONVERSATIONS:
        raise KcscAsrError("conversation selection does not match the approved scope")
    manifest = load_track_manifest(derived_root)
    entries = select_tracks(manifest, conversation_ids)
    for entry in entries:
        verify_track_inputs(entry, derived_root=derived_root)

    started = time.monotonic()
    scores = tuple(score_track(entry, derived_root=derived_root, model=model) for entry in entries)
    total_elapsed = time.monotonic() - started

    source = manifest.get("source")
    revision = source.get("revision") if isinstance(source, dict) else None
    mixed = manifest.get("mixed_set")
    mixed_digest = mixed.get("manifest_sha256") if isinstance(mixed, dict) else None
    return QwenTrackReport(
        model=identity,
        aligner=aligner,
        device=device,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
        source_revision=str(revision),
        manifest_sha256=file_sha256(derived_root / "manifest.json"),
        mixed_manifest_sha256=str(mixed_digest),
        scores=scores,
        egress_attempts=egress_attempts,
        total_elapsed_seconds=round(total_elapsed, 3),
    )


def _identity_block(identity: ModelIdentity) -> dict[str, object]:
    return {
        "repo_id": identity.repo_id,
        "revision": identity.revision,
        "tree_sha256": identity.tree_sha256,
        "total_bytes": identity.total_bytes,
        "files": [dict(entry) for entry in identity.files],
    }


def write_qwen_report(report: QwenTrackReport, path: Path) -> str:
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
            "that speaker's reference rows. Same tracks, references, scorer, and pooling "
            "as the faster-whisper track report, so the character error rates are "
            "directly comparable."
        ),
        "model": {
            "name": report.model.repo_id.removeprefix("Qwen/"),
            "stack": "qwen_asr transformers backend",
            **_identity_block(report.model),
            "device": report.device,
            "dtype": report.dtype,
            "max_new_tokens": report.max_new_tokens,
            "language": LANGUAGE,
            "remote": False,
        },
        "aligner": {
            "name": report.aligner.repo_id.removeprefix("Qwen/"),
            **_identity_block(report.aligner),
            "used_for": "timestamp diagnostics only; not used to build the scored hypothesis",
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
                "inconsistently by annotators and recognisers alike, so this conflates "
                "spacing with recognition and reads as an upper bound."
            ),
            "normalization_steps": list(NORMALIZATION_STEPS),
            "aggregation": "pooled edit operations over pooled reference length",
            "implementation": "rapidfuzz.distance.Levenshtein.editops",
            "shared_with": "kcsc_track_asr_benchmark (identical scorer, imported not restated)",
        },
        "timestamps": {
            "supported": True,
            "source": "Qwen3-ForcedAligner-0.6B via qwen_asr return_time_stamps=True",
            "unit": "word",
            "language": LANGUAGE,
            "note": (
                "Word units carry start_time and end_time in seconds on the track's own "
                "timeline. Reported as diagnostics; the scored hypothesis is the "
                "recogniser's text."
            ),
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
    "LANGUAGE",
    "LIMITATIONS",
    "MAX_NEW_TOKENS",
    "MODEL_FILES",
    "MODEL_REVISIONS",
    "PROFILES",
    "SCHEMA_VERSION",
    "DecodeStats",
    "QwenTrackReport",
    "QwenTrackScore",
    "decode_track",
    "load_qwen_model",
    "run_qwen_track_benchmark",
    "score_track",
    "verify_qwen_identity",
    "write_qwen_report",
]
