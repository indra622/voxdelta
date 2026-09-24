"""Diagnose where G5999's deletions concentrate. Local only, aggregate only.

Reuses the immutable retry-2 record's preserved Precision timeline, the derived WAV, and
the local KCSC reference. Nothing is submitted anywhere and no diarization is recomputed;
the recogniser is decoded locally from the pinned cache with the hub client forced offline
before any model is constructed. Every pinned input is verified by digest first, so a
drifted input stops the run before processing rather than producing a number about
something else.

The artifact this writes is additive: it names the base evaluation by digest and does not
modify or recompute it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Never

from voxdelta.domain.models import SpeakerSegment
from voxdelta.evaluation.kcsc_asr_benchmark import KcscAsrError, configure_offline_cache
from voxdelta.evaluation.kcsc_attributed_asr import (
    KcscAttributedError,
    PinnedInput,
    load_timeline,
    map_speakers,
    sanitize,
    time_records,
)
from voxdelta.evaluation.kcsc_deletion_diagnostic import (
    ASSOCIATION_POLICY,
    BOUNDARY_POLICY,
    BOUNDARY_TOLERANCE_SECONDS,
    OVERLAP_POLICY,
    PRIMARY_PRECEDENCE,
    SCHEMA_VERSION,
    Turn,
    diagnose,
    duration_bin_edges,
    tabulate,
)
from voxdelta.evaluation.kcsc_diarization_benchmark import file_sha256
from voxdelta.evaluation.kcsc_qwen_track_benchmark import load_qwen_model, verify_qwen_identity
from voxdelta.providers.asr_alignment import AlignedWord, align_mixed, point_segment_index
from voxdelta.providers.offline_guard import block_network_egress
from voxdelta.providers.qwen3_asr import ALIGNER_MODEL_ID, DEFAULT_MODEL_ID

CONVERSATION_ID = "A6000_S0005_0"
SPEAKER_UNDER_DIAGNOSIS = "G5999"
RECORD = Path("backend/runtime/poc/kcsc-precision2-qwen-xlsr-e2e/EVALUATION-retry-2.json")
RECORD_SHA256 = "4829014c3756194ce99e70102a35c83b5983db5c6dceb6bca7b481964ba7de34"
AUDIO = Path(f"data/derived/kcsc/audio/{CONVERSATION_ID}.wav")
AUDIO_SHA256 = "01565974fdf7021e0d0afcbae8c040839777434f40416f1e265aeee3926cce27"
REFERENCE = Path(f"data/derived/kcsc/reference/{CONVERSATION_ID}.json")
BASE_ARTIFACT = Path("data/benchmarks/kcsc-precision-qwen-e2e-asr-retry2-v2.json")
MODEL_CACHE = Path("data/models/hf-cache/hub")
DEFAULT_REPORT = Path("data/benchmarks/kcsc-g5999-deletion-diagnostic.json")


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        del message
        raise ValueError("invalid arguments")


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=RECORD)
    parser.add_argument("--audio", type=Path, default=AUDIO)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--base", type=Path, default=BASE_ARTIFACT)
    parser.add_argument("--model-cache", type=Path, default=MODEL_CACHE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    return parser


def _placed_label(word: AlignedWord, segments: Sequence[SpeakerSegment]) -> str | None:
    """Which diarized speaker the alignment would give this word, or None if unplaceable."""

    start, end = word.start, word.end
    if end <= start:
        index = point_segment_index(start, list(segments))
        return segments[index].speaker_id if index is not None else None
    best: tuple[float, int] | None = None
    for index, segment in enumerate(segments):
        overlap = max(0.0, min(end, segment.end) - max(start, segment.start))
        if overlap > 0 and (best is None or overlap > best[0]):
            best = (overlap, index)
    return segments[best[1]].speaker_id if best is not None else None


def _turns(reference_path: Path, speaker: str) -> tuple[list[Turn], list[Turn]]:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    mine: list[Turn] = []
    theirs: list[Turn] = []
    for row in reference["turns"]:
        turn = Turn(float(row["start"]), float(row["end"]), str(row["transcript"]))
        (mine if str(row["speaker"]) == speaker else theirs).append(turn)
    if not mine:
        raise KcscAttributedError(f"reference has no turns for {speaker}")
    return mine, theirs


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
    except ValueError:
        print("diagnostic failed: invalid arguments", file=sys.stderr)
        return 2

    dtype = (
        "bfloat16"
        if arguments.device == "cuda"
        else "float16"
        if arguments.device == "mps"
        else "float32"
    )

    try:
        record_digest = PinnedInput(arguments.record, RECORD_SHA256).verify()
        audio_digest = PinnedInput(arguments.audio, AUDIO_SHA256).verify()
        segments, record = load_timeline(arguments.record)
        mapping, _evidence = map_speakers(segments, arguments.reference)
        mine, theirs = _turns(arguments.reference, SPEAKER_UNDER_DIAGNOSIS)
        mapped_label = next(
            label for label, speaker in mapping.items() if speaker == SPEAKER_UNDER_DIAGNOSIS
        )
        if not arguments.model_cache.is_dir():
            raise KcscAttributedError(f"missing local hub cache: {arguments.model_cache}")
        identity = verify_qwen_identity(arguments.model_cache, DEFAULT_MODEL_ID)
        aligner = verify_qwen_identity(arguments.model_cache, ALIGNER_MODEL_ID)
    except (KcscAttributedError, KcscAsrError) as error:
        print(f"diagnostic failed: {error}", file=sys.stderr)
        return 2

    configure_offline_cache(arguments.model_cache)
    duration = float(record["input"]["duration_seconds"])  # type: ignore[index]
    mapped = [s for s in segments if s.speaker_id == mapped_label]
    other_segments = [s for s in segments if s.speaker_id != mapped_label]

    print("preflight: G5999 deletion diagnostic (local only)")
    print(f"  base artifact       : sha256={file_sha256(arguments.base)}")
    print(f"  retry-2 record      : sha256={record_digest}")
    print(f"  derived audio       : sha256={audio_digest}")
    print(f"  mapped label        : {mapped_label} -> {SPEAKER_UNDER_DIAGNOSIS}")
    print(f"  reference turns     : {len(mine)} (other speaker {len(theirs)})")
    print(f"  mapped segments     : {len(mapped)} of {len(segments)} (reused, not recomputed)")
    print(f"  model tree          : sha256={identity.tree_sha256}")
    print(f"  runtime             : {arguments.device}/{dtype}, hub offline")
    print("  external calls      : 0")
    sys.stdout.flush()

    started = time.monotonic()
    try:
        with block_network_egress() as egress:
            model = load_qwen_model(DEFAULT_MODEL_ID, device=arguments.device, dtype=dtype)
            print("model loaded", flush=True)
            results = list(
                model.transcribe(  # type: ignore[call-overload]
                    str(arguments.audio), language="Korean", return_time_stamps=True
                )
            )
            if len(results) != 1:
                raise KcscAttributedError(f"expected one result, got {len(results)}")
            stamps = getattr(results[0], "time_stamps", None)
            if stamps is None:
                raise KcscAttributedError("timestamps were requested but none returned")
            words, coverage = sanitize(time_records(stamps), duration, segments)
            utterances, omitted = align_mixed(list(words), list(segments), duration)
            # Same placement rule align_mixed uses, so the words binned below are the
            # words the pipeline actually gave this speaker: greatest overlap wins, and a
            # zero-duration instant is placed by strict containment.
            attributed = [word for word in words if _placed_label(word, segments) == mapped_label]
    except (KcscAttributedError, KcscAsrError) as error:
        print(f"diagnostic failed: {error}", file=sys.stderr)
        return 2
    except Exception as error:  # noqa: BLE001 - the type is the diagnosis here
        print(f"diagnostic failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    elapsed = time.monotonic() - started

    if egress.attempted:
        print(f"diagnostic failed: {egress.attempts} egress attempts", file=sys.stderr)
        return 2

    diagnoses, unassigned, unassigned_chars = diagnose(
        turns=mine,
        other_turns=theirs,
        mapped=mapped,
        other_segments=other_segments,
        words=attributed,
    )
    total = tabulate(diagnoses, "primary")
    mapped_seconds = sum(s.end - s.start for s in mapped)
    reference_seconds = sum(turn.duration for turn in mine)

    report = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "kcsc-g5999-deletion-diagnostic",
        "transcript_free": True,
        "additive_to": {
            "path": str(arguments.base),
            "sha256": file_sha256(arguments.base),
            "note": "the base artifact is unmodified and was not recomputed",
        },
        "local_only": {
            "external_calls": 0,
            "network_egress_attempts": egress.attempts,
            "audio_transmitted": False,
            "diarization_recomputed": False,
            "diarization_source": "preserved retry-2 hypothesis_timeline",
            "qwen_decode_rerun_locally": True,
            "hub_offline": True,
        },
        "inputs": {
            "conversation_id": CONVERSATION_ID,
            "speaker_under_diagnosis": SPEAKER_UNDER_DIAGNOSIS,
            "mapped_diarized_label": mapped_label,
            "retry2_record_sha256": record_digest,
            "derived_audio_sha256": audio_digest,
            "reference_sha256": file_sha256(arguments.reference),
            "model_tree_sha256": identity.tree_sha256,
            "aligner_tree_sha256": aligner.tree_sha256,
        },
        "method": {
            "association_policy": ASSOCIATION_POLICY,
            "overlap_policy": OVERLAP_POLICY,
            "boundary_policy": BOUNDARY_POLICY,
            "boundary_tolerance_seconds": BOUNDARY_TOLERANCE_SECONDS,
            "duration_bin_edges": duration_bin_edges(),
            "primary_precedence": list(PRIMARY_PRECEDENCE),
            "non_additive_note": (
                "Marginal factor tables overlap: one turn appears in every table. Only the "
                "'primary' table is mutually exclusive. Per-turn scores come from per-turn "
                "edit paths and do NOT sum to the pooled speaker score in the base artifact."
            ),
        },
        "association_health": {
            "words_attributed_to_mapped_label": len(attributed),
            "words_unassigned_to_any_turn": unassigned,
            "unassigned_characters": unassigned_chars,
            "note": (
                "unassigned words carry real recognised text that no reference turn of this "
                "speaker contains by midpoint; they inflate per-turn deletions and are the "
                "size of this method's own blind spot"
            ),
        },
        "coverage_context": {
            "mapped_segment_seconds": round(mapped_seconds, 3),
            "reference_speech_seconds": round(reference_seconds, 3),
            "mapped_to_reference_ratio": round(mapped_seconds / reference_seconds, 6),
        },
        "pipeline_context": {
            "attributed_utterance_count": len(utterances),
            "words_omitted_by_alignment": omitted,
            "timestamp_coverage": coverage.with_alignment_omissions(omitted).as_dict(),
        },
        "bins": {
            "primary_mutually_exclusive": total,
            "by_duration": tabulate(diagnoses, "duration_bin"),
            "by_reference_overlap": tabulate(diagnoses, "overlap_bin"),
            "by_boundary_proximity": tabulate(diagnoses, "boundary_bin"),
            "by_mapped_coverage": tabulate(diagnoses, "coverage_bin"),
            "by_preceding_silence": tabulate(diagnoses, "preceding_silence_bin"),
        },
        "timing": {
            "elapsed_seconds": round(elapsed, 3),
            "audio_seconds": round(duration, 3),
        },
        "limitations": [
            "Per-bin figures are diagnostic, not a decomposition: per-turn edit paths do "
            "not sum to the pooled speaker score.",
            "Marginal tables overlap and are non-additive; only 'primary' is exclusive.",
            "Word-to-turn association is by midpoint and is imperfect near boundaries; the "
            "unassigned counts in association_health size that error.",
            "Preceding silence is measured in the reference timeline, not in the audio, so "
            "it describes annotation structure rather than what the recogniser heard.",
            "One speaker in one conversation. Directional evidence only.",
        ],
    }

    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    arguments.report.write_text(f"{raw}\n", encoding="utf-8")
    digest = file_sha256(arguments.report)

    print()
    print(f"mapped/reference speech ratio: {mapped_seconds / reference_seconds:.4f}")
    print(f"unassigned words: {unassigned} ({unassigned_chars} chars)")
    for label, row in sorted(total.items(), key=lambda item: -item[1].get("deletions", 0)):
        print(
            f"{label:24s} turns={row['turn_count']:3d} "
            f"ref_chars={row.get('reference_length', 0):5d} "
            f"del={row.get('deletions', 0):4d} cer={row.get('error_rate', 0):.4f}"
        )
    print(f"network egress attempts: {egress.attempts}")
    print(f"report: {arguments.report} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
