"""Score speaker-attributed Qwen ASR on the preserved Precision-2 timeline. Local only.

Nothing leaves this machine. The diarized timeline is read out of the immutable retry-2
record rather than recomputed, so no audio is submitted anywhere; the recogniser is loaded
from the pinned local cache with the hub client forced offline, and every inference call
happens inside the egress guard. Both pinned inputs are verified by digest, and the model
cache is proven complete, before any model is constructed — a cache problem stops the run
while stopping is still free.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Never

from voxdelta.evaluation.kcsc_asr_benchmark import KcscAsrError, configure_offline_cache
from voxdelta.evaluation.kcsc_attributed_asr import (
    KcscAttributedError,
    PinnedInput,
    attribute,
    build_report,
    load_timeline,
    map_speakers,
    reference_rows,
    sanitize,
    score,
    time_records,
    write_report,
)
from voxdelta.evaluation.kcsc_diarization_benchmark import file_sha256
from voxdelta.evaluation.kcsc_qwen_track_benchmark import load_qwen_model, verify_qwen_identity
from voxdelta.providers.base import ProviderError
from voxdelta.providers.offline_guard import block_network_egress
from voxdelta.providers.qwen3_asr import ALIGNER_MODEL_ID, DEFAULT_MODEL_ID

CONVERSATION_ID = "A6000_S0005_0"
RECORD = Path("backend/runtime/poc/kcsc-precision2-qwen-xlsr-e2e/EVALUATION-retry-2.json")
RECORD_SHA256 = "4829014c3756194ce99e70102a35c83b5983db5c6dceb6bca7b481964ba7de34"
AUDIO = Path(f"data/derived/kcsc/audio/{CONVERSATION_ID}.wav")
AUDIO_SHA256 = "01565974fdf7021e0d0afcbae8c040839777434f40416f1e265aeee3926cce27"
REFERENCE = Path(f"data/derived/kcsc/reference/{CONVERSATION_ID}.json")
MODEL_CACHE = Path("data/models/hf-cache/hub")
DEFAULT_REPORT = Path("data/benchmarks/kcsc-precision-qwen-e2e-asr-retry2.json")


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        del message
        raise ValueError("invalid arguments")


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=RECORD)
    parser.add_argument("--audio", type=Path, default=AUDIO)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--model-cache", type=Path, default=MODEL_CACHE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--supersedes",
        type=Path,
        default=None,
        help="a retracted artifact this run replaces, referenced by digest",
    )
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
    except ValueError:
        print("attributed asr failed: invalid arguments", file=sys.stderr)
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
        reference = reference_rows(arguments.reference)
        mapping, evidence = map_speakers(segments, arguments.reference)
        if not arguments.model_cache.is_dir():
            raise KcscAttributedError(f"missing local hub cache: {arguments.model_cache}")
        identity = verify_qwen_identity(arguments.model_cache, DEFAULT_MODEL_ID)
        aligner = verify_qwen_identity(arguments.model_cache, ALIGNER_MODEL_ID)
    except (KcscAttributedError, KcscAsrError) as error:
        print(f"attributed asr failed: {error}", file=sys.stderr)
        return 2

    # Only after the cache is proven complete, so the pin never covers for a missing file.
    configure_offline_cache(arguments.model_cache)
    duration = float(record["input"]["duration_seconds"])  # type: ignore[index]
    reference_digest = file_sha256(arguments.reference)
    event_count = len(reference.events)
    turn_count = reference.turn_count

    print("preflight: speaker-attributed Qwen ASR on a preserved timeline")
    print(f"  conversation        : {CONVERSATION_ID}")
    print(f"  retry-2 record      : sha256={record_digest}")
    print(f"  derived audio       : sha256={audio_digest}")
    print(f"  diarized segments   : {len(segments)} (reused, not recomputed)")
    print(f"  reference turns     : {turn_count} + {event_count} unattributed")
    print(f"  speaker mapping     : {dict(sorted(mapping.items()))} by temporal overlap")
    print(f"  mapping margin      : {evidence['margin_seconds']}s")
    print(f"  model               : {identity.repo_id} @ {identity.revision[:12]}…")
    print(f"  model tree          : sha256={identity.tree_sha256}")
    print(f"  runtime             : {arguments.device}/{dtype}, hub offline")
    print("  external calls      : 0 (no audio leaves this machine)")
    sys.stdout.flush()

    started = time.monotonic()
    try:
        with block_network_egress() as egress:
            model = load_qwen_model(DEFAULT_MODEL_ID, device=arguments.device, dtype=dtype)
            print("model loaded", flush=True)
            outputs = model.transcribe(
                str(arguments.audio), language="Korean", return_time_stamps=True
            )
            results = list(outputs)  # type: ignore[call-overload]
            if len(results) != 1:
                raise KcscAttributedError(f"expected one result, got {len(results)}")
            stamps = getattr(results[0], "time_stamps", None)
            if stamps is None:
                raise KcscAttributedError("timestamps were requested but none returned")
            words, coverage = sanitize(time_records(stamps), duration, segments)
            # The unattributed stream, in word order, is the inherent-error hypothesis.
            stream_text = " ".join(word.text for word in words)
            hypothesis_by_speaker, omitted, utterance_count = attribute(words, segments, duration)
    except (KcscAttributedError, KcscAsrError) as error:
        print(f"attributed asr failed: {error}", file=sys.stderr)
        return 2
    except ProviderError as error:
        print(f"attributed asr failed: provider error {error.code}", file=sys.stderr)
        return 2
    except Exception as error:  # noqa: BLE001 - the type is the diagnosis here
        print(f"attributed asr failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    elapsed = time.monotonic() - started

    if egress.attempted:
        print(f"attributed asr failed: {egress.attempts} egress attempts", file=sys.stderr)
        return 2

    scores = score(
        stream_text=stream_text,
        hypothesis_by_speaker=hypothesis_by_speaker,
        reference=reference,
        mapping=mapping,
    )
    # stream_text and hypothesis_by_speaker go out of scope here; nothing below prints text.
    del stream_text, hypothesis_by_speaker, reference

    report = build_report(
        scores=scores,
        coverage=coverage.with_alignment_omissions(omitted),
        mapping=mapping,
        mapping_evidence=evidence,
        identity=identity,
        aligner=aligner,
        inputs={
            "conversation_id": CONVERSATION_ID,
            "supersedes": str(arguments.supersedes) if arguments.supersedes else "",
            "supersedes_sha256": (
                file_sha256(arguments.supersedes)
                if arguments.supersedes and arguments.supersedes.is_file()
                else ""
            ),
            "retry2_record": str(arguments.record),
            "retry2_record_sha256": record_digest,
            "derived_audio": str(arguments.audio),
            "derived_audio_sha256": audio_digest,
            "reference": str(arguments.reference),
            "reference_sha256": reference_digest,
        },
        segment_count=len(segments),
        reference_turn_count=turn_count,
        reference_event_count=event_count,
        utterance_count=utterance_count,
        omitted_by_alignment=omitted,
        duration_seconds=duration,
        elapsed_seconds=elapsed,
        egress_attempts=egress.attempts,
        device=arguments.device,
        dtype=dtype,
    )
    digest = write_report(report, arguments.report)

    mixed = scores.mixed_stream
    pooled = scores.speaker_attributed_pooled
    print()
    for label, counts in sorted(scores.per_speaker.items()):
        print(
            f"{label} -> {mapping[label]}: cer={counts.error_rate:.4f} "
            f"(sub={counts.substitutions} del={counts.deletions} ins={counts.insertions} "
            f"/ ref_chars={counts.reference_length})"
        )
    print(
        f"mixed-stream (time order) : cer={mixed.error_rate:.4f} "
        f"/ ref_chars={mixed.reference_length}"
    )
    print(
        f"speaker-attributed pooled : cer={pooled.error_rate:.4f} "
        f"/ ref_chars={pooled.reference_length}"
    )
    print(f"e2e vs mixed-stream delta : {scores.e2e_vs_mixed_stream_delta:+.4f} (observed)")
    print(f"words omitted by alignment: {omitted}")
    print(f"network egress attempts   : {egress.attempts}")
    print(f"elapsed                   : {elapsed:.1f}s")
    print(f"report                    : {arguments.report} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
