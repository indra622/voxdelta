"""Local-only Qwen3-ASR benchmark on the six derived KCSC speaker tracks.

Same tracks, references, and Korean-normalised CER scorer as the faster-whisper track
baseline, so the error rates are directly comparable. Model initialization and every
inference call happen inside the network egress guard, so a report that exists is a
report produced without touching the network. Both cached snapshots — the ASR checkpoint
and the forced aligner — are hashed against pinned revisions before the guard opens; a
missing or mismatched cache aborts the run rather than letting the hub client try.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Never

from voxdelta.evaluation.kcsc_asr_benchmark import KcscAsrError, configure_offline_cache
from voxdelta.evaluation.kcsc_qwen_track_benchmark import (
    MAX_NEW_TOKENS,
    PROFILES,
    load_qwen_model,
    run_qwen_track_benchmark,
    verify_qwen_identity,
    write_qwen_report,
)
from voxdelta.evaluation.kcsc_track_asr_benchmark import BENCHMARK_CONVERSATIONS
from voxdelta.providers.base import ProviderError
from voxdelta.providers.offline_guard import block_network_egress
from voxdelta.providers.qwen3_asr import ALIGNER_MODEL_ID

DEFAULT_DERIVED = Path("data/derived/kcsc-tracks")
DEFAULT_CACHE = Path("data/models/hf-cache/hub")
DEFAULT_REPORT_DIR = Path("data/benchmarks")


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        del message
        raise ValueError("invalid arguments")


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--derived", type=Path, default=DEFAULT_DERIVED)
    parser.add_argument("--model-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--device",
        choices=("mps", "cpu", "cuda"),
        default="mps",
        help="what the production provider would select on this host",
    )
    parser.add_argument("--dtype", default=None, help="defaults to the device's usual dtype")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
    except ValueError:
        print("qwen benchmark failed: invalid arguments", file=sys.stderr)
        return 2

    model_id = PROFILES[arguments.profile]
    dtype = arguments.dtype or (
        "bfloat16"
        if arguments.device == "cuda"
        else "float16"
        if arguments.device == "mps"
        else "float32"
    )
    report_path: Path = arguments.report or (
        DEFAULT_REPORT_DIR / f"kcsc-asr-track-{model_id.removeprefix('Qwen/').lower()}.json"
    )

    try:
        identity = verify_qwen_identity(arguments.model_cache, model_id)
        aligner = verify_qwen_identity(arguments.model_cache, ALIGNER_MODEL_ID)
    except KcscAsrError as error:
        print(f"qwen benchmark failed: {error}", file=sys.stderr)
        return 2

    configure_offline_cache(arguments.model_cache)
    print("preflight: KCSC per-speaker ASR, Qwen3-ASR")
    print(f"  conversations   : {', '.join(BENCHMARK_CONVERSATIONS)}")
    print(f"  track set       : {arguments.derived}")
    print(f"  model           : {identity.repo_id} @ {identity.revision[:12]}…")
    print(f"  model tree      : sha256={identity.tree_sha256} ({identity.total_bytes} bytes)")
    print(f"  aligner         : {aligner.repo_id} @ {aligner.revision[:12]}…")
    print(f"  aligner tree    : sha256={aligner.tree_sha256} ({aligner.total_bytes} bytes)")
    print(f"  runtime         : {arguments.device} / {dtype}  max_new_tokens={MAX_NEW_TOKENS}")
    print("  egress          : guarded (model init + inference)")
    sys.stdout.flush()

    try:
        with block_network_egress() as egress:
            model = load_qwen_model(model_id, device=arguments.device, dtype=dtype)
            print("model loaded", flush=True)
            report = run_qwen_track_benchmark(
                derived_root=arguments.derived,
                model=model,
                identity=identity,
                aligner=aligner,
                device=arguments.device,
                dtype=dtype,
                egress_attempts=egress.attempts,
            )
    except KcscAsrError as error:
        print(f"qwen benchmark failed: {error}", file=sys.stderr)
        return 2
    except ProviderError as error:
        print(f"qwen benchmark failed: provider error {error.code}", file=sys.stderr)
        return 2
    except Exception as error:  # noqa: BLE001 - the type is the diagnosis here
        print(f"qwen benchmark failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2

    if egress.attempted:
        print(
            f"qwen benchmark failed: {egress.attempts} network egress attempts to "
            f"{', '.join(egress.hosts)}",
            file=sys.stderr,
        )
        return 2

    for score in report.scores:
        character = score.character
        print(
            f"{score.track_id} "
            f"ref_rows={score.reference_row_count} "
            f"ref_chars={character.reference_length} "
            f"hyp_chars={character.hypothesis_length} "
            f"cer={character.error_rate:.4f} "
            f"(sub={character.substitutions} del={character.deletions} "
            f"ins={character.insertions}) "
            f"eojeol_er={score.eojeol.error_rate:.4f} "
            f"words={score.decode.word_count} "
            f"elapsed={score.elapsed_seconds:.1f}s rtf={score.real_time_factor:.4f}"
        )
    for conversation_id in dict.fromkeys(score.conversation_id for score in report.scores):
        pooled = report.conversation_character(conversation_id)
        print(
            f"{conversation_id} both tracks: "
            f"cer={pooled.error_rate:.4f} "
            f"(sub={pooled.substitutions} del={pooled.deletions} ins={pooled.insertions} "
            f"/ ref_chars={pooled.reference_length})"
        )
    character = report.character
    print(
        f"pooled over {len(report.scores)} tracks: "
        f"cer={character.error_rate:.4f} "
        f"(sub={character.substitutions} del={character.deletions} "
        f"ins={character.insertions} / ref_chars={character.reference_length}) | "
        f"eojeol_er={report.eojeol.error_rate:.4f} [secondary]"
    )
    print(f"network egress attempts: {report.egress_attempts}")
    print(
        f"total elapsed: {report.total_elapsed_seconds:.1f}s "
        f"over {sum(s.duration_seconds for s in report.scores):.1f}s audio"
    )
    digest = write_qwen_report(report, report_path)
    print(f"report: {report_path} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
