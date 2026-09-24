"""Local-only faster-whisper ASR baseline on three derived KCSC conversations.

Model initialization and every inference call happen inside the network egress guard, so
a report that exists is a report produced without touching the network. The cached model
snapshot is hashed before the guard opens; a missing or mismatched cache aborts the run
rather than letting the hub client try to fetch.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Never

from voxdelta.evaluation.kcsc_asr_benchmark import (
    BENCHMARK_CONVERSATIONS,
    KcscAsrError,
    configure_offline_cache,
    load_model,
    run_asr_benchmark,
    verify_model_identity,
    write_report,
)
from voxdelta.providers.base import ProviderError
from voxdelta.providers.offline_guard import block_network_egress

DEFAULT_DERIVED = Path("data/derived/kcsc")
DEFAULT_CACHE = Path("data/models/hf-cache/hub")
DEFAULT_REPORT = Path("data/benchmarks/kcsc-asr-faster-whisper-large-v3-turbo.json")

#: Mirrors the decode configuration VoxDelta's real pipeline uses, recorded so the report
#: states what produced the numbers rather than leaving it to be inferred from source.
DECODE_OPTIONS = {
    "language": "ko",
    "word_timestamps": True,
    "vad_filter": False,
    "beam_size": 5,
}


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        del message
        raise ValueError("invalid arguments")


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description=__doc__)
    parser.add_argument("--derived", type=Path, default=DEFAULT_DERIVED)
    parser.add_argument("--model-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
    except ValueError:
        print("asr benchmark failed: invalid arguments", file=sys.stderr)
        return 2

    try:
        model = verify_model_identity(arguments.model_cache)
    except KcscAsrError as error:
        print(f"asr benchmark failed: {error}", file=sys.stderr)
        return 2

    configure_offline_cache(arguments.model_cache)
    print("preflight: KCSC local ASR baseline")
    print(f"  conversations   : {', '.join(BENCHMARK_CONVERSATIONS)}")
    print(f"  model           : {model.repo_id} @ {model.revision[:12]}…")
    print(f"  model tree      : sha256={model.tree_sha256} ({model.total_bytes} bytes)")
    print(f"  device          : cpu / int8   decode={DECODE_OPTIONS}")
    print("  egress          : guarded (model init + inference)")

    try:
        with block_network_egress() as egress:
            whisper = load_model(device="cpu", compute_type="int8")
            report = run_asr_benchmark(
                derived_root=arguments.derived,
                model=whisper,
                identity=model,
                device="cpu",
                compute_type="int8",
                decode_options=DECODE_OPTIONS,
                egress_attempts=egress.attempts,
            )
    except KcscAsrError as error:
        print(f"asr benchmark failed: {error}", file=sys.stderr)
        return 2
    except ProviderError as error:
        print(f"asr benchmark failed: provider error {error.code}", file=sys.stderr)
        return 2
    except Exception:
        print("asr benchmark failed: unexpected error", file=sys.stderr)
        return 2

    if egress.attempted:
        print(
            f"asr benchmark failed: {egress.attempts} network egress attempts to "
            f"{', '.join(egress.hosts)}",
            file=sys.stderr,
        )
        return 2

    for score in report.scores:
        character = score.character
        overlap = float(str(score.composition["overlapped_speech_fraction"]))
        print(
            f"{score.conversation_id} "
            f"ref_rows={score.reference_row_count} "
            f"ref_chars={character.reference_length} "
            f"hyp_chars={character.hypothesis_length} "
            f"cer={character.error_rate:.4f} "
            f"(sub={character.substitutions} del={character.deletions} "
            f"ins={character.insertions}) "
            f"eojeol_er={score.eojeol.error_rate:.4f} "
            f"overlap={overlap:.4f} "
            f"elapsed={score.elapsed_seconds:.1f}s rtf={score.real_time_factor:.4f}"
        )
    character = report.character
    print(
        f"pooled over {len(report.scores)} conversations: "
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
    digest = write_report(report, arguments.report)
    print(f"report: {arguments.report} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
