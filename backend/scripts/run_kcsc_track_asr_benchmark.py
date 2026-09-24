"""Local-only faster-whisper ASR upper bound on the six derived KCSC speaker tracks.

Same checkpoint, device, compute type, and decode options as the mixed-conversation
baseline, so the gap between the two reports is attributable to speaker separation rather
than to the decoder. Model initialization and every inference call happen inside the
network egress guard, so a report that exists is a report produced without touching the
network. The cached model snapshot is hashed before the guard opens; a missing or
mismatched cache aborts the run rather than letting the hub client try.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Never

from voxdelta.evaluation.kcsc_asr_benchmark import (
    KcscAsrError,
    configure_offline_cache,
    load_model,
    verify_model_identity,
)
from voxdelta.evaluation.kcsc_track_asr_benchmark import (
    BENCHMARK_CONVERSATIONS,
    run_track_asr_benchmark,
    write_track_report,
)
from voxdelta.providers.base import ProviderError
from voxdelta.providers.offline_guard import block_network_egress

DEFAULT_DERIVED = Path("data/derived/kcsc-tracks")
DEFAULT_CACHE = Path("data/models/hf-cache/hub")
DEFAULT_REPORT = Path("data/benchmarks/kcsc-asr-track-faster-whisper-large-v3-turbo.json")

#: Identical to the mixed baseline's options. A difference here would make the two
#: reports non-subtractable, so they are restated rather than tuned.
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
        print("track asr benchmark failed: invalid arguments", file=sys.stderr)
        return 2

    try:
        model = verify_model_identity(arguments.model_cache)
    except KcscAsrError as error:
        print(f"track asr benchmark failed: {error}", file=sys.stderr)
        return 2

    configure_offline_cache(arguments.model_cache)
    print("preflight: KCSC per-speaker ASR upper bound")
    print(f"  conversations   : {', '.join(BENCHMARK_CONVERSATIONS)}")
    print(f"  track set       : {arguments.derived}")
    print(f"  model           : {model.repo_id} @ {model.revision[:12]}…")
    print(f"  model tree      : sha256={model.tree_sha256} ({model.total_bytes} bytes)")
    print(f"  device          : cpu / int8   decode={DECODE_OPTIONS}")
    print("  egress          : guarded (model init + inference)")

    try:
        with block_network_egress() as egress:
            whisper = load_model(device="cpu", compute_type="int8")
            report = run_track_asr_benchmark(
                derived_root=arguments.derived,
                model=whisper,
                identity=model,
                device="cpu",
                compute_type="int8",
                decode_options=DECODE_OPTIONS,
                egress_attempts=egress.attempts,
            )
    except KcscAsrError as error:
        print(f"track asr benchmark failed: {error}", file=sys.stderr)
        return 2
    except ProviderError as error:
        print(f"track asr benchmark failed: provider error {error.code}", file=sys.stderr)
        return 2
    except Exception:
        print("track asr benchmark failed: unexpected error", file=sys.stderr)
        return 2

    if egress.attempted:
        print(
            f"track asr benchmark failed: {egress.attempts} network egress attempts to "
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
            f"speech={score.composition['reference_speech_seconds']}s "
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
    digest = write_track_report(report, arguments.report)
    print(f"report: {arguments.report} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
