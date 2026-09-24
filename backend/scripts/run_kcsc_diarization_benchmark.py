"""Benchmark local diarization on the derived KCSC evaluation set.

Loads pyannote Community-1 from a named local checkpoint, scores the selected
conversations against their derived references, and writes a transcript-free aggregate
report. The whole run — model load included — happens inside the network egress guard,
so a report that exists is a report produced without touching the network.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Never

from voxdelta.credentials import Credentials
from voxdelta.evaluation.kcsc_diarization_benchmark import (
    KcscBenchmarkError,
    run_benchmark,
    write_report,
)
from voxdelta.providers.base import ProviderError
from voxdelta.providers.checkpoints import checkpoint_tree_digest
from voxdelta.providers.offline_guard import block_network_egress
from voxdelta.providers.pyannote_diarization import PyannoteDiarizationProvider

DEFAULT_DERIVED = Path("data/derived/kcsc")
DEFAULT_CHECKPOINT = Path("data/models/speaker-diarization-community-1")
DEFAULT_REPORT = Path("data/benchmarks/kcsc-diarization-community-1.json")

#: Chosen for coverage rather than convenience: the first two are the only conversations
#: whose vendor marker survives trimming and is silenced in the audio, so the derivation's
#: marker path is exercised end to end; the third is the shortest, with a distinct
#: speaker pair and topic.
DEFAULT_CONVERSATIONS = ("A0051_S0001_0", "A0055_S0006_0", "A6000_S0005_0")


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        del message
        raise ValueError("invalid arguments")


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description=__doc__)
    parser.add_argument("--derived", type=Path, default=DEFAULT_DERIVED)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--conversation",
        action="append",
        dest="conversations",
        help="conversation id to score; repeatable, defaults to the standard selection",
    )
    return parser


def _verify_checkpoint(checkpoint: Path) -> str:
    """Fail closed unless the local pipeline is present and hashable."""

    if not checkpoint.is_dir():
        raise KcscBenchmarkError(f"missing local diarization checkpoint: {checkpoint}")
    if not (checkpoint / "config.yaml").is_file():
        raise KcscBenchmarkError(f"local checkpoint has no config.yaml: {checkpoint}")
    try:
        return checkpoint_tree_digest(checkpoint)
    except ValueError:
        raise KcscBenchmarkError(
            f"local checkpoint is not a trustable tree: {checkpoint}"
        ) from None


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
    except ValueError:
        print("benchmark failed: invalid arguments", file=sys.stderr)
        return 2

    conversations = tuple(arguments.conversations or DEFAULT_CONVERSATIONS)
    try:
        model_tree_sha256 = _verify_checkpoint(arguments.checkpoint)
    except KcscBenchmarkError as error:
        print(f"benchmark failed: {error}", file=sys.stderr)
        return 2

    try:
        with block_network_egress() as egress:
            provider = PyannoteDiarizationProvider(Credentials(), model_path=arguments.checkpoint)
            report = run_benchmark(
                derived_root=arguments.derived,
                conversation_ids=conversations,
                diarizer=provider,
                model_name=arguments.checkpoint.name,
                model_tree_sha256=model_tree_sha256,
                egress_attempts=egress.attempts,
            )
    except KcscBenchmarkError as error:
        print(f"benchmark failed: {error}", file=sys.stderr)
        return 2
    except ProviderError as error:
        print(f"benchmark failed: provider error {error.code}", file=sys.stderr)
        return 2
    except Exception:
        print("benchmark failed: unexpected error", file=sys.stderr)
        return 2

    if egress.attempted:
        print(
            f"benchmark failed: {egress.attempts} network egress attempts to "
            f"{', '.join(egress.hosts)}",
            file=sys.stderr,
        )
        return 2

    for score in report.scores:
        collar = score.metrics["collar_250ms"]
        strict = score.metrics["strict"]
        print(
            f"{score.conversation_id} "
            f"ref_turns={score.reference_turn_count} "
            f"ref_speech={score.reference_speech_seconds:.1f}s "
            f"hyp_speakers={score.hypothesis_speaker_count} "
            f"hyp_segments={score.hypothesis_segment_count} "
            f"der={collar['der']:.4f} "
            f"(miss={collar['miss']:.4f} fa={collar['false_alarm']:.4f} "
            f"conf={collar['confusion']:.4f}) "
            f"jer={collar['jer']:.4f} "
            f"strict_der={strict['der']:.4f} "
            f"elapsed={score.elapsed_seconds:.1f}s"
        )
    pooled = report.aggregate["collar_250ms"]
    strict_pooled = report.aggregate["strict"]
    print(
        f"pooled over {report.conversation_count} conversations: "
        f"der={pooled['der']:.4f} miss={pooled['miss']:.4f} "
        f"fa={pooled['false_alarm']:.4f} conf={pooled['confusion']:.4f} "
        f"jer_macro={pooled['jer_macro']:.4f} | strict_der={strict_pooled['der']:.4f}"
    )
    print(f"model: {report.model_name} tree_sha256={report.model_tree_sha256}")
    print(f"source revision: {report.source_revision}")
    print(f"network egress attempts: {report.egress_attempts}")
    digest = write_report(report, arguments.report)
    print(f"report: {arguments.report} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
