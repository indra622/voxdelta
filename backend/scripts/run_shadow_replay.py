"""Replay a local canary through a primary and a shadow candidate, and publish aggregates.

This is a local replay evaluator, not a live shadow: it routes no production traffic and
observes no real request. It re-runs a fixed validation-only or deterministic synthetic
canary through the incumbent emotion2vec rollback baseline (primary) and the verified,
calibrated XLS-R release (candidate), and reports how they compared.

The XLS-R release is never accepted as its own primary. Both sides are built with network
egress blocked, so a primary that cannot load from local assets is refused rather than
compared against a substitute.

There is deliberately no argument for a final, holdout, package, archive, or test split.
A manifest containing any non-validation record is refused before inference.

There is also no candidate timeout option. A timed-out worker cannot be killed, and this
tool's network-egress guard is process-wide and temporary, so an abandoned worker could
reach the network once the guard is restored. The candidate therefore always runs inline.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from voxdelta.evaluation.shadow_replay import (
    PrimaryKind,
    ShadowReplayError,
    run_shadow_replay,
    write_shadow_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new private directory")
    parser.add_argument(
        "--primary",
        choices=("emotion2vec",),
        default="emotion2vec",
        help="the incumbent rollback baseline; refused unless it loads with egress blocked",
    )
    parser.add_argument(
        "--primary-checkpoint",
        type=Path,
        default=None,
        help="absolute local emotion2vec checkpoint directory for the rollback primary",
    )
    parser.add_argument(
        "--primary-encoder-bundle",
        type=Path,
        default=None,
        help="absolute verified local encoder bundle; a raw package cache is refused",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=None,
        help="validation-only manifest; omit to use deterministic synthetic canary audio",
    )
    parser.add_argument("--synthetic-items", type=int, default=7)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = run_shadow_replay(
            release_path=arguments.release.absolute(),
            calibration_path=arguments.calibration.absolute(),
            primary_kind=cast(PrimaryKind, arguments.primary),
            primary_checkpoint=(
                None
                if arguments.primary_checkpoint is None
                else arguments.primary_checkpoint.absolute()
            ),
            # Passed through unchanged so a relative bundle fails closed.
            primary_encoder_bundle=arguments.primary_encoder_bundle,
            device=arguments.device,
            repeats=arguments.repeats,
            validation_manifest=(
                None
                if arguments.validation_manifest is None
                else arguments.validation_manifest.absolute()
            ),
            synthetic_item_count=arguments.synthetic_items,
        )
        published = write_shadow_report(arguments.output, report)
    except ShadowReplayError as error:
        print(error.code)
        return 2
    except Exception:
        print("shadow_replay_error")
        return 2

    comparison = report.comparison
    gates = report.gates
    print("shadow_replay_ok" if gates.overall_ok else "shadow_replay_gate_failed")
    print("mode local-replay")
    print("live_shadow_traffic false")
    print(f"report {published.name}")
    print(f"primary {report.primary.kind}")
    print(f"primary_encoder_bundle_sha256 {report.primary.encoder_bundle_sha256}")
    print(f"candidate {report.candidate.kind}")
    print(f"release_id {report.candidate.release_id}")
    print(f"bundle_tree_sha256 {report.candidate.bundle_tree_sha256}")
    print(f"calibration_id {report.candidate.calibration_id}")
    print(f"attempted {comparison.attempted_count}")
    print(f"primary_completed {comparison.primary_completed_count}")
    print(f"candidate_completed {comparison.candidate_completed_count}")
    print(f"candidate_errors {comparison.candidate_error_count}")
    print(f"primary_invariant_holds {str(comparison.primary_invariant_holds).lower()}")
    print(f"top_label_agreement_rate {comparison.top_label_agreement_rate:.6f}")
    print(f"candidate_abstention_rate {comparison.candidate_abstention_rate:.6f}")
    print(f"candidate_uncertain_rate {comparison.candidate_uncertain_rate:.6f}")
    print(f"primary_median_ms {report.primary_latency.median_ms:.1f}")
    print(f"candidate_median_ms {report.candidate_latency.median_ms:.1f}")
    print(f"peak_rss_mb {report.runtime.peak_rss_mb}")
    for name, value in sorted(gates.model_dump(mode="json").items()):
        print(f"gate.{name} {value}")
    return 0 if gates.overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
