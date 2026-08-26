"""Measure local production readiness for the promoted XLS-R release and its calibration.

Runs the product's own startup path with both opt-in flags enabled, exercises the
calibrated provider on validation-only or deterministic synthetic canary audio, and
publishes one private, aggregate-only report.

There is deliberately no argument for a final-holdout archive, package, or report: the
sealed split is not reachable from this tool. A manifest containing any non-validation
record is refused before inference. Only stable, path-free codes are printed on failure.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from voxdelta.evaluation.readiness import (
    ReadinessError,
    run_readiness_check,
    write_readiness_report,
)
from voxdelta.providers._emotion_runtime import Device


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new private directory")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=None,
        help="validation-only manifest; omit to use deterministic synthetic canary audio",
    )
    parser.add_argument("--synthetic-items", type=int, default=7)
    parser.add_argument(
        "--advisory-p95-latency-ms",
        type=float,
        default=None,
        help="advisory only; recorded and compared but never blocking",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = run_readiness_check(
            release_path=arguments.release.absolute(),
            calibration_path=arguments.calibration.absolute(),
            device=cast(Device, arguments.device),
            repeats=arguments.repeats,
            validation_manifest=(
                None
                if arguments.validation_manifest is None
                else arguments.validation_manifest.absolute()
            ),
            synthetic_item_count=arguments.synthetic_items,
            advisory_p95_latency_ms=arguments.advisory_p95_latency_ms,
        )
        published = write_readiness_report(arguments.output.absolute(), report)
    except ReadinessError as error:
        print(error.code)
        return 2
    except Exception:
        print("readiness_check_error")
        return 2

    gates = report.gates
    print("readiness_ok" if gates.overall_ok else "readiness_gate_failed")
    print(f"report {published.name}")
    print(f"release_id {report.release.release_id}")
    print(f"bundle_tree_sha256 {report.release.bundle_tree_sha256}")
    print(f"calibration_id {report.calibration.calibration_id}")
    print(f"binding_sha256 {report.calibration.binding_sha256}")
    print(f"selected_device {report.runtime.selected_device}")
    print(f"canary_kind {report.canary.kind}")
    print(f"quality_evidence {str(report.canary.quality_evidence).lower()}")
    print(f"cold_first_inference_ms {report.runtime.cold_first_inference_ms:.1f}")
    print(f"warm_median_ms {report.latency.median_ms:.1f}")
    print(f"warm_p95_ms {report.latency.p95_ms:.1f}")
    print(f"peak_rss_mb {report.runtime.peak_rss_mb}")
    print(f"completion_rate {report.outcome.completion_rate:.6f}")
    print(f"abstention_rate {report.outcome.abstention_rate:.6f}")
    print(f"top_label_agreement_rate {report.outcome.top_label_agreement_rate:.6f}")
    for name, value in sorted(gates.model_dump(mode="json").items()):
        print(f"gate.{name} {value}")
    return 0 if gates.overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
