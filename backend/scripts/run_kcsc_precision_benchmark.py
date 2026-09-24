"""Comparative pyannoteAI Precision-2 benchmark on three derived KCSC conversations.

This entry point sends audio to a third party. It prints a preflight summary and, unless
``--confirm-external-upload`` is passed, stops there: the default outcome of running this
script is that nothing leaves the host.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Never

from voxdelta.credentials import load_credentials
from voxdelta.evaluation.kcsc_diarization_benchmark import KcscBenchmarkError
from voxdelta.evaluation.kcsc_precision_benchmark import (
    AUTHORIZATION_SCOPE,
    EXTERNAL_PROCESSING_CAVEAT,
    FROZEN_CONVERSATIONS,
    MAX_DIARIZATION_JOBS,
    CallLedger,
    LedgerClient,
    preflight,
    run_precision_benchmark,
    write_evaluation_record,
    write_precision_report,
)
from voxdelta.providers.base import ProviderError
from voxdelta.providers.pyannote_precision import (
    PRECISION_MODEL_ID,
    PyannotePrecisionProvider,
    _default_client_factory,
)

DEFAULT_DERIVED = Path("data/derived/kcsc")
DEFAULT_REPORT = Path("data/benchmarks/kcsc-diarization-precision-2.json")
DEFAULT_RECORD = Path("backend/runtime/poc/pyannoteai-precision2-kcsc-derived/EVALUATION.json")


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        del message
        raise ValueError("invalid arguments")


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description=__doc__)
    parser.add_argument("--derived", type=Path, default=DEFAULT_DERIVED)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument(
        "--confirm-external-upload",
        action="store_true",
        help="required to transmit audio to pyannoteAI; without it the run stops after preflight",
    )
    parser.add_argument(
        "--authorization",
        choices=("user_limited_override",),
        help=(
            "what permits this transfer; required with --confirm-external-upload. "
            "There is no value for 'rights confirmed': this workflow cannot assert that."
        ),
    )
    return parser


def _print_preflight(summary: dict[str, object], *, credentials: object) -> None:
    conversations = summary["conversations"]
    assert isinstance(conversations, list)
    print("preflight: KCSC Precision-2 comparative diarization")
    print(f"  frozen conversation ids : {', '.join(FROZEN_CONVERSATIONS)}")
    for item in conversations:
        assert isinstance(item, dict)
        print(
            f"    {item['conversation_id']} "
            f"audio_sha256={str(item['audio_sha256'])[:12]}… "
            f"reference_sha256={str(item['reference_sha256'])[:12]}… "
            f"bytes={item['audio_bytes']} "
            f"duration={item['duration_seconds']}s  [checksums verified]"
        )
    print(f"  checksums verified      : {len(conversations)}/{len(FROZEN_CONVERSATIONS)}")
    print(f"  source revision         : {summary['source_revision']}")
    print(f"  configured provider     : pyannoteai / {PRECISION_MODEL_ID} (remote=True)")
    key_present = getattr(credentials, "pyannoteai_api_key", None) is not None
    print(f"  api key                 : {'configured' if key_present else 'MISSING'}")
    print(
        f"  planned diarization jobs: {summary['planned_diarization_jobs']} "
        f"(ceiling {MAX_DIARIZATION_JOBS}, retries 0)"
    )
    print("  transmits               : audio only (transcription disabled)")
    print(f"  rights confirmed        : NO (unknown) — {AUTHORIZATION_SCOPE}")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
    except ValueError:
        print("precision benchmark failed: invalid arguments", file=sys.stderr)
        return 2

    try:
        credentials = load_credentials()
    except Exception:
        print("precision benchmark failed: could not load credentials", file=sys.stderr)
        return 2

    try:
        summary = preflight(arguments.derived)
    except KcscBenchmarkError as error:
        print(f"precision benchmark failed: {error}", file=sys.stderr)
        return 2
    _print_preflight(summary, credentials=credentials)

    if not arguments.confirm_external_upload:
        print()
        print("stopping after preflight: nothing was transmitted.")
        print(f"caveat: {EXTERNAL_PROCESSING_CAVEAT}")
        print("re-run with --confirm-external-upload to submit the three jobs.")
        return 0

    if arguments.authorization is None:
        print(
            "precision benchmark failed: --authorization is required to transmit audio",
            file=sys.stderr,
        )
        return 2

    if credentials.pyannoteai_api_key is None:
        print("precision benchmark failed: missing pyannote api key", file=sys.stderr)
        return 2

    ledger = CallLedger()
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    started = time.monotonic()
    try:
        provider = PyannotePrecisionProvider(
            credentials,
            client_factory=lambda *, timeout_seconds: LedgerClient(
                _default_client_factory(timeout_seconds=timeout_seconds), ledger
            ),
        )
        report = run_precision_benchmark(derived_root=arguments.derived, diarizer=provider)
    except KcscBenchmarkError as error:
        print(f"precision benchmark failed: {error}", file=sys.stderr)
        print(
            f"  jobs submitted before stopping: {ledger.submissions}; no retry attempted",
            file=sys.stderr,
        )
        return 2
    except ProviderError as error:
        print(f"precision benchmark failed: provider error {error.code}", file=sys.stderr)
        print(
            f"  jobs submitted before stopping: {ledger.submissions}; no retry attempted",
            file=sys.stderr,
        )
        return 2
    except Exception:
        print("precision benchmark failed: unexpected error", file=sys.stderr)
        print(
            f"  jobs submitted before stopping: {ledger.submissions}; no retry attempted",
            file=sys.stderr,
        )
        return 2
    total_elapsed = time.monotonic() - started

    if ledger.submissions != MAX_DIARIZATION_JOBS:
        print(
            f"precision benchmark failed: expected {MAX_DIARIZATION_JOBS} jobs, "
            f"ledger counted {ledger.submissions}",
            file=sys.stderr,
        )
        return 2

    for score in report.scores:
        collar = score.metrics["collar_250ms"]
        strict = score.metrics["strict"]
        print(
            f"{score.conversation_id} "
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
    print(f"diarization jobs submitted: {ledger.submissions} (retries 0)")
    print(f"hosts contacted: {', '.join(ledger.hosts)}")
    print(f"total elapsed: {total_elapsed:.1f}s")

    report_digest = write_precision_report(
        report, arguments.report, ledger, authorization=arguments.authorization
    )
    record_digest = write_evaluation_record(
        arguments.record,
        report=report,
        ledger=ledger,
        preflight_summary=summary,
        started_at=started_at,
        total_elapsed_seconds=total_elapsed,
        authorization=arguments.authorization,
    )
    print(f"report: {arguments.report} sha256={report_digest}")
    print(f"local record: {arguments.record} sha256={record_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
