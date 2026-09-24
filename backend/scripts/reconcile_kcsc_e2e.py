"""Write an additive reconciliation for the one-conversation KCSC E2E record.

Local and metadata-only. Reads the evaluation record and the job wrapper's state file,
writes a second artifact beside them, and touches neither. No audio, transcript, segment,
or credential is read, and nothing external is contacted.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Never

from voxdelta.evaluation.e2e_reconciliation import reconcile, write_reconciliation

DEFAULT_RECORD = Path("backend/runtime/poc/kcsc-precision2-qwen-xlsr-e2e/EVALUATION.json")
DEFAULT_OUTPUT = Path("backend/runtime/poc/kcsc-precision2-qwen-xlsr-e2e/RECONCILIATION.json")
DEFAULT_STATE = Path("data/jobs/benchmarks/kcsc-precision-e2e.state.json")

#: The scope that was actually exercised, which is narrower than the programme scope the
#: original record quoted. One conversation, one upload, one job, no retries.
CORRECTED_SCOPE: dict[str, object] = {
    "conversation_ids": ["A6000_S0005_0"],
    "conversation_count": 1,
    "uploads": 1,
    "diarization_jobs": 1,
    "retries": 0,
    "authorization": "user_limited_override",
    "rights_confirmed": False,
    "rights_holder": "Beijing Magic Data Technology Co., Ltd.",
    "note": (
        "Operator override limited to exactly one derived KCSC conversation "
        "(A6000_S0005_0), one upload and one diarization job. Not a general permission "
        "to transmit this corpus, and not evidence that the transfer was licensed."
    ),
}

SCORING_NOTE = (
    "The post-run diarization scorer was called with a manifest entry containing only "
    "conversation_id. _reference_annotation also requires speakers and validates "
    "derived_duration_seconds, so it raised KcscBenchmarkError before any DER or JER was "
    "computed. The reference file itself is valid; the defect was in the caller."
)

RECOVERY_NOTE = (
    "Not recoverable from local state. The hypothesis segments existed only inside the "
    "run's scratch root, which the runner removed on completion, and the evaluation record "
    "keeps segment counts rather than segment boundaries. Recomputing DER would require a "
    "second submission to the external provider, which the authorization does not cover."
)


#: The original record asserted that no model was downloaded. Its run never pinned the
#: hub client offline, and a "Fetching 2 files" progress bar appears in its log, so the
#: assertion was never established either way. It is withdrawn rather than re-argued.
WITHDRAWN_CLAIMS: tuple[dict[str, object], ...] = (
    {
        "field": "local.no_model_downloads",
        "original_value": True,
        "status": "withdrawn_unverified",
        "why": (
            "The run asserted this without configuring HF_HUB_CACHE or the offline flags, "
            "so nothing constrained or observed the hub client. Its log contains a "
            "'Fetching 2 files' progress bar lasting 5m42s. A later local inspection found "
            "no new files in either the default or the project hub cache and no HF "
            "environment variables set, but that is absence of evidence, not verification: "
            "the run had no mechanism that could have established the claim."
        ),
        "what_is_not_claimed": [
            "that no model bytes were downloaded during the run",
            "that the run was network-isolated apart from the one authorized call",
        ],
        "remediation": (
            "verify_local_models now resolves and hashes every checkpoint out of the local "
            "cache and calls configure_offline_cache before any model is constructed, and "
            "fails closed if the cache is incomplete. Future records report "
            "hub_offline_enforced and hub_cache_pinned instead of an unbacked assertion."
        ),
        "applies_to_original_run": True,
        "reverified": False,
    },
)


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        del message
        raise ValueError("invalid arguments")


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
    except ValueError:
        print("reconciliation failed: invalid arguments", file=sys.stderr)
        return 2

    wrapper_state: dict[str, object] = {}
    if arguments.state.is_file():
        loaded = json.loads(arguments.state.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            wrapper_state = loaded

    try:
        record = reconcile(
            arguments.record,
            corrected_scope=CORRECTED_SCOPE,
            wrapper_state=wrapper_state,
            scoring_note=SCORING_NOTE,
            recoverable=False,
            recovery_note=RECOVERY_NOTE,
            withdrawn_claims=WITHDRAWN_CLAIMS,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"reconciliation failed: {error}", file=sys.stderr)
        return 2

    digest = write_reconciliation(record, arguments.output)
    outcome = record["outcome"]
    assert isinstance(outcome, dict)
    reconciles = record["reconciles"]
    assert isinstance(reconciles, dict)

    print("reconciliation: KCSC one-conversation E2E")
    print(f"  reconciles          : {reconciles['path']}")
    print(f"  original sha256     : {reconciles['sha256']}")
    print(f"  pipeline status     : {outcome['pipeline_status']}")
    print(f"  scoring status      : {outcome['scoring_status']}")
    print(f"  overall status      : {outcome['overall_status']}")
    print(f"  scope conversations : {CORRECTED_SCOPE['conversation_ids']}")
    print(f"  rights confirmed    : {CORRECTED_SCOPE['rights_confirmed']}")
    for claim in WITHDRAWN_CLAIMS:
        print(f"  withdrawn claim     : {claim['field']} ({claim['status']})")
    print(f"  written             : {arguments.output} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
