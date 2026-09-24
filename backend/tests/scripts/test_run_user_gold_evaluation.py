"""Contract tests for transcript-free failure reporting in gold evaluation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "run_user_gold_evaluation",
    Path(__file__).resolve().parents[2] / "scripts" / "run_user_gold_evaluation.py",
)
assert _SPEC and _SPEC.loader
script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(script)


def test_public_pipeline_state_keeps_only_status_and_error_code() -> None:
    result = script._public_pipeline_state(
        {
            "status": "failed",
            "stages": {
                "diarize": {
                    "status": "failed",
                    "error_json": json.dumps(
                        {
                            "code": "provider_unavailable",
                            "message": "private audio transcript must not escape",
                        }
                    ),
                },
                "transcribe": {"status": "pending", "error_json": None},
            },
        }
    )

    assert result == {
        "status": "failed",
        "stages": {
            "diarize": {"status": "failed", "error_code": "provider_unavailable"},
            "transcribe": {"status": "pending", "error_code": None},
        },
    }
    assert "private" not in json.dumps(result)
