from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from voxdelta.jobs.artifacts import ArtifactStore
from voxdelta.jobs.logging import PipelineLogger, redact


def test_redact_recurses_through_mappings_and_lists_case_insensitively() -> None:
    value = {
        "ApiKey": "secret-1",
        "nested": [
            {"accessTOKEN": "secret-2", "safe": "visible"},
            {"Authorization": "secret-3", "Transcript": "private speech"},
            {"provider_payload": {"deep": "private response"}},
        ],
    }

    assert redact(value) == {
        "ApiKey": "[REDACTED]",
        "nested": [
            {"accessTOKEN": "[REDACTED]", "safe": "visible"},
            {"Authorization": "[REDACTED]", "Transcript": "[REDACTED]"},
            {"provider_payload": "[REDACTED]"},
        ],
    }


def test_logger_writes_one_safe_json_object_per_event(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "jobs")
    logger = PipelineLogger(store)

    logger.event(
        job_id="j1",
        stage="emotion",
        event="failed",
        duration_ms=1.25,
        provider="fake",
        model="v1",
        error_code="validation_error",
        metadata={"payload": "private", "safe": object()},
    )

    lines = (store.job_dir("j1") / "pipeline.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["job_id"] == "j1"
    assert parsed["stage"] == "emotion"
    assert parsed["event"] == "failed"
    assert parsed["duration_ms"] == 1.25
    assert parsed["provider"] == "fake"
    assert parsed["model"] == "v1"
    assert parsed["error_code"] == "validation_error"
    assert parsed["metadata"] == {"payload": "[REDACTED]", "safe": "[UNSERIALIZABLE]"}


@pytest.mark.skipif(os.name != "posix", reason="mode bits are POSIX-specific")
def test_diagnostics_require_opt_in_use_safe_names_and_mode_0600(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "jobs")
    logger = PipelineLogger(store)

    assert logger.diagnostic("j1", False, "request", {"payload": "private"}) is None
    assert not (store.job_dir("j1") / "diagnostics").exists()

    path = logger.diagnostic(
        "j1",
        True,
        "../Emotion Request!",
        {"api_key": "secret", "payload": "private", "safe": "visible"},
    )

    assert path is not None
    assert path.parent == store.job_dir("j1") / "diagnostics"
    assert ".." not in path.name
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_bytes()) == {
        "api_key": "[REDACTED]",
        "payload": "[REDACTED]",
        "safe": "visible",
    }
