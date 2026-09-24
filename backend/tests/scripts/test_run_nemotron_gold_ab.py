"""Contract tests for the local-only Nemotron user-Gold A/B script."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from voxdelta.providers.nemotron_diarization import CommandResult

_SPEC = importlib.util.spec_from_file_location(
    "run_nemotron_gold_ab",
    Path(__file__).resolve().parents[2] / "scripts" / "run_nemotron_gold_ab.py",
)
assert _SPEC and _SPEC.loader
script = importlib.util.module_from_spec(_SPEC)
# The script declares dataclasses, which resolve their module through sys.modules.
sys.modules[_SPEC.name] = script
_SPEC.loader.exec_module(script)


def _artifact(path: Path, record: dict[str, Any]) -> Path:
    path.write_text(json.dumps({"evaluations": [record]}), encoding="utf-8")
    return path


def _scored_record(**updates: object) -> dict[str, Any]:
    record: dict[str, Any] = {
        "conversation_id": "CASE",
        "gold_sha256": "g" * 64,
        "audio_sha256": "a" * 64,
        "pipeline": {"hypothesis_speakers": 2, "hypothesis_segments": 9},
        "precision_qwen": {
            "diarization": {"strict": {"der": 0.1}, "collar_250ms": {"der": 0.05}},
            "mixed_stream_cer": {"cer": 0.3},
        },
    }
    record.update(updates)
    return record


def test_precision_baseline_is_copied_only_for_identical_inputs(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path / "prior.json", _scored_record())

    copied = script.precision_baseline(artifact, "CASE", "g" * 64, "a" * 64, 2)

    assert copied["status"] == "copied_from_existing_artifact_no_new_remote_call"
    assert copied["metrics"] == {"strict": {"der": 0.1}, "collar_250ms": {"der": 0.05}}
    assert copied["speaker_count_error"] == 0
    assert "mixed_stream_cer" not in json.dumps(copied)
    for gold, audio in (("x" * 64, "a" * 64), ("g" * 64, "x" * 64)):
        refused = script.precision_baseline(artifact, "CASE", gold, audio, 2)
        assert refused["status"] == "not_comparable_gold_or_audio_digest_changed"
        assert "metrics" not in refused


def test_precision_failure_is_reported_not_fabricated(tmp_path: Path) -> None:
    artifact = _artifact(
        tmp_path / "prior.json",
        _scored_record(
            precision_qwen=None,
            evaluation_status="pipeline_failed_before_metrics",
            pipeline={
                "public_state": {"stages": {"diarize": {"error_code": "unsupported_speaker_count"}}}
            },
        ),
    )

    result = script.precision_baseline(artifact, "CASE", "g" * 64, "a" * 64, 3)

    assert result == {
        "status": "no_metrics_in_prior_run",
        "artifact": "prior.json",
        "prior_evaluation_status": "pipeline_failed_before_metrics",
        "prior_diarize_error_code": "unsupported_speaker_count",
    }


def test_precision_absent_or_missing_artifacts_are_explicit(tmp_path: Path) -> None:
    assert script.precision_baseline(None, "CASE", "g", "a", 1)["status"] == (
        "not_available_no_prior_precision_run_for_this_case"
    )
    missing = script.precision_baseline(tmp_path / "absent.json", "CASE", "g", "a", 1)
    assert missing["status"] == "not_available_artifact_missing"
    other = _artifact(tmp_path / "other.json", _scored_record(conversation_id="OTHER"))
    assert script.precision_baseline(other, "CASE", "g", "a", 1)["status"] == (
        "not_available_case_absent_from_artifact"
    )


def test_gold_turns_keep_timing_and_speaker_only() -> None:
    gold = {
        "content": {
            "turns": [
                {"start": 2.0, "end": 3.0, "speaker": "B", "transcript": "private words"},
                {"start": 0.0, "end": 1.5, "speaker": "A", "transcript": "private words"},
            ]
        }
    }
    turns = script.gold_turns(gold, 5.0)
    assert turns == [(0.0, 1.5, "A"), (2.0, 3.0, "B")]
    assert "private" not in json.dumps(turns)


@pytest.mark.parametrize(
    "turn",
    [
        {"start": 0.0, "end": 6.0, "speaker": "A"},
        {"start": 1.0, "end": 1.0, "speaker": "A"},
        {"start": -1.0, "end": 1.0, "speaker": "A"},
        {"start": True, "end": 1.0, "speaker": "A"},
        {"start": 0.0, "end": 1.0, "speaker": ""},
        {"start": "0", "end": 1.0, "speaker": "A"},
    ],
)
def test_gold_turns_reject_invalid_labels(turn: dict[str, object]) -> None:
    with pytest.raises(script.EvaluationError):
        script.gold_turns({"content": {"turns": [turn]}}, 5.0)


def test_sandboxed_runner_denies_network_around_the_unchanged_command() -> None:
    seen: list[list[str]] = []

    class Inner:
        def __call__(
            self, argv: Sequence[str], *, env: Mapping[str, str], timeout_seconds: float
        ) -> CommandResult:
            del env, timeout_seconds
            seen.append(list(argv))
            return CommandResult(0, b"")

    runner = script.SandboxedRunner(Inner())
    runner(["/opt/nemo-speech", "--version"], env={}, timeout_seconds=1.0)

    assert seen == [
        [
            "/usr/bin/sandbox-exec",
            "-p",
            "(version 1)(allow default)(deny network*)",
            "/opt/nemo-speech",
            "--version",
        ]
    ]
    assert runner.invocations == 1


def test_timing_reports_median_wall_time_and_rtfx() -> None:
    assert script.timing([0.5, 0.7, 0.6], 60.0) == {
        "runs": 3,
        "wall_seconds": [0.5, 0.7, 0.6],
        "median_wall_seconds": 0.6,
        "median_rtfx": 100.0,
    }


def test_manifest_entry_rejects_nested_or_absent_items(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "items": [
                    {"id": "A", "file": "a.wav", "sha256": "0" * 64, "duration_seconds": 1.0},
                    {"id": "B", "file": "../b.wav", "sha256": "0" * 64, "duration_seconds": 1},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert script.manifest_entry(manifest, "A") == (tmp_path / "a.wav", "0" * 64, 1.0)
    with pytest.raises(script.EvaluationError):
        script.manifest_entry(manifest, "B")
    with pytest.raises(script.EvaluationError):
        script.manifest_entry(manifest, "C")


def test_existing_report_is_never_overwritten(tmp_path: Path, capsys: Any) -> None:
    output = tmp_path / "report.json"
    output.write_text("prior", encoding="utf-8")

    code = script.main(
        [
            "--executable",
            str(tmp_path / "nemo-speech"),
            "--model",
            str(tmp_path / "m.gguf"),
            "--runtime-source-commit",
            "0" * 40,
            "--output",
            str(output),
        ]
    )

    assert code == 2
    assert output.read_text(encoding="utf-8") == "prior"
    assert "refusing to overwrite" in capsys.readouterr().err
