"""Contract tests for the local-only Nemotron KCSC A/B script."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import wave
from pathlib import Path
from typing import Any

import pytest

from voxdelta.domain.models import AudioAsset, SpeakerSegment
from voxdelta.providers.base import DiarizationTimelines

_SPEC = importlib.util.spec_from_file_location(
    "run_nemotron_kcsc_ab",
    Path(__file__).resolve().parents[2] / "scripts" / "run_nemotron_kcsc_ab.py",
)
assert _SPEC and _SPEC.loader
script = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = script
_SPEC.loader.exec_module(script)

MANIFEST_SHA = "m" * 64
REVISION = "r" * 40
SECRET_TEXT = "private words"


def _metrics(der: float, jer: float, speech: float = 10.0) -> dict[str, dict[str, float]]:
    return {
        variant: {
            "der": der,
            "jer": jer,
            "miss_seconds": der * speech,
            "false_alarm_seconds": 0.0,
            "confusion_seconds": 0.0,
            "scored_speech_seconds": speech,
        }
        for variant in ("strict", "collar_250ms")
    }


def _historical_artifact(path: Path, **dataset: object) -> Path:
    payload = {
        "dataset": {
            "derivation_manifest_sha256": MANIFEST_SHA,
            "source_revision": REVISION,
            **dataset,
        },
        "model": {"name": "precision-2", "checkpoint_tree_sha256": None, "remote": True},
        "conversations": [
            {
                "conversation_id": "C1",
                "duration_seconds": 60.0,
                "reference_turn_count": 12,
                "hypothesis_speaker_count": 2,
                "hypothesis_segment_count": 9,
                "elapsed_seconds": 20.0,
                "metrics": _metrics(0.1, 0.12),
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_sign_test_is_exact_and_drops_ties() -> None:
    assert script.sign_test_p(5, 0) == 0.0625
    assert script.sign_test_p(7, 0) == 0.015625
    assert script.sign_test_p(3, 3) == 1.0
    assert script.sign_test_p(0, 0) is None


def test_historical_artifact_is_cited_only_for_the_same_manifest(tmp_path: Path) -> None:
    artifact = _historical_artifact(tmp_path / "prior.json")

    index = script.load_historical(artifact, manifest_sha256=MANIFEST_SHA, source_revision=REVISION)
    case = script.historical_case(index, "C1", 60.0, 12)

    assert index["status"] == "verified_historical_artifact_no_new_run"
    assert index["remote"] is True
    assert case["status"] == "verified_historical_artifact_no_new_run"
    assert case["metrics"]["strict"]["der"] == 0.1
    assert case["speaker_count_error"] == 0
    for manifest, revision in (("x" * 64, REVISION), (MANIFEST_SHA, "x" * 40)):
        refused = script.load_historical(
            artifact, manifest_sha256=manifest, source_revision=revision
        )
        assert refused["status"] == "not_comparable_manifest_or_revision_changed"
        assert script.historical_case(refused, "C1", 60.0, 12) == {
            "status": "not_comparable_manifest_or_revision_changed"
        }


def test_historical_case_refuses_absent_or_reshaped_references(tmp_path: Path) -> None:
    index = script.load_historical(
        _historical_artifact(tmp_path / "prior.json"),
        manifest_sha256=MANIFEST_SHA,
        source_revision=REVISION,
    )
    assert script.historical_case(index, "C2", 60.0, 12)["status"] == (
        "not_available_case_absent_from_artifact"
    )
    assert script.historical_case(index, "C1", 60.0, 13)["status"] == (
        "not_comparable_reference_shape_changed"
    )
    missing = script.load_historical(
        tmp_path / "absent.json", manifest_sha256=MANIFEST_SHA, source_revision=REVISION
    )
    assert missing["status"] == "not_available_artifact_missing"


def test_reproduction_names_the_matching_timeline_only() -> None:
    historical = {"metrics": _metrics(0.1, 0.12)}
    current = {
        "exclusive": {"metrics": _metrics(0.2, 0.22)},
        "overlap_aware_evidence": {"metrics": _metrics(0.1, 0.12)},
    }
    assert script.reproduced_timeline(historical, current) == "overlap_aware_evidence"
    current["overlap_aware_evidence"] = {"metrics": _metrics(0.1000011, 0.12)}
    assert script.reproduced_timeline(historical, current) == "not_reproduced"
    assert script.reproduced_timeline({"status": "x"}, current) == "not_checked"


def _evaluation(conversation_id: str, pair: str, a: float, b: float) -> dict[str, Any]:
    return {
        "conversation_id": conversation_id,
        "speaker_pair": pair,
        "systems": {
            "a": {"exclusive": {"metrics": _metrics(a, a)}},
            "b": {"exclusive": {"metrics": _metrics(b, b)}},
        },
    }


def test_pair_aggregate_treats_speaker_pairs_not_sessions_as_units() -> None:
    evaluations = [
        _evaluation("S1", "P1", 0.10, 0.20),
        _evaluation("S2", "P1", 0.30, 0.20),
        _evaluation("S3", "P2", 0.10, 0.30),
        {"conversation_id": "S4", "speaker_pair": "P3", "systems": {"a": {}}},
    ]

    result = script.pair_aggregate(evaluations, {"a": "exclusive", "b": "exclusive"}, "a")

    assert result is not None
    assert result["independent_units"] == 2
    assert result["sessions"] == 3
    assert result["per_pair"]["P1"]["sessions"] == ["S1", "S2"]
    assert result["per_pair"]["P1"]["a"]["strict"]["der"] == pytest.approx(0.2)
    delta = result["across_pairs"]["strict"]["a_minus_b_der"]
    assert delta["n"] == 2
    assert delta["reference_better_pairs"] == 1
    assert delta["reference_worse_pairs"] == 0
    assert delta["sign_test_two_sided_p"] == 1.0
    assert script.pair_aggregate([evaluations[3]], {"a": "exclusive"}, "a") is None


def _write_wav(path: Path, seconds: float) -> None:
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16_000)
        target.writeframes(b"\x00\x00" * int(seconds * 16_000))


def _derived_set(root: Path) -> Path:
    derived = root / "derived" / "kcsc"
    (derived / "audio").mkdir(parents=True)
    (derived / "reference").mkdir()
    conversations = []
    for conversation_id, speakers, corrupt in (
        ("C1", ["GA", "GB"], False),
        ("C2", ["GA", "GB"], True),
        ("C3", ["GA", "GB", "GC"], False),
    ):
        audio = derived / "audio" / f"{conversation_id}.wav"
        _write_wav(audio, 10.0)
        reference = derived / "reference" / f"{conversation_id}.json"
        reference.write_text(
            json.dumps(
                {
                    "schema_version": "1",
                    "conversation_id": conversation_id,
                    "duration_seconds": 10.0,
                    "turns": [
                        {"start": 0.0, "end": 4.0, "speaker": "GA", "transcript": SECRET_TEXT},
                        {"start": 5.0, "end": 9.0, "speaker": "GB", "transcript": SECRET_TEXT},
                    ],
                }
            ),
            encoding="utf-8",
        )
        digest = hashlib.sha256(audio.read_bytes()).hexdigest()
        conversations.append(
            {
                "conversation_id": conversation_id,
                "speakers": speakers,
                "derived_duration_seconds": 10.0,
                "outputs": {
                    "audio": f"audio/{conversation_id}.wav",
                    "audio_sha256": "0" * 64 if corrupt else digest,
                    "reference": f"reference/{conversation_id}.json",
                    "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
                },
            }
        )
    (derived / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "kcsc-derived-evaluation-set",
                "source": {"dataset": "corpus", "revision": REVISION},
                "conversations": conversations,
            }
        ),
        encoding="utf-8",
    )
    return derived


class _FakeNemotron:
    def __init__(self) -> None:
        self.calls = 0

    def diarize_unconstrained(self, asset: AudioAsset) -> DiarizationTimelines:
        self.calls += 1
        segments = [
            SpeakerSegment(start=0.0, end=4.0, speaker_id="SPEAKER_00", confidence=1.0),
            SpeakerSegment(start=5.0, end=9.5, speaker_id="SPEAKER_01", confidence=1.0),
        ]
        return DiarizationTimelines(evidence=segments, exclusive=list(segments))


def test_evaluate_scores_valid_inputs_and_records_exclusions(tmp_path: Path) -> None:
    _derived_set(tmp_path)
    provider = _FakeNemotron()

    result = script.evaluate(
        data_root=tmp_path,
        provider=provider,  # type: ignore[arg-type]
        repeats=2,
        community_checkpoint=None,
        community_repeats=1,
    )

    assert provider.calls == 2
    assert [item["conversation_id"] for item in result["evaluations"]] == ["C1"]
    assert result["dataset"]["excluded"] == [
        {"conversation_id": "C2", "reason": "input_verification_failed"},
        {"conversation_id": "C3", "reason": "not_two_speaker"},
    ]
    nemotron = result["evaluations"][0]["systems"]["nemotron_3_local"]
    assert nemotron["repeat_outputs_identical"] is True
    assert nemotron["exclusive"]["speaker_count_error"] == 0
    assert nemotron["exclusive"]["metrics"]["strict"]["false_alarm_seconds"] == 0.5
    historical = result["evaluations"][0]["systems"]["pyannoteai_precision_2_historical"]
    assert historical == {"status": "not_available_artifact_missing"}
    serialized = json.dumps(result)
    assert SECRET_TEXT not in serialized
    assert str(tmp_path) not in serialized
