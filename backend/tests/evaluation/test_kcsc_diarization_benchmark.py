"""Tests for the local KCSC diarization benchmark.

Diarization runs against a stub rather than the real pipeline: the point under test is
the harness's fail-closed behaviour and its arithmetic, both of which must hold without
a 31 MB checkpoint and a minute of inference per case.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from voxdelta.domain.models import AudioAsset, SpeakerSegment
from voxdelta.evaluation.kcsc_diarization_benchmark import (
    KcscBenchmarkError,
    load_manifest,
    run_benchmark,
    score_conversation,
    select_conversations,
    verify_inputs,
    write_report,
)

TRANSCRIPT = "이것은 절대 보고서에 나오면 안 되는 전사입니다"
REVISION = "364fb908b14b9e8383ef6bf9f7ebf5088ffddaf3"


class StubDiarizer:
    """Returns a fixed hypothesis, and records the assets it was handed."""

    def __init__(self, segments: Sequence[SpeakerSegment]) -> None:
        self._segments = list(segments)
        self.assets: list[AudioAsset] = []

    def diarize(self, asset: AudioAsset) -> list[SpeakerSegment]:
        self.assets.append(asset)
        return list(self._segments)


def segment(start: float, end: float, speaker: str) -> SpeakerSegment:
    return SpeakerSegment(start=start, end=end, speaker_id=speaker, confidence=1.0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def derived(tmp_path: Path) -> Path:
    """A minimal derived set: one 10 s conversation, two speakers, four turns."""

    root = tmp_path / "derived"
    (root / "audio").mkdir(parents=True)
    (root / "reference").mkdir(parents=True)

    audio = root / "audio" / "A0001_S0001_0.wav"
    audio.write_bytes(b"RIFF-not-really-audio-but-hashed-all-the-same")
    reference = root / "reference" / "A0001_S0001_0.json"
    reference.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "conversation_id": "A0001_S0001_0",
                "source_revision": REVISION,
                "sample_rate": 16000,
                "duration_seconds": 10.0,
                "trim_offset_seconds": 3.0,
                "speakers": ["G0001", "G0002"],
                "vendor_marker_intervals_silenced": [],
                "turns": [
                    {"start": 0.0, "end": 2.0, "speaker": "G0001", "transcript": TRANSCRIPT},
                    {"start": 2.0, "end": 4.0, "speaker": "G0002", "transcript": TRANSCRIPT},
                    {"start": 5.0, "end": 7.0, "speaker": "G0001", "transcript": TRANSCRIPT},
                    {"start": 7.0, "end": 9.0, "speaker": "G0002", "transcript": TRANSCRIPT},
                ],
                "unattributed_events": [],
            }
        ),
        encoding="utf-8",
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "source": {"revision": REVISION},
                "conversations": [
                    {
                        "conversation_id": "A0001_S0001_0",
                        "speakers": ["G0001", "G0002"],
                        "derived_duration_seconds": 10.0,
                        "outputs": {
                            "audio": "audio/A0001_S0001_0.wav",
                            "audio_sha256": _sha256(audio),
                            "reference": "reference/A0001_S0001_0.json",
                            "reference_sha256": _sha256(reference),
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return root


PERFECT = [
    segment(0.0, 2.0, "G0001"),
    segment(2.0, 4.0, "G0002"),
    segment(5.0, 7.0, "G0001"),
    segment(7.0, 9.0, "G0002"),
]


def test_a_perfect_hypothesis_scores_zero_error(derived: Path) -> None:
    report = run_benchmark(
        derived_root=derived,
        conversation_ids=["A0001_S0001_0"],
        diarizer=StubDiarizer(PERFECT),
        model_name="stub",
        model_tree_sha256="a" * 64,
    )

    assert report.aggregate["strict"]["der"] == 0.0
    assert report.scores[0].hypothesis_speaker_count == 2
    assert report.scores[0].reference_turn_count == 4
    assert report.scores[0].reference_speech_seconds == 8.0


def test_a_swapped_hypothesis_is_mapped_not_penalised(derived: Path) -> None:
    """Speaker labels are arbitrary; only the partition matters."""

    swapped = [segment(s.start, s.end, "X" if s.speaker_id == "G0001" else "Y") for s in PERFECT]

    report = run_benchmark(
        derived_root=derived,
        conversation_ids=["A0001_S0001_0"],
        diarizer=StubDiarizer(swapped),
        model_name="stub",
        model_tree_sha256="a" * 64,
    )

    assert report.aggregate["strict"]["der"] == 0.0


def test_a_missed_turn_is_reported_as_missed_detection(derived: Path) -> None:
    report = run_benchmark(
        derived_root=derived,
        conversation_ids=["A0001_S0001_0"],
        diarizer=StubDiarizer(PERFECT[:-1]),
        model_name="stub",
        model_tree_sha256="a" * 64,
    )

    strict = report.scores[0].metrics["strict"]
    # 2 s of the 8 s reference goes undetected.
    assert strict["miss_seconds"] == pytest.approx(2.0)
    assert strict["der"] == pytest.approx(0.25)
    assert strict["false_alarm_seconds"] == pytest.approx(0.0)


def test_a_single_speaker_hypothesis_is_reported_as_confusion(derived: Path) -> None:
    collapsed = [segment(s.start, s.end, "ONE") for s in PERFECT]

    report = run_benchmark(
        derived_root=derived,
        conversation_ids=["A0001_S0001_0"],
        diarizer=StubDiarizer(collapsed),
        model_name="stub",
        model_tree_sha256="a" * 64,
    )

    assert report.scores[0].hypothesis_speaker_count == 1
    assert report.scores[0].metrics["strict"]["confusion_seconds"] == pytest.approx(4.0)


def test_a_tampered_audio_file_fails_closed(derived: Path) -> None:
    (derived / "audio" / "A0001_S0001_0.wav").write_bytes(b"different bytes entirely")

    with pytest.raises(KcscBenchmarkError, match="audio checksum mismatch"):
        run_benchmark(
            derived_root=derived,
            conversation_ids=["A0001_S0001_0"],
            diarizer=StubDiarizer(PERFECT),
            model_name="stub",
            model_tree_sha256="a" * 64,
        )


def test_a_tampered_reference_file_fails_closed(derived: Path) -> None:
    path = derived / "reference" / "A0001_S0001_0.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["turns"][0]["start"] = 0.5
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(KcscBenchmarkError, match="reference checksum mismatch"):
        run_benchmark(
            derived_root=derived,
            conversation_ids=["A0001_S0001_0"],
            diarizer=StubDiarizer(PERFECT),
            model_name="stub",
            model_tree_sha256="a" * 64,
        )


def test_a_missing_derived_file_fails_closed(derived: Path) -> None:
    (derived / "audio" / "A0001_S0001_0.wav").unlink()

    with pytest.raises(KcscBenchmarkError, match="missing derived audio"):
        verify_inputs(
            load_manifest(derived)["conversations"][0],  # type: ignore[index]
            derived_root=derived,
        )


def test_a_missing_manifest_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(KcscBenchmarkError, match="missing derivation manifest"):
        load_manifest(tmp_path)


def test_an_unknown_conversation_id_fails_closed(derived: Path) -> None:
    with pytest.raises(KcscBenchmarkError, match="unknown conversation ids: A9999_S0001_0"):
        select_conversations(load_manifest(derived), ["A9999_S0001_0"])


def test_a_duplicated_conversation_id_fails_closed(derived: Path) -> None:
    with pytest.raises(KcscBenchmarkError, match="duplicates"):
        select_conversations(load_manifest(derived), ["A0001_S0001_0", "A0001_S0001_0"])


def _rewrite_reference(derived: Path, mutate: object) -> None:
    """Rewrite the reference and reseal the manifest, so only the mutation is under test."""

    path = derived / "reference" / "A0001_S0001_0.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert callable(mutate)
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    manifest_path = derived / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["conversations"][0]["outputs"]["reference_sha256"] = _sha256(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda p: p.update(schema_version="99"), "unsupported reference schema"),
        (lambda p: p.update(conversation_id="OTHER"), "conversation id does not match"),
        (lambda p: p.update(duration_seconds=11.0), "duration disagrees with manifest"),
        (lambda p: p.update(turns=[]), "reference has no turns"),
        (lambda p: p["turns"][0].update(speaker="G9999"), "names an unknown speaker"),
        (lambda p: p["turns"][0].update(end=99.0), "falls outside the audio"),
        (lambda p: p["turns"][0].update(start="x"), "non-numeric timing"),
    ],
)
def test_malformed_references_fail_closed(derived: Path, mutate: object, match: str) -> None:
    _rewrite_reference(derived, mutate)

    with pytest.raises(KcscBenchmarkError, match=match):
        run_benchmark(
            derived_root=derived,
            conversation_ids=["A0001_S0001_0"],
            diarizer=StubDiarizer(PERFECT),
            model_name="stub",
            model_tree_sha256="a" * 64,
        )


def test_a_reference_naming_one_speaker_fails_closed(derived: Path) -> None:
    _rewrite_reference(derived, lambda p: [t.update(speaker="G0001") for t in p["turns"]])

    with pytest.raises(KcscBenchmarkError, match="names 1 speakers"):
        run_benchmark(
            derived_root=derived,
            conversation_ids=["A0001_S0001_0"],
            diarizer=StubDiarizer(PERFECT),
            model_name="stub",
            model_tree_sha256="a" * 64,
        )


def test_an_empty_hypothesis_fails_closed(derived: Path) -> None:
    with pytest.raises(KcscBenchmarkError, match="returned no segments"):
        run_benchmark(
            derived_root=derived,
            conversation_ids=["A0001_S0001_0"],
            diarizer=StubDiarizer([]),
            model_name="stub",
            model_tree_sha256="a" * 64,
        )


def test_the_diarizer_receives_the_verified_audio_and_its_manifest_digest(derived: Path) -> None:
    stub = StubDiarizer(PERFECT)

    run_benchmark(
        derived_root=derived,
        conversation_ids=["A0001_S0001_0"],
        diarizer=stub,
        model_name="stub",
        model_tree_sha256="a" * 64,
    )

    asset = stub.assets[0]
    assert asset.channel_mode == "mixed"
    assert asset.duration_seconds == 10.0
    assert asset.sha256 == _sha256(derived / "audio" / "A0001_S0001_0.wav")


def test_the_report_is_transcript_free(derived: Path, tmp_path: Path) -> None:
    report = run_benchmark(
        derived_root=derived,
        conversation_ids=["A0001_S0001_0"],
        diarizer=StubDiarizer(PERFECT),
        model_name="stub",
        model_tree_sha256="a" * 64,
    )
    path = tmp_path / "report.json"

    digest = write_report(report, path)

    raw = path.read_text(encoding="utf-8")
    assert TRANSCRIPT not in raw
    assert "transcript" not in raw.replace('"transcript_free": true', "")
    assert digest == _sha256(path)
    payload = json.loads(raw)
    assert payload["transcript_free"] is True
    assert payload["model"]["remote"] is False
    assert payload["dataset"]["source_revision"] == REVISION
    assert payload["offline"]["network_egress_attempts"] == 0


def test_the_report_records_a_blocked_egress_attempt(derived: Path, tmp_path: Path) -> None:
    """A run that touched the network must say so rather than look clean."""

    report = run_benchmark(
        derived_root=derived,
        conversation_ids=["A0001_S0001_0"],
        diarizer=StubDiarizer(PERFECT),
        model_name="stub",
        model_tree_sha256="a" * 64,
        egress_attempts=3,
    )
    path = tmp_path / "report.json"
    write_report(report, path)

    assert json.loads(path.read_text(encoding="utf-8"))["offline"]["network_egress_attempts"] == 3


def test_scoring_pools_error_seconds_rather_than_averaging_rates(derived: Path) -> None:
    """A long conversation must not be diluted by a short one, or the reverse."""

    entry = load_manifest(derived)["conversations"][0]  # type: ignore[index]
    perfect = score_conversation(entry, derived_root=derived, diarizer=StubDiarizer(PERFECT))
    missed = score_conversation(entry, derived_root=derived, diarizer=StubDiarizer(PERFECT[:-1]))

    assert perfect.metrics["strict"]["scored_speech_seconds"] == pytest.approx(8.0)
    assert missed.metrics["strict"]["miss_seconds"] == pytest.approx(2.0)
