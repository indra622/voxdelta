"""Tests for the KCSC per-speaker ASR upper bound.

Transcription runs against a stub. What needs pinning is the track selection, the
per-track reference binding, the pooling that makes the figure subtractable from the mixed
baseline, and the guarantee that no transcript reaches an artifact — none of which needs a
1.5 GB decoder to exercise. Normalisation and edit accounting are shared with the mixed
benchmark and are pinned in its own tests.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from voxdelta.evaluation.kcsc_asr_benchmark import (
    MODEL_FILES,
    MODEL_REVISION,
    KcscAsrError,
    characters,
    verify_model_identity,
)
from voxdelta.evaluation.kcsc_diarization_benchmark import file_sha256
from voxdelta.evaluation.kcsc_track_asr_benchmark import (
    BENCHMARK_CONVERSATIONS,
    load_track_manifest,
    run_track_asr_benchmark,
    select_tracks,
    track_composition,
    track_reference_transcript,
    write_track_report,
)

REF_A = "안녕하세요 반갑습니다"
REF_B = "네 그렇습니다"
REVISION = "364fb908b14b9e8383ef6bf9f7ebf5088ffddaf3"
MIXED_MANIFEST_DIGEST = "b" * 64
SPEAKERS = ("G0101", "G0102")


@dataclass(frozen=True)
class FakeWord:
    start: float
    end: float
    word: str


@dataclass(frozen=True)
class FakeSegment:
    start: float
    end: float
    text: str
    words: Sequence[FakeWord] | None


class StubModel:
    """Returns fixed segments per file and records which files were decoded."""

    def __init__(self, segments: dict[str, list[FakeSegment]] | None = None) -> None:
        self.segments = segments or {}
        self.calls: list[str] = []
        self.options: list[dict[str, Any]] = []

    def transcribe(self, path: str, **kwargs: Any) -> tuple[Any, Any]:
        self.calls.append(Path(path).stem)
        self.options.append(dict(kwargs))
        default = [
            FakeSegment(
                0.0,
                2.0,
                REF_A,
                [FakeWord(0.0, 1.0, "안녕하세요"), FakeWord(1.0, 2.0, "반갑습니다")],
            )
        ]
        return self.segments.get(Path(path).stem, default), None


def _cache(tmp_path: Path) -> Path:
    root = tmp_path / "hub"
    repo = root / "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo"
    blobs, snapshot = repo / "blobs", repo / "snapshots" / MODEL_REVISION
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text(MODEL_REVISION, encoding="utf-8")
    for name in MODEL_FILES:
        blob = blobs / f"blob-{name}"
        blob.write_bytes(f"weights for {name}".encode())
        (snapshot / name).symlink_to(Path("..") / ".." / "blobs" / blob.name)
    return root


def _reference(speaker: str, transcript: str) -> dict[str, object]:
    return {
        "schema_version": "1",
        "conversation_id": "",
        "speaker": speaker,
        "source_revision": REVISION,
        "duration_seconds": 10.0,
        "turns": [
            {"start": 0.0, "end": 3.0, "speaker": speaker, "transcript": transcript},
            {"start": 4.0, "end": 5.0, "speaker": speaker, "transcript": transcript},
        ],
        "unattributed_events": [
            {"start": 6.0, "end": 6.4, "speaker": "0", "transcript": "[LAUGHTER]"}
        ],
    }


@pytest.fixture
def tracks(tmp_path: Path) -> Path:
    """A six-track set shaped exactly like the real derivation's output."""

    root = tmp_path / "tracks"
    (root / "audio").mkdir(parents=True)
    (root / "reference").mkdir(parents=True)
    entries = []
    for index, conversation_id in enumerate(BENCHMARK_CONVERSATIONS):
        for offset, speaker in enumerate(SPEAKERS):
            stem = f"{conversation_id}_{speaker}"
            audio = root / "audio" / f"{stem}.wav"
            audio.write_bytes(b"RIFF" + bytes([index * 2 + offset]) * 32)
            reference = root / "reference" / f"{stem}.json"
            payload = _reference(speaker, REF_A if speaker == SPEAKERS[0] else REF_B)
            payload["conversation_id"] = conversation_id
            reference.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            entries.append(
                {
                    "conversation_id": conversation_id,
                    "speaker": speaker,
                    "track_id": stem,
                    "derived_duration_seconds": 10.0,
                    "outputs": {
                        "audio": f"audio/{stem}.wav",
                        "audio_sha256": file_sha256(audio),
                        "reference": f"reference/{stem}.json",
                        "reference_sha256": file_sha256(reference),
                    },
                }
            )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "source": {"revision": REVISION},
                "mixed_set": {"manifest_sha256": MIXED_MANIFEST_DIGEST},
                "track_count": len(entries),
                "tracks": entries,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return root


def _run(tracks: Path, tmp_path: Path, model: StubModel, **kwargs: Any) -> Any:
    return run_track_asr_benchmark(
        derived_root=tracks,
        model=model,
        identity=verify_model_identity(_cache(tmp_path / "cache")),
        device="cpu",
        compute_type="int8",
        decode_options={"language": "ko", "beam_size": 5},
        **kwargs,
    )


# ------------------------------------------------------------------------------- selection


def test_every_track_of_every_conversation_is_decoded_exactly_once(
    tracks: Path, tmp_path: Path
) -> None:
    stub = StubModel()

    _run(tracks, tmp_path, stub)

    assert stub.calls == [
        f"{conversation_id}_{speaker}"
        for conversation_id in BENCHMARK_CONVERSATIONS
        for speaker in SPEAKERS
    ]


def test_a_conversation_missing_one_of_its_speakers_fails_closed(
    tracks: Path, tmp_path: Path
) -> None:
    """Pooling a one-sided call with two-sided ones would tilt towards whoever survived."""

    path = tracks / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    dropped = f"{BENCHMARK_CONVERSATIONS[1]}_{SPEAKERS[1]}"
    manifest["tracks"] = [entry for entry in manifest["tracks"] if entry["track_id"] != dropped]
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    stub = StubModel()

    with pytest.raises(KcscAsrError, match="expected 2 speaker tracks, found 1"):
        _run(tracks, tmp_path, stub)
    assert stub.calls == []


def test_two_tracks_labelled_with_the_same_speaker_fail_closed(tracks: Path) -> None:
    manifest = json.loads((tracks / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["tracks"]:
        entry["speaker"] = SPEAKERS[0]

    with pytest.raises(KcscAsrError, match="not distinctly labelled"):
        select_tracks(manifest, BENCHMARK_CONVERSATIONS)


def test_a_narrowed_selection_is_refused(tracks: Path, tmp_path: Path) -> None:
    with pytest.raises(KcscAsrError, match="does not match the approved scope"):
        _run(tracks, tmp_path, StubModel(), conversation_ids=BENCHMARK_CONVERSATIONS[:2])


def test_a_missing_track_manifest_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(KcscAsrError, match="missing track derivation manifest"):
        load_track_manifest(tmp_path / "absent")


def test_a_manifest_without_a_tracks_list_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"conversations": []}), encoding="utf-8")

    with pytest.raises(KcscAsrError, match="unexpected shape"):
        load_track_manifest(tmp_path)


# ------------------------------------------------------------------------------- integrity


def test_a_tampered_track_fails_closed_before_any_transcription(
    tracks: Path, tmp_path: Path
) -> None:
    (tracks / "audio" / f"{BENCHMARK_CONVERSATIONS[2]}_{SPEAKERS[1]}.wav").write_bytes(b"x")
    stub = StubModel()

    with pytest.raises(KcscAsrError, match=f"track {SPEAKERS[1]}: .*audio checksum mismatch"):
        _run(tracks, tmp_path, stub)
    assert stub.calls == []


def test_a_reference_belonging_to_the_other_speaker_fails_closed(tracks: Path) -> None:
    manifest = json.loads((tracks / "manifest.json").read_text(encoding="utf-8"))
    entry = dict(manifest["tracks"][0])
    entry["speaker"] = SPEAKERS[1]
    path = tracks / "reference" / f"{BENCHMARK_CONVERSATIONS[0]}_{SPEAKERS[0]}.json"

    with pytest.raises(KcscAsrError, match="speaker does not match manifest"):
        track_reference_transcript(path, entry)


def test_the_track_reference_merges_turns_and_unattributed_rows(tracks: Path) -> None:
    manifest = json.loads((tracks / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["tracks"][0]
    path = tracks / "reference" / f"{BENCHMARK_CONVERSATIONS[0]}_{SPEAKERS[0]}.json"

    text, rows = track_reference_transcript(path, entry)

    assert rows == 3
    # The tag row is carried but normalises to nothing, so it costs no reference characters.
    assert "[LAUGHTER]" in text
    assert characters(text) == characters(f"{REF_A} {REF_A}")


# ----------------------------------------------------------------------------- composition


def test_the_composition_reports_no_cross_speaker_overlap_and_quotes_no_text(
    tracks: Path,
) -> None:
    composition = track_composition(
        tracks / "reference" / f"{BENCHMARK_CONVERSATIONS[0]}_{SPEAKERS[0]}.json"
    )

    assert composition["turn_count"] == 2
    assert composition["unattributed_event_count"] == 1
    assert composition["reference_speech_seconds"] == pytest.approx(4.0)
    # A single speaker's own turns must not overlap; a non-zero value would be double counting.
    assert composition["self_overlapped_speech_seconds"] == pytest.approx(0.0)
    assert "overlapped_speech_fraction" not in composition
    assert composition["rows_containing_tag"]["[LAUGHTER]"] == 1
    assert REF_A not in json.dumps(composition, ensure_ascii=False)


def test_self_overlap_is_reported_when_a_tracks_own_rows_overlap(tmp_path: Path) -> None:
    path = tmp_path / "reference.json"
    payload = _reference(SPEAKERS[0], REF_A)
    payload["turns"] = [
        {"start": 0.0, "end": 3.0, "speaker": SPEAKERS[0], "transcript": REF_A},
        {"start": 2.0, "end": 4.0, "speaker": SPEAKERS[0], "transcript": REF_A},
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    assert track_composition(path)["self_overlapped_speech_seconds"] == pytest.approx(1.0)


# -------------------------------------------------------------------------------- pooling


def test_a_conversations_figure_pools_both_tracks_operations_not_their_rates(
    tracks: Path, tmp_path: Path
) -> None:
    report = _run(tracks, tmp_path, StubModel())
    conversation_id = BENCHMARK_CONVERSATIONS[0]

    pooled = report.conversation_character(conversation_id)
    per_track = [
        score.character for score in report.scores if score.conversation_id == conversation_id
    ]

    assert len(per_track) == 2
    assert pooled.reference_length == sum(counts.reference_length for counts in per_track)
    assert pooled.errors == sum(counts.errors for counts in per_track)
    assert pooled.error_rate == pytest.approx(pooled.errors / pooled.reference_length)


def test_the_aggregate_pools_every_track(tracks: Path, tmp_path: Path) -> None:
    report = _run(tracks, tmp_path, StubModel())

    assert len(report.scores) == 6
    assert report.character.reference_length == sum(
        score.character.reference_length for score in report.scores
    )
    assert report.character.errors == sum(score.character.errors for score in report.scores)


def test_a_track_that_matches_its_reference_exactly_scores_zero(
    tracks: Path, tmp_path: Path
) -> None:
    """The stub emits REF_A twice, which is exactly what the G0101 references contain."""

    report = _run(
        tracks,
        tmp_path,
        StubModel(
            {
                f"{BENCHMARK_CONVERSATIONS[0]}_{SPEAKERS[0]}": [
                    FakeSegment(
                        0.0, 2.0, "", [FakeWord(0.0, 1.0, REF_A), FakeWord(1.0, 2.0, REF_A)]
                    )
                ]
            }
        ),
    )

    score = next(
        s for s in report.scores if s.track_id == f"{BENCHMARK_CONVERSATIONS[0]}_{SPEAKERS[0]}"
    )
    assert score.character.errors == 0
    assert score.character.error_rate == 0.0


# --------------------------------------------------------------------------- report safety


def test_the_report_is_transcript_free(tracks: Path, tmp_path: Path) -> None:
    report = _run(tracks, tmp_path, StubModel())
    path = tmp_path / "report.json"

    digest = write_track_report(report, path)

    raw = path.read_text(encoding="utf-8")
    for fragment in (REF_A, REF_B, "안녕하세요", "그렇습니다"):
        assert fragment not in raw
    assert digest == file_sha256(path)
    payload = json.loads(raw)
    assert payload["transcript_free"] is True
    assert payload["model"]["remote"] is False
    assert payload["offline"]["network_egress_attempts"] == 0
    assert payload["task"] == "per-speaker-track-asr-upper-bound"


def test_the_report_binds_the_track_set_to_the_mixed_set_it_is_subtractable_from(
    tracks: Path, tmp_path: Path
) -> None:
    path = tmp_path / "report.json"
    report = _run(tracks, tmp_path, StubModel())

    write_track_report(report, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["dataset"]["source_revision"] == REVISION
    assert payload["dataset"]["mixed_set_manifest_sha256"] == MIXED_MANIFEST_DIGEST
    assert payload["dataset"]["derivation_manifest_sha256"] == file_sha256(tracks / "manifest.json")
    assert payload["dataset"]["track_count"] == 6
    assert payload["dataset"]["conversation_ids"] == list(BENCHMARK_CONVERSATIONS)
    assert payload["model"]["revision"] == MODEL_REVISION
    assert payload["model"]["decode_options"] == {"language": "ko", "beam_size": 5}


def test_the_report_states_what_the_bound_does_and_does_not_cover(
    tracks: Path, tmp_path: Path
) -> None:
    path = tmp_path / "report.json"
    write_track_report(_run(tracks, tmp_path, StubModel()), path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    limitations = " ".join(payload["limitations"])
    assert "no acoustic source separation" in limitations
    assert "No segmentation or diarization error is modelled" in limitations
    assert "same annotated rows with the same decoder" in limitations
    assert payload["scoring"]["primary_metric"] == "korean_normalized_character_error_rate"


def test_the_report_carries_both_per_track_and_per_conversation_figures(
    tracks: Path, tmp_path: Path
) -> None:
    path = tmp_path / "report.json"
    report = _run(tracks, tmp_path, StubModel())

    write_track_report(report, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert [t["track_id"] for t in payload["tracks"]] == [s.track_id for s in report.scores]
    assert [c["conversation_id"] for c in payload["conversations"]] == list(BENCHMARK_CONVERSATIONS)
    first = payload["conversations"][0]["character"]
    assert first["reference_length"] == sum(
        t["character"]["reference_length"]
        for t in payload["tracks"]
        if t["conversation_id"] == BENCHMARK_CONVERSATIONS[0]
    )


def test_the_report_is_deterministic_apart_from_timing(tracks: Path, tmp_path: Path) -> None:
    report = _run(tracks, tmp_path, StubModel())
    first, second = tmp_path / "a.json", tmp_path / "b.json"

    write_track_report(report, first)
    write_track_report(report, second)

    assert file_sha256(first) == file_sha256(second)


def test_a_run_that_attempted_egress_says_so_in_the_report(tracks: Path, tmp_path: Path) -> None:
    report = _run(tracks, tmp_path, StubModel(), egress_attempts=3)
    path = tmp_path / "report.json"

    write_track_report(report, path)

    assert json.loads(path.read_text(encoding="utf-8"))["offline"]["network_egress_attempts"] == 3
