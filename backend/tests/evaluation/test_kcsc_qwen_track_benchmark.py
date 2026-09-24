"""Tests for the Qwen3-ASR per-speaker track benchmark.

Transcription runs against a stub. What needs pinning is the decode contract with the
Qwen stack — timestamps actually requested and actually returned, the scored hypothesis
taken from the recogniser rather than the aligner — plus the model identity checks and the
guarantee that no transcript reaches an artifact. Normalisation, edit accounting, track
selection, and pooling are imported from the faster-whisper track benchmark and are pinned
in its own tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from voxdelta.evaluation.kcsc_asr_benchmark import KcscAsrError, characters
from voxdelta.evaluation.kcsc_diarization_benchmark import file_sha256
from voxdelta.evaluation.kcsc_qwen_track_benchmark import (
    MODEL_FILES,
    MODEL_REVISIONS,
    PROFILES,
    decode_track,
    run_qwen_track_benchmark,
    verify_qwen_identity,
    write_qwen_report,
)
from voxdelta.evaluation.kcsc_track_asr_benchmark import BENCHMARK_CONVERSATIONS
from voxdelta.providers.qwen3_asr import ALIGNER_MODEL_ID, LOW_MEMORY_MODEL_ID

REF = "안녕하세요 반갑습니다"
REVISION = "364fb908b14b9e8383ef6bf9f7ebf5088ffddaf3"
SPEAKERS = ("G0101", "G0102")


@dataclass(frozen=True)
class FakeUnit:
    text: str
    start_time: float
    end_time: float


@dataclass(frozen=True)
class FakeResult:
    text: str
    time_stamps: Any
    language: str = "Korean"


class StubModel:
    """Returns a fixed result per file and records how it was asked."""

    def __init__(self, results: dict[str, FakeResult] | None = None) -> None:
        self.results = results or {}
        self.calls: list[str] = []
        self.kwargs: list[dict[str, Any]] = []

    def transcribe(self, audio: Any, **kwargs: Any) -> list[FakeResult]:
        stem = Path(str(audio)).stem
        self.calls.append(stem)
        self.kwargs.append(dict(kwargs))
        default = FakeResult(
            text=REF,
            time_stamps=[FakeUnit("안녕하세요", 0.0, 1.0), FakeUnit("반갑습니다", 1.0, 2.0)],
        )
        return [self.results.get(stem, default)]


def _qwen_cache(tmp_path: Path, *, model_id: str = LOW_MEMORY_MODEL_ID, omit: str = "") -> Path:
    """A cache laid out the way the Hugging Face hub client writes one."""

    root = tmp_path / "hub"
    for repo_id in (model_id, ALIGNER_MODEL_ID):
        revision = MODEL_REVISIONS[repo_id]
        repo = root / f"models--{repo_id.replace('/', '--')}"
        blobs, snapshot = repo / "blobs", repo / "snapshots" / revision
        blobs.mkdir(parents=True, exist_ok=True)
        snapshot.mkdir(parents=True, exist_ok=True)
        (repo / "refs").mkdir(parents=True, exist_ok=True)
        (repo / "refs" / "main").write_text(revision, encoding="utf-8")
        for name in MODEL_FILES[repo_id]:
            if repo_id == model_id and name == omit:
                continue
            blob = blobs / f"blob-{name}"
            blob.write_bytes(f"{repo_id}:{name}".encode())
            (snapshot / name).symlink_to(Path("..") / ".." / "blobs" / blob.name)
    return root


@pytest.fixture
def tracks(tmp_path: Path) -> Path:
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
            reference.write_text(
                json.dumps(
                    {
                        "schema_version": "1",
                        "conversation_id": conversation_id,
                        "speaker": speaker,
                        "duration_seconds": 10.0,
                        "turns": [
                            {"start": 0.0, "end": 3.0, "speaker": speaker, "transcript": REF}
                        ],
                        "unattributed_events": [],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
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
                "mixed_set": {"manifest_sha256": "b" * 64},
                "tracks": entries,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return root


def _run(tracks: Path, tmp_path: Path, model: StubModel, **kwargs: Any) -> Any:
    cache = _qwen_cache(tmp_path / "cache")
    return run_qwen_track_benchmark(
        derived_root=tracks,
        model=model,
        identity=verify_qwen_identity(cache, LOW_MEMORY_MODEL_ID),
        aligner=verify_qwen_identity(cache, ALIGNER_MODEL_ID),
        device="mps",
        dtype="float16",
        **kwargs,
    )


# --------------------------------------------------------------------------- decode contract


def test_timestamps_are_actually_requested_for_every_track(tracks: Path, tmp_path: Path) -> None:
    stub = StubModel()

    _run(tracks, tmp_path, stub)

    assert stub.calls == [
        f"{conversation_id}_{speaker}"
        for conversation_id in BENCHMARK_CONVERSATIONS
        for speaker in SPEAKERS
    ]
    assert all(call == {"language": "Korean", "return_time_stamps": True} for call in stub.kwargs)


def test_a_result_without_timestamps_fails_closed_rather_than_scoring_untimed_text() -> None:
    stub = StubModel({"x": FakeResult(text=REF, time_stamps=None)})

    with pytest.raises(KcscAsrError, match="timestamps were requested but none returned"):
        decode_track(stub, Path("x.wav"), 10.0)


def test_an_empty_transcription_fails_closed() -> None:
    stub = StubModel({"x": FakeResult(text="   ", time_stamps=[FakeUnit("a", 0.0, 1.0)])})

    with pytest.raises(KcscAsrError, match="returned no text"):
        decode_track(stub, Path("x.wav"), 10.0)


def test_an_unreadable_timestamp_structure_fails_closed() -> None:
    stub = StubModel({"x": FakeResult(text=REF, time_stamps=object())})

    with pytest.raises(KcscAsrError, match="unreadable timestamp structure"):
        decode_track(stub, Path("x.wav"), 10.0)


def test_the_scored_hypothesis_is_the_recognisers_text_not_the_aligner_units() -> None:
    """A weak aligner must not be able to move the error rate."""

    stub = StubModel({"x": FakeResult(text=REF, time_stamps=[FakeUnit("전혀다른말", 0.0, 1.0)])})

    text, stats = decode_track(stub, Path("x.wav"), 10.0)

    assert text == REF
    assert stats.word_count == 1


def test_the_decode_stats_describe_the_returned_timeline() -> None:
    stub = StubModel(
        {
            "x": FakeResult(
                text=REF,
                time_stamps=[
                    FakeUnit("하나", 0.0, 1.0),
                    FakeUnit("둘", 5.0, 5.0),
                    FakeUnit("셋", 3.0, 12.0),
                ],
            )
        }
    )

    _text, stats = decode_track(stub, Path("x.wav"), 10.0)

    assert stats.word_count == 3
    assert stats.zero_length_spans == 1
    assert stats.words_past_declared_duration == 1
    assert stats.non_monotonic_starts == 1
    assert stats.last_word_end_seconds == pytest.approx(12.0)
    assert stats.max_word_gap_seconds == pytest.approx(4.0)


# --------------------------------------------------------------------------- model identity


def test_both_checkpoints_are_hashed_against_their_pinned_revisions(tmp_path: Path) -> None:
    cache = _qwen_cache(tmp_path)

    model = verify_qwen_identity(cache, LOW_MEMORY_MODEL_ID)
    aligner = verify_qwen_identity(cache, ALIGNER_MODEL_ID)

    assert model.revision == MODEL_REVISIONS[LOW_MEMORY_MODEL_ID]
    assert aligner.revision == MODEL_REVISIONS[ALIGNER_MODEL_ID]
    assert [entry["name"] for entry in model.files] == list(MODEL_FILES[LOW_MEMORY_MODEL_ID])
    assert model.tree_sha256 != aligner.tree_sha256


def test_a_missing_weight_file_fails_closed(tmp_path: Path) -> None:
    cache = _qwen_cache(tmp_path, omit="model.safetensors")

    with pytest.raises(KcscAsrError, match="model file missing from cache: model.safetensors"):
        verify_qwen_identity(cache, LOW_MEMORY_MODEL_ID)


def test_a_cache_pinned_to_another_revision_fails_closed(tmp_path: Path) -> None:
    cache = _qwen_cache(tmp_path)
    repo = cache / f"models--{LOW_MEMORY_MODEL_ID.replace('/', '--')}"
    (repo / "refs" / "main").write_text("f" * 40, encoding="utf-8")

    with pytest.raises(KcscAsrError, match="does not match pinned"):
        verify_qwen_identity(cache, LOW_MEMORY_MODEL_ID)


def test_an_unpinned_model_is_refused(tmp_path: Path) -> None:
    with pytest.raises(KcscAsrError, match="no pinned revision"):
        verify_qwen_identity(tmp_path, "Qwen/Qwen3-ASR-72B")


def test_both_profiles_are_pinned() -> None:
    assert set(PROFILES) == {"0.6b", "1.7b"}
    for model_id in PROFILES.values():
        assert len(MODEL_REVISIONS[model_id]) == 40
        assert MODEL_FILES[model_id]


# ---------------------------------------------------------------------------------- scoring


def test_a_narrowed_selection_is_refused(tracks: Path, tmp_path: Path) -> None:
    with pytest.raises(KcscAsrError, match="does not match the approved scope"):
        _run(tracks, tmp_path, StubModel(), conversation_ids=BENCHMARK_CONVERSATIONS[:2])


def test_a_tampered_track_fails_closed_before_any_transcription(
    tracks: Path, tmp_path: Path
) -> None:
    (tracks / "audio" / f"{BENCHMARK_CONVERSATIONS[1]}_{SPEAKERS[0]}.wav").write_bytes(b"x")
    stub = StubModel()

    with pytest.raises(KcscAsrError, match="audio checksum mismatch"):
        _run(tracks, tmp_path, stub)
    assert stub.calls == []


def test_an_exact_transcription_scores_zero_and_pooling_sums_operations(
    tracks: Path, tmp_path: Path
) -> None:
    report = _run(tracks, tmp_path, StubModel())

    assert len(report.scores) == 6
    assert report.character.errors == 0
    assert report.character.reference_length == 6 * len(characters(REF))
    assert report.character.reference_length == sum(
        score.character.reference_length for score in report.scores
    )


# ---------------------------------------------------------------------------- report safety


def test_the_report_is_transcript_free_and_names_both_checkpoints(
    tracks: Path, tmp_path: Path
) -> None:
    report = _run(tracks, tmp_path, StubModel())
    path = tmp_path / "report.json"

    digest = write_qwen_report(report, path)

    raw = path.read_text(encoding="utf-8")
    for fragment in (REF, "안녕하세요", "반갑습니다"):
        assert fragment not in raw
    assert digest == file_sha256(path)
    payload = json.loads(raw)
    assert payload["transcript_free"] is True
    assert payload["model"]["remote"] is False
    assert payload["model"]["repo_id"] == LOW_MEMORY_MODEL_ID
    assert payload["model"]["revision"] == MODEL_REVISIONS[LOW_MEMORY_MODEL_ID]
    assert payload["aligner"]["repo_id"] == ALIGNER_MODEL_ID
    assert payload["offline"]["network_egress_attempts"] == 0


def test_the_report_records_that_timestamps_are_genuinely_supported(
    tracks: Path, tmp_path: Path
) -> None:
    path = tmp_path / "report.json"
    write_qwen_report(_run(tracks, tmp_path, StubModel()), path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["timestamps"]["supported"] is True
    assert payload["timestamps"]["unit"] == "word"
    assert payload["timestamps"]["language"] == "Korean"
    assert "not used to build the scored hypothesis" in payload["aligner"]["used_for"]


def test_the_report_is_comparable_to_the_faster_whisper_track_report(
    tracks: Path, tmp_path: Path
) -> None:
    """Same task tag, same metric, same dataset binding — otherwise it is not a comparison."""

    path = tmp_path / "report.json"
    write_qwen_report(_run(tracks, tmp_path, StubModel()), path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["task"] == "per-speaker-track-asr-upper-bound"
    assert payload["scoring"]["primary_metric"] == "korean_normalized_character_error_rate"
    assert payload["dataset"]["track_count"] == 6
    assert payload["dataset"]["derivation_manifest_sha256"] == file_sha256(tracks / "manifest.json")
    limitations = " ".join(payload["limitations"])
    assert "NOT comparable to the faster-whisper track report" in limitations


def test_the_report_is_deterministic_apart_from_timing(tracks: Path, tmp_path: Path) -> None:
    report = _run(tracks, tmp_path, StubModel())
    first, second = tmp_path / "a.json", tmp_path / "b.json"

    write_qwen_report(report, first)
    write_qwen_report(report, second)

    assert file_sha256(first) == file_sha256(second)


def test_a_run_that_attempted_egress_says_so_in_the_report(tracks: Path, tmp_path: Path) -> None:
    report = _run(tracks, tmp_path, StubModel(), egress_attempts=2)
    path = tmp_path / "report.json"

    write_qwen_report(report, path)

    assert json.loads(path.read_text(encoding="utf-8"))["offline"]["network_egress_attempts"] == 2
