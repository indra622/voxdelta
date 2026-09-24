"""Tests for the local KCSC ASR baseline.

Transcription runs against a stub. What needs pinning is the normalisation, the error
accounting, the offline guard, and the guarantee that no transcript reaches an artifact —
none of which needs a 1.5 GB decoder to exercise.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from voxdelta.evaluation.kcsc_asr_benchmark import (
    BENCHMARK_CONVERSATIONS,
    MODEL_FILES,
    MODEL_REVISION,
    ErrorCounts,
    KcscAsrError,
    characters,
    decode_chronological,
    eojeol,
    normalize,
    reference_composition,
    reference_transcript,
    run_asr_benchmark,
    score_sequences,
    verify_model_identity,
    write_report,
)
from voxdelta.evaluation.kcsc_diarization_benchmark import KcscBenchmarkError, file_sha256

REF_A = "안녕하세요 반갑습니다"
REF_B = "네 그렇습니다"
REVISION = "364fb908b14b9e8383ef6bf9f7ebf5088ffddaf3"


@dataclass(frozen=True)
class FakeWord:
    """Mirrors faster_whisper's Word, whose text field is named ``word``."""

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
    """Returns fixed segments per file and counts how many times each file was decoded."""

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
                "안녕하세요 반갑습니다",
                [FakeWord(0.0, 1.0, "안녕하세요"), FakeWord(1.0, 2.0, "반갑습니다")],
            )
        ]
        return self.segments.get(Path(path).stem, default), None


# --------------------------------------------------------------------------- normalization


def test_event_tags_are_removed_so_a_cough_is_not_charged_as_missed_speech() -> None:
    assert normalize("[LAUGHTER]") == ""
    assert normalize("[SONANT] 네 [ENS]") == "네"
    assert characters("[MUSIC][SYSTEM][*]") == ""


def test_the_overlap_marker_is_removed() -> None:
    assert normalize("네+ 그렇습니다") == "네 그렇습니다"


def test_punctuation_and_symbols_are_removed() -> None:
    assert normalize("네, 그렇습니다! (정말)") == "네 그렇습니다 정말"
    assert characters("네, 그렇습니다!") == "네그렇습니다"


def test_normalization_is_nfc_and_case_folding() -> None:
    decomposed = "한"  # 한 in decomposed jamo
    assert normalize(decomposed) == "한"
    assert normalize("OK 네") == "ok 네"


def test_the_character_metric_ignores_whitespace_but_eojeol_does_not() -> None:
    assert characters("안녕 하세요") == characters("안녕하세요")
    assert eojeol("안녕 하세요") == ("안녕", "하세요")
    assert eojeol("안녕하세요") == ("안녕하세요",)


def test_normalizing_empty_or_tag_only_text_yields_no_tokens() -> None:
    assert eojeol("[LAUGHTER]") == ()
    assert characters("   ") == ""


# ------------------------------------------------------------------------------ accounting


def test_an_exact_match_scores_zero() -> None:
    counts = score_sequences(characters(REF_A), characters(REF_A))

    assert counts.errors == 0
    assert counts.error_rate == 0.0


def test_substitutions_deletions_and_insertions_are_counted_separately() -> None:
    assert score_sequences("abcd", "abxd").substitutions == 1
    assert score_sequences("abcd", "abd").deletions == 1
    assert score_sequences("abd", "abcd").insertions == 1


def test_the_rate_divides_by_the_reference_length_not_the_hypothesis() -> None:
    counts = score_sequences("abcd", "")

    assert counts.deletions == 4
    assert counts.reference_length == 4
    assert counts.error_rate == 1.0


def test_a_rate_over_an_empty_reference_fails_closed() -> None:
    with pytest.raises(KcscAsrError, match="empty reference"):
        _ = score_sequences("", "abc").error_rate


def test_counts_pool_by_summing_operations_and_reference_length() -> None:
    """Pooling must not average two rates computed over different denominators."""

    small = ErrorCounts(1, 0, 0, reference_length=2, hypothesis_length=2)
    large = ErrorCounts(1, 0, 0, reference_length=98, hypothesis_length=98)

    merged = small.merged(large)

    assert merged.reference_length == 100
    assert merged.error_rate == pytest.approx(0.02)
    assert merged.error_rate != pytest.approx((small.error_rate + large.error_rate) / 2)


def test_the_hypothesis_is_ordered_by_onset_not_emission_order() -> None:
    stub = StubModel(
        {"x": [FakeSegment(0.0, 3.0, "", [FakeWord(2.0, 3.0, "둘"), FakeWord(0.0, 1.0, "하나")])]}
    )

    text, stats = decode_chronological(stub, Path("x.wav"), 10.0, {})

    assert text == "하나 둘"
    assert stats.word_count == 2


def test_a_zero_length_word_span_is_kept_and_counted_not_dropped() -> None:
    """Dropping it would discard real transcribed text and inflate deletions."""

    stub = StubModel(
        {"x": [FakeSegment(0.0, 2.0, "", [FakeWord(1.0, 1.0, "네"), FakeWord(1.5, 2.0, "그렇죠")])]}
    )

    text, stats = decode_chronological(stub, Path("x.wav"), 10.0, {})

    assert text == "네 그렇죠"
    assert stats.word_count == 2
    assert stats.zero_length_spans == 1


def test_a_word_past_the_declared_duration_is_counted() -> None:
    stub = StubModel({"x": [FakeSegment(0.0, 12.0, "", [FakeWord(9.0, 12.0, "끝")])]})

    _text, stats = decode_chronological(stub, Path("x.wav"), 10.0, {})

    assert stats.words_past_declared_duration == 1


def test_a_segment_without_word_timestamps_falls_back_to_segment_text() -> None:
    stub = StubModel({"x": [FakeSegment(0.0, 2.0, "안녕하세요", None)]})

    text, stats = decode_chronological(stub, Path("x.wav"), 10.0, {})

    assert text == "안녕하세요"
    assert stats.segments_without_word_timestamps == 1


def test_the_decoder_receives_the_pipeline_decode_options() -> None:
    stub = StubModel()

    decode_chronological(stub, Path("x.wav"), 10.0, {"language": "ko", "beam_size": 5})

    assert stub.options[0] == {"language": "ko", "beam_size": 5}


def test_an_empty_transcription_fails_closed() -> None:
    with pytest.raises(KcscAsrError, match="returned no words"):
        decode_chronological(StubModel({"x": []}), Path("x.wav"), 10.0, {})


# ------------------------------------------------------------------------- model identity


def _cache(tmp_path: Path, *, revision: str = MODEL_REVISION, omit: str = "") -> Path:
    root = tmp_path / "hub"
    repo = root / "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo"
    blobs, snapshot = repo / "blobs", repo / "snapshots" / revision
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text(revision, encoding="utf-8")
    for name in MODEL_FILES:
        if name == omit:
            continue
        blob = blobs / f"blob-{name}"
        blob.write_bytes(f"weights for {name}".encode())
        # The real cache links snapshot entries at blobs; the digest must follow them.
        (snapshot / name).symlink_to(Path("..") / ".." / "blobs" / blob.name)
    return root


def test_model_identity_follows_cache_symlinks_and_hashes_the_blobs(tmp_path: Path) -> None:
    identity = verify_model_identity(_cache(tmp_path))

    assert identity.revision == MODEL_REVISION
    assert [entry["name"] for entry in identity.files] == list(MODEL_FILES)
    assert len(identity.tree_sha256) == 64
    assert identity.total_bytes > 0


def test_an_absent_snapshot_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(KcscAsrError, match="model snapshot not cached locally"):
        verify_model_identity(tmp_path / "empty")


def test_a_missing_model_file_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(KcscAsrError, match="model file missing from cache: model.bin"):
        verify_model_identity(_cache(tmp_path, omit="model.bin"))


def test_a_cache_pinned_to_another_revision_fails_closed(tmp_path: Path) -> None:
    root = _cache(tmp_path)
    repo = root / "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo"
    (repo / "refs" / "main").write_text("f" * 40, encoding="utf-8")

    with pytest.raises(KcscAsrError, match="does not match pinned"):
        verify_model_identity(root)


def test_model_identity_changes_when_a_weight_file_changes(tmp_path: Path) -> None:
    """The digest is what makes 'the same model' checkable across runs."""

    root = _cache(tmp_path)
    before = verify_model_identity(root).tree_sha256
    blob = (
        root / "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo" / "blobs" / "blob-model.bin"
    )
    blob.write_bytes(b"tampered weights")

    assert verify_model_identity(root).tree_sha256 != before


# ------------------------------------------------------------------------------- end to end


@pytest.fixture
def derived(tmp_path: Path) -> Path:
    root = tmp_path / "derived"
    (root / "audio").mkdir(parents=True)
    (root / "reference").mkdir(parents=True)
    conversations = []
    for index, cid in enumerate(BENCHMARK_CONVERSATIONS):
        audio = root / "audio" / f"{cid}.wav"
        audio.write_bytes(b"RIFF" + bytes([index]) * 32)
        reference = root / "reference" / f"{cid}.json"
        reference.write_text(
            json.dumps(
                {
                    "schema_version": "1",
                    "conversation_id": cid,
                    "source_revision": REVISION,
                    "duration_seconds": 10.0,
                    "speakers": ["G0001", "G0002"],
                    "turns": [
                        {"start": 0.0, "end": 3.0, "speaker": "G0001", "transcript": REF_A},
                        {"start": 2.0, "end": 5.0, "speaker": "G0002", "transcript": REF_B},
                    ],
                    "unattributed_events": [
                        {"start": 6.0, "end": 6.4, "speaker": "0", "transcript": "[LAUGHTER]"}
                    ],
                }
            ),
            encoding="utf-8",
        )
        conversations.append(
            {
                "conversation_id": cid,
                "speakers": ["G0001", "G0002"],
                "derived_duration_seconds": 10.0,
                "outputs": {
                    "audio": f"audio/{cid}.wav",
                    "audio_sha256": file_sha256(audio),
                    "reference": f"reference/{cid}.json",
                    "reference_sha256": file_sha256(reference),
                },
            }
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "source": {"revision": REVISION},
                "conversations": conversations,
            }
        ),
        encoding="utf-8",
    )
    return root


def _identity(tmp_path: Path) -> Any:
    return verify_model_identity(_cache(tmp_path / "cache"))


def _run(derived: Path, tmp_path: Path, model: StubModel) -> Any:
    return run_asr_benchmark(
        derived_root=derived,
        model=model,
        identity=_identity(tmp_path),
        device="cpu",
        compute_type="int8",
        decode_options={"language": "ko", "beam_size": 5},
    )


def test_the_reference_merges_turns_and_unattributed_rows_chronologically(
    derived: Path,
) -> None:
    entry = json.loads((derived / "manifest.json").read_text())["conversations"][0]
    path = derived / "reference" / f"{BENCHMARK_CONVERSATIONS[0]}.json"

    text, rows = reference_transcript(path, entry)

    assert rows == 3
    assert text.startswith(REF_A)
    assert REF_B in text
    # The tag row is carried but normalises to nothing, so it costs no reference characters.
    assert "[LAUGHTER]" in text
    assert characters(text) == characters(f"{REF_A} {REF_B}")


def test_each_conversation_is_transcribed_exactly_once(derived: Path, tmp_path: Path) -> None:
    stub = StubModel()

    _run(derived, tmp_path, stub)

    assert stub.calls == list(BENCHMARK_CONVERSATIONS)


def test_a_narrowed_selection_is_refused(derived: Path, tmp_path: Path) -> None:
    with pytest.raises(KcscAsrError, match="does not match the approved scope"):
        run_asr_benchmark(
            derived_root=derived,
            model=StubModel(),
            identity=_identity(tmp_path),
            device="cpu",
            compute_type="int8",
            decode_options={},
            conversation_ids=BENCHMARK_CONVERSATIONS[:2],
        )


def test_a_tampered_audio_file_fails_closed_before_transcription(
    derived: Path, tmp_path: Path
) -> None:
    (derived / "audio" / f"{BENCHMARK_CONVERSATIONS[2]}.wav").write_bytes(b"tampered")
    stub = StubModel()

    # verify_inputs raises the shared base class; KcscAsrError is a subclass of it.
    with pytest.raises(KcscBenchmarkError, match="audio checksum mismatch"):
        _run(derived, tmp_path, stub)
    assert stub.calls == []


def test_reference_composition_measures_overlap_without_quoting_text(derived: Path) -> None:
    composition = reference_composition(
        derived / "reference" / f"{BENCHMARK_CONVERSATIONS[0]}.json"
    )

    # Turns 0.0-3.0 (G0001) and 2.0-5.0 (G0002) overlap for 1.0 s of 6.0 s total speech.
    assert composition["reference_speech_seconds"] == pytest.approx(6.0)
    assert composition["overlapped_speech_seconds"] == pytest.approx(1.0)
    assert composition["overlapped_speech_fraction"] == pytest.approx(1 / 6, abs=1e-6)
    assert composition["rows_containing_tag"]["[LAUGHTER]"] == 1
    assert REF_A not in json.dumps(composition, ensure_ascii=False)


# ---------------------------------------------------------------------------- report safety


def test_the_report_is_transcript_free(derived: Path, tmp_path: Path) -> None:
    report = _run(derived, tmp_path, StubModel())
    path = tmp_path / "report.json"

    digest = write_report(report, path)

    raw = path.read_text(encoding="utf-8")
    for fragment in (REF_A, REF_B, "안녕하세요", "그렇습니다"):
        assert fragment not in raw
    assert digest == file_sha256(path)
    payload = json.loads(raw)
    assert payload["transcript_free"] is True
    assert payload["model"]["remote"] is False
    assert payload["offline"]["network_egress_attempts"] == 0
    assert payload["task"] == "input-level-mixed-conversation-asr"


def test_the_report_states_the_metric_normalization_and_its_limits(
    derived: Path, tmp_path: Path
) -> None:
    path = tmp_path / "report.json"
    write_report(_run(derived, tmp_path, StubModel()), path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    scoring = payload["scoring"]
    assert scoring["primary_metric"] == "korean_normalized_character_error_rate"
    assert scoring["secondary_metric"] == "eojeol_error_rate"
    assert any("NFC" in step for step in scoring["normalization_steps"])
    assert any("event tags" in step for step in scoring["normalization_steps"])
    limitations = " ".join(payload["limitations"])
    assert "not turn-level speaker ASR" in limitations
    assert "NOT stratified" in limitations
    assert "deletion floor" in limitations


def test_the_report_records_model_identity_and_input_checksums(
    derived: Path, tmp_path: Path
) -> None:
    path = tmp_path / "report.json"
    report = _run(derived, tmp_path, StubModel())
    write_report(report, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["model"]["revision"] == MODEL_REVISION
    assert payload["model"]["tree_sha256"] == report.model.tree_sha256
    assert [f["name"] for f in payload["model"]["files"]] == list(MODEL_FILES)
    assert payload["dataset"]["source_revision"] == REVISION
    manifest_entry = json.loads((derived / "manifest.json").read_text())["conversations"][0]
    assert payload["conversations"][0]["audio_sha256"] == manifest_entry["outputs"]["audio_sha256"]


def test_the_report_is_deterministic_apart_from_timing(derived: Path, tmp_path: Path) -> None:
    report = _run(derived, tmp_path, StubModel())
    first, second = tmp_path / "a.json", tmp_path / "b.json"

    write_report(report, first)
    write_report(report, second)

    assert file_sha256(first) == file_sha256(second)


# ----------------------------------------------------------------------------- offline guard


def test_the_offline_guard_blocks_and_counts_a_connection_attempt() -> None:
    import socket

    from voxdelta.providers.offline_guard import NetworkEgressAttempted, block_network_egress

    with block_network_egress() as log:
        with pytest.raises(NetworkEgressAttempted):
            socket.socket().connect(("huggingface.co", 443))
        with pytest.raises(NetworkEgressAttempted):
            socket.getaddrinfo("huggingface.co", 443)

    assert log.attempts == 2
    assert log.attempted is True
    assert "huggingface.co" in log.hosts


def test_the_guard_is_lifted_afterwards_so_it_cannot_leak_into_other_tests() -> None:
    import socket

    from voxdelta.providers.offline_guard import block_network_egress

    with block_network_egress():
        pass

    assert socket.socket.connect is not None
    assert callable(socket.getaddrinfo)


def test_a_run_that_attempted_egress_says_so_in_the_report(derived: Path, tmp_path: Path) -> None:
    """A report that reached the network must not look like a clean local run."""

    report = run_asr_benchmark(
        derived_root=derived,
        model=StubModel(),
        identity=_identity(tmp_path),
        device="cpu",
        compute_type="int8",
        decode_options={},
        egress_attempts=4,
    )
    path = tmp_path / "report.json"
    write_report(report, path)

    assert json.loads(path.read_text(encoding="utf-8"))["offline"]["network_egress_attempts"] == 4


def test_configure_offline_cache_pins_the_hub_client_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from voxdelta.evaluation.kcsc_asr_benchmark import configure_offline_cache

    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    configure_offline_cache(tmp_path)

    import os

    assert os.environ["HF_HUB_CACHE"] == str(tmp_path.resolve())
    assert os.environ["HF_HUB_OFFLINE"] == "1"
