"""Tests for the per-speaker KCSC track derivation.

Every fixture is synthetic. The raw corpus is licensed, evaluation-only material that must
not be copied into the test tree, and synthetic tracks let each rule be exercised on its
own — the shared trim, a marker annotated on the *other* speaker, a reference that has
drifted from the mixed set it must agree with.

The corpus is built under the three approved conversation ids because the derivation
refuses any other selection, and the mixed set the tracks are tied to is produced by the
real mixed derivation rather than a hand-written stand-in: the cross-check is only worth
testing against the artifact it will actually meet.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from voxdelta.evaluation.kcsc_derivation import (
    EXPECTED_SAMPLE_RATE,
    KcscDerivationError,
    derive_evaluation_set,
    read_track_audio,
)
from voxdelta.evaluation.kcsc_track_derivation import (
    BENCHMARK_CONVERSATIONS,
    conversation_trim_samples,
    cross_check_against_mixed,
    derive_track_set,
)

REVISION = "364fb908b14b9e8383ef6bf9f7ebf5088ffddaf3"
TREE_DIGEST = "415cec754d23e70ceb449eda371f270c1b49b7fb6c540eac498800c5e826c47b"
PROVENANCE = f"""# ASR-KCSC provenance

- **Source:** `MagicHub/korean-conversational-speech-corpus`
- **Immutable revision:** `{REVISION}`
- **Local tree SHA-256:** `{TREE_DIGEST}`
"""

FIRST, SECOND, THIRD = BENCHMARK_CONVERSATIONS


def write_wav(path: Path, samples: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(EXPECTED_SAMPLE_RATE)
        handle.writeframes(samples.astype("<i2").tobytes())
    return path


def write_annotations(path: Path, rows: list[tuple[float, float, str, str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            f"[{s:.3f},{e:.3f}]\t{spk}\t{gender}\t{text}\n" for s, e, spk, gender, text in rows
        ),
        encoding="utf-8",
    )
    return path


def tone(seconds: float, *, amplitude: int) -> np.ndarray:
    """A constant-amplitude signal, so a track can be told from the mix sample by sample."""

    return np.full(int(round(seconds * EXPECTED_SAMPLE_RATE)), amplitude, dtype=np.int32)


def at(audio: np.ndarray, seconds: float) -> int:
    return int(audio[int(seconds * EXPECTED_SAMPLE_RATE)])


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A synthetic corpus under the three approved ids, one distinct level per speaker.

    ``A0055_S0006_0`` is the interesting one: both tracks run the full length and each
    speaker carries a vendor marker inside the other's audio, so cross-track silencing is
    observable. ``A0051_S0001_0`` has unequal track lengths and an unattributed row.
    """

    root = tmp_path / "kcsc"
    root.mkdir(parents=True)
    (root / "PROVENANCE.md").write_text(PROVENANCE, encoding="utf-8")

    write_wav(root / "WAV" / f"{FIRST}_G0101.wav", tone(10.0, amplitude=1000))
    write_wav(root / "WAV" / f"{FIRST}_G0102.wav", tone(8.0, amplitude=2000))
    write_annotations(
        root / "TXT" / f"{FIRST}_G0101.txt",
        [
            (1.0, 2.0, "G0101", "male", "매직 데이터"),
            (3.0, 4.0, "G0101", "male", "첫 번째 발화"),
            (9.0, 9.5, "G0101", "male", "매직 데이터"),
        ],
    )
    write_annotations(
        root / "TXT" / f"{FIRST}_G0102.txt",
        [
            (0.5, 1.5, "G0102", "female", "매직 데이터"),
            (5.0, 6.0, "G0102", "female", "두 번째 발화"),
            (7.0, 7.4, "0", "none", "[LAUGHTER]"),
        ],
    )

    write_wav(root / "WAV" / f"{SECOND}_G0201.wav", tone(10.0, amplitude=1000))
    write_wav(root / "WAV" / f"{SECOND}_G0202.wav", tone(10.0, amplitude=3000))
    write_annotations(
        root / "TXT" / f"{SECOND}_G0201.txt",
        [
            (0.0, 0.5, "G0201", "male", "매직 데이터"),
            (2.0, 3.0, "G0201", "male", "세 번째 발화"),
            (8.0, 8.5, "G0201", "male", "매직 데이터"),
        ],
    )
    write_annotations(
        root / "TXT" / f"{SECOND}_G0202.txt",
        [
            (4.0, 5.0, "G0202", "female", "네 번째 발화"),
            (6.0, 6.5, "G0202", "female", "매직 데이타"),
        ],
    )

    write_wav(root / "WAV" / f"{THIRD}_G0301.wav", tone(6.0, amplitude=500))
    write_wav(root / "WAV" / f"{THIRD}_G0302.wav", tone(6.0, amplitude=700))
    write_annotations(
        root / "TXT" / f"{THIRD}_G0301.txt",
        [
            (0.2, 0.8, "G0301", "male", "매직 데이타"),
            (2.0, 2.5, "G0301", "male", "다섯 번째 발화"),
        ],
    )
    write_annotations(
        root / "TXT" / f"{THIRD}_G0302.txt",
        [(1.0, 1.5, "G0302", "female", "여섯 번째 발화")],
    )
    return root


@pytest.fixture
def mixed(corpus: Path, tmp_path: Path) -> Path:
    """The mixed evaluation set the tracks are cut against, from the same raw corpus."""

    root = tmp_path / "mixed"
    derive_evaluation_set(corpus, root)
    return root


# ------------------------------------------------------------------------------ the timeline


def test_both_tracks_are_cut_at_the_conversations_shared_trim_offset(
    corpus: Path, mixed: Path, tmp_path: Path
) -> None:
    """A per-track trim would move each track's clock away from the mixed reference."""

    summary = derive_track_set(corpus, mixed, tmp_path / "tracks")

    second = [track for track in summary.tracks if track.conversation_id == SECOND]
    # Earliest attributed non-vendor onset across both speakers is G0201 at 2.0 s.
    assert [track.trim_offset_seconds for track in second] == [2.0, 2.0]
    assert [track.derived_duration_seconds for track in second] == [8.0, 8.0]
    manifest = json.loads((mixed / "manifest.json").read_text(encoding="utf-8"))
    entry = next(e for e in manifest["conversations"] if e["conversation_id"] == SECOND)
    assert entry["trim_offset_seconds"] == pytest.approx(2.0)


def test_a_track_is_the_speakers_own_microphone_and_not_the_mix(
    corpus: Path, mixed: Path, tmp_path: Path
) -> None:
    summary = derive_track_set(corpus, mixed, tmp_path / "tracks")

    by_id = {f"{t.conversation_id}_{t.speaker}": t for t in summary.tracks}
    quiet = read_track_audio(by_id[f"{SECOND}_G0201"].audio_path)
    loud = read_track_audio(by_id[f"{SECOND}_G0202"].audio_path)

    assert at(quiet, 0.5) == 1000
    assert at(loud, 0.5) == 3000
    # The mixed set sums to 4000 at the same instant; neither track may show that.
    assert at(quiet, 0.5) + at(loud, 0.5) == 4000


def test_a_shorter_track_keeps_its_own_length_rather_than_being_padded(
    corpus: Path, mixed: Path, tmp_path: Path
) -> None:
    summary = derive_track_set(corpus, mixed, tmp_path / "tracks")

    first = {t.speaker: t for t in summary.tracks if t.conversation_id == FIRST}
    assert first["G0101"].derived_duration_seconds == pytest.approx(7.0)  # 10 s track, trim 3 s
    assert first["G0102"].derived_duration_seconds == pytest.approx(5.0)  # 8 s track, trim 3 s


# ------------------------------------------------------------------------- vendor treatment


def test_a_marker_annotated_on_the_other_speaker_is_silenced_on_this_track(
    corpus: Path, mixed: Path, tmp_path: Path
) -> None:
    """One room, two microphones: the other speaker's marker is audible here too."""

    summary = derive_track_set(corpus, mixed, tmp_path / "tracks")

    by_id = {f"{t.conversation_id}_{t.speaker}": t for t in summary.tracks}
    loud = read_track_audio(by_id[f"{SECOND}_G0202"].audio_path)
    # G0201's terminal marker at 8.0-8.5 s lands at 6.0-6.5 s on the trimmed timeline.
    assert not loud[int(6.0 * EXPECTED_SAMPLE_RATE) : int(6.5 * EXPECTED_SAMPLE_RATE)].any()
    assert at(loud, 5.9) == 3000
    assert at(loud, 6.6) == 3000
    # And this track's own marker, at 6.0-6.5 s source time, lands at 4.0-4.5 s.
    assert not loud[int(4.0 * EXPECTED_SAMPLE_RATE) : int(4.5 * EXPECTED_SAMPLE_RATE)].any()
    assert by_id[f"{SECOND}_G0202"].silenced_intervals == ((4.0, 4.5), (6.0, 6.5))


def test_the_same_intervals_are_silenced_on_both_tracks_of_a_conversation(
    corpus: Path, mixed: Path, tmp_path: Path
) -> None:
    summary = derive_track_set(corpus, mixed, tmp_path / "tracks")

    second = [track for track in summary.tracks if track.conversation_id == SECOND]
    assert second[0].silenced_intervals == second[1].silenced_intervals


# ------------------------------------------------------------------------------- references


def test_the_per_track_reference_carries_only_that_speakers_turns(
    corpus: Path, mixed: Path, tmp_path: Path
) -> None:
    summary = derive_track_set(corpus, mixed, tmp_path / "tracks")

    by_id = {f"{t.conversation_id}_{t.speaker}": t for t in summary.tracks}
    reference = json.loads(by_id[f"{FIRST}_G0102"].reference_path.read_text(encoding="utf-8"))

    assert reference["conversation_id"] == FIRST
    assert reference["speaker"] == "G0102"
    assert reference["source_revision"] == REVISION
    assert reference["trim_offset_seconds"] == pytest.approx(3.0)
    assert [(t["start"], t["end"], t["speaker"]) for t in reference["turns"]] == [
        (2.0, 3.0, "G0102")
    ]
    # Transcript is kept locally because ASR scoring needs it.
    assert reference["turns"][0]["transcript"] == "두 번째 발화"
    # The corpus's unattributed rows stay with the track that annotated them.
    assert [e["speaker"] for e in reference["unattributed_events"]] == ["0"]
    assert reference["unattributed_events"][0]["start"] == pytest.approx(4.0)


def test_the_two_tracks_reference_rows_reunite_into_the_mixed_reference(
    corpus: Path, mixed: Path, tmp_path: Path
) -> None:
    """The upper bound is only a bound if both benchmarks score the same ground truth."""

    summary = derive_track_set(corpus, mixed, tmp_path / "tracks")

    def rows(payload: dict[str, object]) -> list[tuple[object, ...]]:
        collected = []
        for key in ("turns", "unattributed_events"):
            block = payload[key]
            assert isinstance(block, list)
            collected.extend(
                (r["start"], r["end"], r["speaker"], r["gender"], r["transcript"]) for r in block
            )
        return sorted(collected)

    for conversation_id in BENCHMARK_CONVERSATIONS:
        union: list[tuple[object, ...]] = []
        for track in summary.tracks:
            if track.conversation_id == conversation_id:
                union.extend(rows(json.loads(track.reference_path.read_text(encoding="utf-8"))))
        mixed_reference = json.loads(
            (mixed / "reference" / f"{conversation_id}.json").read_text(encoding="utf-8")
        )
        assert sorted(union) == rows(mixed_reference)


def test_a_re_derived_reference_that_disagrees_with_the_mixed_set_fails_closed() -> None:
    mixed_reference = {
        "turns": [
            {
                "start": 0.0,
                "end": 1.0,
                "speaker": "G0101",
                "gender": "male",
                "transcript": "첫 번째 발화",
            }
        ]
    }

    with pytest.raises(KcscDerivationError, match="disagree with the mixed reference"):
        cross_check_against_mixed(
            [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "speaker": "G0101",
                    "gender": "male",
                    "transcript": "다른 발화",
                }
            ],
            mixed_reference,
            conversation_id=FIRST,
            speaker="G0101",
        )


def test_the_cross_check_reports_counts_and_never_the_rows_that_differed() -> None:
    secret = "민감한전사내용"
    mixed_reference = {
        "turns": [
            {"start": 0.0, "end": 1.0, "speaker": "G0101", "gender": "male", "transcript": secret}
        ]
    }

    with pytest.raises(KcscDerivationError) as raised:
        cross_check_against_mixed([], mixed_reference, conversation_id=FIRST, speaker="G0101")

    assert secret not in str(raised.value)
    assert "0 rows derived, 1 rows in the mixed set" in str(raised.value)


def test_a_mixed_reference_that_drifted_from_its_manifest_fails_closed(
    corpus: Path, mixed: Path, tmp_path: Path
) -> None:
    path = mixed / "reference" / f"{FIRST}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["turns"][0]["transcript"] = "조작된 전사"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(KcscDerivationError, match="mixed reference checksum mismatch"):
        derive_track_set(corpus, mixed, tmp_path / "tracks")


def test_a_conversation_absent_from_the_mixed_manifest_fails_closed(
    corpus: Path, mixed: Path, tmp_path: Path
) -> None:
    path = mixed / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["conversations"] = [
        entry for entry in manifest["conversations"] if entry["conversation_id"] != FIRST
    ]
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(KcscDerivationError, match=f"conversation {FIRST} is absent"):
        derive_track_set(corpus, mixed, tmp_path / "tracks")


def test_a_trim_offset_disagreeing_with_the_mixed_manifest_fails_closed(
    corpus: Path, mixed: Path, tmp_path: Path
) -> None:
    """A silently different trim would make the two benchmarks measure different audio."""

    path = mixed / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for entry in manifest["conversations"]:
        if entry["conversation_id"] == FIRST:
            entry["trim_offset_seconds"] = 1.25
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(KcscDerivationError, match="disagrees with the mixed manifest"):
        derive_track_set(corpus, mixed, tmp_path / "tracks")


def test_a_missing_mixed_manifest_fails_closed(corpus: Path, tmp_path: Path) -> None:
    with pytest.raises(KcscDerivationError, match="missing mixed derivation manifest"):
        derive_track_set(corpus, tmp_path / "absent", tmp_path / "tracks")


def test_a_track_with_no_scorable_turns_fails_closed(corpus: Path, tmp_path: Path) -> None:
    write_annotations(
        corpus / "TXT" / f"{THIRD}_G0302.txt",
        [(1.0, 1.5, "0", "none", "[LAUGHTER]")],
    )
    mixed_root = tmp_path / "mixed"
    derive_evaluation_set(corpus, mixed_root)

    with pytest.raises(KcscDerivationError, match="no scorable turns"):
        derive_track_set(corpus, mixed_root, tmp_path / "tracks")


# ---------------------------------------------------------------------------------- scope


def test_a_narrowed_selection_is_refused(corpus: Path, mixed: Path, tmp_path: Path) -> None:
    with pytest.raises(KcscDerivationError, match="does not match the approved scope"):
        derive_track_set(
            corpus,
            mixed,
            tmp_path / "tracks",
            conversation_ids=BENCHMARK_CONVERSATIONS[:2],
        )


def test_the_trim_anchors_on_speech_not_on_a_vendor_marker_or_a_cough(corpus: Path) -> None:
    from voxdelta.evaluation.kcsc_derivation import discover_conversations, parse_annotations

    conversation = next(c for c in discover_conversations(corpus) if c.conversation_id == SECOND)
    annotations = [
        parse_annotations(track.annotation_path, expected_speaker=track.speaker)
        for track in conversation.tracks
    ]

    samples = conversation_trim_samples(annotations, 10 * EXPECTED_SAMPLE_RATE)

    # G0201's vendor marker opens at 0.0 s; the first real utterance is at 2.0 s.
    assert samples == 2 * EXPECTED_SAMPLE_RATE


# -------------------------------------------------------------------------------- manifest


def test_the_manifest_records_source_checksums_and_the_mixed_reference_digest(
    corpus: Path, mixed: Path, tmp_path: Path
) -> None:
    output = tmp_path / "tracks"

    summary = derive_track_set(corpus, mixed, output)
    manifest = json.loads(summary.manifest_path.read_text(encoding="utf-8"))

    assert manifest["track_count"] == 6
    assert manifest["conversation_ids"] == list(BENCHMARK_CONVERSATIONS)
    assert manifest["source"]["revision"] == REVISION
    assert manifest["source"]["local_tree_sha256"] == TREE_DIGEST
    assert manifest["usage"]["training"] == "prohibited"
    assert manifest["usage"]["third_party_upload"].startswith("prohibited")
    assert "not source-separated" in manifest["acoustic_note"]

    entry = manifest["tracks"][0]
    assert entry["track_id"] == f"{FIRST}_G0101"
    assert entry["source"]["audio"] == f"{FIRST}_G0101.wav"
    assert entry["source"]["annotation"] == f"{FIRST}_G0101.txt"
    assert entry["outputs"]["audio"] == f"audio/{FIRST}_G0101.wav"
    assert entry["outputs"]["audio_sha256"] == summary.tracks[0].audio_sha256
    assert entry["outputs"]["reference_sha256"] == summary.tracks[0].reference_sha256
    mixed_manifest = json.loads((mixed / "manifest.json").read_text(encoding="utf-8"))
    mixed_entry = next(e for e in mixed_manifest["conversations"] if e["conversation_id"] == FIRST)
    assert entry["mixed_reference_sha256"] == mixed_entry["outputs"]["reference_sha256"]


def test_the_manifest_carries_no_transcript_text(corpus: Path, mixed: Path, tmp_path: Path) -> None:
    summary = derive_track_set(corpus, mixed, tmp_path / "tracks")

    raw = summary.manifest_path.read_text(encoding="utf-8")
    assert "첫 번째 발화" not in raw
    assert "매직" not in raw


def test_the_local_readme_states_the_restrictions_and_the_acoustic_caveat(
    corpus: Path, mixed: Path, tmp_path: Path
) -> None:
    output = tmp_path / "tracks"

    derive_track_set(corpus, mixed, output)

    readme = (output / "README.md").read_text(encoding="utf-8")
    assert "Research-only" in readme
    assert "Training" in readme
    assert "Redistribution" in readme
    assert "pyannoteAI" in readme
    assert "NOT clean single-speaker audio" in readme


def test_derivation_leaves_the_raw_tree_untouched(
    corpus: Path, mixed: Path, tmp_path: Path
) -> None:
    before = {
        path.relative_to(corpus).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(corpus.rglob("*"))
        if path.is_file()
    }

    derive_track_set(corpus, mixed, tmp_path / "tracks")

    after = {
        path.relative_to(corpus).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(corpus.rglob("*"))
        if path.is_file()
    }
    assert after == before


def test_track_derivation_is_deterministic(corpus: Path, mixed: Path, tmp_path: Path) -> None:
    first = derive_track_set(corpus, mixed, tmp_path / "one")
    second = derive_track_set(corpus, mixed, tmp_path / "two")

    assert first.manifest_sha256 == second.manifest_sha256
    assert [track.audio_sha256 for track in first.tracks] == [
        track.audio_sha256 for track in second.tracks
    ]
    assert first.track_count == 6
    assert first.turn_count == second.turn_count
