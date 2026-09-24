"""Tests for the local-only KCSC derived evaluation-set workflow.

Every fixture here is synthetic. The raw corpus is licensed, evaluation-only material
that must not be copied into the test tree, and synthetic tracks let each rule be
exercised on its own — a mismatched pair, an off-spec sample rate, a marker that runs
past the end of the file.
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
    discover_conversations,
    is_vendor_marker,
    limit_to_int16,
    mix_tracks,
    parse_annotations,
    read_provenance,
    read_track_audio,
    silence_intervals,
)

REVISION = "364fb908b14b9e8383ef6bf9f7ebf5088ffddaf3"
TREE_DIGEST = "415cec754d23e70ceb449eda371f270c1b49b7fb6c540eac498800c5e826c47b"
PROVENANCE = f"""# ASR-KCSC provenance

- **Source:** `MagicHub/korean-conversational-speech-corpus`
- **Immutable revision:** `{REVISION}`
- **Local tree SHA-256:** `{TREE_DIGEST}`
"""


def write_wav(
    path: Path,
    samples: np.ndarray,
    *,
    sample_rate: int = EXPECTED_SAMPLE_RATE,
    channels: int = 1,
    sample_width: int = 2,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sample_width)
        handle.setframerate(sample_rate)
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


def tone(seconds: float, *, amplitude: int, sample_rate: int = EXPECTED_SAMPLE_RATE) -> np.ndarray:
    """A constant-amplitude square-ish signal: easy to assert on sample by sample."""

    return np.full(int(round(seconds * sample_rate)), amplitude, dtype=np.int32)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A two-conversation synthetic corpus mirroring the real tree's shape.

    ``A0001_S0001_0`` carries a leading vendor marker on both tracks plus a terminal
    marker on one, and unequal track lengths. ``A0002_S0001_0`` uses the alternate
    marker spelling and has no terminal marker.
    """

    root = tmp_path / "kcsc"
    (root / "PROVENANCE.md").parent.mkdir(parents=True, exist_ok=True)
    (root / "PROVENANCE.md").write_text(PROVENANCE, encoding="utf-8")

    write_wav(root / "WAV" / "A0001_S0001_0_G0001.wav", tone(10.0, amplitude=1000))
    write_wav(root / "WAV" / "A0001_S0001_0_G0002.wav", tone(8.0, amplitude=2000))
    write_annotations(
        root / "TXT" / "A0001_S0001_0_G0001.txt",
        [
            (1.0, 2.0, "G0001", "male", "매직 데이터"),
            (3.0, 4.0, "G0001", "male", "첫 번째 발화"),
            (9.0, 9.5, "G0001", "male", "매직 데이터"),
        ],
    )
    write_annotations(
        root / "TXT" / "A0001_S0001_0_G0002.txt",
        [
            (0.5, 1.5, "G0002", "female", "매직 데이터"),
            (5.0, 6.0, "G0002", "female", "두 번째 발화"),
            (7.0, 7.4, "0", "none", "[LAUGHTER]"),
        ],
    )

    write_wav(root / "WAV" / "A0002_S0001_0_G0003.wav", tone(6.0, amplitude=500))
    write_wav(root / "WAV" / "A0002_S0001_0_G0004.wav", tone(6.0, amplitude=500))
    write_annotations(
        root / "TXT" / "A0002_S0001_0_G0003.txt",
        [
            (0.2, 0.8, "G0003", "male", "매직 데이타"),
            (2.0, 2.5, "G0003", "male", "세 번째 발화"),
        ],
    )
    write_annotations(
        root / "TXT" / "A0002_S0001_0_G0004.txt",
        [(1.0, 1.5, "G0004", "female", "네 번째 발화")],
    )
    return root


def test_vendor_marker_detection_covers_both_spellings_and_spacing() -> None:
    assert is_vendor_marker("매직 데이터")
    assert is_vendor_marker("매직 데이타")
    assert is_vendor_marker("  매직데이터 ")
    assert not is_vendor_marker("매직 데이터를 소개합니다")
    assert not is_vendor_marker("")


def test_tracks_group_into_conversations_by_stripping_the_speaker_suffix(corpus: Path) -> None:
    conversations = discover_conversations(corpus)

    assert [conversation.conversation_id for conversation in conversations] == [
        "A0001_S0001_0",
        "A0002_S0001_0",
    ]
    assert [track.speaker for track in conversations[0].tracks] == ["G0001", "G0002"]


def test_an_unpaired_track_fails_closed(corpus: Path) -> None:
    (corpus / "WAV" / "A0001_S0001_0_G0002.wav").unlink()

    with pytest.raises(KcscDerivationError, match="has 1 tracks"):
        discover_conversations(corpus)


def test_a_track_without_its_annotation_file_fails_closed(corpus: Path) -> None:
    (corpus / "TXT" / "A0001_S0001_0_G0002.txt").unlink()

    with pytest.raises(KcscDerivationError, match="missing annotation file"):
        discover_conversations(corpus)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sample_rate": 8000}, "expected 16000 Hz"),
        ({"channels": 2}, "expected mono"),
        ({"sample_width": 1}, "expected 16-bit"),
    ],
)
def test_off_spec_audio_fails_closed(tmp_path: Path, kwargs: dict[str, int], match: str) -> None:
    path = write_wav(tmp_path / "track.wav", tone(1.0, amplitude=100), **kwargs)

    with pytest.raises(KcscDerivationError, match=match):
        read_track_audio(path)


@pytest.mark.parametrize(
    ("line", "match"),
    [
        ("[1.0,2.0]\tG0001\tmale", "expected 4 tab-separated fields"),
        ("1.0-2.0\tG0001\tmale\t발화", "malformed interval"),
        ("[2.0,1.0]\tG0001\tmale\t발화", "negative or inverted"),
        ("[1.0,2.0]\tG0009\tmale\t발화", "neither track speaker"),
    ],
)
def test_malformed_annotation_rows_fail_closed(tmp_path: Path, line: str, match: str) -> None:
    path = tmp_path / "A0001_S0001_0_G0001.txt"
    path.write_text(f"{line}\n", encoding="utf-8")

    with pytest.raises(KcscDerivationError, match=match):
        parse_annotations(path, expected_speaker="G0001")


def test_annotation_parse_errors_never_quote_the_transcript(tmp_path: Path) -> None:
    secret = "민감한전사내용"
    path = tmp_path / "A0001_S0001_0_G0001.txt"
    path.write_text(f"[2.0,1.0]\tG0001\tmale\t{secret}\n", encoding="utf-8")

    with pytest.raises(KcscDerivationError) as raised:
        parse_annotations(path, expected_speaker="G0001")
    assert secret not in str(raised.value)


def test_an_empty_annotation_file_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "A0001_S0001_0_G0001.txt"
    path.write_text("\n", encoding="utf-8")

    with pytest.raises(KcscDerivationError, match="contains no rows"):
        parse_annotations(path, expected_speaker="G0001")


def test_missing_provenance_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(KcscDerivationError, match="missing provenance record"):
        read_provenance(tmp_path)


def test_provenance_without_a_pinned_revision_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "PROVENANCE.md").write_text("# no revision here\n", encoding="utf-8")

    with pytest.raises(KcscDerivationError, match="missing source, revision, or digest"):
        read_provenance(tmp_path)


def test_the_shorter_track_is_padded_with_silence_not_truncated() -> None:
    long_track = tone(1.0, amplitude=100)
    short_track = tone(0.5, amplitude=100)

    mixed, padded = mix_tracks([long_track, short_track])

    assert mixed.size == long_track.size
    assert padded == (0, long_track.size - short_track.size)
    assert mixed[0] == 200
    assert mixed[-1] == 100


def test_a_summed_overshoot_is_scaled_not_clipped() -> None:
    mixed = np.array([30000, -60000, 15000], dtype=np.int32)

    samples, gain = limit_to_int16(mixed)

    assert gain == pytest.approx(32767 / 60000)
    assert int(np.max(np.abs(samples))) == 32767
    # A uniform gain keeps every relative level; hard clipping would flatten the peak
    # and leave the quieter samples untouched.
    assert samples[0] / samples[1] == pytest.approx(mixed[0] / mixed[1], rel=1e-3)


def test_a_mix_within_range_is_left_bit_exact() -> None:
    mixed = np.array([1000, -2000, 3000], dtype=np.int32)

    samples, gain = limit_to_int16(mixed)

    assert gain == 1.0
    assert samples.tolist() == mixed.tolist()


def test_a_terminal_marker_running_past_the_buffer_is_still_silenced() -> None:
    audio = np.full(EXPECTED_SAMPLE_RATE, 500, dtype=np.int32)

    applied = silence_intervals(audio, [(0.9, 2.0)], sample_rate=EXPECTED_SAMPLE_RATE)

    assert applied == ((0.9, 1.0),)
    assert not audio[int(0.9 * EXPECTED_SAMPLE_RATE) :].any()
    assert audio[: int(0.9 * EXPECTED_SAMPLE_RATE)].all()


def test_derivation_trims_silences_and_writes_an_auditable_manifest(
    corpus: Path, tmp_path: Path
) -> None:
    output = tmp_path / "derived"

    summary = derive_evaluation_set(corpus, output)

    assert summary.conversation_count == 2
    assert summary.provenance.revision == REVISION
    assert summary.provenance.local_tree_sha256 == TREE_DIGEST

    first = summary.conversations[0]
    # Earliest non-vendor onset across both speakers is G0001 at 3.0 s, not G0002's
    # vendor marker at 0.5 s.
    assert first.trim_offset_seconds == pytest.approx(3.0)
    assert first.source_duration_seconds == pytest.approx(10.0)
    assert first.derived_duration_seconds == pytest.approx(7.0)
    assert first.padded_samples == (0, 2 * EXPECTED_SAMPLE_RATE)
    assert first.vendor_marker_count == 3
    # Only the terminal marker survives the trim; the two leading markers are cut away.
    assert first.silenced_intervals == ((6.0, 6.5),)

    audio = read_track_audio(first.audio_path)
    assert audio.size == 7 * EXPECTED_SAMPLE_RATE
    assert not audio[6 * EXPECTED_SAMPLE_RATE : int(6.5 * EXPECTED_SAMPLE_RATE)].any()
    assert audio[0] == 3000  # 1000 + 2000, both tracks still running at t = 3 s
    assert audio[int(5.5 * EXPECTED_SAMPLE_RATE)] == 1000  # past the shorter track's end


def test_reference_files_shift_turn_timings_and_drop_vendor_rows(
    corpus: Path, tmp_path: Path
) -> None:
    output = tmp_path / "derived"

    summary = derive_evaluation_set(corpus, output)
    reference = json.loads(summary.conversations[0].reference_path.read_text(encoding="utf-8"))

    assert reference["conversation_id"] == "A0001_S0001_0"
    assert reference["source_revision"] == REVISION
    assert reference["trim_offset_seconds"] == pytest.approx(3.0)
    assert [(turn["start"], turn["end"], turn["speaker"]) for turn in reference["turns"]] == [
        (0.0, 1.0, "G0001"),
        (2.0, 3.0, "G0002"),
    ]
    assert not any(is_vendor_marker(turn["transcript"]) for turn in reference["turns"])
    # The corpus's own unattributed rows stay out of the scored turn list.
    assert [event["speaker"] for event in reference["unattributed_events"]] == ["0"]
    assert reference["unattributed_events"][0]["start"] == pytest.approx(4.0)
    # Transcript is preserved locally because ASR scoring needs it.
    assert reference["turns"][0]["transcript"] == "첫 번째 발화"


def test_the_manifest_records_provenance_source_names_and_output_checksums(
    corpus: Path, tmp_path: Path
) -> None:
    output = tmp_path / "derived"

    summary = derive_evaluation_set(corpus, output)
    manifest = json.loads(summary.manifest_path.read_text(encoding="utf-8"))

    assert manifest["source"]["revision"] == REVISION
    assert manifest["source"]["local_tree_sha256"] == TREE_DIGEST
    assert manifest["conversation_count"] == 2
    assert manifest["usage"]["training"] == "prohibited"
    assert manifest["usage"]["third_party_upload"].startswith("prohibited")

    entry = manifest["conversations"][0]
    assert [source["audio"] for source in entry["sources"]] == [
        "A0001_S0001_0_G0001.wav",
        "A0001_S0001_0_G0002.wav",
    ]
    assert [source["annotation"] for source in entry["sources"]] == [
        "A0001_S0001_0_G0001.txt",
        "A0001_S0001_0_G0002.txt",
    ]
    assert entry["outputs"]["audio"] == "audio/A0001_S0001_0.wav"
    assert entry["outputs"]["audio_sha256"] == summary.conversations[0].audio_sha256
    assert entry["outputs"]["reference_sha256"] == summary.conversations[0].reference_sha256


def test_the_manifest_carries_no_transcript_text(corpus: Path, tmp_path: Path) -> None:
    output = tmp_path / "derived"

    summary = derive_evaluation_set(corpus, output)

    raw = summary.manifest_path.read_text(encoding="utf-8")
    assert "첫 번째 발화" not in raw
    assert "매직" not in raw


def test_derivation_leaves_the_raw_tree_untouched(corpus: Path, tmp_path: Path) -> None:
    before = {
        path.relative_to(corpus).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(corpus.rglob("*"))
        if path.is_file()
    }

    derive_evaluation_set(corpus, tmp_path / "derived")

    after = {
        path.relative_to(corpus).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(corpus.rglob("*"))
        if path.is_file()
    }
    assert after == before


def test_derivation_is_deterministic(corpus: Path, tmp_path: Path) -> None:
    first = derive_evaluation_set(corpus, tmp_path / "one")
    second = derive_evaluation_set(corpus, tmp_path / "two")

    assert first.manifest_sha256 == second.manifest_sha256
    assert [entry.audio_sha256 for entry in first.conversations] == [
        entry.audio_sha256 for entry in second.conversations
    ]


def test_the_local_readme_states_the_use_restrictions(corpus: Path, tmp_path: Path) -> None:
    output = tmp_path / "derived"

    derive_evaluation_set(corpus, output)

    readme = (output / "README.md").read_text(encoding="utf-8")
    assert "Research-only" in readme
    assert "Training" in readme
    assert "Redistribution" in readme
    assert "pyannoteAI" in readme


def test_a_conversation_of_pure_vendor_markers_fails_closed(corpus: Path, tmp_path: Path) -> None:
    for speaker in ("G0003", "G0004"):
        write_annotations(
            corpus / "TXT" / f"A0002_S0001_0_{speaker}.txt",
            [(0.2, 0.8, speaker, "male", "매직 데이터")],
        )

    with pytest.raises(KcscDerivationError, match="no non-vendor speech rows"):
        derive_evaluation_set(corpus, tmp_path / "derived")


def test_unattributed_rows_are_accepted_and_flagged(tmp_path: Path) -> None:
    path = write_annotations(
        tmp_path / "A0001_S0001_0_G0001.txt",
        [
            (1.0, 2.0, "G0001", "male", "발화"),
            (3.0, 3.4, "0", "none", "[LAUGHTER]"),
        ],
    )

    rows = parse_annotations(path, expected_speaker="G0001")

    assert [row.attributed for row in rows] == [True, False]
    assert [row.speaker for row in rows] == ["G0001", "0"]


def test_an_unattributed_row_never_anchors_the_trim(tmp_path: Path) -> None:
    """The trim must land on real speech, not on a cough that precedes it."""

    root = tmp_path / "kcsc"
    root.mkdir(parents=True)
    (root / "PROVENANCE.md").write_text(PROVENANCE, encoding="utf-8")
    for speaker in ("G0001", "G0002"):
        write_wav(root / "WAV" / f"A0001_S0001_0_{speaker}.wav", tone(10.0, amplitude=1000))
    write_annotations(
        root / "TXT" / "A0001_S0001_0_G0001.txt",
        [
            (1.0, 1.4, "0", "none", "[SONANT]"),
            (5.0, 6.0, "G0001", "male", "첫 번째 발화"),
        ],
    )
    write_annotations(
        root / "TXT" / "A0001_S0001_0_G0002.txt",
        [(6.5, 7.0, "G0002", "female", "두 번째 발화")],
    )

    summary = derive_evaluation_set(root, tmp_path / "derived")

    conversation = summary.conversations[0]
    assert conversation.trim_offset_seconds == pytest.approx(5.0)
    assert conversation.turn_count == 2
    # The event at 1.0 s falls before the trim and is dropped rather than clamped to 0.
    assert conversation.unattributed_event_count == 0


def test_an_unknown_speaker_id_still_fails_closed(tmp_path: Path) -> None:
    path = write_annotations(
        tmp_path / "A0001_S0001_0_G0001.txt", [(1.0, 2.0, "G0777", "male", "발화")]
    )

    with pytest.raises(KcscDerivationError, match="neither track speaker"):
        parse_annotations(path, expected_speaker="G0001")
