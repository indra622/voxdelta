"""Tests for resolving a draft's recording and cutting clips out of it without copying.

The properties worth holding are the ones that would be invisible if they broke:

* A conversation id can only ever name one file under one root. Traversal, absolute
  paths, and symlinks out of the root are refused rather than resolved.
* A recording that is not the one the draft was made from is refused rather than played,
  because the draft would no longer describe what a reviewer hears.
* Gaps are the complement of the draft's valid turns over the real audio duration, with
  overlaps merged first — not the intervals of the turns validation dropped, which the
  artifact does not record.
"""

from __future__ import annotations

import hashlib
import io
import wave
from pathlib import Path

import pytest

from voxdelta.annotation.audio import (
    MAX_CLIP_SECONDS,
    ReviewGap,
    clip_bytes,
    merge_intervals,
    plan_clip,
    resolve_source,
    review_gaps,
    source_sha256,
)
from voxdelta.annotation.gemini_silver import SilverTurn
from voxdelta.annotation.review import ReviewRejected

FRAME_RATE = 16_000


def write_wav(path: Path, *, seconds: float = 10.0, frame_rate: int = FRAME_RATE) -> str:
    """A recording whose samples encode their own frame index, so a clip can be located."""

    frames = int(seconds * frame_rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(frame_rate)
        handle.writeframes(
            b"".join((index % 30_000).to_bytes(2, "little") for index in range(frames))
        )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def turn(start: float, end: float) -> SilverTurn:
    return SilverTurn(
        start=start,
        end=end,
        speaker="SPEAKER_00",
        transcript="…",
        emotion="neutral",
        emotion_rationale="",
        confidence=0.5,
    )


class TestMergeIntervals:
    def test_overlapping_turns_become_one_covered_span(self) -> None:
        assert merge_intervals([(0.0, 11.9), (12.0, 20.3), (19.3, 28.1)]) == (
            (0.0, 11.9),
            (12.0, 28.1),
        )

    def test_unordered_input_is_sorted_before_merging(self) -> None:
        assert merge_intervals([(5.0, 6.0), (0.0, 1.0)]) == ((0.0, 1.0), (5.0, 6.0))

    def test_touching_spans_merge_and_leave_no_zero_length_gap(self) -> None:
        assert merge_intervals([(0.0, 2.0), (2.0, 4.0)]) == ((0.0, 4.0),)

    def test_degenerate_and_non_finite_spans_are_dropped(self) -> None:
        assert merge_intervals([(1.0, 1.0), (2.0, 1.0), (float("nan"), 3.0)]) == ()


class TestReviewGaps:
    def test_gap_is_the_complement_of_the_covered_time(self) -> None:
        gaps = review_gaps([turn(0.0, 2.0), turn(5.0, 7.0)], duration_seconds=10.0)
        assert gaps == (ReviewGap(2.0, 5.0), ReviewGap(7.0, 10.0))

    def test_overlapping_turns_do_not_manufacture_a_gap(self) -> None:
        assert review_gaps([turn(0.0, 6.0), turn(5.0, 10.0)], duration_seconds=10.0) == ()

    def test_leading_silence_before_the_first_turn_is_a_gap(self) -> None:
        assert review_gaps([turn(3.0, 10.0)], duration_seconds=10.0) == (ReviewGap(0.0, 3.0),)

    def test_gaps_below_the_minimum_are_not_offered(self) -> None:
        gaps = review_gaps(
            [turn(0.0, 2.0), turn(2.2, 10.0)],
            duration_seconds=10.0,
            minimum_seconds=0.5,
        )
        assert gaps == ()

    def test_a_turn_running_past_the_recording_leaves_no_trailing_gap(self) -> None:
        assert review_gaps([turn(0.0, 12.0)], duration_seconds=10.0) == ()

    def test_a_draft_with_no_turns_is_one_whole_gap(self) -> None:
        assert review_gaps([], duration_seconds=10.0) == (ReviewGap(0.0, 10.0),)

    def test_unknown_duration_offers_nothing_rather_than_guessing(self) -> None:
        assert review_gaps([turn(0.0, 2.0)], duration_seconds=0.0) == ()


class TestResolveSource:
    def test_resolves_the_recording_named_by_the_conversation_id(self, tmp_path: Path) -> None:
        digest = write_wav(tmp_path / "audio" / "A6000_S0005_0.wav")
        source = resolve_source(tmp_path / "audio", "A6000_S0005_0", expected_sha256=digest)
        assert source.frame_rate == FRAME_RATE
        assert source.duration_seconds == pytest.approx(10.0)

    @pytest.mark.parametrize(
        ("collection", "conversation_id"),
        [
            ("kcsc/audio", "A6000_S0005_0"),
            ("022-finance-silver/audio", "O022_FIN_CLAIM_20210128_231556"),
            ("user-provided/2026-09-11-korean-evaluation-audio", "USER_EVAL_20260911_01"),
            (
                "user-provided/2026-09-11-korean-evaluation-audio/gold-splits-v1",
                "USER_EVAL_20260911_02A",
            ),
        ],
    )
    def test_resolves_only_named_derived_review_collections(
        self, tmp_path: Path, collection: str, conversation_id: str
    ) -> None:
        derived = tmp_path / "derived"
        digest = write_wav(derived / collection / f"{conversation_id}.wav")

        source = resolve_source(derived, conversation_id, expected_sha256=digest)

        assert source.duration_seconds == pytest.approx(10.0)

    @pytest.mark.parametrize(
        "conversation_id",
        ["../etc/passwd", "..", "a/b", "/absolute", "with space", "", "a" * 65],
    )
    def test_an_id_that_could_name_another_file_is_refused(
        self, tmp_path: Path, conversation_id: str
    ) -> None:
        write_wav(tmp_path / "audio" / "A6000_S0005_0.wav")
        with pytest.raises(ReviewRejected) as raised:
            resolve_source(tmp_path / "audio", conversation_id, expected_sha256="0" * 64)
        assert raised.value.code == "invalid_conversation_id"

    def test_a_symlink_out_of_the_root_is_refused(self, tmp_path: Path) -> None:
        outside = tmp_path / "elsewhere" / "private.wav"
        digest = write_wav(outside)
        root = tmp_path / "audio"
        root.mkdir()
        (root / "Escapee.wav").symlink_to(outside)
        with pytest.raises(ReviewRejected) as raised:
            resolve_source(root, "Escapee", expected_sha256=digest)
        assert raised.value.code == "audio_source_unavailable"

    def test_a_missing_recording_is_refused_without_naming_a_path(self, tmp_path: Path) -> None:
        (tmp_path / "audio").mkdir()
        with pytest.raises(ReviewRejected) as raised:
            resolve_source(tmp_path / "audio", "A6000_S0005_0", expected_sha256="0" * 64)
        assert raised.value.code == "audio_source_unavailable"
        assert str(tmp_path) not in raised.value.message

    def test_a_recording_swapped_since_annotation_is_refused(self, tmp_path: Path) -> None:
        write_wav(tmp_path / "audio" / "A6000_S0005_0.wav")
        with pytest.raises(ReviewRejected) as raised:
            resolve_source(tmp_path / "audio", "A6000_S0005_0", expected_sha256="a" * 64)
        assert raised.value.code == "audio_source_mismatch"

    def test_a_draft_recording_no_digest_was_recorded_for_is_refused(
        self, tmp_path: Path
    ) -> None:
        write_wav(tmp_path / "audio" / "A6000_S0005_0.wav")
        with pytest.raises(ReviewRejected) as raised:
            resolve_source(tmp_path / "audio", "A6000_S0005_0", expected_sha256="")
        assert raised.value.code == "audio_source_mismatch"

    def test_a_file_that_is_not_wav_is_refused(self, tmp_path: Path) -> None:
        root = tmp_path / "audio"
        root.mkdir()
        path = root / "A6000_S0005_0.wav"
        path.write_bytes(b"not a wav at all")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with pytest.raises(ReviewRejected) as raised:
            resolve_source(root, "A6000_S0005_0", expected_sha256=digest)
        assert raised.value.code == "audio_source_unreadable"


class TestSourceSha256:
    def test_reads_the_digest_the_draft_recorded(self) -> None:
        assert source_sha256({"source": {"input_sha256": "b" * 64}}) == "b" * 64

    def test_a_draft_without_a_source_block_yields_no_digest(self) -> None:
        assert source_sha256({}) == ""
        assert source_sha256({"source": {"input_sha256": 5}}) == ""


class TestPlanClip:
    @pytest.fixture
    def source(self, tmp_path: Path):
        digest = write_wav(tmp_path / "audio" / "A6000_S0005_0.wav")
        return resolve_source(tmp_path / "audio", "A6000_S0005_0", expected_sha256=digest)

    def test_a_range_becomes_the_frames_it_names(self, source) -> None:
        plan = plan_clip(source, start=1.0, end=3.0)
        assert plan.start_frame == FRAME_RATE
        assert plan.frame_count == 2 * FRAME_RATE

    @pytest.mark.parametrize(
        ("start", "end"),
        [(-1.0, 2.0), (2.0, 2.0), (3.0, 1.0), (float("nan"), 1.0), (0.0, float("inf"))],
    )
    def test_a_range_that_is_not_a_forward_span_is_refused(self, source, start, end) -> None:
        with pytest.raises(ReviewRejected) as raised:
            plan_clip(source, start=start, end=end)
        assert raised.value.code in {"invalid_clip_range", "clip_too_long"}

    def test_a_range_longer_than_the_ceiling_is_refused(self, source) -> None:
        with pytest.raises(ReviewRejected) as raised:
            plan_clip(source, start=0.0, end=MAX_CLIP_SECONDS + 1)
        assert raised.value.code == "clip_too_long"

    def test_a_range_past_the_end_is_clamped_rather_than_refused(self, source) -> None:
        plan = plan_clip(source, start=9.0, end=12.0)
        assert plan.end_seconds == pytest.approx(10.0)

    def test_a_range_starting_past_the_end_is_refused(self, source) -> None:
        with pytest.raises(ReviewRejected) as raised:
            plan_clip(source, start=11.0, end=12.0)
        assert raised.value.code == "invalid_clip_range"


class TestClipBytes:
    def test_the_clip_is_a_playable_wav_holding_exactly_the_requested_audio(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "audio" / "A6000_S0005_0.wav"
        digest = write_wav(path)
        source = resolve_source(tmp_path / "audio", "A6000_S0005_0", expected_sha256=digest)
        plan = plan_clip(source, start=1.0, end=1.5)

        rendered = b"".join(clip_bytes(plan))
        assert len(rendered) == plan.total_bytes

        with wave.open(io.BytesIO(rendered), "rb") as clip:
            assert clip.getframerate() == FRAME_RATE
            assert clip.getnchannels() == 1
            assert clip.getsampwidth() == 2
            assert clip.getnframes() == int(0.5 * FRAME_RATE)
            first = int.from_bytes(clip.readframes(1), "little")
        # The samples encode their own index, so this proves the clip starts where it was
        # asked to rather than at the beginning of the file.
        assert first == FRAME_RATE % 30_000

    def test_the_source_file_is_never_modified_or_copied(self, tmp_path: Path) -> None:
        root = tmp_path / "audio"
        path = root / "A6000_S0005_0.wav"
        digest = write_wav(path)
        source = resolve_source(root, "A6000_S0005_0", expected_sha256=digest)
        before = sorted(entry.name for entry in root.iterdir())

        b"".join(clip_bytes(plan_clip(source, start=0.0, end=2.0)))

        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        assert sorted(entry.name for entry in root.iterdir()) == before
