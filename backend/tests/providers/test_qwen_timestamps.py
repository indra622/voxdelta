"""Tests for the Qwen zero-duration timestamp sanitation contract.

The contract's whole job is to keep real text without inventing timing, so what is pinned
here is where the line falls: genuine spans survive untouched, an instant strictly inside
one segment is attributed at the moment the aligner reported, and an instant that could
land on either side of a boundary is dropped and counted rather than guessed at.

The alignment and validation these tests exercise are shared with faster-whisper, so the
last group checks that none of the relaxations reach it and that the default ASR provider
is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr

from voxdelta.api.dependencies import ProviderFactories, build_dependencies
from voxdelta.config import Settings
from voxdelta.domain.models import SpeakerSegment
from voxdelta.providers.asr_alignment import (
    AlignedWord,
    align_mixed,
    align_separate,
    point_segment_index,
    validated_point_words,
    validated_words,
)
from voxdelta.providers.base import ProviderError
from voxdelta.providers.qwen_timestamps import (
    SANITATION_POLICY,
    TimestampCoverage,
    sanitize_words,
)

DURATION = 10.0


def _timeline() -> list[SpeakerSegment]:
    """Two abutting speakers, so 5.0 is a boundary shared by both."""

    return [
        SpeakerSegment(start=0.0, end=5.0, speaker_id="SPEAKER_00", confidence=1.0),
        SpeakerSegment(start=5.0, end=10.0, speaker_id="SPEAKER_01", confidence=1.0),
    ]


# ------------------------------------------------------------------ genuine spans survive


def test_positive_spans_pass_through_untouched() -> None:
    records = [(0.5, 1.0, "하나"), (1.5, 2.25, "둘"), (6.0, 7.0, "셋")]

    words, coverage = sanitize_words(records, DURATION, _timeline())

    assert words == [
        AlignedWord(0.5, 1.0, "하나"),
        AlignedWord(1.5, 2.25, "둘"),
        AlignedWord(6.0, 7.0, "셋"),
    ]
    assert coverage.zero_spans == 0
    assert coverage.omitted_words == 0
    assert coverage.uncertain is False
    assert coverage.attributed_ratio == 1.0


def test_a_span_running_backwards_still_fails_closed() -> None:
    """A reversed span is a broken aligner, not a point event, and is never sanitized."""

    with pytest.raises(ProviderError) as raised:
        sanitize_words([(2.0, 1.0, "역방향")], DURATION, _timeline())
    assert raised.value.code == "invalid_provider_output"


def test_a_span_past_the_asset_duration_still_fails_closed() -> None:
    with pytest.raises(ProviderError) as raised:
        sanitize_words([(9.0, 11.0, "너무김")], DURATION, _timeline())
    assert raised.value.code == "invalid_provider_output"


# ------------------------------------------------------------- zero spans and the boundary


def test_an_instant_strictly_inside_one_segment_is_kept() -> None:
    words, coverage = sanitize_words([(2.0, 2.0, "안쪽")], DURATION, _timeline())

    assert words == [AlignedWord(2.0, 2.0, "안쪽")]
    assert coverage.zero_spans == 1
    assert coverage.zero_spans_unplaceable == 0
    assert coverage.uncertain is False


@pytest.mark.parametrize("instant", [0.0, 5.0, 10.0])
def test_an_instant_on_a_segment_edge_is_omitted_not_guessed(instant: float) -> None:
    """5.0 belongs equally to both speakers; placing it would be a coin flip."""

    words, coverage = sanitize_words([(instant, instant, "경계")], DURATION, _timeline())

    assert words == []
    assert coverage.zero_spans == 1
    assert coverage.zero_spans_unplaceable == 1
    assert coverage.uncertain is True
    assert coverage.attributed_ratio == 0.0


def test_an_instant_in_a_gap_between_segments_is_omitted() -> None:
    timeline = [
        SpeakerSegment(start=0.0, end=2.0, speaker_id="SPEAKER_00", confidence=1.0),
        SpeakerSegment(start=8.0, end=10.0, speaker_id="SPEAKER_01", confidence=1.0),
    ]

    words, coverage = sanitize_words([(5.0, 5.0, "공백")], DURATION, timeline)

    assert words == []
    assert coverage.zero_spans_unplaceable == 1


def test_an_instant_inside_two_overlapping_segments_is_omitted() -> None:
    timeline = [
        SpeakerSegment(start=0.0, end=6.0, speaker_id="SPEAKER_00", confidence=1.0),
        SpeakerSegment(start=4.0, end=10.0, speaker_id="SPEAKER_01", confidence=1.0),
    ]

    assert point_segment_index(5.0, timeline) is None
    words, coverage = sanitize_words([(5.0, 5.0, "중첩")], DURATION, timeline)
    assert words == []
    assert coverage.zero_spans_unplaceable == 1


def test_without_a_timeline_no_instant_can_be_misattributed_so_none_is_dropped() -> None:
    """Separate-channel audio has one speaker per channel; there is no boundary to cross."""

    words, coverage = sanitize_words([(0.5, 1.0, "하나"), (5.0, 5.0, "점")], DURATION, None)

    assert [word.text for word in words] == ["하나", "점"]
    assert coverage.zero_spans == 1
    assert coverage.zero_spans_unplaceable == 0


# ------------------------------------------------------------------------------- ordering


def test_word_order_is_preserved_and_never_re_sorted() -> None:
    records = [(0.5, 1.0, "하나"), (2.0, 2.0, "점"), (2.5, 3.0, "둘"), (4.0, 4.0, "점둘")]

    words, _coverage = sanitize_words(records, DURATION, _timeline())

    assert [word.text for word in words] == ["하나", "점", "둘", "점둘"]
    assert [word.start for word in words] == [0.5, 2.0, 2.5, 4.0]


def test_words_arriving_out_of_order_fail_closed() -> None:
    with pytest.raises(ProviderError) as raised:
        sanitize_words([(3.0, 4.0, "둘"), (1.0, 2.0, "하나")], DURATION, _timeline())
    assert raised.value.code == "invalid_provider_output"


def test_two_instants_at_the_same_moment_are_both_kept() -> None:
    """An aligner that cannot separate two adjacent tokens reports both at one moment."""

    words, coverage = sanitize_words([(2.0, 2.0, "가"), (2.0, 2.0, "나")], DURATION, _timeline())

    assert [word.text for word in words] == ["가", "나"]
    assert coverage.zero_spans == 2


def test_duplicate_positive_spans_still_fail_closed() -> None:
    """The duplicate relaxation is for instants only."""

    with pytest.raises(ProviderError) as raised:
        sanitize_words([(1.0, 2.0, "가"), (1.0, 2.0, "나")], DURATION, _timeline())
    assert raised.value.code == "invalid_provider_output"


# ---------------------------------------------------------------------------- attribution


def test_an_interior_instant_joins_its_speakers_turn_at_its_own_moment() -> None:
    words = validated_point_words(
        [(0.5, 1.0, "하나"), (2.0, 2.0, "점"), (6.0, 7.0, "둘")], DURATION
    )

    utterances, omitted = align_mixed(words, _timeline(), DURATION)

    assert omitted == 0
    assert [(u.speaker_id, u.start, u.end, u.transcript) for u in utterances] == [
        ("SPEAKER_00", 0.5, 2.0, "하나 점"),
        ("SPEAKER_01", 6.0, 7.0, "둘"),
    ]


def test_an_instant_never_extends_a_turn_past_its_speakers_boundary() -> None:
    """The turn may reach the instant, but clamping still stops it at the next speaker."""

    words = validated_point_words([(0.5, 1.0, "하나"), (4.9, 4.9, "끝")], DURATION)

    utterances, _omitted = align_mixed(words, _timeline(), DURATION)

    assert len(utterances) == 1
    assert utterances[0].speaker_id == "SPEAKER_00"
    assert utterances[0].end <= 5.0


def test_a_turn_made_only_of_instants_is_omitted_rather_than_widened() -> None:
    """An utterance must have real extent; inventing one would be fabricated timing."""

    words = validated_point_words(
        [(0.5, 1.0, "하나"), (7.0, 7.0, "점"), (8.0, 8.0, "점둘")], DURATION
    )

    utterances, omitted = align_mixed(words, _timeline(), DURATION)

    assert omitted == 2
    assert [u.speaker_id for u in utterances] == ["SPEAKER_00"]
    assert "점" not in utterances[0].transcript


def test_separate_channels_omit_instant_only_runs_and_report_the_count() -> None:
    left = validated_point_words([(0.5, 1.0, "왼쪽")], DURATION)
    right = validated_point_words([(3.0, 3.0, "점")], DURATION)

    utterances, omitted = align_separate([left, right])

    assert omitted == 1
    assert [u.transcript for u in utterances] == ["왼쪽"]


# ------------------------------------------------------------------ coverage / uncertainty


def test_coverage_folds_alignment_omissions_into_one_authoritative_figure() -> None:
    coverage = TimestampCoverage(
        total_words=10,
        positive_spans=7,
        zero_spans=3,
        zero_spans_unplaceable=1,
        omitted_words=1,
    )

    final = coverage.with_alignment_omissions(2)

    assert final.omitted_words == 3
    assert final.attributed_words == 7
    assert final.attributed_ratio == pytest.approx(0.7)
    assert final.zero_span_ratio == pytest.approx(0.3)
    assert final.uncertain is True


def test_coverage_of_a_clean_result_reports_certainty() -> None:
    coverage = TimestampCoverage(
        total_words=4, positive_spans=4, zero_spans=0, zero_spans_unplaceable=0, omitted_words=0
    ).with_alignment_omissions(0)

    assert coverage.uncertain is False
    assert coverage.attributed_ratio == 1.0
    assert coverage.as_dict()["policy"] == SANITATION_POLICY


def test_coverage_merges_across_channels() -> None:
    left = TimestampCoverage(3, 2, 1, 0, 0)
    right = TimestampCoverage(5, 4, 1, 1, 1)

    merged = left.merged(right)

    assert (merged.total_words, merged.zero_spans, merged.omitted_words) == (8, 2, 1)


def test_an_empty_result_reports_full_coverage_rather_than_dividing_by_zero() -> None:
    coverage = TimestampCoverage(0, 0, 0, 0, 0)

    assert coverage.attributed_ratio == 1.0
    assert coverage.zero_span_ratio == 0.0
    assert coverage.uncertain is False


def test_negative_alignment_omissions_are_refused() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        TimestampCoverage(1, 1, 0, 0, 0).with_alignment_omissions(-1)


# ------------------------------------------------- the relaxation does not reach elsewhere


def test_the_shared_validator_still_refuses_zero_duration_spans() -> None:
    """faster-whisper must keep failing closed; the relaxation is opt-in per provider."""

    with pytest.raises(ProviderError) as raised:
        validated_words([(2.0, 2.0, "점")], DURATION)
    assert raised.value.code == "invalid_provider_output"


def test_qwen_is_the_canonical_default_with_a_bounded_fallback() -> None:
    assert Settings.model_fields["asr_provider"].default == "qwen3"
    assert Settings.model_fields["asr_fallback_provider"].default == "faster-whisper"
    # Both remain selectable, which is what makes the rollback a config change.
    assert {"fake", "faster-whisper"} <= set(
        Settings.model_fields["asr_provider"].annotation.__args__
    )


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(
        storage_root=tmp_path / "storage",
        database_path=tmp_path / "data" / "db.sqlite3",
        api_capability_token=SecretStr("t" * 43),
        # These tests cover ASR wiring; keep diarization off the local runtime.
        **{"diarization_provider": "fake", **overrides},
    )


def test_the_default_configuration_selects_qwen_behind_the_fallback(tmp_path: Path) -> None:
    dependencies = build_dependencies(_settings(tmp_path), provider_factories=ProviderFactories())

    transcription = dependencies.runner._transcription
    assert type(transcription).__name__ == "FallbackTranscriptionProvider"
    # Before any transcription the provenance names what will be attempted.
    assert transcription.provenance.model == "Qwen3-ASR-1.7B"


def test_the_fake_provider_is_still_selectable_for_tests(tmp_path: Path) -> None:
    dependencies = build_dependencies(
        _settings(tmp_path, asr_provider="fake"), provider_factories=ProviderFactories()
    )

    assert type(dependencies.runner._transcription).__name__ == "FakeTranscriptionProvider"


def test_qwen_is_selectable_and_defaults_to_the_1_7b_checkpoint(tmp_path: Path) -> None:
    dependencies = build_dependencies(
        _settings(tmp_path, asr_provider="qwen3", asr_device="cpu", asr_fallback_provider="none"),
        provider_factories=ProviderFactories(),
    )

    provider = dependencies.runner._transcription
    assert provider.provenance.name == "qwen3-asr"
    assert provider.provenance.model == "Qwen3-ASR-1.7B"
    assert provider.provenance.remote is False


def test_the_low_memory_profile_selects_the_0_6b_checkpoint(tmp_path: Path) -> None:
    dependencies = build_dependencies(
        _settings(
            tmp_path,
            asr_provider="qwen3",
            asr_device="cpu",
            qwen_profile="low-memory",
            asr_fallback_provider="none",
        ),
        provider_factories=ProviderFactories(),
    )

    assert dependencies.runner._transcription.provenance.model == "Qwen3-ASR-0.6B"
