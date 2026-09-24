"""Tests for the G5999 deletion diagnostic's bin definitions and its artifact contract.

Every bin here is a claim about where error concentrates, so the edges are what need
pinning: a turn exactly on a bin boundary, two turns that merely touch, a segment edge
exactly at the tolerance. Those are the cases where a diagnostic quietly starts telling a
different story than its own documentation.

Nothing here loads a model or opens a socket.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from voxdelta.domain.models import SpeakerSegment
from voxdelta.evaluation.kcsc_asr_benchmark import ErrorCounts
from voxdelta.evaluation.kcsc_deletion_diagnostic import (
    BOUNDARY_TOLERANCE_SECONDS,
    DURATION_BINS,
    PRIMARY_PRECEDENCE,
    Turn,
    TurnDiagnosis,
    associate,
    boundary_bin,
    coverage_bin,
    diagnose,
    duration_bin,
    duration_bin_edges,
    overlap_bin,
    preceding_silence,
    primary_category,
    tabulate,
)
from voxdelta.providers.asr_alignment import AlignedWord


def _segment(start: float, end: float, speaker: str = "SPEAKER_00") -> SpeakerSegment:
    return SpeakerSegment(start=start, end=end, speaker_id=speaker, confidence=1.0)


# ------------------------------------------------------------------------ duration bins


@pytest.mark.parametrize(
    ("duration", "expected"),
    [
        (0.0, "lt_1s"),
        (0.999, "lt_1s"),
        (1.0, "1_3s"),  # half-open: the edge belongs to the upper bin
        (2.999, "1_3s"),
        (3.0, "3_8s"),
        (7.999, "3_8s"),
        (8.0, "gte_8s"),
        (120.0, "gte_8s"),
    ],
)
def test_duration_bins_are_half_open_with_no_gaps(duration: float, expected: str) -> None:
    assert duration_bin(duration) == expected


def test_every_duration_bin_edge_is_contiguous() -> None:
    """A gap between bins would silently drop turns out of the diagnostic."""

    edges = [(low, high) for _name, low, high in DURATION_BINS]
    for (_low, high), (next_low, _next_high) in zip(edges, edges[1:], strict=False):
        assert high == next_low


# ------------------------------------------------------------------------- overlap bins


def test_touching_turns_are_not_overlap() -> None:
    """The documented policy: a shared instant is not shared speech."""

    turn = Turn(1.0, 2.0, "가")
    assert overlap_bin(turn, [Turn(2.0, 3.0, "나")]) == "none"
    assert overlap_bin(turn, [Turn(0.0, 1.0, "나")]) == "none"


def test_any_positive_intersection_is_overlap() -> None:
    turn = Turn(1.0, 2.0, "가")

    assert overlap_bin(turn, [Turn(1.999, 3.0, "나")]) == "any"
    assert overlap_bin(turn, [Turn(0.0, 1.001, "나")]) == "any"


def test_a_turn_with_no_other_speaker_is_never_overlapped() -> None:
    assert overlap_bin(Turn(1.0, 2.0, "가"), []) == "none"


# ------------------------------------------------------------------------ boundary bins


def test_a_turn_edge_exactly_at_the_tolerance_counts_as_near() -> None:
    """Inclusive at the tolerance, as the policy string states."""

    turn = Turn(10.0, 12.0, "가")
    edge = 10.0 + BOUNDARY_TOLERANCE_SECONDS

    label, distance = boundary_bin(turn, [_segment(edge, edge + 5.0)])

    assert label == "near_boundary"
    assert distance == pytest.approx(BOUNDARY_TOLERANCE_SECONDS)


def test_a_turn_well_inside_a_segment_is_interior() -> None:
    label, distance = boundary_bin(Turn(10.0, 12.0, "가"), [_segment(0.0, 30.0)])

    assert label == "interior"
    assert distance > BOUNDARY_TOLERANCE_SECONDS


def test_boundary_distance_uses_the_nearest_edge_to_either_turn_edge() -> None:
    turn = Turn(10.0, 20.0, "가")

    _label, distance = boundary_bin(turn, [_segment(0.0, 5.0), _segment(19.9, 30.0)])

    assert distance == pytest.approx(0.1)


# ------------------------------------------------------------------------ coverage bins


def test_a_fully_covered_turn_is_full() -> None:
    label, mapped, other = coverage_bin(Turn(10.0, 12.0, "가"), [_segment(9.0, 13.0)], [])

    assert label == "full"
    assert mapped == pytest.approx(1.0)
    assert other == 0.0


def test_a_turn_the_other_speaker_covers_is_mixed_regardless_of_mapped_coverage() -> None:
    """Precedence: the diarizer giving speech to the wrong speaker is the severe case."""

    label, _mapped, other = coverage_bin(
        Turn(10.0, 12.0, "가"), [_segment(10.0, 11.0)], [_segment(11.0, 12.0, "SPEAKER_01")]
    )

    assert label == "mixed_other_speaker"
    assert other == pytest.approx(0.5)


def test_a_turn_in_a_diarization_gap_is_uncovered() -> None:
    label, mapped, _other = coverage_bin(Turn(10.0, 12.0, "가"), [_segment(0.0, 5.0)], [])

    assert label == "uncovered_gap"
    assert mapped == 0.0


def test_a_partly_clipped_turn_is_same_speaker_partial() -> None:
    label, mapped, _other = coverage_bin(Turn(10.0, 12.0, "가"), [_segment(10.0, 11.0)], [])

    assert label == "same_speaker_partial"
    assert mapped == pytest.approx(0.5)


def test_a_trace_of_other_speaker_bleed_does_not_reclassify_a_covered_turn() -> None:
    label, _mapped, _other = coverage_bin(
        Turn(0.0, 100.0, "가"), [_segment(0.0, 100.0)], [_segment(0.0, 0.5, "SPEAKER_01")]
    )

    assert label == "full"


# --------------------------------------------------------------------- primary precedence


@pytest.mark.parametrize(
    ("coverage", "boundary", "expected"),
    [
        ("mixed_other_speaker", "interior", "mixed_other_speaker"),
        ("mixed_other_speaker", "near_boundary", "mixed_other_speaker"),
        ("uncovered_gap", "near_boundary", "uncovered_gap"),
        ("same_speaker_partial", "near_boundary", "same_speaker_partial"),
        ("full", "near_boundary", "near_boundary"),
        ("full", "interior", "covered_interior"),
    ],
)
def test_primary_category_follows_the_documented_precedence(
    coverage: str, boundary: str, expected: str
) -> None:
    assert primary_category(coverage, boundary) == expected


def test_every_primary_label_is_reachable() -> None:
    """A precedence entry nothing can produce would be documentation, not a category."""

    produced = {
        primary_category(coverage, boundary)
        for coverage in ("mixed_other_speaker", "uncovered_gap", "same_speaker_partial", "full")
        for boundary in ("near_boundary", "interior")
    }
    assert produced == set(PRIMARY_PRECEDENCE)


# ------------------------------------------------------------------------- association


def test_a_word_is_assigned_to_the_turn_containing_its_midpoint() -> None:
    turns = [Turn(0.0, 2.0, "가"), Turn(3.0, 5.0, "나")]
    words = [AlignedWord(0.5, 1.5, "하나"), AlignedWord(3.5, 4.5, "둘")]

    assigned, unassigned, chars = associate(words, turns)

    assert assigned == {0: ["하나"], 1: ["둘"]}
    assert unassigned == 0
    assert chars == 0


def test_a_word_whose_midpoint_falls_in_no_turn_is_counted_unassigned() -> None:
    """That bucket is the size of the method's own blind spot; it must not be hidden."""

    turns = [Turn(0.0, 2.0, "가")]

    assigned, unassigned, chars = associate([AlignedWord(9.0, 10.0, "바깥")], turns)

    assert assigned == {}
    assert unassigned == 1
    assert chars == 2


def test_a_word_straddling_a_boundary_follows_its_midpoint_not_its_span() -> None:
    turns = [Turn(0.0, 2.0, "가"), Turn(2.0, 4.0, "나")]

    assigned, _unassigned, _chars = associate([AlignedWord(1.5, 2.4, "걸침")], turns)

    assert assigned == {0: ["걸침"]}


# --------------------------------------------------------------------- preceding silence


def test_preceding_silence_measures_the_gap_to_the_last_reference_speech() -> None:
    turn = Turn(10.0, 11.0, "가")
    others = [Turn(0.0, 2.0, "나"), Turn(5.0, 7.5, "다"), turn]

    assert preceding_silence(turn, others) == pytest.approx(2.5)


def test_the_first_turn_measures_silence_from_the_start_of_the_recording() -> None:
    turn = Turn(3.0, 4.0, "가")

    assert preceding_silence(turn, [turn]) == pytest.approx(3.0)


# --------------------------------------------------------------------------- tabulation


def _diagnoses() -> list:
    return diagnose(
        turns=[Turn(0.0, 0.5, "가나"), Turn(10.0, 14.0, "다라마")],
        other_turns=[],
        mapped=[_segment(0.0, 0.5)],
        other_segments=[],
        words=[AlignedWord(0.1, 0.4, "가나")],
    )[0]


def test_empty_duration_bins_are_reported_rather_than_omitted() -> None:
    """A missing bin reads as 'no data collected'; an empty one reads as 'no turns here'."""

    table = tabulate(_diagnoses(), "duration_bin")

    assert set(table) == {name for name, _low, _high in DURATION_BINS}
    assert table["gte_8s"]["turn_count"] == 0


def test_each_turn_appears_exactly_once_in_the_primary_table() -> None:
    diagnoses = _diagnoses()

    table = tabulate(diagnoses, "primary")

    assert sum(row["turn_count"] for row in table.values()) == len(diagnoses)


def test_a_turn_with_no_associated_words_scores_as_all_deletions() -> None:
    diagnoses = _diagnoses()

    uncovered = next(d for d in diagnoses if d.coverage_bin == "uncovered_gap")
    assert uncovered.counts.deletions == uncovered.counts.reference_length
    assert uncovered.counts.insertions == 0


# ------------------------------------------------------------------- artifact contract


ARTIFACT = Path("data/benchmarks/kcsc-g5999-deletion-diagnostic.json")


def _artifact() -> dict:
    root = Path(__file__).resolve().parents[3]
    path = root / ARTIFACT
    if not path.is_file():
        pytest.skip("diagnostic artifact not present")
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_artifact_is_transcript_free() -> None:
    report = _artifact()

    def keys(node: object) -> set[str]:
        if isinstance(node, dict):
            return set(node) | {key for value in node.values() for key in keys(value)}
        if isinstance(node, list):
            return {key for value in node for key in keys(value)}
        return set()

    present = keys(report)
    assert "transcript" not in present
    assert "text" not in present
    assert "hypothesis" not in present
    assert report["transcript_free"] is True


def test_the_artifact_pins_the_immutable_input_digests() -> None:
    report = _artifact()

    assert (
        report["inputs"]["retry2_record_sha256"]
        == "4829014c3756194ce99e70102a35c83b5983db5c6dceb6bca7b481964ba7de34"
    )
    assert (
        report["inputs"]["derived_audio_sha256"]
        == "01565974fdf7021e0d0afcbae8c040839777434f40416f1e265aeee3926cce27"
    )


def test_the_artifact_declares_the_offline_no_egress_contract() -> None:
    boundary = _artifact()["local_only"]

    assert boundary["external_calls"] == 0
    assert boundary["network_egress_attempts"] == 0
    assert boundary["audio_transmitted"] is False
    assert boundary["diarization_recomputed"] is False
    assert boundary["hub_offline"] is True


def test_the_artifact_is_additive_and_names_its_base_by_digest() -> None:
    report = _artifact()

    assert report["additive_to"]["sha256"]
    assert "unmodified" in report["additive_to"]["note"]


def test_the_artifact_labels_the_marginal_tables_non_additive() -> None:
    """Overlapping tables presented as a decomposition would be the headline mistake."""

    method = _artifact()["method"]

    assert "non_additive_note" in method
    assert "mutually exclusive" in method["non_additive_note"]
    assert "do NOT sum" in method["non_additive_note"]


def test_the_published_bin_edges_are_json_safe() -> None:
    """inf is a valid float and an invalid JSON number; the artifact writer rejects it."""

    edges = duration_bin_edges()

    assert edges[-1] == ["gte_8s", 8.0, None]
    # Exactly the call the artifact writer makes.
    json.dumps({"duration_bin_edges": edges}, allow_nan=False)


def test_the_python_bin_definition_keeps_its_unbounded_edge() -> None:
    """The JSON form must not leak back into the comparison logic."""

    assert DURATION_BINS[-1][2] == float("inf")
    assert duration_bin(10_000.0) == "gte_8s"


def test_a_bin_with_zero_reference_characters_serialises_finitely() -> None:
    """Latent second failure on the same write path, found while tracing the first.

    A bin whose turns all normalise to empty text has no denominator. ErrorCounts refuses
    to invent a rate for that, so pooling one used to raise while building the artifact.
    The bin must still serialise: null rate, real counts, no inf.
    """

    empty = TurnDiagnosis(
        duration_bin="lt_1s",
        overlap_bin="none",
        boundary_bin="interior",
        coverage_bin="uncovered_gap",
        primary="uncovered_gap",
        preceding_silence_bin="lt_0.5s",
        counts=ErrorCounts(0, 0, 0, reference_length=0, hypothesis_length=0),
    )

    table = tabulate([empty], "primary")

    assert table["uncovered_gap"]["error_rate"] is None
    assert table["uncovered_gap"]["reference_length"] == 0
    assert table["uncovered_gap"]["turn_count"] == 1
    json.dumps(table, allow_nan=False)
