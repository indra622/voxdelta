"""Tests for the speaker-attributed ASR evaluation.

Three contracts matter here and none of them involve a model. The inputs are pinned by
digest, so a changed record or a changed WAV stops the run instead of silently scoring
something else. The arithmetic is internally consistent, so the attribution cost is a real
subtraction over one denominator rather than two numbers placed next to each other. And
the artifact is transcript-free and states its own local-only boundary, because that
boundary is the whole reason this evaluation may run at all.

Nothing here loads a checkpoint or opens a socket.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from voxdelta.evaluation.kcsc_asr_benchmark import ModelIdentity, characters, score_sequences
from voxdelta.evaluation.kcsc_attributed_asr import (
    DELTA_CONFOUNDS,
    SPEAKER_MAPPING_RULE,
    KcscAttributedError,
    PinnedInput,
    ReferenceRows,
    attribute,
    build_report,
    load_timeline,
    map_speakers,
    reference_rows,
    score,
    write_report,
)
from voxdelta.evaluation.kcsc_diarization_benchmark import file_sha256
from voxdelta.providers.asr_alignment import AlignedWord
from voxdelta.providers.qwen_timestamps import TimestampCoverage

REF_A = "안녕하세요 반갑습니다"
REF_B = "네 그렇습니다"
DURATION = 20.0


def _record(tmp_path: Path, timeline: list[dict[str, Any]] | None = None) -> Path:
    path = tmp_path / "EVALUATION-retry-2.json"
    default = [
        {"start": 0.0, "end": 10.0, "speaker_id": "SPEAKER_00"},
        {"start": 10.0, "end": 20.0, "speaker_id": "SPEAKER_01"},
    ]
    path.write_text(
        json.dumps(
            {
                "input": {"duration_seconds": DURATION},
                "hypothesis_timeline": default if timeline is None else timeline,
            }
        ),
        encoding="utf-8",
    )
    return path


def _reference(tmp_path: Path, *, crossed: bool = False) -> Path:
    path = tmp_path / "reference.json"
    first, second = ("G5999", "G6000") if not crossed else ("G6000", "G5999")
    path.write_text(
        json.dumps(
            {
                "conversation_id": "A6000_S0005_0",
                "turns": [
                    {"start": 1.0, "end": 9.0, "speaker": first, "transcript": REF_A},
                    {"start": 11.0, "end": 19.0, "speaker": second, "transcript": REF_B},
                ],
                "unattributed_events": [
                    {"start": 9.5, "end": 9.8, "speaker": "0", "transcript": "[LAUGHTER]"}
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------- pinned inputs


def test_a_pinned_input_that_matches_returns_its_digest(tmp_path: Path) -> None:
    path = _record(tmp_path)

    assert PinnedInput(path, file_sha256(path)).verify() == file_sha256(path)


def test_a_changed_pinned_input_fails_closed(tmp_path: Path) -> None:
    """The record and the WAV are the evaluation's identity; a drifted one is a different run."""

    path = _record(tmp_path)

    with pytest.raises(KcscAttributedError, match="does not match the pinned"):
        PinnedInput(path, "0" * 64).verify()


def test_a_missing_pinned_input_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(KcscAttributedError, match="missing pinned input"):
        PinnedInput(tmp_path / "absent.json", "0" * 64).verify()


# ------------------------------------------------------------------- the preserved timeline


def test_the_timeline_is_read_from_the_record_not_recomputed(tmp_path: Path) -> None:
    segments, record = load_timeline(_record(tmp_path))

    assert [segment.speaker_id for segment in segments] == ["SPEAKER_00", "SPEAKER_01"]
    assert record["input"]["duration_seconds"] == DURATION


def test_a_record_without_a_timeline_fails_rather_than_producing_one(tmp_path: Path) -> None:
    """There is no fallback that would recompute diarization; that would need an upload."""

    path = tmp_path / "EVALUATION-retry-2.json"
    path.write_text(json.dumps({"input": {"duration_seconds": DURATION}}), encoding="utf-8")

    with pytest.raises(KcscAttributedError, match="no preserved hypothesis timeline"):
        load_timeline(path)


def test_a_timeline_with_one_speaker_fails_closed(tmp_path: Path) -> None:
    single = [{"start": 0.0, "end": 10.0, "speaker_id": "SPEAKER_00"}]

    with pytest.raises(KcscAttributedError, match="expected two diarized speakers"):
        load_timeline(_record(tmp_path, single))


# -------------------------------------------------------------------------- speaker mapping


def test_the_mapping_follows_temporal_overlap(tmp_path: Path) -> None:
    segments, _loaded = load_timeline(_record(tmp_path))

    mapping, evidence = map_speakers(segments, _reference(tmp_path))

    assert mapping == {"SPEAKER_00": "G5999", "SPEAKER_01": "G6000"}
    assert evidence["straight_overlap_seconds"] > evidence["crossed_overlap_seconds"]


def test_the_mapping_swaps_when_the_reference_speakers_are_reversed(tmp_path: Path) -> None:
    segments, _loaded = load_timeline(_record(tmp_path))

    mapping, evidence = map_speakers(segments, _reference(tmp_path, crossed=True))

    assert mapping == {"SPEAKER_00": "G6000", "SPEAKER_01": "G5999"}
    assert evidence["margin_seconds"] > 0


def test_the_mapping_rule_is_time_only(tmp_path: Path) -> None:
    """A mapping chosen by error rate would be choosing the answer it then reports."""

    assert "timings only" in SPEAKER_MAPPING_RULE
    assert "overlap" in SPEAKER_MAPPING_RULE


# -------------------------------------------------------------------------- attribution


def test_words_are_grouped_by_the_speaker_the_timeline_places_them_under(
    tmp_path: Path,
) -> None:
    segments, _loaded = load_timeline(_record(tmp_path))
    words = [
        AlignedWord(1.0, 2.0, "가"),
        AlignedWord(3.0, 4.0, "나"),
        AlignedWord(12.0, 13.0, "다"),
    ]

    by_speaker, omitted, utterances = attribute(words, segments, DURATION)

    assert by_speaker["SPEAKER_00"] == "가 나"
    assert by_speaker["SPEAKER_01"] == "다"
    assert omitted == 0
    assert utterances == 2


def test_words_the_timeline_cannot_place_are_counted_as_omitted(tmp_path: Path) -> None:
    """Those words become deletions in the attributed score; the count makes that visible."""

    timeline = [
        {"start": 0.0, "end": 5.0, "speaker_id": "SPEAKER_00"},
        {"start": 15.0, "end": 20.0, "speaker_id": "SPEAKER_01"},
    ]
    segments, _loaded = load_timeline(_record(tmp_path, timeline))
    words = [AlignedWord(1.0, 2.0, "가"), AlignedWord(8.0, 9.0, "구멍")]

    _by_speaker, omitted, _utterances = attribute(words, segments, DURATION)

    assert omitted == 1


# -------------------------------------------------------------------------- score contract


def _scores(hypothesis: dict[str, str] | None = None, *, reference: Any = None) -> Any:
    return score(
        stream_text=f"{REF_A} {REF_B}",
        hypothesis_by_speaker=hypothesis or {"SPEAKER_00": REF_A, "SPEAKER_01": REF_B},
        reference=reference
        or ReferenceRows(
            by_speaker={"G5999": [REF_A], "G6000": [REF_B]},
            events=["[LAUGHTER]"],
            turn_count=2,
            chronological=[REF_A, "[LAUGHTER]", REF_B],
            chronological_attributed=[REF_A, REF_B],
        ),
        mapping={"SPEAKER_00": "G5999", "SPEAKER_01": "G6000"},
    )


def test_a_perfect_attribution_scores_zero_everywhere() -> None:
    scores = _scores()

    assert scores.speaker_attributed_pooled.errors == 0
    assert scores.mixed_stream.errors == 0
    assert scores.e2e_vs_mixed_stream_delta == 0.0


def test_the_pooled_score_is_the_sum_of_the_per_speaker_operations() -> None:
    scores = _scores({"SPEAKER_00": REF_B, "SPEAKER_01": REF_A})

    pooled = scores.speaker_attributed_pooled
    assert pooled.reference_length == sum(
        counts.reference_length for counts in scores.per_speaker.values()
    )
    assert pooled.errors == sum(counts.errors for counts in scores.per_speaker.values())


def test_the_delta_is_a_subtraction_over_one_denominator() -> None:
    """Two rates over different reference lengths could not honestly be subtracted."""

    scores = _scores({"SPEAKER_00": REF_B, "SPEAKER_01": REF_A})

    assert scores.mixed_stream.reference_length == (
        scores.speaker_attributed_pooled.reference_length
    )
    assert scores.e2e_vs_mixed_stream_delta == pytest.approx(
        scores.speaker_attributed_pooled.error_rate - scores.mixed_stream.error_rate
    )


# ------------------------------------------------------------------------ artifact contract


def _report(tmp_path: Path) -> dict[str, Any]:
    identity = ModelIdentity(
        repo_id="Qwen/Qwen3-ASR-1.7B",
        revision="a" * 40,
        files=(),
        tree_sha256="b" * 64,
        total_bytes=1,
    )
    aligner = ModelIdentity(
        repo_id="Qwen/Qwen3-ForcedAligner-0.6B",
        revision="c" * 40,
        files=(),
        tree_sha256="d" * 64,
        total_bytes=1,
    )
    return build_report(
        scores=_scores({"SPEAKER_00": REF_B, "SPEAKER_01": REF_A}),
        coverage=TimestampCoverage(10, 9, 1, 0, 0).with_alignment_omissions(2),
        mapping={"SPEAKER_00": "G5999", "SPEAKER_01": "G6000"},
        mapping_evidence={"straight_overlap_seconds": 489.3, "margin_seconds": 418.2},
        identity=identity,
        aligner=aligner,
        inputs={"retry2_record_sha256": "e" * 64, "derived_audio_sha256": "f" * 64},
        segment_count=181,
        reference_turn_count=290,
        reference_event_count=3,
        utterance_count=80,
        omitted_by_alignment=2,
        duration_seconds=583.91,
        elapsed_seconds=300.0,
        egress_attempts=0,
        device="mps",
        dtype="float16",
    )


def test_the_artifact_declares_the_local_only_boundary(tmp_path: Path) -> None:
    report = _report(tmp_path)

    boundary = report["local_only"]
    assert boundary["external_calls"] == 0
    assert boundary["audio_transmitted"] is False
    assert boundary["diarization_recomputed"] is False
    assert boundary["network_egress_attempts"] == 0
    assert boundary["hub_offline"] is True


def test_the_artifact_carries_the_input_digests(tmp_path: Path) -> None:
    report = _report(tmp_path)

    assert report["inputs"]["retry2_record_sha256"] == "e" * 64
    assert report["inputs"]["derived_audio_sha256"] == "f" * 64
    assert report["model"]["tree_sha256"] == "b" * 64


def test_the_artifact_is_transcript_free(tmp_path: Path) -> None:
    report = _report(tmp_path)
    path = tmp_path / "report.json"
    digest = write_report(report, path)

    raw = path.read_text(encoding="utf-8")
    for fragment in (REF_A, REF_B, "안녕하세요", "그렇습니다"):
        assert fragment not in raw

    def keys(node: object) -> set[str]:
        if isinstance(node, dict):
            return set(node) | {key for value in node.values() for key in keys(value)}
        if isinstance(node, list):
            return {key for value in node for key in keys(value)}
        return set()

    present = keys(json.loads(raw))
    assert "transcript" not in present
    assert "text" not in present
    assert report["transcript_free"] is True
    assert digest == file_sha256(path)


def test_the_artifact_states_the_measurement_scope_and_its_limits(tmp_path: Path) -> None:
    report = _report(tmp_path)

    assert report["scope"]["conversation_count"] == 1
    assert report["scope"]["diarized_segment_count"] == 181
    assert report["scope"]["words_omitted_by_alignment"] == 2
    limitations = " ".join(report["limitations"])
    # The headline number must not be sold as a decomposition or as diarization error.
    assert "NOT a decomposition" in limitations
    assert "One conversation" in limitations
    assert report["metrics"]["delta_confounds"]
    assert "NOT diarization error" in report["metrics"]["delta_note"]


def test_the_report_is_deterministic(tmp_path: Path) -> None:
    first, second = tmp_path / "a.json", tmp_path / "b.json"

    assert write_report(_report(tmp_path), first) == write_report(_report(tmp_path), second)


# ------------------------------------------------------- root cause of the retracted run


def test_a_perfect_chronological_transcript_is_not_penalised_by_the_baseline() -> None:
    """Root cause of the retracted artifact: the baseline was ordered by speaker, not time.

    A speaker-grouped reference scores a perfect time-ordered decode as roughly half
    wrong. That is what produced the invalid 0.6636 baseline and its negative delta.
    """

    alternating = ReferenceRows(
        by_speaker={"G5999": ["가가가", "다다다"], "G6000": ["나나나", "라라라"]},
        events=[],
        turn_count=4,
        chronological=["가가가", "나나나", "다다다", "라라라"],
        chronological_attributed=["가가가", "나나나", "다다다", "라라라"],
    )
    perfect_stream = "가가가 나나나 다다다 라라라"

    scores = score(
        stream_text=perfect_stream,
        hypothesis_by_speaker={"SPEAKER_00": "가가가 다다다", "SPEAKER_01": "나나나 라라라"},
        reference=alternating,
        mapping={"SPEAKER_00": "G5999", "SPEAKER_01": "G6000"},
    )

    assert scores.mixed_stream.errors == 0
    # The speaker-grouped ordering the retracted run used would have scored this ~0.5.
    grouped = " ".join(["가가가", "다다다", "나나나", "라라라"])
    assert score_sequences(characters(grouped), characters(perfect_stream)).error_rate > 0.4


def test_reference_rows_orders_the_stream_by_time_not_by_speaker(tmp_path: Path) -> None:
    """The defect lived in this seam: only by_speaker was returned, so score() regrouped."""

    path = tmp_path / "chronology.json"
    path.write_text(
        json.dumps(
            {
                "turns": [
                    {"start": 1.0, "end": 2.0, "speaker": "G5999", "transcript": "하나"},
                    {"start": 3.0, "end": 4.0, "speaker": "G6000", "transcript": "둘"},
                    {"start": 5.0, "end": 6.0, "speaker": "G5999", "transcript": "셋"},
                ],
                "unattributed_events": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    rows = reference_rows(path)

    assert rows.chronological_attributed == ["하나", "둘", "셋"]
    assert rows.by_speaker["G5999"] == ["하나", "셋"]


def test_the_delta_is_not_labelled_an_attribution_cost() -> None:
    """The name must not promise a decomposition the metric cannot deliver."""

    assert not hasattr(_scores(), "attribution_cost")
    assert hasattr(_scores(), "e2e_vs_mixed_stream_delta")
    assert len(DELTA_CONFOUNDS) >= 3
