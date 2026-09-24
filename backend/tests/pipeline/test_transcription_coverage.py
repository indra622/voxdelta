"""Tests for how timestamp-sanitation coverage crosses from a provider into the contract.

The propagation through a real pipeline run is covered in ``test_runner.py``. What is
pinned here is the boundary itself: the mapping refuses to publish a figure it cannot
read, the model refuses to hold one that contradicts itself, and the public JSON always
carries the field so a client can tell "no sanitation applied" from "field missing".
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from voxdelta.domain.models import (
    AnalysisReport,
    CallSummary,
    ProviderProvenance,
    TranscriptionCoverage,
    Utterance,
)
from voxdelta.pipeline.runner import _coverage_warning, _transcription_coverage
from voxdelta.pipeline.stages import TranscribeArtifact
from voxdelta.providers.qwen_timestamps import SANITATION_POLICY, TimestampCoverage


class _Provider:
    def __init__(self, coverage: object) -> None:
        self.last_timestamp_coverage = coverage


def _coverage(*, omitted: int = 0, total: int = 10) -> TimestampCoverage:
    return TimestampCoverage(
        total_words=total,
        positive_spans=total - 2,
        zero_spans=2,
        zero_spans_unplaceable=omitted,
        omitted_words=omitted,
    )


# --------------------------------------------------------------------------------- mapping


def test_a_provider_that_reports_no_coverage_maps_to_none() -> None:
    """faster-whisper has no such attribute and must not be made to look like it does."""

    assert _transcription_coverage(object()) is None
    assert _transcription_coverage(_Provider(None)) is None


def test_a_reported_coverage_is_mapped_field_for_field() -> None:
    mapped = _transcription_coverage(_Provider(_coverage(omitted=3, total=20)))

    assert mapped is not None
    assert mapped.policy == SANITATION_POLICY
    assert mapped.omitted_words == 3
    assert mapped.attributed_words == 17
    assert mapped.attributed_ratio == pytest.approx(0.85)
    assert mapped.uncertain is True


def test_an_unreadable_coverage_object_fails_rather_than_publishing_a_guess() -> None:
    with pytest.raises(ValueError, match="unreadable figure"):
        _transcription_coverage(_Provider(object()))


def test_a_coverage_missing_required_fields_fails_closed() -> None:
    class _Partial:
        def as_dict(self) -> dict[str, Any]:
            return {"policy": "x", "omitted_words": 1}

    with pytest.raises(ValueError, match="unreadable figure"):
        _transcription_coverage(_Provider(_Partial()))


def test_the_warning_names_the_count_and_the_share_that_survived() -> None:
    warning = _coverage_warning(
        TranscriptionCoverage(
            policy=SANITATION_POLICY,
            attributed_words=88,
            attributed_ratio=0.88,
            omitted_words=12,
            uncertain=True,
        )
    )

    assert "12개 단어" in warning
    assert "88.0%" in warning


# ----------------------------------------------------------------------------------- model


def test_uncertain_must_agree_with_the_omission_count() -> None:
    with pytest.raises(ValidationError, match="uncertain must be set exactly"):
        TranscriptionCoverage(
            policy=SANITATION_POLICY,
            attributed_words=10,
            attributed_ratio=1.0,
            omitted_words=5,
            uncertain=False,
        )


def test_a_clean_coverage_must_not_claim_uncertainty() -> None:
    with pytest.raises(ValidationError, match="uncertain must be set exactly"):
        TranscriptionCoverage(
            policy=SANITATION_POLICY,
            attributed_words=10,
            attributed_ratio=1.0,
            omitted_words=0,
            uncertain=True,
        )


def test_a_ratio_outside_zero_to_one_is_refused() -> None:
    with pytest.raises(ValidationError):
        TranscriptionCoverage(
            policy=SANITATION_POLICY,
            attributed_words=10,
            attributed_ratio=1.5,
            omitted_words=0,
            uncertain=False,
        )


# -------------------------------------------------------------------------------- artifact


def _artifact_fields() -> dict[str, Any]:
    return {
        "cache_key": "c" * 64,
        "upstream_hashes": (),
        "provider": ProviderProvenance(name="p", model="m", remote=False),
        "utterances": [],
    }


def test_an_artifact_written_before_this_field_still_loads() -> None:
    """The field is additive, which is why the artifact version stays at 1."""

    persisted = TranscribeArtifact(**_artifact_fields()).model_dump_json()
    without = json.loads(persisted)
    without.pop("timestamp_coverage")

    restored = TranscribeArtifact.model_validate(without)

    assert restored.timestamp_coverage is None
    assert restored.schema_version == "1"


def test_the_artifact_round_trips_a_reported_coverage() -> None:
    mapped = _transcription_coverage(_Provider(_coverage(omitted=2)))
    artifact = TranscribeArtifact(**_artifact_fields(), timestamp_coverage=mapped)

    restored = TranscribeArtifact.model_validate_json(artifact.model_dump_json())

    assert restored.timestamp_coverage == mapped


# ------------------------------------------------------------------------------ public JSON


def _report(coverage: TranscriptionCoverage | None) -> AnalysisReport:
    return AnalysisReport(
        job_id="j1",
        summary=CallSummary(
            start_state="stable",
            end_state="stable",
            peak_customer_utterance_id="u1",
            overall_delta=0.0,
            valid_coverage=1.0,
            recovery_count=0,
            worsening_count=0,
        ),
        utterances=[
            Utterance(id="u1", start=0.0, end=1.0, speaker_id="S0", confidence=1.0, transcript="네")
        ],
        emotions=[],
        strategies=[],
        transitions=[],
        transcription_coverage=coverage,
    )


def test_the_public_report_always_carries_the_field_even_when_absent() -> None:
    """A client must be able to read "no sanitation" without guessing at a missing key."""

    payload = json.loads(_report(None).model_dump_json())

    assert "transcription_coverage" in payload
    assert payload["transcription_coverage"] is None


def test_the_public_report_exposes_the_four_figures_a_reader_needs() -> None:
    mapped = _transcription_coverage(_Provider(_coverage(omitted=1, total=10)))
    payload = json.loads(_report(mapped).model_dump_json())

    assert payload["transcription_coverage"] == {
        "policy": SANITATION_POLICY,
        "attributed_words": 9,
        "attributed_ratio": 0.9,
        "omitted_words": 1,
        "uncertain": True,
    }
