"""Tests for the Gemini silver-to-gold workflow.

The workflow's whole point is that a machine's guess cannot become ground truth without a
person, and that audio does not leave this machine without a specific yes. So the tests
are organised around the two gates and the one irreversible act: consent before anything
is sent, review before anything counts as gold, and deletion of the uploaded file whatever
else happened.

No test reaches the network. The HTTP client is a stub throughout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from voxdelta.annotation.gemini_silver import (
    EMOTION_LABELS,
    GEMINI_MODEL,
    PROMPT,
    RESPONSE_SCHEMA,
    GeminiAnnotator,
    GeminiBudgetExceeded,
    GeminiConsentError,
    SilverTurn,
    _extract_json,
    provenance,
    validate,
    validate_silver,
)
from voxdelta.annotation.store import (
    REVIEW_REQUIRED,
    REVIEWED_GOLD,
    AnnotationError,
    assert_gold_eligible,
    is_gold_eligible,
    promote,
    read_silver,
    summarise,
    verify_gold,
    write_silver,
)
from voxdelta.config import Settings
from voxdelta.providers.base import ProviderError

TRANSCRIPT = "환불 문의드립니다"


def _turn(**overrides: Any) -> dict[str, Any]:
    row = {
        "start": 0.0,
        "end": 2.0,
        "speaker": "SPEAKER_00",
        "transcript": TRANSCRIPT,
        "emotion": "neutral",
        "emotion_rationale": "flat prosody",
        "confidence": 0.6,
    }
    row.update(overrides)
    return row


def _payload(turns: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "speakers": ["SPEAKER_00", "SPEAKER_01"],
        "turns": turns if turns is not None else [_turn()],
        "notes": "check the overlap at 12s",
    }


class StubResponse:
    def __init__(self, status: int, body: Any = None, headers: dict[str, str] | None = None):
        self.status_code = status
        self._body = body if body is not None else {}
        self.headers = headers or {}
        self.text = json.dumps(self._body)

    def json(self) -> Any:
        return self._body


class StubClient:
    """Records every request and replies with a scripted Gemini conversation."""

    def __init__(
        self,
        *,
        interaction: Any = None,
        delete_status: int = 200,
        delete_raises: bool = False,
        interaction_status: int = 200,
    ) -> None:
        self.requests: list[tuple[str, str]] = []
        self._interaction = interaction if interaction is not None else _payload()
        self._delete_status = delete_status
        self._delete_raises = delete_raises
        self._interaction_status = interaction_status

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Any = None,
        json: Any = None,
        content: bytes | None = None,
    ) -> StubResponse:
        self.requests.append((method, url))
        if method == "DELETE":
            if self._delete_raises:
                raise RuntimeError("network down")
            return StubResponse(self._delete_status)
        if url.endswith("/upload/v1beta/files"):
            return StubResponse(200, {}, {"x-goog-upload-url": "https://upload.example/session"})
        if url == "https://upload.example/session":
            return StubResponse(200, {"file": {"uri": "files/abc", "name": "files/abc"}})
        if url.endswith("/v1beta/interactions"):
            import json as _json

            return StubResponse(
                self._interaction_status,
                {"output_text": _json.dumps(self._interaction)},
            )
        raise AssertionError(f"unexpected request {method} {url}")

    def kinds(self) -> list[str]:
        kinds = []
        for method, url in self.requests:
            if method == "DELETE":
                kinds.append("delete")
            elif "interactions" in url:
                kinds.append("interaction")
            else:
                kinds.append("upload")
        return kinds


@pytest.fixture
def audio(tmp_path: Path) -> Path:
    path = tmp_path / "A6000_S0005_0.wav"
    path.write_bytes(b"RIFF" + b"\0" * 64)
    return path


def _annotator(client: StubClient) -> GeminiAnnotator:
    return GeminiAnnotator(api_key="test-key", client=client, consent_granted=True)


# ------------------------------------------------------------------------------- consent


def test_without_gemini_consent_the_annotator_refuses_to_exist() -> None:
    """Constructed before the key is read: no consent means nothing to authenticate."""

    with pytest.raises(GeminiConsentError):
        GeminiAnnotator(api_key="test-key", client=StubClient(), consent_granted=False)


def test_no_request_is_made_when_consent_is_absent(audio: Path) -> None:
    client = StubClient()

    with pytest.raises(GeminiConsentError):
        GeminiAnnotator(api_key="k", client=client, consent_granted=False)

    assert client.requests == []


def test_enabling_gemini_without_its_own_consent_is_a_configuration_error(
    tmp_path: Path,
) -> None:
    """Another provider's consent must not be usable as Gemini's."""

    with pytest.raises(ValueError, match="does not extend to Gemini"):
        Settings(
            data_root=tmp_path,
            database_path=tmp_path / "db.sqlite3",
            diarization_provider="pyannoteai-precision",
            gemini_annotation_enabled=True,
        )


def test_gemini_is_off_by_default(tmp_path: Path) -> None:
    settings = Settings(data_root=tmp_path, database_path=tmp_path / "db.sqlite3")

    assert settings.gemini_annotation_enabled is False
    assert settings.gemini_consent_granted is False


def test_the_disclosure_declares_remote_audio_transmission() -> None:
    declared = provenance()

    assert declared.remote is True
    assert declared.transmits == ("audio",)
    assert declared.model == GEMINI_MODEL
    assert declared.retention_policy_url


# ------------------------------------------------------------- structured response validation


def test_a_well_formed_response_validates() -> None:
    annotation = validate(_payload())

    assert annotation.turn_count == 1
    assert annotation.turns[0].transcript == TRANSCRIPT
    assert annotation.speakers == ("SPEAKER_00", "SPEAKER_01")


def test_extract_json_accepts_the_interactions_steps_envelope() -> None:
    """The REST API returns structured output inside model_output steps."""

    payload = _payload()
    body = {
        "status": "completed",
        "steps": [
            {
                "type": "model_output",
                "content": [{"type": "text", "text": json.dumps(payload)}],
            }
        ],
    }

    assert _extract_json(body) == payload


@pytest.mark.parametrize("emotion", sorted(EMOTION_LABELS))
def test_every_declared_emotion_label_is_accepted(emotion: str) -> None:
    annotation = validate(_payload([_turn(emotion=emotion)]))

    assert annotation.turns[0].emotion == emotion


def test_an_unknown_emotion_label_fails_closed() -> None:
    with pytest.raises(ProviderError):
        validate(_payload([_turn(emotion="ecstatic")]))


def test_silver_salvage_keeps_valid_turns_and_counts_invalid_rules() -> None:
    """A reviewer can inspect the usable draft, but omissions must be explicit."""

    annotation = validate_silver(
        _payload([_turn(), _turn(start=4.0, end=4.0, transcript="비공개 전사")])
    )

    assert annotation.turn_count == 1
    assert annotation.dropped_turn_count == 1
    assert annotation.dropped_turns_by_rule == {"turn.invalid_interval": 1}


def test_silver_salvage_still_refuses_when_no_turn_is_usable() -> None:
    with pytest.raises(ProviderError):
        validate_silver(_payload([_turn(start=4.0, end=4.0)]))


@pytest.mark.parametrize(
    "override",
    [
        {"end": 0.0},  # backwards interval
        {"start": -1.0},
        {"confidence": 1.5},
        {"transcript": None},
    ],
)
def test_a_malformed_turn_fails_closed(override: dict[str, Any]) -> None:
    with pytest.raises(ProviderError):
        validate(_payload([_turn(**override)]))


def test_a_turn_past_the_audio_duration_fails_closed() -> None:
    with pytest.raises(ProviderError):
        validate(_payload([_turn(end=900.0)]), duration_seconds=100.0)


def test_an_empty_or_shapeless_response_fails_closed() -> None:
    for payload in ({}, {"turns": []}, {"turns": [_turn()]}, "not json"):
        with pytest.raises(ProviderError):
            validate(payload)


def test_the_schema_requires_a_rationale_and_offers_uncertain() -> None:
    """A reviewer needs a reason to check, and the model needs a way to decline."""

    turn_schema = RESPONSE_SCHEMA["properties"]["turns"]["items"]

    assert "emotion_rationale" in turn_schema["required"]
    assert "uncertain" in turn_schema["properties"]["emotion"]["enum"]
    assert "uncertain" in PROMPT


# ------------------------------------------------------------------------- call ceilings


def test_one_upload_one_interaction_one_delete(audio: Path) -> None:
    client = StubClient()

    outcome = _annotator(client).annotate(audio)

    assert outcome.call_counts == {"uploads": 1, "interactions": 1, "deletes": 1}
    assert client.kinds().count("interaction") == 1
    assert client.kinds().count("delete") == 1


def test_model_turn_defect_becomes_review_required_salvage_not_a_full_failure(audio: Path) -> None:
    client = StubClient(interaction=_payload([_turn(), _turn(start=3.0, end=3.0)]))

    outcome = _annotator(client).annotate(audio)

    assert outcome.annotation.turn_count == 1
    assert outcome.annotation.dropped_turns_by_rule == {"turn.invalid_interval": 1}


def test_a_second_interaction_is_refused_by_the_ledger(audio: Path) -> None:
    annotator = _annotator(StubClient())
    annotator.annotate(audio)

    with pytest.raises(GeminiBudgetExceeded):
        annotator.ledger.reserve("interaction")


# --------------------------------------------------------------------- remote file deletion


def test_the_uploaded_file_is_deleted_after_a_successful_run(audio: Path) -> None:
    client = StubClient()

    outcome = _annotator(client).annotate(audio)

    assert outcome.remote_file_deleted is True
    assert client.kinds()[-1] == "delete"


def test_deletion_is_attempted_even_when_the_interaction_fails(audio: Path) -> None:
    """The finally is the point: a failed run must not leave audio sitting on a server."""

    client = StubClient(interaction_status=500)

    with pytest.raises(ProviderError):
        _annotator(client).annotate(audio)

    assert "delete" in client.kinds()


def test_deletion_is_attempted_even_when_the_response_is_malformed(audio: Path) -> None:
    client = StubClient(interaction={"turns": []})

    with pytest.raises(ProviderError):
        _annotator(client).annotate(audio)

    assert "delete" in client.kinds()


def test_a_failed_deletion_is_recorded_rather_than_raised(audio: Path) -> None:
    """An upload that could not be withdrawn is a fact the reviewer has to be told."""

    outcome = _annotator(StubClient(delete_status=500)).annotate(audio)

    assert outcome.remote_file_deleted is False
    assert "500" in outcome.deletion_detail


def test_a_deletion_that_raises_does_not_mask_the_result(audio: Path) -> None:
    outcome = _annotator(StubClient(delete_raises=True)).annotate(audio)

    assert outcome.remote_file_deleted is False
    assert "raised" in outcome.deletion_detail
    assert outcome.annotation.turn_count == 1


# ------------------------------------------------------------------------ silver artifacts


def _write(tmp_path: Path, turns: list[dict[str, Any]] | None = None) -> tuple[Path, str]:
    annotation = validate(_payload(turns))
    path, digest, _summary = write_silver(
        annotation,
        root=tmp_path / "annotations",
        conversation_id="A6000_S0005_0",
        input_sha256="a" * 64,
        model=GEMINI_MODEL,
        prompt=PROMPT,
        config={"temperature": None},
        remote_file_deleted=True,
        deletion_detail="deleted",
        call_counts={"uploads": 1, "interactions": 1, "deletes": 1},
    )
    return path, digest


def test_a_silver_artifact_records_everything_a_reviewer_needs(tmp_path: Path) -> None:
    path, digest = _write(tmp_path)
    record = read_silver(path)

    assert record["review_state"] == REVIEW_REQUIRED
    assert record["promotable"] is False
    assert record["source"]["input_sha256"] == "a" * 64
    assert record["source"]["model"] == GEMINI_MODEL
    assert record["source"]["prompt"] == PROMPT
    assert record["source"]["remote_audio_transmitted"] is True
    assert record["remote_file"]["deleted"] is True
    assert record["validation"] == {
        "mode": "strict",
        "dropped_turn_count": 0,
        "dropped_turns_by_rule": {},
    }
    assert record["created_at"]
    assert record["content_sha256"] == digest


def test_writing_over_an_existing_silver_artifact_is_refused(tmp_path: Path) -> None:
    _write(tmp_path)

    with pytest.raises(AnnotationError, match="already exists"):
        _write(tmp_path)


def test_the_safe_summary_carries_counts_and_never_transcript(tmp_path: Path) -> None:
    annotation = validate(_payload())

    summary = summarise(annotation.turns)

    assert summary["turn_count"] == 1
    assert summary["transcript_characters"] == len(TRANSCRIPT)
    assert TRANSCRIPT not in json.dumps(summary, ensure_ascii=False)


def test_salvage_metadata_is_private_count_only_and_reaches_the_artifact(tmp_path: Path) -> None:
    annotation = validate_silver(_payload([_turn(), _turn(start=4.0, end=4.0)]))
    path, _digest, summary = write_silver(
        annotation,
        root=tmp_path / "annotations",
        conversation_id="A6000_S0005_0",
        input_sha256="a" * 64,
        model=GEMINI_MODEL,
        prompt=PROMPT,
        config={},
        remote_file_deleted=True,
        deletion_detail="deleted",
        call_counts={"uploads": 1, "interactions": 1, "deletes": 1},
    )
    record = read_silver(path)

    assert record["validation"]["dropped_turns_by_rule"] == {"turn.invalid_interval": 1}
    assert summary["dropped_turn_count"] == 1
    assert "비공개 전사" not in json.dumps(record["validation"], ensure_ascii=False)


# ---------------------------------------------------------- silver is never ground truth


def test_silver_is_not_gold_eligible(tmp_path: Path) -> None:
    path, _digest = _write(tmp_path)

    assert is_gold_eligible(read_silver(path)) is False


def test_evaluating_against_silver_fails_closed(tmp_path: Path) -> None:
    """The gate anything scoring against ground truth is expected to call."""

    path, _digest = _write(tmp_path)

    with pytest.raises(AnnotationError, match="not reviewed gold"):
        assert_gold_eligible(read_silver(path))


def test_there_is_no_way_to_mark_silver_reviewed_in_place(tmp_path: Path) -> None:
    path, _digest = _write(tmp_path)

    record = read_silver(path)
    record["review_state"] = REVIEWED_GOLD  # a caller tampering with the loaded dict

    # Eligibility is decided by kind as well as state, so flipping the state is not enough.
    assert is_gold_eligible(record) is False


# ----------------------------------------------------------------------- promotion to gold


def _corrected() -> list[SilverTurn]:
    return [SilverTurn(0.0, 2.0, "SPEAKER_00", "환불 문의드립니다.", "anger", "raised pitch", 0.9)]


def test_a_reviewer_correction_becomes_immutable_gold(tmp_path: Path) -> None:
    silver_path, silver_digest = _write(tmp_path)

    gold_path, gold_digest, summary = promote(
        read_silver(silver_path),
        corrected_turns=_corrected(),
        reviewer="reviewer@example.test",
        root=tmp_path / "annotations",
        conversation_id="A6000_S0005_0",
        review_note="fixed emotion on turn 1",
    )
    record = verify_gold(gold_path)

    assert record["review_state"] == REVIEWED_GOLD
    assert record["reviewer"] == "reviewer@example.test"
    assert record["parent_silver_sha256"] == silver_digest
    assert record["content_sha256"] == gold_digest
    assert record["reviewed_at"]
    assert is_gold_eligible(record) is True
    assert summary["turn_count"] == 1


def test_promotion_without_a_reviewer_is_refused(tmp_path: Path) -> None:
    silver_path, _digest = _write(tmp_path)

    with pytest.raises(AnnotationError, match="reviewer identity"):
        promote(
            read_silver(silver_path),
            corrected_turns=_corrected(),
            reviewer="   ",
            root=tmp_path / "annotations",
            conversation_id="A6000_S0005_0",
        )


def test_promotion_without_a_correction_is_refused(tmp_path: Path) -> None:
    """A reviewer who submits nothing has asserted nothing; that is not a sign-off."""

    silver_path, _digest = _write(tmp_path)

    with pytest.raises(AnnotationError, match="corrected annotation"):
        promote(
            read_silver(silver_path),
            corrected_turns=[],
            reviewer="reviewer@example.test",
            root=tmp_path / "annotations",
            conversation_id="A6000_S0005_0",
        )


def test_gemini_output_cannot_be_promoted_without_going_through_a_reviewer(
    tmp_path: Path,
) -> None:
    """There is no API that turns the model's own artifact into gold by itself."""

    silver_path, _digest = _write(tmp_path)
    record = read_silver(silver_path)

    with pytest.raises(AnnotationError):
        promote(
            record,
            corrected_turns=[],
            reviewer="",
            root=tmp_path / "annotations",
            conversation_id="A6000_S0005_0",
        )
    assert not (tmp_path / "annotations" / "A6000_S0005_0" / "gold.json").exists()


def test_gold_cannot_be_overwritten(tmp_path: Path) -> None:
    silver_path, _digest = _write(tmp_path)
    arguments = {
        "corrected_turns": _corrected(),
        "reviewer": "reviewer@example.test",
        "root": tmp_path / "annotations",
        "conversation_id": "A6000_S0005_0",
    }
    promote(read_silver(silver_path), **arguments)

    with pytest.raises(AnnotationError, match="immutable"):
        promote(read_silver(silver_path), **arguments)


def test_an_edited_gold_file_fails_verification(tmp_path: Path) -> None:
    silver_path, _digest = _write(tmp_path)
    gold_path, _gold_digest, _summary = promote(
        read_silver(silver_path),
        corrected_turns=_corrected(),
        reviewer="reviewer@example.test",
        root=tmp_path / "annotations",
        conversation_id="A6000_S0005_0",
    )
    record = json.loads(gold_path.read_text(encoding="utf-8"))
    record["content"]["turns"][0]["emotion"] = "happiness"
    gold_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(AnnotationError, match="does not match its recorded digest"):
        verify_gold(gold_path)


def test_a_reviewer_agreeing_verbatim_is_recorded_as_such(tmp_path: Path) -> None:
    """Agreement is still a reviewer's assertion, and the artifact says it was unchanged."""

    silver_path, _digest = _write(tmp_path)
    annotation = validate(_payload())

    gold_path, _gold_digest, _summary = promote(
        read_silver(silver_path),
        corrected_turns=list(annotation.turns),
        reviewer="reviewer@example.test",
        root=tmp_path / "annotations",
        conversation_id="A6000_S0005_0",
    )

    record = verify_gold(gold_path)
    assert record["reviewer"] == "reviewer@example.test"
    assert isinstance(record["unchanged_from_silver"], bool)


# ------------------------------------------------------------------ api / ui disclosure


def test_the_gemini_disclosure_is_separate_from_the_pipeline_stage_disclosures() -> None:
    """A client rendering stage consent must not thereby render Gemini as consented."""

    from voxdelta.api.schemas import GeminiDisclosure, ProviderConfiguration

    assert "gemini" not in str(ProviderConfiguration.model_fields).lower()
    fields = set(GeminiDisclosure.model_fields)
    assert {"enabled", "consent_granted", "requires_human_review", "transmits"} <= fields


def test_the_disclosure_states_that_output_needs_review() -> None:
    from voxdelta.api.schemas import GeminiDisclosure

    disclosure = GeminiDisclosure(
        enabled=True,
        consent_granted=True,
        provider="gemini-annotation",
        model=GEMINI_MODEL,
        remote=True,
        transmits=("audio",),
        retention_policy_url="https://ai.google.dev/gemini-api/terms",
        produces="silver",
        requires_human_review=True,
    )

    assert disclosure.requires_human_review is True
    assert disclosure.produces == "silver"
    assert disclosure.remote is True


def test_the_review_state_schema_exposes_counts_not_transcript() -> None:
    from voxdelta.api.schemas import AnnotationReviewState

    fields = set(AnnotationReviewState.model_fields)

    assert "transcript" not in fields
    assert "turns" not in fields
    assert {"review_state", "promotable", "gold_present", "turn_count"} <= fields


# --------------------------------------------------------------- failure diagnosability


def test_a_failed_upload_start_records_that_no_audio_was_sent(audio: Path) -> None:
    """The distinction that matters after a failure: did the audio actually leave?"""

    class RejectingClient(StubClient):
        def request(self, method: str, url: str, **kwargs: Any) -> StubResponse:
            self.requests.append((method, url))
            if url.endswith("/upload/v1beta/files"):
                return StubResponse(403)
            raise AssertionError("nothing should follow a rejected upload start")

    client = RejectingClient()
    annotator = GeminiAnnotator(api_key="k", client=client, consent_granted=True)

    with pytest.raises(ProviderError):
        annotator.annotate(audio)

    assert annotator.trace.stage == "upload_start"
    assert annotator.trace.last_status == 403
    assert annotator.trace.audio_bytes_sent == 0
    # No delete either: there is no remote file to withdraw.
    assert "delete" not in client.kinds()


def test_a_failure_after_the_bytes_are_sent_records_them_and_still_deletes(
    audio: Path,
) -> None:
    client = StubClient(interaction_status=500)
    annotator = GeminiAnnotator(api_key="k", client=client, consent_granted=True)

    with pytest.raises(ProviderError):
        annotator.annotate(audio)

    assert annotator.trace.stage == "interaction"
    assert annotator.trace.audio_bytes_sent > 0
    assert "delete" in client.kinds()


def test_contract_failure_records_safe_response_shape_not_response_text(audio: Path) -> None:
    """A future retry can distinguish envelope and schema failures without retaining text."""

    class StepsClient(StubClient):
        def request(self, method: str, url: str, **kwargs: Any) -> StubResponse:
            if url.endswith("/v1beta/interactions"):
                self.requests.append((method, url))
                return StubResponse(
                    200,
                    {
                        "status": "completed",
                        "steps": [
                            {
                                "type": "model_output",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": json.dumps(
                                            _payload(
                                                [
                                                    _turn(
                                                        emotion="not-a-label",
                                                        transcript="비공개 전사",
                                                    )
                                                ]
                                            )
                                        ),
                                    }
                                ],
                            }
                        ],
                    },
                )
            return super().request(method, url, **kwargs)

    annotator = GeminiAnnotator(api_key="k", client=StepsClient(), consent_granted=True)

    with pytest.raises(ProviderError):
        annotator.annotate(audio)

    trace = annotator.trace.as_dict()
    assert trace["output_stage"] == "contract_invalid"
    assert trace["response_shape"] == {
        "body_type": "object",
        "top_level_keys": ["status", "steps"],
        "steps": [{"content_type": "list", "part_types": ["text"], "type": "model_output"}],
    }
    assert "비공개 전사" not in json.dumps(trace, ensure_ascii=False)


def test_contract_failure_records_counts_and_first_rule_without_model_text(audio: Path) -> None:
    """Schema diagnostics must identify the broken rule without copying a payload value."""

    class InvalidContractClient(StubClient):
        def request(self, method: str, url: str, **kwargs: Any) -> StubResponse:
            if url.endswith("/v1beta/interactions"):
                self.requests.append((method, url))
                return StubResponse(
                    200,
                    {
                        "steps": [
                            {
                                "type": "model_output",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": json.dumps(
                                            _payload(
                                                [
                                                    _turn(
                                                        emotion="private-unknown-label",
                                                        transcript="비공개 전사",
                                                    )
                                                ]
                                            )
                                        ),
                                    }
                                ],
                            }
                        ]
                    },
                )
            return super().request(method, url, **kwargs)

    annotator = GeminiAnnotator(
        api_key="k", client=InvalidContractClient(), consent_granted=True
    )
    with pytest.raises(ProviderError):
        annotator.annotate(audio)

    diagnostic = annotator.trace.as_dict()["contract_diagnostic"]
    assert diagnostic == {
        "payload_type": "object",
        "top_level_keys": ["notes", "speakers", "turns"],
        "turns": {"type": "list", "count": 1},
        "speakers": {"type": "list", "count": 2},
        "notes": {"type": "str"},
        "first_violation": {"rule": "turn.emotion_unknown", "turn_index": 0},
    }
    assert "비공개 전사" not in json.dumps(diagnostic, ensure_ascii=False)
    assert "private-unknown-label" not in json.dumps(diagnostic, ensure_ascii=False)


def test_duration_contract_failure_records_only_an_excess_bucket(audio: Path) -> None:
    """Timestamp drift is useful to diagnose, but exact model timestamps stay private."""

    client = StubClient(interaction=_payload([_turn(end=107.0)]))
    annotator = GeminiAnnotator(api_key="k", client=client, consent_granted=True)
    with pytest.raises(ProviderError):
        annotator.annotate(audio, duration_seconds=100.0)

    diagnostic = annotator.trace.as_dict()["contract_diagnostic"]
    assert diagnostic is not None
    assert diagnostic["first_violation"] == {
        "rule": "turn.exceeds_duration",
        "turn_index": 0,
        "duration_excess_bucket": "5_to_15_seconds",
    }
    encoded = json.dumps(diagnostic, ensure_ascii=False)
    assert "107" not in encoded
    assert "100" not in encoded


def test_contract_failure_records_delete_status_in_the_trace(audio: Path) -> None:
    annotator = GeminiAnnotator(
        api_key="k",
        client=StubClient(interaction={"turns": []}, delete_status=503),
        consent_granted=True,
    )
    with pytest.raises(ProviderError):
        annotator.annotate(audio)

    trace = annotator.trace.as_dict()
    assert trace["delete_attempted"] is True
    assert trace["delete_http_status"] == 503
    assert trace["delete_outcome"] == "http_error"


def test_the_trace_never_captures_a_response_body() -> None:
    from voxdelta.annotation.gemini_silver import CallTrace

    fields = set(CallTrace().as_dict())

    assert fields == {
        "stage_reached",
        "last_http_status",
        "audio_bytes_sent",
        "output_stage",
        "response_shape",
        "contract_diagnostic",
        "delete_attempted",
        "delete_http_status",
        "delete_outcome",
    }
