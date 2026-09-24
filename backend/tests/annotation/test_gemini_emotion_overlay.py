"""Tests for the remote Gemini emotion-only overlay candidate.

The overlay's value depends on three claims, so they are what these tests are organised
around rather than the happy path:

* **Nothing it touches changes.** ``silver.json``, gold, and the local XLS-R candidate are
  byte-compared across a successful generation and a rejected one.
* **The positional mapping is exact or the response is discarded.** There is no salvage
  here, unlike Silver drafting, so every way a mapping can be quietly wrong — missing,
  duplicated, renumbered, out of range, short, long — is asserted to reject the whole
  reply rather than produce a partial overlay.
* **The call budget holds.** One upload, one interaction, one delete, and the delete
  happens whether the interaction succeeded or raised.

No test reaches the network. The HTTP client is a stub throughout.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from voxdelta.annotation.gemini_emotion_overlay import (
    FILENAME,
    KIND,
    MAX_RATIONALE_CHARACTERS,
    RESPONSE_SCHEMA,
    GeminiEmotionOverlayError,
    OverlayCandidateError,
    OverlayRow,
    apply_rows,
    build_prompt,
    load_candidate,
    provenance,
    source_turns,
    validate_rows,
    write_candidate,
)
from voxdelta.annotation.gemini_silver import (
    EMOTION_LABELS,
    GeminiAnnotator,
    GeminiBudgetExceeded,
    GeminiConsentError,
)
from voxdelta.annotation.store import AnnotationError
from voxdelta.providers.base import ProviderError

CONVERSATION = "A6000_S0005_0"
TRANSCRIPT = "환불 처리가 아직 안 됐습니다"
OTHER_TRANSCRIPT = "네 확인해 드리겠습니다"


def silver_turn(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "start": 0.0,
        "end": 2.5,
        "speaker": "G5999",
        "transcript": TRANSCRIPT,
        "emotion": "uncertain",
        "emotion_rationale": "reviewer judgment required",
        "confidence": 0.0,
    }
    row.update(overrides)
    return row


def write_silver_fixture(root: Path, *, turn_count: int = 3) -> dict[str, Any]:
    """A clean reference-style Silver draft: real timing and transcript, no emotion yet."""

    turns = [
        silver_turn(
            start=float(index) * 2.5,
            end=float(index) * 2.5 + 2.5,
            speaker="G5999" if index % 2 == 0 else "G6000",
            transcript=TRANSCRIPT if index % 2 == 0 else OTHER_TRANSCRIPT,
        )
        for index in range(turn_count)
    ]
    content = {
        "turns": turns,
        "speakers": ["G5999", "G6000"],
        "notes": "Emotion requires reviewer judgment.",
    }
    digest = hashlib.sha256(
        json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    record: dict[str, Any] = {
        "schema_version": "1",
        "kind": "silver-annotation",
        "review_state": "review_required",
        "promotable": False,
        "conversation_id": CONVERSATION,
        "created_at": "2026-09-04T01:50:00+00:00",
        "source": {
            "input_sha256": "0" * 64,
            "model": "kcsc-human-reference",
            "prompt": "not_applicable_local_human_reference",
            "config": {},
            "remote_audio_transmitted": False,
        },
        "remote_file": {"deleted": False, "detail": "not_applicable_local_reference"},
        "call_counts": {"uploads": 0, "interactions": 0, "deletes": 0},
        "validation": {"mode": "reference_normalized"},
        "content": content,
        "content_sha256": digest,
        "contains_transcript": True,
        "handling": "private local reference artifact",
    }
    directory = root / CONVERSATION
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "silver.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record


def write_neighbour_fixtures(root: Path) -> dict[Path, bytes]:
    """The artifacts the overlay must never touch, captured for byte comparison."""

    directory = root / CONVERSATION
    local = directory / "emotion-candidate.calibrated-xlsr.json"
    local.write_text(json.dumps({"kind": "emotion-overlay-candidate"}), encoding="utf-8")
    gold = directory / "gold.json"
    gold.write_text(json.dumps({"kind": "gold-annotation"}), encoding="utf-8")
    return {path: path.read_bytes() for path in (directory / "silver.json", local, gold)}


def rows_payload(count: int, **overrides: Any) -> dict[str, Any]:
    rows = [
        {
            "position": position,
            "emotion": "neutral",
            "confidence": 0.5,
            "rationale": "flat prosody throughout",
        }
        for position in range(1, count + 1)
    ]
    payload: dict[str, Any] = {"rows": rows}
    payload.update(overrides)
    return payload


# --------------------------------------------------------------------------------------
# Positional validation: exact or discarded.
# --------------------------------------------------------------------------------------


def test_complete_exact_mapping_is_accepted_in_position_order() -> None:
    payload = rows_payload(3)
    # Out of order on the wire is fine; the mapping is by position, not by arrival.
    payload["rows"] = list(reversed(payload["rows"]))

    rows = validate_rows(payload, source_turn_count=3)

    assert [row.position for row in rows] == [1, 2, 3]
    assert all(row.emotion in EMOTION_LABELS for row in rows)


@pytest.mark.parametrize(
    ("mutate", "rule"),
    [
        pytest.param(
            lambda payload: payload["rows"].pop(), "rows.missing_positions", id="truncated"
        ),
        pytest.param(
            lambda payload: payload["rows"].__setitem__(2, dict(payload["rows"][0])),
            "row.duplicate_position",
            id="duplicate",
        ),
        pytest.param(
            lambda payload: payload["rows"].append(
                {"position": 2, "emotion": "anger", "confidence": 0.9, "rationale": "extra"}
            ),
            "row.duplicate_position",
            id="extra_row_must_collide",
        ),
        pytest.param(
            lambda payload: payload["rows"][1].update({"position": 0}),
            "row.position_out_of_range",
            id="position_zero",
        ),
        pytest.param(
            lambda payload: payload["rows"][1].update({"position": 9}),
            "row.position_out_of_range",
            id="position_past_end",
        ),
        pytest.param(
            lambda payload: payload["rows"][1].update({"emotion": "frustration"}),
            "row.emotion_unknown",
            id="unknown_label",
        ),
        pytest.param(
            lambda payload: payload["rows"][1].update({"emotion": None}),
            "row.required_text_not_string",
            id="null_label",
        ),
        pytest.param(
            lambda payload: payload["rows"][1].update({"confidence": 1.5}),
            "row.confidence_out_of_range",
            id="confidence_high",
        ),
        pytest.param(
            lambda payload: payload["rows"][1].update({"confidence": float("nan")}),
            "row.confidence_out_of_range",
            id="confidence_nan",
        ),
        pytest.param(
            lambda payload: payload["rows"][1].update({"rationale": "   "}),
            "row.rationale_empty",
            id="blank_rationale",
        ),
        pytest.param(
            lambda payload: payload["rows"][1].update(
                {"rationale": "가" * (MAX_RATIONALE_CHARACTERS + 1)}
            ),
            "row.rationale_too_long",
            id="long_rationale",
        ),
        pytest.param(
            lambda payload: payload["rows"][1].pop("confidence"),
            "row.missing_or_invalid_field",
            id="missing_field",
        ),
        pytest.param(
            lambda payload: payload["rows"].__setitem__(1, "neutral"),
            "row.not_object",
            id="row_not_object",
        ),
    ],
)
def test_broken_positional_mapping_rejects_the_whole_response(mutate: Any, rule: str) -> None:
    payload = rows_payload(3)
    mutate(payload)

    with pytest.raises(GeminiEmotionOverlayError) as raised:
        validate_rows(payload, source_turn_count=3)

    assert raised.value.diagnostic["first_violation"]["rule"] == rule


def test_renumbered_response_of_the_right_length_is_still_rejected() -> None:
    """A count check would pass this; the positional rules are what catch it."""

    payload = rows_payload(3)
    payload["rows"][2]["position"] = 2

    with pytest.raises(GeminiEmotionOverlayError) as raised:
        validate_rows(payload, source_turn_count=3)

    assert raised.value.diagnostic["first_violation"]["rule"] == "row.duplicate_position"


def test_missing_position_is_reported_without_any_model_text() -> None:
    payload = rows_payload(3)
    payload["rows"][1]["position"] = 3
    payload["rows"][2]["position"] = 1

    with pytest.raises(GeminiEmotionOverlayError) as raised:
        validate_rows(payload, source_turn_count=3)

    violation = raised.value.diagnostic["first_violation"]
    assert violation["rule"] == "row.duplicate_position"
    assert TRANSCRIPT not in json.dumps(raised.value.diagnostic, ensure_ascii=False)


def test_a_gap_in_the_middle_names_the_first_missing_position_and_the_gap_size() -> None:
    payload = {
        "rows": [
            {"position": 1, "emotion": "neutral", "confidence": 0.4, "rationale": "flat"},
            {"position": 3, "emotion": "neutral", "confidence": 0.4, "rationale": "flat"},
            {"position": 4, "emotion": "neutral", "confidence": 0.4, "rationale": "flat"},
        ]
    }

    with pytest.raises(GeminiEmotionOverlayError) as raised:
        validate_rows(payload, source_turn_count=4)

    violation = raised.value.diagnostic["first_violation"]
    assert violation["rule"] == "rows.missing_positions"
    assert violation["position"] == 2
    assert violation["missing_count"] == 1
    assert (violation["expected"], violation["received"]) == (4, 3)


@pytest.mark.parametrize(
    "payload", [None, [], "rows", {"rows": {}}, {"rows": None}, {}], ids=lambda value: str(value)
)
def test_malformed_envelopes_are_rejected(payload: object) -> None:
    with pytest.raises(GeminiEmotionOverlayError):
        validate_rows(payload, source_turn_count=2)


def test_rejection_diagnostics_never_carry_model_text() -> None:
    payload = rows_payload(2)
    payload["rows"][0]["rationale"] = TRANSCRIPT + " 라고 말했다"
    payload["rows"][0]["emotion"] = "furious"

    with pytest.raises(GeminiEmotionOverlayError) as raised:
        validate_rows(payload, source_turn_count=2)

    serialized = json.dumps(raised.value.diagnostic, ensure_ascii=False)
    assert TRANSCRIPT not in serialized
    assert "furious" not in serialized


# --------------------------------------------------------------------------------------
# The overlay preserves everything it is not allowed to change.
# --------------------------------------------------------------------------------------


def test_apply_rows_changes_only_emotion_confidence_and_rationale(tmp_path: Path) -> None:
    record = write_silver_fixture(tmp_path, turn_count=3)
    turns = source_turns(record)
    rows = validate_rows(rows_payload(3), source_turn_count=3)

    overlaid = apply_rows(turns, rows)

    for original, result in zip(turns, overlaid, strict=True):
        assert (result.start, result.end) == (original.start, original.end)
        assert result.speaker == original.speaker
        assert result.transcript == original.transcript
    assert [turn.emotion for turn in overlaid] == ["neutral"] * 3
    assert [turn.confidence for turn in overlaid] == [0.5] * 3
    assert all(turn.emotion_rationale == "flat prosody throughout" for turn in overlaid)


def test_apply_rows_refuses_a_length_mismatch(tmp_path: Path) -> None:
    record = write_silver_fixture(tmp_path, turn_count=3)
    turns = source_turns(record)

    with pytest.raises(GeminiEmotionOverlayError):
        apply_rows(turns, [OverlayRow(1, "neutral", 0.5, "flat")])


def test_prompt_asks_for_every_position_and_forbids_restating_the_source(
    tmp_path: Path,
) -> None:
    record = write_silver_fixture(tmp_path, turn_count=3)

    prompt = build_prompt(source_turns(record), duration_seconds=583.909562)

    assert "EXACTLY 3 entries" in prompt
    assert "1 through 3" in prompt
    assert "EMOTION ONLY" in prompt
    # The transcript is transmitted on purpose so the model can key its answer, and every
    # label the model may return is enumerated in the prompt.
    assert TRANSCRIPT in prompt
    for label in EMOTION_LABELS:
        assert label in prompt


def test_declared_provenance_admits_the_remote_transmission() -> None:
    declared = provenance()

    assert declared.remote is True
    assert set(declared.transmits) == {"audio", "text"}
    assert declared.retention_policy_url


def test_response_schema_enumerates_only_known_labels() -> None:
    row = RESPONSE_SCHEMA["properties"]["rows"]["items"]
    assert row["properties"]["emotion"]["enum"] == list(EMOTION_LABELS)
    assert set(row["required"]) == {"position", "emotion", "confidence", "rationale"}


# --------------------------------------------------------------------------------------
# The artifact, and the artifacts it must not disturb.
# --------------------------------------------------------------------------------------


def test_writing_a_candidate_mutates_nothing_else(tmp_path: Path) -> None:
    record = write_silver_fixture(tmp_path, turn_count=3)
    before = write_neighbour_fixtures(tmp_path)

    path, summary = write_candidate(
        annotation_root=tmp_path,
        conversation_id=CONVERSATION,
        silver_record=record,
        rows=validate_rows(rows_payload(3), source_turn_count=3),
        input_sha256="1" * 64,
        model="gemini-3.7-flash",
        remote_file_deleted=True,
        deletion_detail="deleted",
        call_counts={"uploads": 1, "interactions": 1, "deletes": 1},
    )

    assert path.name == FILENAME
    assert summary["turn_count"] == 3
    for neighbour, content in before.items():
        assert neighbour.read_bytes() == content, f"{neighbour.name} was modified"


def test_candidate_is_review_required_not_promotable_and_declares_transmission(
    tmp_path: Path,
) -> None:
    record = write_silver_fixture(tmp_path, turn_count=2)

    path, _ = write_candidate(
        annotation_root=tmp_path,
        conversation_id=CONVERSATION,
        silver_record=record,
        rows=validate_rows(rows_payload(2), source_turn_count=2),
        input_sha256="1" * 64,
        model="gemini-3.7-flash",
        remote_file_deleted=True,
        deletion_detail="deleted",
        call_counts={"uploads": 1, "interactions": 1, "deletes": 1},
    )
    candidate = json.loads(path.read_text(encoding="utf-8"))

    assert candidate["kind"] == KIND
    assert candidate["review_state"] == "review_required"
    assert candidate["promotable"] is False
    assert candidate["source"]["remote_audio_transmitted"] is True
    assert candidate["source"]["consent"] == "gemini_audio_transfer"
    assert candidate["source"]["model"] == "gemini-3.7-flash"
    assert candidate["source"]["silver_content_sha256"] == record["content_sha256"]
    assert candidate["remote_file"] == {"deleted": True, "detail": "deleted"}
    assert candidate["call_counts"] == {"uploads": 1, "interactions": 1, "deletes": 1}
    assert candidate["validation"]["salvage"] == "not_permitted"


def test_candidate_file_is_private_and_written_once(tmp_path: Path) -> None:
    record = write_silver_fixture(tmp_path, turn_count=2)
    rows = validate_rows(rows_payload(2), source_turn_count=2)
    arguments = {
        "annotation_root": tmp_path,
        "conversation_id": CONVERSATION,
        "silver_record": record,
        "rows": rows,
        "input_sha256": "1" * 64,
        "model": "gemini-3.7-flash",
        "remote_file_deleted": True,
        "deletion_detail": "deleted",
        "call_counts": {"uploads": 1, "interactions": 1, "deletes": 1},
    }

    path, _ = write_candidate(**arguments)  # type: ignore[arg-type]
    assert path.stat().st_mode & 0o777 == 0o600

    with pytest.raises(AnnotationError):
        write_candidate(**arguments)  # type: ignore[arg-type]


def test_write_refuses_a_silver_without_a_usable_digest(tmp_path: Path) -> None:
    record = write_silver_fixture(tmp_path, turn_count=2)
    record = {**record, "content_sha256": "short"}

    with pytest.raises(AnnotationError):
        write_candidate(
            annotation_root=tmp_path,
            conversation_id=CONVERSATION,
            silver_record=record,
            rows=validate_rows(rows_payload(2), source_turn_count=2),
            input_sha256="1" * 64,
            model="gemini-3.7-flash",
            remote_file_deleted=True,
            deletion_detail="deleted",
            call_counts={"uploads": 1, "interactions": 1, "deletes": 1},
        )


def _write_for_load(tmp_path: Path, *, turn_count: int = 2) -> tuple[Path, dict[str, Any]]:
    record = write_silver_fixture(tmp_path, turn_count=turn_count)
    path, _ = write_candidate(
        annotation_root=tmp_path,
        conversation_id=CONVERSATION,
        silver_record=record,
        rows=validate_rows(rows_payload(turn_count), source_turn_count=turn_count),
        input_sha256="1" * 64,
        model="gemini-3.7-flash",
        remote_file_deleted=True,
        deletion_detail="deleted",
        call_counts={"uploads": 1, "interactions": 1, "deletes": 1},
    )
    return path, record


def test_load_is_bound_to_the_exact_silver_and_leaves_the_file_alone(tmp_path: Path) -> None:
    path, record = _write_for_load(tmp_path)
    before = path.read_bytes()

    loaded = load_candidate(
        annotation_root=tmp_path,
        conversation_id=CONVERSATION,
        source_silver_content_sha256=record["content_sha256"],
    )

    assert loaded is not None
    assert loaded["content"]["turns"][0]["transcript"] == TRANSCRIPT
    assert path.read_bytes() == before
    assert (
        load_candidate(
            annotation_root=tmp_path,
            conversation_id=CONVERSATION,
            source_silver_content_sha256="b" * 64,
        )
        is None
    )


def test_load_returns_none_when_no_candidate_exists(tmp_path: Path) -> None:
    write_silver_fixture(tmp_path)
    assert (
        load_candidate(
            annotation_root=tmp_path,
            conversation_id=CONVERSATION,
            source_silver_content_sha256="a" * 64,
        )
        is None
    )


@pytest.mark.parametrize(
    "tamper",
    [
        pytest.param(
            lambda record: record["content"]["turns"][0].update({"emotion": "anger"}),
            id="content_edited",
        ),
        pytest.param(lambda record: record.update({"promotable": True}), id="made_promotable"),
        pytest.param(
            lambda record: record.update({"review_state": "reviewed_gold"}),
            id="review_state_forged",
        ),
        pytest.param(lambda record: record.update({"kind": "silver-annotation"}), id="kind_forged"),
    ],
)
def test_a_tampered_candidate_is_refused_rather_than_trusted(tmp_path: Path, tamper: Any) -> None:
    path, record = _write_for_load(tmp_path)
    candidate = json.loads(path.read_text(encoding="utf-8"))
    tamper(candidate)
    path.write_text(json.dumps(candidate, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(OverlayCandidateError):
        load_candidate(
            annotation_root=tmp_path,
            conversation_id=CONVERSATION,
            source_silver_content_sha256=record["content_sha256"],
        )


def test_unreadable_candidate_is_refused(tmp_path: Path) -> None:
    path, record = _write_for_load(tmp_path)
    path.write_text("{ not json", encoding="utf-8")

    with pytest.raises(OverlayCandidateError):
        load_candidate(
            annotation_root=tmp_path,
            conversation_id=CONVERSATION,
            source_silver_content_sha256=record["content_sha256"],
        )


# --------------------------------------------------------------------------------------
# Call and deletion boundaries.
# --------------------------------------------------------------------------------------


class StubResponse:
    def __init__(self, status: int, body: Any = None, headers: dict[str, str] | None = None):
        self.status_code = status
        self._body = body if body is not None else {}
        self.headers = headers or {}
        self.text = json.dumps(self._body)

    def json(self) -> Any:
        return self._body


class StubClient:
    """Replies with a scripted overlay conversation and records every request made."""

    def __init__(self, *, rows: Any = None, interaction_status: int = 200) -> None:
        self.requests: list[tuple[str, str]] = []
        self._rows = rows if rows is not None else rows_payload(3)
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
            return StubResponse(200)
        if url.endswith("/upload/v1beta/files"):
            return StubResponse(200, {}, {"x-goog-upload-url": "https://upload.example/session"})
        if url == "https://upload.example/session":
            return StubResponse(200, {"file": {"uri": "files/abc", "name": "files/abc"}})
        if url.endswith("/v1beta/interactions"):
            if self._interaction_status >= 400:
                return StubResponse(self._interaction_status)
            import json as _json

            return StubResponse(200, {"output_text": _json.dumps(self._rows)})
        raise AssertionError(f"unexpected request: {method} {url}")

    @property
    def kinds(self) -> list[str]:
        kinds = []
        for method, url in self.requests:
            if method == "DELETE":
                kinds.append("delete")
            elif url.endswith("/upload/v1beta/files") or url.startswith("https://upload.example"):
                kinds.append("upload")
            else:
                kinds.append("interaction")
        return kinds


def _overlay_run(
    annotator: GeminiAnnotator, audio: Path, *, source_turn_count: int
) -> tuple[Any, bool]:
    """The exact upload/interact/delete shape the generation script performs."""

    uri, name = annotator._upload(audio, "audio/wav")
    try:
        payload = annotator._interact(
            uri,
            "audio/wav",
            prompt="emotion only",
            response_schema=RESPONSE_SCHEMA,
        )
        rows = validate_rows(payload, source_turn_count=source_turn_count)
    finally:
        deleted, _detail = annotator._delete(name)
    return rows, deleted


@pytest.fixture
def audio(tmp_path: Path) -> Path:
    path = tmp_path / "conversation.wav"
    path.write_bytes(b"RIFF0000WAVEfmt ")
    return path


def test_a_successful_run_uses_exactly_one_of_each_call(audio: Path) -> None:
    client = StubClient()
    annotator = GeminiAnnotator(api_key="k", client=client, consent_granted=True)

    rows, deleted = _overlay_run(annotator, audio, source_turn_count=3)

    assert len(rows) == 3
    assert deleted is True
    assert annotator.ledger.counts() == {"uploads": 1, "interactions": 1, "deletes": 1}
    assert client.kinds == ["upload", "upload", "interaction", "delete"]


def test_the_upload_is_deleted_even_when_the_interaction_fails(audio: Path) -> None:
    client = StubClient(interaction_status=500)
    annotator = GeminiAnnotator(api_key="k", client=client, consent_granted=True)

    with pytest.raises(ProviderError):
        _overlay_run(annotator, audio, source_turn_count=3)

    assert annotator.ledger.counts() == {"uploads": 1, "interactions": 1, "deletes": 1}
    assert client.kinds[-1] == "delete"
    assert annotator.trace.delete_outcome == "deleted"


def test_the_upload_is_deleted_even_when_validation_rejects_the_response(audio: Path) -> None:
    client = StubClient(rows=rows_payload(2))
    annotator = GeminiAnnotator(api_key="k", client=client, consent_granted=True)

    with pytest.raises(GeminiEmotionOverlayError):
        _overlay_run(annotator, audio, source_turn_count=3)

    assert annotator.ledger.counts() == {"uploads": 1, "interactions": 1, "deletes": 1}
    assert client.kinds[-1] == "delete"


def test_a_second_call_of_any_kind_is_refused_by_the_ledger(audio: Path) -> None:
    client = StubClient()
    annotator = GeminiAnnotator(api_key="k", client=client, consent_granted=True)
    _overlay_run(annotator, audio, source_turn_count=3)
    before = len(client.requests)

    for retry in (
        lambda: annotator._upload(audio, "audio/wav"),
        lambda: annotator._interact("files/abc", "audio/wav", prompt="again"),
    ):
        with pytest.raises(GeminiBudgetExceeded):
            retry()
    # A refused delete is swallowed by design, but it must still not reach the network.
    annotator._delete("files/abc")

    assert len(client.requests) == before
    assert annotator.ledger.counts() == {"uploads": 1, "interactions": 1, "deletes": 1}


def test_without_gemini_consent_nothing_is_constructed_or_sent(audio: Path) -> None:
    client = StubClient()

    with pytest.raises(GeminiConsentError):
        GeminiAnnotator(api_key="k", client=client, consent_granted=False)

    assert client.requests == []
