"""Tests for the capability-fenced silver review API and the gold it can produce.

Three properties are worth more than the endpoint's happy path, so they are what these
tests are organised around:

* Gold is only ever what a named person asserted. Every route that skips the reviewer,
  the acknowledgement, or the corrections is refused and writes nothing.
* Silver never changes. It is byte-compared across a successful promotion, a refused one,
  and a repeat attempt.
* Nothing leaks. No response, including every error body, carries a filesystem path, and
  the listing carries no transcript at all.
"""

from __future__ import annotations

import hashlib
import json
import stat
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from pydantic import SecretStr

from voxdelta.annotation.review import CONVERSATION_ID
from voxdelta.annotation.store import verify_gold
from voxdelta.api.app import create_app
from voxdelta.api.dependencies import (
    default_annotation_audio_root,
    default_annotation_root,
)
from voxdelta.audio.service import AudioService
from voxdelta.config import Settings
from voxdelta.jobs.artifacts import ArtifactStore
from voxdelta.jobs.repository import JobRepository
from voxdelta.pipeline.runner import PipelineRunner

TEST_CAPABILITY_TOKEN = "annotation-review-capability-token-entropy"
CONVERSATION = "A6000_S0005_0"
TRANSCRIPT = "환불 처리가 아직도 안 됐습니다"
OTHER_TRANSCRIPT = "네 확인해 드리겠습니다"


def silver_turn(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "start": 0.0,
        "end": 2.5,
        "speaker": "SPEAKER_00",
        "transcript": TRANSCRIPT,
        "emotion": "anger",
        "emotion_rationale": "raised pitch",
        "confidence": 0.62,
    }
    row.update(overrides)
    return row


def write_silver_fixture(
    root: Path,
    *,
    conversation_id: str = CONVERSATION,
    turns: list[dict[str, Any]] | None = None,
    dropped: int = 31,
    remote_file_deleted: bool = True,
) -> Path:
    """Write a silver artifact shaped exactly like the one the Gemini run produced."""

    content = {
        "turns": turns
        if turns is not None
        else [
            silver_turn(),
            silver_turn(
                start=2.5,
                end=5.0,
                speaker="SPEAKER_01",
                transcript=OTHER_TRANSCRIPT,
                emotion="neutral",
                confidence=0.71,
            ),
            silver_turn(start=5.0, end=7.0, emotion="uncertain", confidence=0.3),
        ],
        "speakers": ["SPEAKER_00", "SPEAKER_01"],
        "notes": "Check the overlapping section near the end.",
    }
    digest = hashlib.sha256(
        json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    record = {
        "schema_version": "1",
        "kind": "silver-annotation",
        "review_state": "review_required",
        "promotable": False,
        "conversation_id": conversation_id,
        "created_at": "2026-09-02T02:26:24+00:00",
        "source": {
            "input_sha256": "0" * 64,
            "model": "gemini-3.7-flash",
            "prompt": "provisional first-pass annotation",
            "config": {"response_format": "structured-json"},
            "remote_audio_transmitted": True,
        },
        "remote_file": {
            "deleted": remote_file_deleted,
            "detail": "deleted" if remote_file_deleted else "delete returned HTTP 503",
        },
        "call_counts": {"uploads": 1, "interactions": 1, "deletes": 1},
        "validation": {
            "mode": "salvaged" if dropped else "strict",
            "dropped_turn_count": dropped,
            "dropped_turns_by_rule": {"turn.exceeds_duration": dropped} if dropped else {},
        },
        "content": content,
        "content_sha256": digest,
        "contains_transcript": True,
        "handling": "private artifact; never copy into logs or benchmark output",
    }
    directory = root / conversation_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "silver.json"
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def build_app(
    tmp_path: Path,
    annotation_root: Path,
    *,
    annotation_audio_root: Path | None = None,
) -> FastAPI:
    jobs_root = tmp_path / "jobs"
    repository = JobRepository(tmp_path / "voxdelta.sqlite3")
    artifacts = ArtifactStore(jobs_root)
    runner = PipelineRunner(repository, artifacts, AudioService(jobs_root, 60, 3600))
    return create_app(
        repository=repository,
        artifacts=artifacts,
        runner=runner,
        api_capability_token=SecretStr(TEST_CAPABILITY_TOKEN),
        annotation_root=annotation_root,
        annotation_audio_root=annotation_audio_root,
    )


@asynccontextmanager
async def client_for(app: FastAPI, *, token: str | None = TEST_CAPABILITY_TOKEN):
    headers = {"X-VoxDelta-Token": token} if token is not None else {}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://localhost",
        headers=headers,
    ) as client:
        yield client


@pytest.fixture
def annotation_root(tmp_path: Path) -> Path:
    root = tmp_path / "annotations"
    root.mkdir()
    return root


@pytest.fixture
def app(tmp_path: Path, annotation_root: Path) -> FastAPI:
    write_silver_fixture(annotation_root)
    return build_app(tmp_path, annotation_root)


@asynccontextmanager
async def reviewing(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with client_for(app) as client:
        yield client


def assert_no_paths(body: object, root: Path) -> None:
    serialized = json.dumps(body, ensure_ascii=False)
    assert str(root) not in serialized
    assert str(root.parent) not in serialized
    assert "silver.json" not in serialized
    assert "gold.json" not in serialized


def promotion_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "reviewer": "reviewer@example.test",
        "acknowledged": True,
        "review_note": "Listened end to end.",
        "change_reasons": [],
        "turns": [
            silver_turn(transcript="환불 처리가 아직 안 됐습니다", emotion="anger", confidence=0.9),
            silver_turn(
                start=2.5,
                end=5.0,
                speaker="SPEAKER_01",
                transcript=OTHER_TRANSCRIPT,
                emotion="neutral",
                confidence=0.8,
            ),
        ],
    }
    body.update(overrides)
    return body


def write_alignment_proposal_fixture(root: Path) -> None:
    """A transcript-free sidecar bound to the fixture's immutable Silver digest."""

    silver = json.loads((root / CONVERSATION / "silver.json").read_text(encoding="utf-8"))
    proposal = {
        "schema_version": "1",
        "kind": "alignment-proposal",
        "source_silver_content_sha256": silver["content_sha256"],
        "content": {
            "target_start_position": 10,
            "rows": [{"position": 10, "start": 12.3, "end": 15.4, "confidence": 0.9}],
        },
        "validation": {"dropped_row_count": 0, "dropped_rows_by_rule": {}},
    }
    (root / CONVERSATION / "alignment-proposal-10.json").write_text(
        json.dumps(proposal, ensure_ascii=False), encoding="utf-8"
    )


def write_reference_fixture(root: Path) -> Path:
    reference_root = root / "reference"
    reference_root.mkdir(parents=True)
    (reference_root / f"{CONVERSATION}.json").write_text(
        json.dumps(
            {
                "conversation_id": CONVERSATION,
                "duration_seconds": 10.0,
                "speakers": ["G5999", "G6000"],
                "turns": [
                    {"start": 0.0, "end": 2.0, "speaker": "G5999", "transcript": TRANSCRIPT},
                    {"start": 2.0, "end": 4.0, "speaker": "G6000", "transcript": OTHER_TRANSCRIPT},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return reference_root


@pytest.mark.asyncio
async def test_reference_resegmentation_candidate_is_optional_and_capability_fenced(
    tmp_path: Path,
    annotation_root: Path,
) -> None:
    write_silver_fixture(annotation_root)
    reference_root = write_reference_fixture(tmp_path / "derived" / "kcsc")
    app = build_app(
        tmp_path,
        annotation_root,
        annotation_audio_root=reference_root.parent / "audio",
    )

    async with client_for(app) as client:
        response = await client.get(
            f"/api/annotations/{CONVERSATION}/reference-resegmentation-candidate"
        )

    assert response.status_code == 200
    body = response.json()
    assert body["reference_turn_count"] == 2
    assert body["turns"][0]["speaker"] == "G5999"
    assert body["turns"][0]["emotion"] == "uncertain"
    assert_no_paths(body, annotation_root)


def write_gemini_overlay_fixture(
    root: Path,
    *,
    silver_digest: str | None = None,
    review_state: str = "review_required",
    promotable: bool = False,
) -> Path:
    """A remote Gemini overlay bound to the fixture Silver, written the way the script does."""

    silver = json.loads((root / CONVERSATION / "silver.json").read_text(encoding="utf-8"))
    turns = [
        {**turn, "emotion": "sadness", "emotion_rationale": "falling pitch", "confidence": 0.44}
        for turn in silver["content"]["turns"]
    ]
    content = {"turns": turns}
    digest = hashlib.sha256(
        json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    candidate = {
        "schema_version": "1",
        "kind": "gemini-emotion-overlay-candidate",
        "review_state": review_state,
        "promotable": promotable,
        "conversation_id": CONVERSATION,
        "source": {
            "silver_content_sha256": silver_digest or silver["content_sha256"],
            "input_sha256": "1" * 64,
            "model": "gemini-3.7-flash",
            "provider": "gemini-emotion-overlay",
            "remote_audio_transmitted": True,
            "consent": "gemini_audio_transfer",
        },
        "content": content,
        "content_sha256": digest,
        "summary": {
            "turn_count": len(turns),
            "emotion_histogram": {"sadness": len(turns), "uncertain": 0},
            "uncertain_turns": 0,
            "mean_confidence": 0.44,
        },
        "contains_transcript": True,
    }
    path = root / CONVERSATION / "emotion-candidate.gemini-remote.json"
    path.write_text(json.dumps(candidate, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_gemini_emotion_overlay_is_served_marked_remote_and_unpromotable(
    app: FastAPI, annotation_root: Path
) -> None:
    path = write_gemini_overlay_fixture(annotation_root)
    before = path.read_bytes()
    silver = (annotation_root / CONVERSATION / "silver.json").read_bytes()

    async with reviewing(app) as client:
        response = await client.get(
            f"/api/annotations/{CONVERSATION}/gemini-emotion-overlay-candidate"
        )

    assert response.status_code == 200
    body = response.json()
    assert body["remote_audio_transmitted"] is True
    assert body["review_required"] is True
    assert body["promotable"] is False
    assert body["model"] == "gemini-3.7-flash"
    assert body["uncertain_turns"] == 0
    assert body["mean_confidence"] == 0.44
    assert [turn["emotion"] for turn in body["turns"]] == ["sadness"] * 3
    # Reading a candidate is a read. Nothing on disk moves.
    assert path.read_bytes() == before
    assert (annotation_root / CONVERSATION / "silver.json").read_bytes() == silver
    assert_no_paths(body, annotation_root)


@pytest.mark.asyncio
async def test_gemini_overlay_preserves_the_draft_timing_speaker_and_transcript(
    app: FastAPI, annotation_root: Path
) -> None:
    write_gemini_overlay_fixture(annotation_root)

    async with reviewing(app) as client:
        draft = await client.get(f"/api/annotations/{CONVERSATION}")
        overlay = await client.get(
            f"/api/annotations/{CONVERSATION}/gemini-emotion-overlay-candidate"
        )

    for source, candidate in zip(draft.json()["turns"], overlay.json()["turns"], strict=True):
        assert (source["start"], source["end"]) == (candidate["start"], candidate["end"])
        assert source["speaker"] == candidate["speaker"]
        assert source["transcript"] == candidate["transcript"]


@pytest.mark.asyncio
async def test_no_gemini_overlay_is_not_an_error(app: FastAPI) -> None:
    async with reviewing(app) as client:
        response = await client.get(
            f"/api/annotations/{CONVERSATION}/gemini-emotion-overlay-candidate"
        )

    assert response.status_code == 200
    assert response.json() is None


@pytest.mark.asyncio
async def test_a_gemini_overlay_for_a_different_silver_is_not_served(
    app: FastAPI, annotation_root: Path
) -> None:
    write_gemini_overlay_fixture(annotation_root, silver_digest="c" * 64)

    async with reviewing(app) as client:
        response = await client.get(
            f"/api/annotations/{CONVERSATION}/gemini-emotion-overlay-candidate"
        )

    assert response.status_code == 200
    assert response.json() is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"promotable": True}, id="claims_promotable"),
        pytest.param({"review_state": "reviewed_gold"}, id="claims_reviewed"),
    ],
)
async def test_an_overlay_claiming_to_be_promotable_is_refused(
    app: FastAPI, annotation_root: Path, overrides: dict[str, Any]
) -> None:
    write_gemini_overlay_fixture(annotation_root, **overrides)

    async with reviewing(app) as client:
        response = await client.get(
            f"/api/annotations/{CONVERSATION}/gemini-emotion-overlay-candidate"
        )

    assert response.status_code == 409
    body = response.json()
    assert body["detail"]["code"] == "gemini_emotion_candidate_unreadable"
    assert_no_paths(body, annotation_root)


@pytest.mark.asyncio
async def test_a_tampered_gemini_overlay_is_refused(app: FastAPI, annotation_root: Path) -> None:
    path = write_gemini_overlay_fixture(annotation_root)
    candidate = json.loads(path.read_text(encoding="utf-8"))
    candidate["content"]["turns"][0]["emotion"] = "anger"
    path.write_text(json.dumps(candidate, ensure_ascii=False), encoding="utf-8")

    async with reviewing(app) as client:
        response = await client.get(
            f"/api/annotations/{CONVERSATION}/gemini-emotion-overlay-candidate"
        )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "gemini_emotion_candidate_unreadable"


@pytest.mark.asyncio
async def test_the_gemini_overlay_route_is_capability_fenced(
    app: FastAPI, annotation_root: Path
) -> None:
    write_gemini_overlay_fixture(annotation_root)

    async with client_for(app, token=None) as client:
        response = await client.get(
            f"/api/annotations/{CONVERSATION}/gemini-emotion-overlay-candidate"
        )

    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "capability_required"
    assert TRANSCRIPT not in response.text


@pytest.mark.asyncio
async def test_the_api_offers_no_way_to_generate_a_gemini_overlay(app: FastAPI) -> None:
    """Generating one transmits audio. That stays an operator action, not a web request."""

    routes = {getattr(route, "path", "") for route in app.routes if hasattr(route, "methods")}
    overlay_route = "/api/annotations/{conversation_id}/gemini-emotion-overlay-candidate"
    assert overlay_route in routes
    for route in app.routes:
        if getattr(route, "path", "") == overlay_route:
            assert set(getattr(route, "methods", set())) == {"GET"}


# ------------------------------------------------------------------------- access fence


@pytest.mark.asyncio
async def test_the_review_routes_require_the_local_capability(app: FastAPI) -> None:
    async with client_for(app, token=None) as client:
        listing = await client.get("/api/annotations")
        draft = await client.get(f"/api/annotations/{CONVERSATION}")
        alignment = await client.get(f"/api/annotations/{CONVERSATION}/alignment-proposal")
        promotion = await client.post(
            f"/api/annotations/{CONVERSATION}/gold", json=promotion_body()
        )

    for response in (listing, draft, alignment, promotion):
        assert response.status_code == 401
        assert response.json()["detail"]["code"] == "capability_required"


@pytest.mark.asyncio
async def test_a_wrong_capability_reaches_no_transcript(app: FastAPI) -> None:
    async with client_for(app, token="not-the-launch-token") as client:
        response = await client.get(f"/api/annotations/{CONVERSATION}")

    assert response.status_code == 401
    assert TRANSCRIPT not in response.text


@pytest.mark.asyncio
async def test_a_non_local_host_cannot_reach_the_review_routes(app: FastAPI) -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://annotations.example.test",
        headers={"X-VoxDelta-Token": TEST_CAPABILITY_TOKEN},
    ) as client:
        response = await client.get(f"/api/annotations/{CONVERSATION}")

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_host"
    assert TRANSCRIPT not in response.text


# ------------------------------------------------------------------------------ listing


@pytest.mark.asyncio
async def test_alignment_proposal_is_timing_only_and_bound_to_silver(
    app: FastAPI, annotation_root: Path
) -> None:
    write_alignment_proposal_fixture(annotation_root)

    async with reviewing(app) as client:
        response = await client.get(f"/api/annotations/{CONVERSATION}/alignment-proposal")

    assert response.status_code == 200
    body = response.json()
    assert body["target_start_position"] == 10
    assert body["target_end_position"] == 3
    assert body["rows"] == [{"position": 10, "start": 12.3, "end": 15.4, "confidence": 0.9}]
    assert TRANSCRIPT not in response.text
    assert OTHER_TRANSCRIPT not in response.text
    assert_no_paths(body, annotation_root)


@pytest.mark.asyncio
async def test_alignment_proposal_collection_exposes_available_ranges_only(
    app: FastAPI, annotation_root: Path
) -> None:
    write_alignment_proposal_fixture(annotation_root)

    async with reviewing(app) as client:
        response = await client.get(f"/api/annotations/{CONVERSATION}/alignment-proposals")

    assert response.status_code == 200
    body = response.json()
    assert len(body["proposals"]) == 1
    assert body["proposals"][0]["target_start_position"] == 10
    assert body["proposals"][0]["target_end_position"] == 3
    assert TRANSCRIPT not in response.text
    assert_no_paths(body, annotation_root)


@pytest.mark.asyncio
async def test_the_listing_is_counts_only_and_carries_no_transcript(
    app: FastAPI, annotation_root: Path
) -> None:
    async with reviewing(app) as client:
        response = await client.get("/api/annotations")

    assert response.status_code == 200
    body = response.json()
    assert TRANSCRIPT not in response.text
    assert OTHER_TRANSCRIPT not in response.text
    assert_no_paths(body, annotation_root)
    (state,) = body["annotations"]
    assert state["conversation_id"] == CONVERSATION
    assert state["review_state"] == "review_required"
    assert state["promotable"] is False
    assert state["gold_present"] is False
    assert state["turn_count"] == 3
    assert state["uncertain_turns"] == 1
    assert "turns" not in state
    assert body["unreadable_count"] == 0


@pytest.mark.asyncio
async def test_an_empty_annotation_root_lists_nothing_rather_than_failing(
    tmp_path: Path,
) -> None:
    app = build_app(tmp_path, tmp_path / "never-written")

    async with reviewing(app) as client:
        response = await client.get("/api/annotations")

    assert response.status_code == 200
    assert response.json() == {"annotations": [], "unreadable_count": 0}


@pytest.mark.asyncio
async def test_a_provenance_draft_hidden_from_the_queue_remains_stored(
    tmp_path: Path, annotation_root: Path
) -> None:
    path = write_silver_fixture(annotation_root)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["review_listing_visible"] = False
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    app = build_app(tmp_path, annotation_root)

    async with reviewing(app) as client:
        listing = await client.get("/api/annotations")
        draft = await client.get(f"/api/annotations/{CONVERSATION}")

    assert listing.status_code == 200
    assert listing.json() == {"annotations": [], "unreadable_count": 0}
    assert draft.status_code == 200


@pytest.mark.asyncio
async def test_a_draft_that_cannot_be_read_is_counted_rather_than_hidden(
    tmp_path: Path, annotation_root: Path
) -> None:
    write_silver_fixture(annotation_root)
    broken = annotation_root / "B0001_S0001_0"
    broken.mkdir()
    (broken / "silver.json").write_text("{not json", encoding="utf-8")
    app = build_app(tmp_path, annotation_root)

    async with reviewing(app) as client:
        body = (await client.get("/api/annotations")).json()

    assert [state["conversation_id"] for state in body["annotations"]] == [CONVERSATION]
    assert body["unreadable_count"] == 1


# -------------------------------------------------------------------------------- draft


@pytest.mark.asyncio
async def test_the_draft_carries_every_reviewable_field(
    app: FastAPI, annotation_root: Path
) -> None:
    async with reviewing(app) as client:
        response = await client.get(f"/api/annotations/{CONVERSATION}")

    assert response.status_code == 200
    body = response.json()
    assert_no_paths(body, annotation_root)
    assert body["state"]["review_state"] == "review_required"
    assert body["speakers"] == ["SPEAKER_00", "SPEAKER_01"]
    assert body["notes"]
    assert [turn["transcript"] for turn in body["turns"]][:2] == [TRANSCRIPT, OTHER_TRANSCRIPT]
    assert {"start", "end", "speaker", "transcript", "emotion", "confidence"} <= set(
        body["turns"][0]
    )
    assert "uncertain" in body["emotion_labels"]


@pytest.mark.asyncio
async def test_the_draft_reports_review_required_and_the_salvage_loss(app: FastAPI) -> None:
    async with reviewing(app) as client:
        body = (await client.get(f"/api/annotations/{CONVERSATION}")).json()

    warnings = {warning["code"]: warning for warning in body["warnings"]}
    assert "review_required" in warnings
    assert warnings["salvaged_dropped_turns"]["count"] == 31
    assert warnings["salvaged_dropped_turns"]["by_rule"] == {"turn.exceeds_duration": 31}
    assert "remote_audio_transmitted" in warnings


@pytest.mark.asyncio
async def test_a_draft_without_dropped_turns_reports_no_salvage_warning(
    tmp_path: Path, annotation_root: Path
) -> None:
    write_silver_fixture(annotation_root, dropped=0)
    app = build_app(tmp_path, annotation_root)

    async with reviewing(app) as client:
        body = (await client.get(f"/api/annotations/{CONVERSATION}")).json()

    codes = {warning["code"] for warning in body["warnings"]}
    assert "review_required" in codes
    assert "salvaged_dropped_turns" not in codes


@pytest.mark.asyncio
async def test_an_undeleted_remote_upload_is_reported_to_the_reviewer(
    tmp_path: Path, annotation_root: Path
) -> None:
    write_silver_fixture(annotation_root, remote_file_deleted=False)
    app = build_app(tmp_path, annotation_root)

    async with reviewing(app) as client:
        body = (await client.get(f"/api/annotations/{CONVERSATION}")).json()

    assert "remote_file_not_deleted" in {warning["code"] for warning in body["warnings"]}
    assert body["state"]["remote_file_deleted"] is False


@pytest.mark.asyncio
async def test_an_unknown_conversation_is_a_path_free_not_found(
    app: FastAPI, annotation_root: Path
) -> None:
    async with reviewing(app) as client:
        response = await client.get("/api/annotations/B9999_S0001_0")

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "annotation_not_found"
    assert_no_paths(response.json(), annotation_root)


@pytest.mark.parametrize(
    "conversation_id",
    # An empty id is left out on purpose: it is the listing route with a trailing slash,
    # not an identifier, and the shape rule below covers it directly.
    ["..", "a.b", "A6000%2F..%2F..%2Fetc", "a" * 65, "-leading-hyphen"],
)
@pytest.mark.asyncio
async def test_an_identifier_that_is_not_a_conversation_name_is_refused(
    app: FastAPI, annotation_root: Path, conversation_id: str
) -> None:
    async with reviewing(app) as client:
        response = await client.get(f"/api/annotations/{conversation_id}")

    assert response.status_code in {404, 422}
    body = response.json()
    if response.status_code == 422:
        assert body["detail"]["code"] in {"invalid_conversation_id", "invalid_request"}
    assert_no_paths(body, annotation_root)


def test_the_accepted_conversation_id_shape_excludes_traversal() -> None:
    assert CONVERSATION_ID.fullmatch(CONVERSATION)
    for rejected in ("..", "../etc", "a/b", "a\\b", ".hidden", "a.b", ""):
        assert CONVERSATION_ID.fullmatch(rejected) is None


@pytest.mark.asyncio
async def test_a_silver_artifact_that_is_not_silver_is_refused_without_its_path(
    tmp_path: Path, annotation_root: Path
) -> None:
    directory = annotation_root / CONVERSATION
    directory.mkdir()
    (directory / "silver.json").write_text(
        json.dumps({"kind": "gold-annotation"}), encoding="utf-8"
    )
    app = build_app(tmp_path, annotation_root)

    async with reviewing(app) as client:
        response = await client.get(f"/api/annotations/{CONVERSATION}")

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "annotation_unreadable"
    assert_no_paths(response.json(), annotation_root)


# ---------------------------------------------------------------------------- promotion


@pytest.mark.asyncio
async def test_a_reviewed_correction_becomes_immutable_gold(
    app: FastAPI, annotation_root: Path
) -> None:
    silver_path = annotation_root / CONVERSATION / "silver.json"
    before = silver_path.read_bytes()

    async with reviewing(app) as client:
        response = await client.post(f"/api/annotations/{CONVERSATION}/gold", json=promotion_body())

    assert response.status_code == 201
    body = response.json()
    assert_no_paths(body, annotation_root)
    assert body["reviewer"] == "reviewer@example.test"
    assert body["turn_count"] == 2
    assert body["silver_unmodified"] is True
    assert len(body["content_sha256"]) == 64
    assert len(body["parent_silver_sha256"]) == 64
    assert body["change_reason_count"] == 0

    gold = verify_gold(annotation_root / CONVERSATION / "gold.json")
    assert gold["review_state"] == "reviewed_gold"
    assert gold["reviewer"] == "reviewer@example.test"
    assert silver_path.read_bytes() == before


@pytest.mark.asyncio
async def test_a_change_reason_is_frozen_with_the_reviewed_gold(
    app: FastAPI, annotation_root: Path
) -> None:
    async with reviewing(app) as client:
        response = await client.post(
            f"/api/annotations/{CONVERSATION}/gold",
            json=promotion_body(
                change_reasons=[{"position": 1, "reason": "transcript_mismatch"}],
            ),
        )

    assert response.status_code == 201
    assert response.json()["change_reason_count"] == 1
    gold = verify_gold(annotation_root / CONVERSATION / "gold.json")
    assert gold["content"]["review_change_reasons"] == [
        {"position": 1, "reason": "transcript_mismatch"}
    ]


@pytest.mark.asyncio
async def test_reading_a_draft_never_produces_gold(app: FastAPI, annotation_root: Path) -> None:
    """There is no automatic promotion: only the reviewer's own POST writes gold."""

    async with reviewing(app) as client:
        for _ in range(3):
            assert (await client.get(f"/api/annotations/{CONVERSATION}")).status_code == 200
        listing = (await client.get("/api/annotations")).json()

    assert not (annotation_root / CONVERSATION / "gold.json").exists()
    assert listing["annotations"][0]["gold_present"] is False
    assert listing["annotations"][0]["promotable"] is False


@pytest.mark.parametrize(
    ("overrides", "expected_code"),
    [
        ({"reviewer": "   "}, "reviewer_required"),
        ({"acknowledged": False}, "review_acknowledgement_required"),
        ({"turns": []}, "corrected_turns_required"),
        ({"turns": [silver_turn(emotion="frustrated")]}, "invalid_turn_emotion"),
        ({"turns": [silver_turn(start=4.0, end=4.0)]}, "invalid_turn_interval"),
        ({"turns": [silver_turn(start=-1.0, end=2.0)]}, "invalid_turn_interval"),
        ({"turns": [silver_turn(confidence=1.4)]}, "invalid_turn_confidence"),
        ({"turns": [silver_turn(transcript="   ")]}, "invalid_turn_transcript"),
        ({"turns": [silver_turn(speaker=" ")]}, "invalid_turn_speaker"),
    ],
)
@pytest.mark.asyncio
async def test_a_promotion_that_is_not_earned_is_refused_and_writes_nothing(
    app: FastAPI,
    annotation_root: Path,
    overrides: dict[str, Any],
    expected_code: str,
) -> None:
    silver_path = annotation_root / CONVERSATION / "silver.json"
    before = silver_path.read_bytes()

    async with reviewing(app) as client:
        response = await client.post(
            f"/api/annotations/{CONVERSATION}/gold", json=promotion_body(**overrides)
        )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == expected_code
    assert_no_paths(response.json(), annotation_root)
    assert not (annotation_root / CONVERSATION / "gold.json").exists()
    assert silver_path.read_bytes() == before


@pytest.mark.asyncio
async def test_the_refusal_names_the_turn_without_quoting_it(app: FastAPI) -> None:
    async with reviewing(app) as client:
        response = await client.post(
            f"/api/annotations/{CONVERSATION}/gold",
            json=promotion_body(
                turns=[
                    silver_turn(),
                    silver_turn(transcript=TRANSCRIPT, emotion="frustrated"),
                ]
            ),
        )

    message = response.json()["detail"]["message"]
    assert "Turn 2" in message
    assert TRANSCRIPT not in message


@pytest.mark.asyncio
async def test_gold_is_written_once_and_a_repeat_is_refused(
    app: FastAPI, annotation_root: Path
) -> None:
    gold_path = annotation_root / CONVERSATION / "gold.json"

    async with reviewing(app) as client:
        first = await client.post(f"/api/annotations/{CONVERSATION}/gold", json=promotion_body())
        frozen = gold_path.read_bytes()
        second = await client.post(
            f"/api/annotations/{CONVERSATION}/gold",
            json=promotion_body(reviewer="second@example.test"),
        )

    assert first.status_code == 201
    assert second.status_code == 409
    assert second.json()["detail"]["code"] == "gold_already_exists"
    assert_no_paths(second.json(), annotation_root)
    assert gold_path.read_bytes() == frozen


@pytest.mark.asyncio
async def test_gold_is_created_exclusively_so_a_concurrent_write_cannot_replace_it(
    app: FastAPI, annotation_root: Path
) -> None:
    """The refusal is the create failing, not a check that a second writer could race past."""

    gold_path = annotation_root / CONVERSATION / "gold.json"
    gold_path.write_text('{"kind": "gold-annotation"}', encoding="utf-8")
    frozen = gold_path.read_bytes()

    async with reviewing(app) as client:
        response = await client.post(f"/api/annotations/{CONVERSATION}/gold", json=promotion_body())

    assert response.status_code == 409
    assert gold_path.read_bytes() == frozen


@pytest.mark.asyncio
async def test_gold_is_written_private_because_it_holds_transcript(
    app: FastAPI, annotation_root: Path
) -> None:
    async with reviewing(app) as client:
        assert (
            await client.post(f"/api/annotations/{CONVERSATION}/gold", json=promotion_body())
        ).status_code == 201

    mode = (annotation_root / CONVERSATION / "gold.json").stat().st_mode
    assert stat.S_IMODE(mode) == 0o600


@pytest.mark.asyncio
async def test_an_existing_gold_is_disclosed_before_a_reviewer_starts(
    app: FastAPI, annotation_root: Path
) -> None:
    async with reviewing(app) as client:
        await client.post(f"/api/annotations/{CONVERSATION}/gold", json=promotion_body())
        draft = (await client.get(f"/api/annotations/{CONVERSATION}")).json()
        listing = (await client.get("/api/annotations")).json()

    assert draft["state"]["gold_present"] is True
    assert "gold_already_exists" in {warning["code"] for warning in draft["warnings"]}
    assert listing["annotations"][0]["gold_present"] is True


@pytest.mark.asyncio
async def test_promoting_an_absent_conversation_creates_no_directory(
    app: FastAPI, annotation_root: Path
) -> None:
    async with reviewing(app) as client:
        response = await client.post("/api/annotations/B9999_S0001_0/gold", json=promotion_body())

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "annotation_not_found"
    assert not (annotation_root / "B9999_S0001_0").exists()


@pytest.mark.asyncio
async def test_silver_still_says_review_required_after_gold_exists(
    app: FastAPI, annotation_root: Path
) -> None:
    async with reviewing(app) as client:
        await client.post(f"/api/annotations/{CONVERSATION}/gold", json=promotion_body())
        draft = (await client.get(f"/api/annotations/{CONVERSATION}")).json()

    record = json.loads((annotation_root / CONVERSATION / "silver.json").read_text("utf-8"))
    assert record["review_state"] == "review_required"
    assert record["promotable"] is False
    assert draft["state"]["review_state"] == "review_required"


@pytest.mark.asyncio
async def test_an_oversized_submission_is_bounded_before_it_is_interpreted(
    app: FastAPI,
) -> None:
    async with reviewing(app) as client:
        response = await client.post(
            f"/api/annotations/{CONVERSATION}/gold",
            json=promotion_body(turns=[silver_turn(transcript="가" * 4_001)]),
        )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_request"


@pytest.mark.asyncio
async def test_unknown_submission_fields_are_refused(app: FastAPI) -> None:
    async with reviewing(app) as client:
        response = await client.post(
            f"/api/annotations/{CONVERSATION}/gold",
            json=promotion_body(review_state="reviewed_gold"),
        )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_request"


# ----------------------------------------------------------------------- default wiring


def test_the_annotation_root_defaults_under_the_data_root(tmp_path: Path) -> None:
    settings = Settings(data_root=tmp_path, database_path=tmp_path / "db.sqlite3")

    assert default_annotation_root(settings) == tmp_path / "annotations"


def test_a_configured_annotation_root_is_used_verbatim(tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    settings = Settings(
        data_root=tmp_path,
        database_path=tmp_path / "db.sqlite3",
        annotation_root=elsewhere,
    )

    assert default_annotation_root(settings) == elsewhere


def test_the_annotation_audio_root_defaults_to_the_derived_review_collections(
    tmp_path: Path,
) -> None:
    """Review recordings use an explicitly allowlisted pair of children below this root."""

    settings = Settings(data_root=tmp_path, database_path=tmp_path / "db.sqlite3")

    assert default_annotation_audio_root(settings) == tmp_path / "derived"


def test_a_configured_annotation_audio_root_is_used_verbatim(tmp_path: Path) -> None:
    elsewhere = tmp_path / "recordings"
    settings = Settings(
        data_root=tmp_path,
        database_path=tmp_path / "db.sqlite3",
        annotation_audio_root=elsewhere,
    )

    assert default_annotation_audio_root(settings) == elsewhere
