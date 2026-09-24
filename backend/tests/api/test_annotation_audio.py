"""Tests for the capability-fenced route a reviewer listens through.

Playing audio is the one thing on the review screen that reads a file outside the
annotation root, so these tests are organised around what that must never become:

* It sits behind the same capability as the draft itself, and refuses before it reads.
* No request names a file. The conversation id is the only address, and one that could
  mean anything else is refused.
* It only ever serves audio the draft already vouches for, digest-checked, and it writes
  nothing while doing it.
* Nothing it returns carries transcript, and no refusal carries a path.
"""

from __future__ import annotations

import hashlib
import io
import json
import wave
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from pydantic import SecretStr

from voxdelta.annotation.audio import MAX_CLIP_SECONDS
from voxdelta.api.app import create_app
from voxdelta.audio.service import AudioService
from voxdelta.jobs.artifacts import ArtifactStore
from voxdelta.jobs.repository import JobRepository
from voxdelta.pipeline.runner import PipelineRunner

TEST_CAPABILITY_TOKEN = "annotation-audio-capability-token-entropy"
CONVERSATION = "A6000_S0005_0"
TRANSCRIPT = "환불 처리가 아직도 안 됐습니다"
FRAME_RATE = 16_000
DURATION_SECONDS = 10.0


def write_wav(path: Path, *, seconds: float = DURATION_SECONDS) -> str:
    frames = int(seconds * FRAME_RATE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(FRAME_RATE)
        handle.writeframes(b"\x00\x01" * frames)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def silver_turn(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "start": 0.0,
        "end": 2.0,
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
    input_sha256: str,
    turns: list[dict[str, Any]] | None = None,
) -> Path:
    """A silver artifact shaped like the Gemini run's, pointed at a given recording."""

    content = {
        "turns": turns
        if turns is not None
        else [
            silver_turn(),
            # Overlapping on purpose: two speakers talking over each other must not read
            # as a gap running backwards.
            silver_turn(start=1.5, end=3.0, speaker="SPEAKER_01", emotion="neutral"),
            silver_turn(start=6.0, end=8.0, emotion="uncertain", confidence=0.3),
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
            "input_sha256": input_sha256,
            "model": "gemini-3.7-flash",
            "prompt": "provisional first-pass annotation",
            "config": {"response_format": "structured-json"},
            "remote_audio_transmitted": True,
        },
        "remote_file": {"deleted": True, "detail": "deleted"},
        "call_counts": {"uploads": 1, "interactions": 1, "deletes": 1},
        "validation": {
            "mode": "salvaged",
            "dropped_turn_count": 31,
            "dropped_turns_by_rule": {"turn.exceeds_duration": 31},
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


def build_app(tmp_path: Path, annotation_root: Path, audio_root: Path) -> FastAPI:
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
        annotation_audio_root=audio_root,
    )


@asynccontextmanager
async def client_for(
    app: FastAPI, *, token: str | None = TEST_CAPABILITY_TOKEN
) -> AsyncIterator[httpx.AsyncClient]:
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
def audio_root(tmp_path: Path) -> Path:
    root = tmp_path / "recordings"
    root.mkdir()
    return root


@pytest.fixture
def app(tmp_path: Path, annotation_root: Path, audio_root: Path) -> FastAPI:
    digest = write_wav(audio_root / f"{CONVERSATION}.wav")
    write_silver_fixture(annotation_root, input_sha256=digest)
    return build_app(tmp_path, annotation_root, audio_root)


class TestAuthorization:
    @pytest.mark.parametrize(
        "path",
        [
            f"/api/annotations/{CONVERSATION}/audio",
            f"/api/annotations/{CONVERSATION}/audio/clip?start=0&end=1",
        ],
    )
    @pytest.mark.anyio
    async def test_audio_needs_the_same_capability_as_the_draft(
        self, app: FastAPI, path: str
    ) -> None:
        async with client_for(app, token=None) as client:
            response = await client.get(path)
        assert response.status_code == 401
        assert response.json()["detail"]["code"] == "capability_required"

    @pytest.mark.anyio
    async def test_a_wrong_capability_is_refused(self, app: FastAPI) -> None:
        async with client_for(app, token="x" * 48) as client:
            response = await client.get(f"/api/annotations/{CONVERSATION}/audio")
        assert response.status_code == 401

    @pytest.mark.anyio
    async def test_a_non_local_origin_is_refused_before_any_audio_is_read(
        self, app: FastAPI
    ) -> None:
        async with client_for(app) as client:
            response = await client.get(
                f"/api/annotations/{CONVERSATION}/audio",
                headers={"Origin": "https://elsewhere.example"},
            )
        assert response.status_code == 403
        assert response.json()["detail"]["code"] == "origin_not_allowed"


class TestOverview:
    @pytest.mark.anyio
    async def test_describes_the_recording_and_the_playable_turns(self, app: FastAPI) -> None:
        async with client_for(app) as client:
            response = await client.get(f"/api/annotations/{CONVERSATION}/audio")
        assert response.status_code == 200
        body = response.json()
        assert body["conversation_id"] == CONVERSATION
        assert body["duration_seconds"] == pytest.approx(DURATION_SECONDS)
        assert body["sample_rate"] == FRAME_RATE
        assert body["max_clip_seconds"] == MAX_CLIP_SECONDS
        assert [clip["position"] for clip in body["turn_clips"]] == [1, 2, 3]

    @pytest.mark.anyio
    async def test_gaps_are_the_complement_of_the_valid_turns(self, app: FastAPI) -> None:
        async with client_for(app) as client:
            body = (await client.get(f"/api/annotations/{CONVERSATION}/audio")).json()
        # Turns cover 0-3 (the first two overlap) and 6-8, in a 10 second recording.
        assert body["gaps"] == [
            {"start": 3.0, "end": 6.0},
            {"start": 8.0, "end": 10.0},
        ]

    @pytest.mark.anyio
    async def test_the_overview_carries_no_transcript(self, app: FastAPI) -> None:
        async with client_for(app) as client:
            response = await client.get(f"/api/annotations/{CONVERSATION}/audio")
        assert TRANSCRIPT not in response.text

    @pytest.mark.anyio
    async def test_a_turn_clip_is_clamped_to_audio_that_exists(
        self, tmp_path: Path, annotation_root: Path, audio_root: Path
    ) -> None:
        digest = write_wav(audio_root / f"{CONVERSATION}.wav")
        write_silver_fixture(
            annotation_root,
            input_sha256=digest,
            turns=[silver_turn(start=9.0, end=14.0)],
        )
        app = build_app(tmp_path, annotation_root, audio_root)
        async with client_for(app) as client:
            body = (await client.get(f"/api/annotations/{CONVERSATION}/audio")).json()
        assert body["turn_clips"] == [{"position": 1, "start": 9.0, "end": 10.0}]


class TestSourceResolution:
    @pytest.mark.anyio
    async def test_a_draft_without_its_recording_refuses_playback_only(
        self, tmp_path: Path, annotation_root: Path, audio_root: Path
    ) -> None:
        write_silver_fixture(annotation_root, input_sha256="0" * 64)
        app = build_app(tmp_path, annotation_root, audio_root)
        async with client_for(app) as client:
            audio = await client.get(f"/api/annotations/{CONVERSATION}/audio")
            draft = await client.get(f"/api/annotations/{CONVERSATION}")
        assert audio.status_code == 409
        assert audio.json()["detail"]["code"] == "audio_source_unavailable"
        # The draft itself stays reviewable; only listening is unavailable.
        assert draft.status_code == 200

    @pytest.mark.anyio
    async def test_a_recording_swapped_since_annotation_is_refused(
        self, tmp_path: Path, annotation_root: Path, audio_root: Path
    ) -> None:
        write_wav(audio_root / f"{CONVERSATION}.wav")
        write_silver_fixture(annotation_root, input_sha256="a" * 64)
        app = build_app(tmp_path, annotation_root, audio_root)
        async with client_for(app) as client:
            response = await client.get(f"/api/annotations/{CONVERSATION}/audio")
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "audio_source_mismatch"

    @pytest.mark.anyio
    async def test_audio_without_a_silver_draft_is_never_served(
        self, tmp_path: Path, annotation_root: Path, audio_root: Path
    ) -> None:
        # A recording sitting in the audio root that no draft describes is not reachable.
        write_wav(audio_root / "A0051_S0001_0.wav")
        app = build_app(tmp_path, annotation_root, audio_root)
        async with client_for(app) as client:
            overview = await client.get("/api/annotations/A0051_S0001_0/audio")
            clip = await client.get("/api/annotations/A0051_S0001_0/audio/clip?start=0&end=1")
        assert overview.status_code == 404
        assert overview.json()["detail"]["code"] == "annotation_not_found"
        assert clip.status_code == 404

    @pytest.mark.parametrize(
        "conversation_id",
        ["..%2F..%2Fetc%2Fpasswd", "..", "%2Fetc%2Fpasswd", "a%00b", "with%20space"],
    )
    @pytest.mark.anyio
    async def test_an_id_that_could_name_another_file_is_refused(
        self, app: FastAPI, conversation_id: str
    ) -> None:
        async with client_for(app) as client:
            response = await client.get(f"/api/annotations/{conversation_id}/audio")
        assert response.status_code in {404, 422}
        if response.status_code == 422:
            assert response.json()["detail"]["code"] == "invalid_conversation_id"

    @pytest.mark.anyio
    async def test_no_refusal_names_a_filesystem_path(
        self, tmp_path: Path, annotation_root: Path, audio_root: Path
    ) -> None:
        write_silver_fixture(annotation_root, input_sha256="0" * 64)
        app = build_app(tmp_path, annotation_root, audio_root)
        async with client_for(app) as client:
            responses = [
                await client.get(f"/api/annotations/{CONVERSATION}/audio"),
                await client.get(f"/api/annotations/{CONVERSATION}/audio/clip?start=0&end=1"),
                await client.get("/api/annotations/..%2F..%2Fetc/audio"),
            ]
        for response in responses:
            assert str(tmp_path) not in response.text
            assert str(audio_root) not in response.text


class TestClip:
    @pytest.mark.anyio
    async def test_serves_a_playable_wav_of_exactly_the_requested_range(
        self, app: FastAPI
    ) -> None:
        async with client_for(app) as client:
            response = await client.get(
                f"/api/annotations/{CONVERSATION}/audio/clip", params={"start": 1.0, "end": 3.0}
            )
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.headers["cache-control"] == "no-store"
        assert int(response.headers["content-length"]) == len(response.content)
        with wave.open(io.BytesIO(response.content), "rb") as clip:
            assert clip.getframerate() == FRAME_RATE
            assert clip.getnframes() == 2 * FRAME_RATE

    @pytest.mark.anyio
    async def test_a_gap_range_plays_the_same_way_a_turn_does(self, app: FastAPI) -> None:
        async with client_for(app) as client:
            overview = (await client.get(f"/api/annotations/{CONVERSATION}/audio")).json()
            gap = overview["gaps"][0]
            response = await client.get(
                f"/api/annotations/{CONVERSATION}/audio/clip",
                params={"start": gap["start"], "end": gap["end"]},
            )
        assert response.status_code == 200
        with wave.open(io.BytesIO(response.content), "rb") as clip:
            assert clip.getnframes() == 3 * FRAME_RATE

    @pytest.mark.parametrize(
        ("start", "end", "code"),
        [
            (-1.0, 2.0, "invalid_clip_range"),
            (2.0, 2.0, "invalid_clip_range"),
            (3.0, 1.0, "invalid_clip_range"),
            (11.0, 12.0, "invalid_clip_range"),
            (0.0, MAX_CLIP_SECONDS + 1, "clip_too_long"),
        ],
    )
    @pytest.mark.anyio
    async def test_an_unusable_range_is_refused_with_a_code(
        self, app: FastAPI, start: float, end: float, code: str
    ) -> None:
        async with client_for(app) as client:
            response = await client.get(
                f"/api/annotations/{CONVERSATION}/audio/clip",
                params={"start": start, "end": end},
            )
        assert response.status_code == 422
        assert response.json()["detail"]["code"] == code

    @pytest.mark.parametrize("params", [{}, {"start": 0.0}, {"start": "x", "end": "y"}])
    @pytest.mark.anyio
    async def test_a_range_that_is_not_two_numbers_is_refused(
        self, app: FastAPI, params: dict[str, object]
    ) -> None:
        async with client_for(app) as client:
            response = await client.get(
                f"/api/annotations/{CONVERSATION}/audio/clip", params=params
            )
        assert response.status_code == 422

    @pytest.mark.anyio
    async def test_a_range_past_the_end_is_clamped_rather_than_refused(
        self, app: FastAPI
    ) -> None:
        async with client_for(app) as client:
            response = await client.get(
                f"/api/annotations/{CONVERSATION}/audio/clip", params={"start": 9.0, "end": 12.0}
            )
        assert response.status_code == 200
        with wave.open(io.BytesIO(response.content), "rb") as clip:
            assert clip.getnframes() == FRAME_RATE

    @pytest.mark.anyio
    async def test_the_clip_carries_no_transcript(self, app: FastAPI) -> None:
        async with client_for(app) as client:
            response = await client.get(
                f"/api/annotations/{CONVERSATION}/audio/clip", params={"start": 0.0, "end": 2.0}
            )
        assert TRANSCRIPT.encode("utf-8") not in response.content

    @pytest.mark.anyio
    async def test_serving_a_clip_writes_nothing(
        self, app: FastAPI, audio_root: Path, annotation_root: Path
    ) -> None:
        source = audio_root / f"{CONVERSATION}.wav"
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        silver = annotation_root / CONVERSATION / "silver.json"
        silver_before = hashlib.sha256(silver.read_bytes()).hexdigest()

        async with client_for(app) as client:
            await client.get(
                f"/api/annotations/{CONVERSATION}/audio/clip", params={"start": 0.0, "end": 2.0}
            )

        assert hashlib.sha256(source.read_bytes()).hexdigest() == before
        assert hashlib.sha256(silver.read_bytes()).hexdigest() == silver_before
        assert [entry.name for entry in audio_root.iterdir()] == [f"{CONVERSATION}.wav"]
        assert sorted(entry.name for entry in (annotation_root / CONVERSATION).iterdir()) == [
            "silver.json"
        ]

    @pytest.mark.anyio
    async def test_listening_does_not_make_a_draft_promotable(self, app: FastAPI) -> None:
        async with client_for(app) as client:
            await client.get(
                f"/api/annotations/{CONVERSATION}/audio/clip", params={"start": 0.0, "end": 2.0}
            )
            draft = (await client.get(f"/api/annotations/{CONVERSATION}")).json()
        assert draft["state"]["promotable"] is False
        assert draft["state"]["review_state"] == "review_required"
        assert draft["state"]["gold_present"] is False
