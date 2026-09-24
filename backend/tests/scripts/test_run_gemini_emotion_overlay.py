"""Tests for the one-shot Gemini emotion-overlay generation script.

The script is the only thing in the repository that can cause audio to be sent to Google
for an emotion overlay, so what is tested here is the gates in front of that, not the
formatting of its output: preflight transmits nothing, the Gemini-specific consent is
required by name, and a run that gets past both makes exactly one upload, one interaction,
and one delete with no retry.

No test reaches the network. ``httpx.Client`` is replaced by a stub throughout.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from voxdelta.credentials import Credentials

_SPEC = importlib.util.spec_from_file_location(
    "run_gemini_emotion_overlay",
    Path(__file__).resolve().parents[2] / "scripts" / "run_gemini_emotion_overlay.py",
)
assert _SPEC and _SPEC.loader
script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(script)

CONVERSATION = "A6000_S0005_0"
TRANSCRIPT = "환불 처리가 아직 안 됐습니다"


def write_silver(root: Path, *, turn_count: int = 3) -> dict[str, Any]:
    turns = [
        {
            "start": float(index) * 2.0,
            "end": float(index) * 2.0 + 2.0,
            "speaker": "G5999" if index % 2 == 0 else "G6000",
            "transcript": TRANSCRIPT,
            "emotion": "uncertain",
            "emotion_rationale": "reviewer judgment required",
            "confidence": 0.0,
        }
        for index in range(turn_count)
    ]
    content = {"turns": turns, "speakers": ["G5999", "G6000"], "notes": ""}
    digest = hashlib.sha256(
        json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    record = {
        "schema_version": "1",
        "kind": "silver-annotation",
        "review_state": "review_required",
        "promotable": False,
        "conversation_id": CONVERSATION,
        "source": {"input_sha256": "0" * 64, "model": "kcsc-human-reference"},
        "content": content,
        "content_sha256": digest,
        "contains_transcript": True,
    }
    directory = root / CONVERSATION
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "silver.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


class StubResponse:
    def __init__(self, status: int, body: Any = None, headers: dict[str, str] | None = None):
        self.status_code = status
        self._body = body if body is not None else {}
        self.headers = headers or {}
        self.text = json.dumps(self._body)

    def json(self) -> Any:
        return self._body


class StubClient:
    calls: list[tuple[str, str]] = []
    rows: dict[str, Any] = {}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> StubClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Any = None,
        json: Any = None,
        content: bytes | None = None,
    ) -> StubResponse:
        type(self).calls.append((method, url))
        if method == "DELETE":
            return StubResponse(200)
        if url.endswith("/upload/v1beta/files"):
            return StubResponse(200, {}, {"x-goog-upload-url": "https://upload.example/session"})
        if url == "https://upload.example/session":
            return StubResponse(200, {"file": {"uri": "files/abc", "name": "files/abc"}})
        if url.endswith("/v1beta/interactions"):
            import json as _json

            return StubResponse(200, {"output_text": _json.dumps(type(self).rows)})
        raise AssertionError(f"unexpected request: {method} {url}")


@pytest.fixture
def bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A pinned-audio, clean-Silver bench with the network and credentials stubbed out."""

    audio = tmp_path / "conversation.wav"
    audio.write_bytes(b"RIFF-fixture-bytes")
    monkeypatch.setattr(script, "AUDIO_SHA256", hashlib.sha256(audio.read_bytes()).hexdigest())
    root = tmp_path / "annotations"
    record = write_silver(root, turn_count=3)
    monkeypatch.setattr(
        script, "load_credentials", lambda *a, **k: Credentials(GEMINI_API_KEY=SecretStr("key"))
    )
    StubClient.calls = []
    StubClient.rows = {
        "rows": [
            {
                "position": position,
                "emotion": "neutral",
                "confidence": 0.5,
                "rationale": "flat prosody",
            }
            for position in (1, 2, 3)
        ]
    }
    monkeypatch.setattr(script.httpx, "Client", StubClient)
    return {
        "audio": audio,
        "root": root,
        "state": tmp_path / "state.json",
        "record": record,
        "silver": root / CONVERSATION / "silver.json",
    }


def run(bench: dict[str, Any], *extra: str) -> int:
    return script.main(
        [
            "--audio",
            str(bench["audio"]),
            "--root",
            str(bench["root"]),
            "--state",
            str(bench["state"]),
            *extra,
        ]
    )


def test_preflight_alone_transmits_nothing(bench: dict[str, Any], capsys: Any) -> None:
    before = bench["silver"].read_bytes()

    assert run(bench) == 0

    output = capsys.readouterr().out
    assert "stopping after preflight: nothing was transmitted." in output
    assert StubClient.calls == []
    assert not bench["state"].exists()
    assert bench["silver"].read_bytes() == before
    # The preflight is where an operator decides. It must show what will be sent, and no
    # transcript and no key.
    assert "audio, text" in output
    assert "key" not in output.replace("api key             : configured", "")


def test_consent_for_another_provider_does_not_extend_to_gemini(bench: dict[str, Any]) -> None:
    assert run(bench, "--confirm-external-upload") == 2
    assert StubClient.calls == []

    # argparse restricts --consent to the one accepted value; anything else is refused
    # before any credential is read.
    assert run(bench, "--confirm-external-upload", "--consent", "pyannoteai_audio_transfer") == 2
    assert StubClient.calls == []


def test_an_approved_run_makes_exactly_one_of_each_call(bench: dict[str, Any]) -> None:
    before = bench["silver"].read_bytes()

    assert run(bench, "--confirm-external-upload", "--consent", "gemini_audio_transfer") == 0

    methods = [method for method, _url in StubClient.calls]
    assert methods.count("DELETE") == 1
    assert len(StubClient.calls) == 4  # upload start, upload bytes, interaction, delete
    state = json.loads(bench["state"].read_text(encoding="utf-8"))
    assert state["state"] == "completed"
    assert state["call_counts"] == {"uploads": 1, "interactions": 1, "deletes": 1}
    assert state["remote_file_deleted"] is True
    assert state["retry_attempted"] is False
    assert state["consent"] == "gemini_audio_transfer"
    assert state["remote_audio_transmitted"] is True
    assert state["review_state"] == "review_required"
    assert state["promotable"] is False
    assert state["source_silver_content_sha256"] == bench["record"]["content_sha256"]
    assert bench["silver"].read_bytes() == before


def test_the_state_file_records_no_transcript_or_secret(bench: dict[str, Any]) -> None:
    run(bench, "--confirm-external-upload", "--consent", "gemini_audio_transfer")

    raw = bench["state"].read_text(encoding="utf-8")
    assert TRANSCRIPT not in raw
    assert "key" not in json.loads(raw).values()
    assert "flat prosody" not in raw


def test_the_written_candidate_is_bound_review_required_and_preserves_the_draft(
    bench: dict[str, Any],
) -> None:
    run(bench, "--confirm-external-upload", "--consent", "gemini_audio_transfer")

    path = bench["root"] / CONVERSATION / "emotion-candidate.gemini-remote.json"
    candidate = json.loads(path.read_text(encoding="utf-8"))
    assert candidate["review_state"] == "review_required"
    assert candidate["promotable"] is False
    assert candidate["source"]["remote_audio_transmitted"] is True
    assert candidate["source"]["silver_content_sha256"] == bench["record"]["content_sha256"]
    for source, overlaid in zip(
        bench["record"]["content"]["turns"], candidate["content"]["turns"], strict=True
    ):
        assert (source["start"], source["end"]) == (overlaid["start"], overlaid["end"])
        assert source["speaker"] == overlaid["speaker"]
        assert source["transcript"] == overlaid["transcript"]
        assert overlaid["emotion"] == "neutral"


def test_a_second_run_is_refused_before_anything_is_transmitted(bench: dict[str, Any]) -> None:
    assert run(bench, "--confirm-external-upload", "--consent", "gemini_audio_transfer") == 0
    first = list(StubClient.calls)

    assert run(bench, "--confirm-external-upload", "--consent", "gemini_audio_transfer") == 2

    assert StubClient.calls == first, "the second run must not reach the network at all"


def test_a_broken_positional_response_is_not_retried_and_the_upload_is_deleted(
    bench: dict[str, Any],
) -> None:
    StubClient.rows = {"rows": StubClient.rows["rows"][:2]}

    assert run(bench, "--confirm-external-upload", "--consent", "gemini_audio_transfer") == 2

    methods = [method for method, _url in StubClient.calls]
    assert methods.count("DELETE") == 1
    assert StubClient.calls[-1][0] == "DELETE"
    state = json.loads(bench["state"].read_text(encoding="utf-8"))
    assert state["state"] == "failed"
    assert state["reason"] == "GeminiEmotionOverlayError"
    assert state["retry_attempted"] is False
    assert state["remote_file_deleted"] is True
    assert state["call_counts"] == {"uploads": 1, "interactions": 1, "deletes": 1}
    assert not (bench["root"] / CONVERSATION / "emotion-candidate.gemini-remote.json").exists()


def test_unpinned_audio_stops_before_credentials_are_read(
    bench: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(script, "AUDIO_SHA256", "f" * 64)

    assert run(bench, "--confirm-external-upload", "--consent", "gemini_audio_transfer") == 2
    assert StubClient.calls == []
    assert not bench["state"].exists()


def test_a_missing_silver_draft_stops_the_run(
    bench: dict[str, Any],
) -> None:
    bench["silver"].unlink()

    assert run(bench, "--confirm-external-upload", "--consent", "gemini_audio_transfer") == 2
    assert StubClient.calls == []
