"""Tests for the Precision-2 comparative benchmark.

Every test runs against a fake transport. The budget, the endpoint allowlist, and the
fail-closed paths are exactly the properties that must hold before any real audio is
transmitted, so they are the properties that must be testable without transmitting any.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from voxdelta.domain.models import AudioAsset, SpeakerSegment
from voxdelta.evaluation.kcsc_diarization_benchmark import (
    KcscBenchmarkError,
    file_sha256,
    run_benchmark,
)
from voxdelta.evaluation.kcsc_precision_benchmark import (
    AUTHORIZATION_SCOPE,
    CORRECTION_NOTE,
    EXTERNAL_PROCESSING_CAVEAT,
    FROZEN_CONVERSATIONS,
    MAX_DIARIZATION_JOBS,
    RIGHTS_HOLDER,
    CallLedger,
    LedgerClient,
    PrecisionBudgetExceeded,
    UnexpectedRemoteCall,
    amend_authorization,
    preflight,
    run_precision_benchmark,
    write_evaluation_record,
    write_precision_report,
)

API = "https://api.pyannote.ai"
UPLOAD = "https://storage.example.invalid/presigned/abc"
TRANSCRIPT = "이것은 절대 아티팩트에 나오면 안 되는 전사입니다"
REVISION = "364fb908b14b9e8383ef6bf9f7ebf5088ffddaf3"


class FakeResponse:
    def __init__(self, status_code: int, body: object) -> None:
        self.status_code = status_code
        self._body = body

    def json(self) -> Any:
        return self._body


class FakeTransport:
    """Answers the four endpoints the workflow uses, and records nothing else."""

    def __init__(self, *, statuses: list[str] | None = None) -> None:
        self.statuses = statuses or ["succeeded"]
        self.requests: list[tuple[str, str]] = []
        self.jobs = 0
        self.closed = False

    def request(
        self,
        method: str,
        url: str,
        *,
        json: Any = None,
        content: bytes | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> FakeResponse:
        self.requests.append((method, url))
        if url.endswith("/v1/media/input"):
            return FakeResponse(200, {"url": UPLOAD})
        if url.startswith(UPLOAD):
            return FakeResponse(200, {})
        if url.endswith("/v1/diarize"):
            self.jobs += 1
            return FakeResponse(200, {"jobId": f"job-{self.jobs}"})
        if "/v1/jobs/" in url:
            status = self.statuses[min(self.jobs - 1, len(self.statuses) - 1)]
            if status != "succeeded":
                return FakeResponse(200, {"status": status})
            return FakeResponse(
                200,
                {
                    "status": "succeeded",
                    "output": {
                        "diarization": [
                            {"start": 0.0, "end": 2.0, "speaker": "A"},
                            {"start": 2.0, "end": 4.0, "speaker": "B"},
                            {"start": 5.0, "end": 7.0, "speaker": "A"},
                            {"start": 7.0, "end": 9.0, "speaker": "B"},
                        ],
                        "exclusiveDiarization": [
                            {"start": 0.0, "end": 2.0, "speaker": "A"},
                            {"start": 2.0, "end": 4.0, "speaker": "B"},
                            {"start": 5.0, "end": 7.0, "speaker": "A"},
                            {"start": 7.0, "end": 9.0, "speaker": "B"},
                        ],
                    },
                },
            )
        raise AssertionError(f"unexpected url {url}")

    def close(self) -> None:
        self.closed = True


def make_client(ledger: CallLedger, transport: FakeTransport) -> LedgerClient:
    return LedgerClient(transport, ledger)  # type: ignore[arg-type]


@pytest.fixture
def derived(tmp_path: Path) -> Path:
    """A derived set holding all three frozen conversations."""

    root = tmp_path / "derived"
    (root / "audio").mkdir(parents=True)
    (root / "reference").mkdir(parents=True)
    conversations = []
    for index, cid in enumerate(FROZEN_CONVERSATIONS):
        audio = root / "audio" / f"{cid}.wav"
        audio.write_bytes(b"RIFF" + bytes([index]) * 64)
        reference = root / "reference" / f"{cid}.json"
        reference.write_text(
            json.dumps(
                {
                    "schema_version": "1",
                    "conversation_id": cid,
                    "source_revision": REVISION,
                    "sample_rate": 16000,
                    "duration_seconds": 10.0,
                    "trim_offset_seconds": 3.0,
                    "speakers": ["G0001", "G0002"],
                    "vendor_marker_intervals_silenced": [],
                    "turns": [
                        {"start": 0.0, "end": 2.0, "speaker": "G0001", "transcript": TRANSCRIPT},
                        {"start": 2.0, "end": 4.0, "speaker": "G0002", "transcript": TRANSCRIPT},
                        {"start": 5.0, "end": 7.0, "speaker": "G0001", "transcript": TRANSCRIPT},
                        {"start": 7.0, "end": 9.0, "speaker": "G0002", "transcript": TRANSCRIPT},
                    ],
                    "unattributed_events": [],
                }
            ),
            encoding="utf-8",
        )
        conversations.append(
            {
                "conversation_id": cid,
                "speakers": ["G0001", "G0002"],
                "derived_duration_seconds": 10.0,
                "outputs": {
                    "audio": f"audio/{cid}.wav",
                    "audio_sha256": file_sha256(audio),
                    "reference": f"reference/{cid}.json",
                    "reference_sha256": file_sha256(reference),
                },
            }
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "source": {"revision": REVISION},
                "conversations": conversations,
            }
        ),
        encoding="utf-8",
    )
    return root


class StubDiarizer:
    def __init__(self) -> None:
        self.calls = 0

    def diarize(self, asset: AudioAsset) -> list[SpeakerSegment]:
        self.calls += 1
        return [
            SpeakerSegment(start=0.0, end=2.0, speaker_id="A", confidence=1.0),
            SpeakerSegment(start=2.0, end=4.0, speaker_id="B", confidence=1.0),
            SpeakerSegment(start=5.0, end=7.0, speaker_id="A", confidence=1.0),
            SpeakerSegment(start=7.0, end=9.0, speaker_id="B", confidence=1.0),
        ]


def test_the_conversation_set_is_frozen_in_code() -> None:
    assert FROZEN_CONVERSATIONS == ("A0051_S0001_0", "A0055_S0006_0", "A6000_S0005_0")
    assert MAX_DIARIZATION_JOBS == 3


def test_preflight_verifies_every_checksum_and_plans_three_jobs(derived: Path) -> None:
    summary = preflight(derived)

    assert summary["planned_diarization_jobs"] == 3
    assert summary["source_revision"] == REVISION
    conversations = summary["conversations"]
    assert isinstance(conversations, list)
    assert [c["conversation_id"] for c in conversations] == list(FROZEN_CONVERSATIONS)


def test_preflight_rejects_a_tampered_third_file(derived: Path) -> None:
    """A corrupt third file must stop the run before the first byte leaves the host."""

    (derived / "audio" / f"{FROZEN_CONVERSATIONS[2]}.wav").write_bytes(b"tampered")

    with pytest.raises(KcscBenchmarkError, match="audio checksum mismatch"):
        preflight(derived)


def test_a_narrowed_or_widened_selection_is_refused(derived: Path) -> None:
    with pytest.raises(KcscBenchmarkError, match="does not match the approved scope"):
        run_precision_benchmark(
            derived_root=derived,
            diarizer=StubDiarizer(),
            conversation_ids=FROZEN_CONVERSATIONS[:2],
        )


def test_the_ledger_refuses_a_fourth_diarization_job() -> None:
    ledger = CallLedger()
    for _ in range(MAX_DIARIZATION_JOBS):
        ledger.reserve("diarize")

    with pytest.raises(PrecisionBudgetExceeded, match="ceiling is 3"):
        ledger.reserve("diarize")


def test_the_ledger_refuses_a_fourth_upload() -> None:
    ledger = CallLedger()
    for _ in range(MAX_DIARIZATION_JOBS):
        ledger.reserve("upload")

    with pytest.raises(PrecisionBudgetExceeded, match="refused upload 4"):
        ledger.reserve("upload")


def test_the_ledger_refuses_an_endpoint_outside_the_workflow() -> None:
    ledger = CallLedger()

    with pytest.raises(UnexpectedRemoteCall, match="refused unexpected remote call"):
        ledger.classify("POST", "https://api.pyannote.ai/v1/transcribe")


def test_the_ledger_refuses_an_upload_to_a_host_the_api_never_named() -> None:
    ledger = CallLedger()

    with pytest.raises(UnexpectedRemoteCall):
        ledger.classify("PUT", "https://attacker.example.invalid/x")


def test_an_upload_host_is_allowed_only_after_the_api_returns_it() -> None:
    ledger = CallLedger()
    ledger.allow_upload_host(UPLOAD)

    assert ledger.classify("PUT", UPLOAD) == "upload"


def test_the_ledger_classifies_each_workflow_endpoint() -> None:
    ledger = CallLedger()

    assert ledger.classify("POST", f"{API}/v1/media/input") == "media_input"
    assert ledger.classify("POST", f"{API}/v1/diarize") == "diarize"
    assert ledger.classify("GET", f"{API}/v1/jobs/abc") == "job_poll"


def test_the_ledger_records_job_ids_and_statuses_without_bodies() -> None:
    ledger = CallLedger()
    transport = FakeTransport()
    client = make_client(ledger, transport)

    client.request("POST", f"{API}/v1/media/input")
    client.request("PUT", UPLOAD, content=b"audio")
    client.request("POST", f"{API}/v1/diarize")
    client.request("GET", f"{API}/v1/jobs/job-1")

    assert ledger.job_ids == ["job-1"]
    assert ledger.job_statuses == ["succeeded"]
    assert ledger.submissions == 1
    assert ledger.uploads == 1
    assert {call.kind for call in ledger.calls} == {
        "media_input",
        "upload",
        "diarize",
        "job_poll",
    }
    # Nothing on a recorded call can carry a payload.
    assert all(
        set(vars(call) if hasattr(call, "__dict__") else call.__slots__)
        <= {"kind", "method", "host", "status_code", "elapsed_seconds"}
        for call in ledger.calls
    )


def test_the_public_report_is_transcript_free_and_carries_no_job_ids(
    derived: Path, tmp_path: Path
) -> None:
    ledger = CallLedger()
    ledger.reserve("diarize")
    ledger.note_job("job-secret-1")
    report = run_precision_benchmark(derived_root=derived, diarizer=StubDiarizer())
    path = tmp_path / "report.json"

    write_precision_report(report, path, ledger, authorization="user_limited_override")

    raw = path.read_text(encoding="utf-8")
    assert TRANSCRIPT not in raw
    assert "job-secret-1" not in raw
    payload = json.loads(raw)
    assert payload["model"]["remote"] is True
    assert payload["model"]["name"] == "precision-2"
    assert payload["transcript_free"] is True
    assert payload["processing"]["location"] == "remote"
    assert payload["processing"]["transcription_requested"] is False
    assert payload["processing"]["caveat"] == EXTERNAL_PROCESSING_CAVEAT
    assert payload["dataset"]["source_revision"] == REVISION


def test_the_public_report_is_field_compatible_with_the_local_benchmark(
    derived: Path, tmp_path: Path
) -> None:
    """A comparison is only mechanical if both reports carry the same keys."""

    from voxdelta.evaluation.kcsc_diarization_benchmark import write_report

    local = run_benchmark(
        derived_root=derived,
        conversation_ids=list(FROZEN_CONVERSATIONS),
        diarizer=StubDiarizer(),
        model_name="speaker-diarization-community-1",
        model_tree_sha256="a" * 64,
    )
    remote = run_precision_benchmark(derived_root=derived, diarizer=StubDiarizer())
    local_path, remote_path = tmp_path / "local.json", tmp_path / "remote.json"

    write_report(local, local_path)
    write_precision_report(remote, remote_path, CallLedger(), authorization="user_limited_override")

    local_payload = json.loads(local_path.read_text(encoding="utf-8"))
    remote_payload = json.loads(remote_path.read_text(encoding="utf-8"))
    assert local_payload.keys() == remote_payload.keys()
    assert local_payload["model"].keys() == remote_payload["model"].keys()
    assert local_payload["aggregate"].keys() == remote_payload["aggregate"].keys()
    for variant in ("collar_250ms", "strict"):
        assert (
            local_payload["aggregate"][variant].keys()
            == remote_payload["aggregate"][variant].keys()
        )
    assert (
        local_payload["conversations"][0]["metrics"]["strict"].keys()
        == remote_payload["conversations"][0]["metrics"]["strict"].keys()
    )


def test_the_local_record_keeps_job_ids_and_declares_the_licensed_upload(
    derived: Path, tmp_path: Path
) -> None:
    ledger = CallLedger()
    ledger.reserve("diarize")
    ledger.note_job("job-1")
    ledger.note_status("succeeded")
    report = run_precision_benchmark(derived_root=derived, diarizer=StubDiarizer())
    path = tmp_path / "EVALUATION.json"

    write_evaluation_record(
        path,
        report=report,
        ledger=ledger,
        preflight_summary=preflight(derived),
        started_at="2026-09-01T00:00:00+00:00",
        total_elapsed_seconds=12.5,
        authorization="user_limited_override",
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["remote_jobs"]["job_ids"] == ["job-1"]
    assert payload["remote_jobs"]["terminal_statuses"] == ["succeeded"]
    assert payload["licensed_or_private_audio_transmitted"] is True
    assert payload["budget"]["max_diarization_jobs"] == 3
    assert payload["budget"]["retries"] == 0
    assert payload["provider"]["request"]["transcription"] is False
    assert payload["timing"]["total_elapsed_seconds"] == 12.5
    # The operational record must not carry content either.
    assert TRANSCRIPT not in path.read_text(encoding="utf-8")


def test_the_record_is_deterministic(derived: Path, tmp_path: Path) -> None:
    report = run_precision_benchmark(derived_root=derived, diarizer=StubDiarizer())
    summary = preflight(derived)
    digests = []
    for name in ("one.json", "two.json"):
        digests.append(
            write_evaluation_record(
                tmp_path / name,
                report=report,
                ledger=CallLedger(),
                preflight_summary=summary,
                started_at="2026-09-01T00:00:00+00:00",
                total_elapsed_seconds=1.0,
                authorization="user_limited_override",
            )
        )

    assert digests[0] == digests[1]


def _provider(ledger: CallLedger, transport: FakeTransport) -> Any:
    """The real provider, wired to the fake transport through the ledger."""

    from pydantic import SecretStr

    from voxdelta.credentials import Credentials
    from voxdelta.providers.pyannote_precision import PyannotePrecisionProvider

    credentials = Credentials(PYANNOTEAI_API_KEY=SecretStr("test-key"))  # type: ignore[call-arg]
    return PyannotePrecisionProvider(
        credentials,
        client_factory=lambda *, timeout_seconds: make_client(ledger, transport),
        poll_interval_seconds=0.0,
        sleep=lambda _seconds: None,
    )


def test_a_full_run_submits_exactly_three_jobs_through_the_real_provider(derived: Path) -> None:
    ledger = CallLedger()
    transport = FakeTransport()

    report = run_precision_benchmark(derived_root=derived, diarizer=_provider(ledger, transport))

    assert ledger.submissions == MAX_DIARIZATION_JOBS
    assert ledger.uploads == MAX_DIARIZATION_JOBS
    assert ledger.job_ids == ["job-1", "job-2", "job-3"]
    assert ledger.counts()["diarize"] == 3
    assert report.conversation_count == 3
    assert report.remote is True
    assert report.model_tree_sha256 is None
    # Only the API host and the presigned upload host were ever contacted.
    assert set(ledger.hosts) == {"api.pyannote.ai", "storage.example.invalid"}


def test_a_failed_job_stops_the_run_without_submitting_the_rest(derived: Path) -> None:
    """No retry, and no quiet continuation onto the remaining conversations."""

    from voxdelta.providers.base import ProviderError

    ledger = CallLedger()
    transport = FakeTransport(statuses=["failed"])

    with pytest.raises(ProviderError):
        run_precision_benchmark(derived_root=derived, diarizer=_provider(ledger, transport))

    assert ledger.submissions == 1
    assert transport.jobs == 1
    assert ledger.counts().get("diarize") == 1


def test_a_provider_returning_one_speaker_fails_closed(derived: Path) -> None:
    from voxdelta.providers.base import ProviderError

    class OneSpeaker(FakeTransport):
        def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
            response = super().request(method, url, **kwargs)
            body = response.json()
            if isinstance(body, dict) and body.get("status") == "succeeded":
                for key in ("diarization", "exclusiveDiarization"):
                    for turn in body["output"][key]:
                        turn["speaker"] = "A"
            return response

    ledger = CallLedger()

    with pytest.raises(ProviderError) as raised:
        run_precision_benchmark(derived_root=derived, diarizer=_provider(ledger, OneSpeaker()))
    assert raised.value.code == "unsupported_speaker_count"
    assert ledger.submissions == 1


def test_no_artifact_can_claim_confirmed_rights(derived: Path, tmp_path: Path) -> None:
    """The only authorization this workflow can express is the limited override."""

    report = run_precision_benchmark(derived_root=derived, diarizer=StubDiarizer())
    report_path, record_path = tmp_path / "r.json", tmp_path / "e.json"

    write_precision_report(report, report_path, CallLedger(), authorization="user_limited_override")
    write_evaluation_record(
        record_path,
        report=report,
        ledger=CallLedger(),
        preflight_summary=preflight(derived),
        started_at="2026-09-01T00:00:00+00:00",
        total_elapsed_seconds=1.0,
        authorization="user_limited_override",
    )

    for path in (report_path, record_path):
        raw = path.read_text(encoding="utf-8")
        assert "third_party_processing_confirmed" not in raw
        assert '"rights_confirmed": false' in raw
    public = json.loads(report_path.read_text(encoding="utf-8"))["processing"]
    record = json.loads(record_path.read_text(encoding="utf-8"))["rights"]
    for block in (public, record):
        assert block["external_processing_authorization"] == "user_limited_override"
        assert block["rights_confirmed"] is False
        assert block["third_party_processing_rights"] == "unknown"
        assert block["rights_holder"] == RIGHTS_HOLDER
        assert block["authorization_scope"] == AUTHORIZATION_SCOPE


def test_the_caveat_states_the_licensing_uncertainty_and_the_retention_window() -> None:
    assert "have NOT been" in EXTERNAL_PROCESSING_CAVEAT
    assert "48 hours" in EXTERNAL_PROCESSING_CAVEAT
    assert "no deletion endpoint" in EXTERNAL_PROCESSING_CAVEAT
    assert "limited operator override" in EXTERNAL_PROCESSING_CAVEAT
    assert RIGHTS_HOLDER in EXTERNAL_PROCESSING_CAVEAT


def test_amending_an_artifact_corrects_authorization_without_touching_measurements(
    derived: Path, tmp_path: Path
) -> None:
    """A correction must not change a single measured number."""

    report = run_precision_benchmark(derived_root=derived, diarizer=StubDiarizer())
    path = tmp_path / "r.json"
    write_precision_report(report, path, CallLedger(), authorization="user_limited_override")
    # Simulate the artifact as it was originally written, claiming confirmed rights.
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["processing"].pop("external_processing_authorization")
    payload["processing"]["third_party_processing_confirmed"] = True
    before_metrics = json.loads(json.dumps(payload["conversations"]))
    before_aggregate = json.loads(json.dumps(payload["aggregate"]))
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))

    amend_authorization(path, authorization="user_limited_override")

    after = json.loads(path.read_text(encoding="utf-8"))
    assert after["conversations"] == before_metrics
    assert after["aggregate"] == before_aggregate
    assert "third_party_processing_confirmed" not in json.dumps(after)
    assert after["processing"]["rights_confirmed"] is False
    assert after["processing"]["external_processing_authorization"] == "user_limited_override"
    assert after["correction"] == CORRECTION_NOTE


def test_amending_a_local_record_rewrites_its_rights_block(derived: Path, tmp_path: Path) -> None:
    path = tmp_path / "e.json"
    write_evaluation_record(
        path,
        report=run_precision_benchmark(derived_root=derived, diarizer=StubDiarizer()),
        ledger=CallLedger(),
        preflight_summary=preflight(derived),
        started_at="2026-09-01T00:00:00+00:00",
        total_elapsed_seconds=1.0,
        authorization="user_limited_override",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rights"] = {"holder": RIGHTS_HOLDER, "third_party_processing_confirmed": True}
    jobs_before = payload["remote_jobs"]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))

    amend_authorization(path, authorization="user_limited_override")

    after = json.loads(path.read_text(encoding="utf-8"))
    assert after["rights"]["rights_confirmed"] is False
    assert after["rights"]["third_party_processing_rights"] == "unknown"
    assert after["remote_jobs"] == jobs_before
    assert after["licensed_or_private_audio_transmitted"] is True


def test_amending_a_file_with_no_authorization_block_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"unrelated": True}), encoding="utf-8")

    with pytest.raises(KcscBenchmarkError, match="carries no authorization block"):
        amend_authorization(path, authorization="user_limited_override")
