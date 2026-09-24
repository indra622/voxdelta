from __future__ import annotations

import json as json_module
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import pytest

import voxdelta.providers.pyannote_precision as pyannote_precision
from voxdelta.credentials import Credentials
from voxdelta.domain.models import AudioAsset
from voxdelta.providers.base import DiarizationProvider, ProviderError
from voxdelta.providers.pyannote_precision import (
    PYANNOTE_DATA_RETENTION_URL,
    PYANNOTE_MEDIA_RETENTION_HOURS,
    PyannotePrecisionProvider,
    _default_client_factory,
)

SECRET = "pyannoteai_private_sentinel"


class FakeResponse:
    def __init__(self, status_code: int, body: object = None, *, raises: bool = False) -> None:
        self.status_code = status_code
        self._body = body
        self._raises = raises

    def json(self) -> Any:
        if self._raises:
            raise ValueError("not json")
        return self._body


class ExitingResponse:
    def __init__(self, status_code: int, error: BaseException) -> None:
        self.status_code = status_code
        self._error = error

    def json(self) -> Any:
        raise self._error


class FakeClient:
    """Record every call so tests can assert on transmission and on secret handling."""

    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str, Any, bytes | None, Mapping[str, str] | None]] = []
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
        self.calls.append((method, url, json, content, headers))
        if not self._responses:
            raise AssertionError("unexpected extra request")
        nxt = self._responses.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt  # type: ignore[return-value]

    def close(self) -> None:
        self.closed = True


class FakeFactory:
    def __init__(self, client: FakeClient) -> None:
        self.client = client
        self.timeouts: list[float] = []

    def __call__(self, *, timeout_seconds: float) -> FakeClient:
        self.timeouts.append(timeout_seconds)
        return self.client


def _asset(
    *,
    mode: str = "mixed",
    paths: tuple[str, ...] = ("normalized/mixed.wav",),
    duration: float = 30.0,
) -> AudioAsset:
    return AudioAsset(
        source_name="private-call.wav",
        source_path="private/input/private-call.wav",
        normalized_paths=paths,
        channel_mode=mode,  # type: ignore[arg-type]
        duration_seconds=duration,
        channels=max(1, len(paths)),
        sha256="0" * 64,
    )


def _diarization(items: list[tuple[float, float, str]]) -> list[dict[str, object]]:
    return [{"speaker": label, "start": start, "end": end} for start, end, label in items]


def _succeeded(
    evidence: list[tuple[float, float, str]],
    exclusive: list[tuple[float, float, str]] | None = None,
) -> FakeResponse:
    output: dict[str, object] = {"diarization": _diarization(evidence)}
    if exclusive is not None:
        output["exclusiveDiarization"] = _diarization(exclusive)
    return FakeResponse(200, {"jobId": "job-1", "status": "succeeded", "output": output})


def _upload_pair() -> list[object]:
    return [
        FakeResponse(201, {"url": "https://storage.example/presigned"}),
        FakeResponse(200, {}),
    ]


def _provider(
    responses: list[object],
    *,
    token: str | None = SECRET,
    **kwargs: Any,
) -> tuple[PyannotePrecisionProvider, FakeClient, FakeFactory]:
    client = FakeClient(responses)
    factory = FakeFactory(client)
    credentials = Credentials(PYANNOTEAI_API_KEY=token) if token is not None else Credentials()
    provider = PyannotePrecisionProvider(
        credentials,
        client_factory=factory,
        clock=iter_clock(),
        sleep=lambda _seconds: None,
        **kwargs,
    )
    return provider, client, factory


def iter_clock() -> Any:
    state = {"now": 0.0}

    def clock() -> float:
        state["now"] += 1.0
        return state["now"]

    return clock


def _write_audio(tmp_path: Any, name: str = "mixed.wav", payload: bytes = b"RIFFfake") -> str:
    path = tmp_path / name
    path.write_bytes(payload)
    return str(path)


def test_default_client_bypasses_environment_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """The provider API is contacted directly, not through a transient host proxy."""

    recorded: dict[str, object] = {}

    class StubHttpxClient:
        def __init__(self, **kwargs: object) -> None:
            recorded.update(kwargs)

    monkeypatch.setattr(
        pyannote_precision,
        "import_module",
        lambda _name: SimpleNamespace(Client=StubHttpxClient),
    )

    _default_client_factory(timeout_seconds=12.5)

    assert recorded == {"timeout": 12.5, "trust_env": False}


def test_provider_conforms_to_protocol_and_declares_the_remote_contract() -> None:
    provider, _client, _factory = _provider([])

    assert isinstance(provider, DiarizationProvider)
    assert provider.provenance.name == "pyannoteai"
    assert provider.provenance.model == "precision-2"
    assert provider.provenance.remote is True
    assert provider.provenance.transmits == ("audio",)
    assert provider.provenance.retention_policy_url == PYANNOTE_DATA_RETENTION_URL
    # The consent prompt states this window, so the provider has to publish it.
    assert provider.provenance.retention_window_hours == PYANNOTE_MEDIA_RETENTION_HOURS
    assert PYANNOTE_MEDIA_RETENTION_HOURS == 48


def test_successful_diarization_uploads_once_and_maps_both_timelines(tmp_path: Any) -> None:
    audio = _write_audio(tmp_path)
    provider, client, factory = _provider(
        [
            *_upload_pair(),
            FakeResponse(200, {"jobId": "job-1", "status": "created"}),
            _succeeded(
                [(0.0, 6.0, "SPEAKER_00"), (4.0, 10.0, "SPEAKER_01")],
                [(0.0, 4.0, "SPEAKER_00"), (4.0, 10.0, "SPEAKER_01")],
            ),
        ]
    )

    timelines = provider.diarize_timelines(_asset(paths=(audio,)))

    methods = [(method, url) for method, url, _j, _c, _h in client.calls]
    assert methods[0] == ("POST", "https://api.pyannote.ai/v1/media/input")
    assert methods[1] == ("PUT", "https://storage.example/presigned")
    assert methods[2] == ("POST", "https://api.pyannote.ai/v1/diarize")
    assert methods[3] == ("GET", "https://api.pyannote.ai/v1/jobs/job-1")
    # Exactly one upload of exactly the normalized bytes.
    assert client.calls[1][3] == b"RIFFfake"
    assert factory.timeouts == [60.0]
    assert client.closed is True

    assert [
        (item.start, item.end, item.speaker_id, item.overlap) for item in timelines.evidence
    ] == [
        (0.0, 6.0, "SPEAKER_00", True),
        (4.0, 10.0, "SPEAKER_01", True),
    ]
    assert [(item.start, item.end, item.speaker_id) for item in timelines.exclusive] == [
        (0.0, 4.0, "SPEAKER_00"),
        (4.0, 10.0, "SPEAKER_01"),
    ]


def test_submitted_job_requests_pure_diarization_with_exclusive_timeline(tmp_path: Any) -> None:
    """transcription must be off: this POC diarizes and never asks for speech content."""

    audio = _write_audio(tmp_path)
    provider, client, _factory = _provider(
        [
            *_upload_pair(),
            FakeResponse(200, {"jobId": "job-1", "status": "created"}),
            _succeeded([(0.0, 5.0, "a"), (5.0, 10.0, "b")], [(0.0, 5.0, "a"), (5.0, 10.0, "b")]),
        ]
    )

    provider.diarize_timelines(_asset(paths=(audio,)))

    submitted = client.calls[2][2]
    assert submitted["transcription"] is False
    assert submitted["exclusive"] is True
    assert submitted["model"] == "precision-2"
    assert submitted["url"].startswith("media://voxdelta/")
    assert "numSpeakers" not in submitted
    assert "transcriptionConfig" not in submitted


def test_separate_channels_diarize_each_side_as_one_speaker(tmp_path: Any) -> None:
    left = _write_audio(tmp_path, "left.wav", b"LEFT")
    right = _write_audio(tmp_path, "right.wav", b"RIGHT")
    provider, client, _factory = _provider(
        [
            *_upload_pair(),
            FakeResponse(200, {"jobId": "job-1", "status": "created"}),
            _succeeded([(0.0, 4.0, "SPEAKER_00")]),
            *_upload_pair(),
            FakeResponse(200, {"jobId": "job-2", "status": "created"}),
            _succeeded([(5.0, 9.0, "SPEAKER_00")]),
        ]
    )

    timelines = provider.diarize_timelines(_asset(mode="separate", paths=(left, right)))

    assert [
        call[2] and call[2].get("numSpeakers")
        for call in client.calls
        if call[0] == "POST" and call[1].endswith("/v1/diarize")
    ] == [1, 1]
    assert [(item.start, item.end, item.speaker_id) for item in timelines.evidence] == [
        (0.0, 4.0, "SPEAKER_00"),
        (5.0, 9.0, "SPEAKER_01"),
    ]
    assert timelines.exclusive == timelines.evidence


def test_missing_api_key_fails_before_any_audio_is_transmitted(tmp_path: Any) -> None:
    audio = _write_audio(tmp_path)
    provider, client, factory = _provider([], token=None)

    with pytest.raises(ProviderError) as failure:
        provider.diarize_timelines(_asset(paths=(audio,)))

    assert failure.value.code == "missing_pyannote_api_key"
    assert client.calls == []
    assert factory.timeouts == []


def test_more_or_fewer_than_two_speakers_is_refused(tmp_path: Any) -> None:
    audio = _write_audio(tmp_path)
    provider, _client, _factory = _provider(
        [
            *_upload_pair(),
            FakeResponse(200, {"jobId": "job-1", "status": "created"}),
            _succeeded(
                [(0.0, 5.0, "a"), (5.0, 8.0, "b"), (8.0, 10.0, "c")],
                [(0.0, 5.0, "a"), (5.0, 8.0, "b"), (8.0, 10.0, "c")],
            ),
        ]
    )

    with pytest.raises(ProviderError) as failure:
        provider.diarize_timelines(_asset(paths=(audio,)))

    assert failure.value.code == "unsupported_speaker_count"


@pytest.mark.parametrize(
    "output",
    [
        {"diarization": []},
        {"diarization": [{"speaker": "a", "start": 1.0}]},
        {"diarization": [{"speaker": "", "start": 0.0, "end": 1.0}]},
        {"diarization": [{"speaker": "a", "start": 2.0, "end": 1.0}]},
        {"diarization": [{"speaker": "a", "start": "x", "end": 1.0}]},
        {"diarization": [{"speaker": "a", "start": 0.0, "end": 1.0}], "exclusiveDiarization": []},
        {"notDiarization": []},
    ],
)
def test_malformed_provider_output_is_rejected(tmp_path: Any, output: dict[str, object]) -> None:
    audio = _write_audio(tmp_path)
    provider, _client, _factory = _provider(
        [
            *_upload_pair(),
            FakeResponse(200, {"jobId": "job-1", "status": "created"}),
            FakeResponse(200, {"jobId": "job-1", "status": "succeeded", "output": output}),
        ]
    )

    with pytest.raises(ProviderError) as failure:
        provider.diarize_timelines(_asset(paths=(audio,)))

    assert failure.value.code in {"invalid_provider_output", "unsupported_speaker_count"}


def test_unparseable_response_body_is_reported_as_invalid_output(tmp_path: Any) -> None:
    audio = _write_audio(tmp_path)
    provider, _client, _factory = _provider([FakeResponse(201, raises=True)])

    with pytest.raises(ProviderError) as failure:
        provider.diarize_timelines(_asset(paths=(audio,)))

    assert failure.value.code == "invalid_provider_output"


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, "provider_unavailable"),
        (403, "provider_unavailable"),
        (429, "provider_unavailable"),
        (500, "provider_unavailable"),
        (408, "provider_timeout"),
        (504, "provider_timeout"),
    ],
)
def test_http_failures_map_to_typed_codes(tmp_path: Any, status: int, expected: str) -> None:
    audio = _write_audio(tmp_path)
    provider, _client, _factory = _provider([FakeResponse(status, {"message": "nope"})])

    with pytest.raises(ProviderError) as failure:
        provider.diarize_timelines(_asset(paths=(audio,)))

    assert failure.value.code == expected


def test_transport_timeout_maps_to_provider_timeout(tmp_path: Any) -> None:
    audio = _write_audio(tmp_path)
    provider, _client, _factory = _provider([TimeoutError("connect timed out")])

    with pytest.raises(ProviderError) as failure:
        provider.diarize_timelines(_asset(paths=(audio,)))

    assert failure.value.code == "provider_timeout"


def test_failed_job_never_falls_back_to_a_local_provider(tmp_path: Any) -> None:
    audio = _write_audio(tmp_path)
    provider, client, _factory = _provider(
        [
            *_upload_pair(),
            FakeResponse(200, {"jobId": "job-1", "status": "created"}),
            FakeResponse(200, {"jobId": "job-1", "status": "failed"}),
        ]
    )

    with pytest.raises(ProviderError) as failure:
        provider.diarize_timelines(_asset(paths=(audio,)))

    assert failure.value.code == "provider_unavailable"
    assert client.closed is True


def test_polling_stops_at_the_job_deadline(tmp_path: Any) -> None:
    audio = _write_audio(tmp_path)
    running = [FakeResponse(200, {"jobId": "job-1", "status": "running"}) for _ in range(50)]
    provider, _client, _factory = _provider(
        [
            *_upload_pair(),
            FakeResponse(200, {"jobId": "job-1", "status": "created"}),
            *running,
        ],
        job_timeout_seconds=5.0,
    )

    with pytest.raises(ProviderError) as failure:
        provider.diarize_timelines(_asset(paths=(audio,)))

    assert failure.value.code == "provider_timeout"


def test_invalid_asset_is_refused_before_any_request(tmp_path: Any) -> None:
    provider, client, _factory = _provider([])

    with pytest.raises(ProviderError) as failure:
        provider.diarize_timelines(_asset(paths=(), duration=30.0))

    assert failure.value.code == "invalid_audio_asset"
    assert client.calls == []


@pytest.mark.parametrize(
    ("name", "payload"),
    [("missing.wav", None), ("empty.wav", b"")],
)
def test_unusable_local_audio_reserves_no_remote_media_slot(
    tmp_path: Any, name: str, payload: bytes | None
) -> None:
    """A bad local file must fail before POST /v1/media/input, leaving no orphan slot."""

    if payload is None:
        audio = str(tmp_path / name)
    else:
        audio = _write_audio(tmp_path, name, payload)
    provider, client, _factory = _provider([])

    with pytest.raises(ProviderError) as failure:
        provider.diarize_timelines(_asset(paths=(audio,)))

    assert failure.value.code == "invalid_audio_asset"
    assert client.calls == []


@pytest.mark.parametrize("exit_error", [KeyboardInterrupt(), SystemExit(1)])
def test_interpreter_exits_are_never_swallowed_by_the_transport(
    tmp_path: Any, exit_error: BaseException
) -> None:
    """Only ordinary exceptions become provider codes; Ctrl-C and exit must propagate."""

    audio = _write_audio(tmp_path)
    provider, _client, _factory = _provider([exit_error])

    with pytest.raises(type(exit_error)):
        provider.diarize_timelines(_asset(paths=(audio,)))


@pytest.mark.parametrize("exit_error", [KeyboardInterrupt(), SystemExit(1)])
def test_interpreter_exits_are_never_swallowed_while_decoding(
    tmp_path: Any, exit_error: BaseException
) -> None:
    audio = _write_audio(tmp_path)
    provider, _client, _factory = _provider([ExitingResponse(201, exit_error)])

    with pytest.raises(type(exit_error)):
        provider.diarize_timelines(_asset(paths=(audio,)))


def test_secret_is_sent_only_as_a_bearer_header_and_never_leaks(tmp_path: Any) -> None:
    """The key may appear in the Authorization header and absolutely nowhere else."""

    audio = _write_audio(tmp_path)
    provider, client, _factory = _provider(
        [
            *_upload_pair(),
            FakeResponse(200, {"jobId": "job-1", "status": "created"}),
            _succeeded([(0.0, 5.0, "a"), (5.0, 10.0, "b")], [(0.0, 5.0, "a"), (5.0, 10.0, "b")]),
        ]
    )

    provider.diarize_timelines(_asset(paths=(audio,)))

    authorized = [headers for _m, _u, _j, _c, headers in client.calls if headers]
    assert authorized, "at least one request must be authorized"
    assert all(h.get("Authorization") == f"Bearer {SECRET}" for h in authorized)
    # The presigned upload carries its own credentials, so the key is withheld there.
    presigned = [h for m, _u, _j, _c, h in client.calls if m == "PUT"]
    assert presigned == [{}]

    # Nothing that a caller or log could see may contain the secret.
    assert SECRET not in repr(provider)
    assert SECRET not in str(provider.provenance)
    assert SECRET not in json_module.dumps(provider.provenance.model_dump(mode="json"))
    bodies = [json_module.dumps(body) for _m, _u, body, _c, _h in client.calls if body]
    assert all(SECRET not in body for body in bodies)


def test_provider_error_text_never_carries_the_secret_or_a_payload(tmp_path: Any) -> None:
    audio = _write_audio(tmp_path)
    provider, _client, _factory = _provider(
        [FakeResponse(500, {"message": f"boom {SECRET}", "detail": "internal-trace"})]
    )

    with pytest.raises(ProviderError) as failure:
        provider.diarize_timelines(_asset(paths=(audio,)))

    rendered = f"{failure.value}{failure.value.args!r}"
    assert SECRET not in rendered
    assert "boom" not in rendered
    assert "internal-trace" not in rendered


def _diagnostic_of(failure: pytest.ExceptionInfo[ProviderError]) -> Mapping[str, object] | None:
    return failure.value.diagnostic


def _fail_at(tmp_path: Any, responses: list[object]) -> pytest.ExceptionInfo[ProviderError]:
    audio = _write_audio(tmp_path)
    provider, _client, _factory = _provider(responses)
    with pytest.raises(ProviderError) as failure:
        provider.diarize_timelines(_asset(paths=(audio,)))
    return failure


class ConnectError(Exception):
    """Stands in for httpx.ConnectError, whose class name is reportable."""


@pytest.mark.parametrize(
    ("responses", "boundary"),
    [
        ([FakeResponse(403, {"message": "no"})], "media_input"),
        (
            [
                FakeResponse(201, {"url": "https://storage.example/presigned"}),
                FakeResponse(403, {}),
            ],
            "media_upload",
        ),
        ([*_upload_pair(), FakeResponse(403, {})], "diarize_submit"),
        (
            [
                *_upload_pair(),
                FakeResponse(200, {"jobId": "job-1", "status": "created"}),
                FakeResponse(403, {}),
            ],
            "job_poll",
        ),
    ],
)
def test_each_remote_boundary_is_named_in_the_diagnostic(
    tmp_path: Any, responses: list[object], boundary: str
) -> None:
    """provider_unavailable alone cannot say which hop failed; the diagnostic must."""

    failure = _fail_at(tmp_path, responses)

    assert failure.value.code == "provider_unavailable"
    assert _diagnostic_of(failure) == {
        "boundary": boundary,
        "failure": "http_status",
        "status": 403,
    }


def test_a_transport_failure_is_classified_apart_from_an_http_status(tmp_path: Any) -> None:
    failure = _fail_at(tmp_path, [ConnectError("connection refused to 1.2.3.4")])

    assert failure.value.code == "provider_unavailable"
    assert _diagnostic_of(failure) == {
        "boundary": "media_input",
        "failure": "transport_error",
        "error_type": "ConnectError",
    }


def test_a_transport_timeout_is_classified_apart_from_a_timeout_status(tmp_path: Any) -> None:
    timed_out = _fail_at(tmp_path, [TimeoutError("connect timed out")])
    status_408 = _fail_at(tmp_path, [FakeResponse(408, {})])

    assert timed_out.value.code == status_408.value.code == "provider_timeout"
    assert _diagnostic_of(timed_out) == {
        "boundary": "media_input",
        "failure": "transport_timeout",
        "error_type": "TimeoutError",
    }
    assert _diagnostic_of(status_408) == {
        "boundary": "media_input",
        "failure": "http_status",
        "status": 408,
    }


def test_a_remote_side_job_outcome_is_distinguished_from_an_unreachable_boundary(
    tmp_path: Any,
) -> None:
    failure = _fail_at(
        tmp_path,
        [
            *_upload_pair(),
            FakeResponse(200, {"jobId": "job-1", "status": "created"}),
            FakeResponse(200, {"jobId": "job-1", "status": "failed"}),
        ],
    )

    assert failure.value.code == "provider_unavailable"
    assert _diagnostic_of(failure) == {"boundary": "job_poll", "failure": "job_failed"}


def test_an_undecodable_body_is_classified_as_a_decode_failure(tmp_path: Any) -> None:
    failure = _fail_at(tmp_path, [FakeResponse(201, raises=True)])

    assert failure.value.code == "invalid_provider_output"
    assert _diagnostic_of(failure) == {
        "boundary": "media_input",
        "failure": "response_decode",
        "status": 201,
    }


def test_the_diagnostic_carries_no_secret_body_or_free_text(tmp_path: Any) -> None:
    """Only enumerated labels, a status number and an exception class name may escape."""

    failure = _fail_at(
        tmp_path,
        [FakeResponse(500, {"message": f"boom {SECRET}", "detail": "internal-trace"})],
    )
    diagnostic = _diagnostic_of(failure)
    assert diagnostic is not None
    rendered = json_module.dumps(diagnostic)

    assert SECRET not in rendered
    assert "boom" not in rendered
    assert "internal-trace" not in rendered
    assert set(diagnostic) <= {"boundary", "failure", "status", "error_type"}
    assert all(isinstance(value, (str, int)) for value in diagnostic.values())


def test_an_exception_class_name_can_never_smuggle_a_payload_into_the_diagnostic(
    tmp_path: Any,
) -> None:
    """Only known transport failure names are reportable, so a name cannot carry text."""

    smuggler = type(f"Boom {SECRET} " + "x" * 200, (Exception,), {})
    failure = _fail_at(tmp_path, [smuggler()])
    diagnostic = _diagnostic_of(failure)
    assert diagnostic is not None

    assert diagnostic["error_type"] == "OtherError"
    assert SECRET not in json_module.dumps(diagnostic)


def test_a_successful_run_reports_no_diagnostic(tmp_path: Any) -> None:
    audio = _write_audio(tmp_path)
    provider, _client, _factory = _provider(
        [
            *_upload_pair(),
            FakeResponse(200, {"jobId": "job-1", "status": "created"}),
            _succeeded([(0.0, 5.0, "a"), (5.0, 10.0, "b")], [(0.0, 5.0, "a"), (5.0, 10.0, "b")]),
        ]
    )

    provider.diarize_timelines(_asset(paths=(audio,)))

    assert ProviderError("provider_unavailable").diagnostic is None


def test_a_poll_that_never_settles_is_distinguished_from_an_unreachable_boundary(
    tmp_path: Any,
) -> None:
    audio = _write_audio(tmp_path)
    running = [FakeResponse(200, {"jobId": "job-1", "status": "running"}) for _ in range(50)]
    provider, _client, _factory = _provider(
        [
            *_upload_pair(),
            FakeResponse(200, {"jobId": "job-1", "status": "created"}),
            *running,
        ],
        job_timeout_seconds=5.0,
    )

    with pytest.raises(ProviderError) as failure:
        provider.diarize_timelines(_asset(paths=(audio,)))

    assert failure.value.code == "provider_timeout"
    assert _diagnostic_of(failure) == {"boundary": "job_poll", "failure": "job_poll_timeout"}
