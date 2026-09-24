from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import wave
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from voxdelta.domain.models import AudioAsset
from voxdelta.providers.base import (
    DiarizationProvider,
    DiarizationTimelineProvider,
    ProviderError,
)
from voxdelta.providers.nemotron_diarization import (
    OFFLINE_30_4S_GEOMETRY,
    CommandResult,
    NemotronDiarizationProvider,
    Turn,
    exclusive_segments,
    parse_segments,
    speaker_map,
)

PRIVATE_MARKER = "private-runtime-detail"


class FakeRunner:
    """Replays a scripted diarization result; records every invocation."""

    def __init__(
        self,
        outputs: Sequence[object] | object = (),
        *,
        version: bytes = b"nemo-speech 0.1.0\n",
    ) -> None:
        self.outputs = list(outputs) if isinstance(outputs, list) else [outputs]
        self.version = version
        self.calls: list[tuple[list[str], dict[str, str], float]] = []

    def __call__(
        self, argv: Sequence[str], *, env: Mapping[str, str], timeout_seconds: float
    ) -> CommandResult:
        self.calls.append((list(argv), dict(env), timeout_seconds))
        if argv[1:] == ["--version"]:
            return CommandResult(0, self.version)
        index = min(len(self.calls) - 2, len(self.outputs) - 1)
        output = self.outputs[index]
        if isinstance(output, BaseException):
            raise output
        if isinstance(output, CommandResult):
            return output
        return CommandResult(0, _document(output))  # type: ignore[arg-type]


def _document(segments: list[dict[str, object]], *, file: str = "/private/call.wav") -> bytes:
    return json.dumps({"file": file, "segments": segments}).encode()


def _seg(start: float, end: float, speaker: int) -> dict[str, object]:
    return {"start": start, "end": end, "speaker": speaker}


def _wav(path: Path, *, rate: int = 16_000, channels: int = 1, seconds: float = 1.0) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\0\0" * channels * int(rate * seconds))
    return path


@pytest.fixture()
def runtime(tmp_path: Path) -> tuple[Path, Path]:
    executable = tmp_path / "bin" / "nemo-speech"
    executable.parent.mkdir()
    executable.write_text("#!/bin/sh\nexit 1\n")
    executable.chmod(0o755)
    model = tmp_path / "models" / "Nemotron-3-Diarization.q8_0.gguf"
    model.parent.mkdir()
    model.write_bytes(b"synthetic-gguf")
    return executable, model


def _provider(
    runtime: tuple[Path, Path], runner: FakeRunner, **kwargs: object
) -> NemotronDiarizationProvider:
    executable, model = runtime
    return NemotronDiarizationProvider(
        executable_path=executable,
        model_path=model,
        runner=runner,
        **kwargs,  # type: ignore[arg-type]
    )


def _asset(tmp_path: Path, *, mode: str = "mixed", duration: float = 10.0) -> AudioAsset:
    count = 2 if mode == "separate" else 1
    paths = tuple(str(_wav(tmp_path / f"normalized-{index}.wav")) for index in range(count))
    return AudioAsset(
        source_name="call.wav",
        source_path=str(tmp_path / "call.wav"),
        normalized_paths=paths,
        channel_mode=mode,  # type: ignore[arg-type]
        duration_seconds=duration,
        channels=count,
        sha256="0" * 64,
    )


def test_provider_is_local_and_records_runtime_model_and_preset(
    runtime: tuple[Path, Path],
) -> None:
    provider = _provider(runtime, FakeRunner())

    assert isinstance(provider, DiarizationProvider)
    assert isinstance(provider, DiarizationTimelineProvider)
    assert provider.provenance.remote is False
    assert provider.provenance.transmits == ()
    assert provider.provenance.name == "nemo-speech"
    assert provider.provenance.model == "Nemotron-3-Diarization"
    digest = hashlib.sha256(b"synthetic-gguf").hexdigest()
    assert provider.provenance.revision == (
        f"nemo-speech-0.1.0+gguf-sha256-{digest}+model-card-offline-30.4s"
    )
    configuration = provider.configuration
    assert configuration["geometry_80ms_frames"] == {
        "chunk": 340,
        "right_context": 40,
        "left_context": 0,
        "fifo": 40,
        "spkcache": 264,
        "update_period": 300,
    }
    assert str(runtime[0].parent) not in json.dumps(configuration)
    assert str(runtime[1].parent) not in json.dumps(configuration)


def test_invocation_is_a_scrubbed_local_command_with_the_30_4s_geometry(
    runtime: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid")
    monkeypatch.setenv("NEMO_SPEECH_DIAR_PRESET", "v3-streaming")
    runner = FakeRunner([[_seg(0.0, 2.0, 1), _seg(2.5, 4.0, 2)]])
    provider = _provider(runtime, runner, device="metal", timeout_seconds=42)
    asset = _asset(tmp_path)

    provider.diarize_timelines(asset)

    argv, env, timeout = runner.calls[-1]
    assert argv == [
        str(runtime[0].resolve()),
        "diarize",
        asset.normalized_paths[0],
        "--model",
        str(runtime[1].resolve()),
        "--device",
        "metal",
        "--format",
        "json",
    ]
    assert timeout == 42
    assert "HTTPS_PROXY" not in env and "NEMO_SPEECH_DIAR_PRESET" not in env
    assert {key: env[key] for key in env if key.startswith("NEMO_SPEECH_")} == {
        f"NEMO_SPEECH_DIAR_{key.upper()}": str(value)
        for key, value in OFFLINE_30_4S_GEOMETRY.items()
    }
    assert not any(part.startswith("http") for part in argv)


def test_speakers_map_deterministically_by_first_activity_with_overlap_evidence(
    runtime: tuple[Path, Path], tmp_path: Path
) -> None:
    runner = FakeRunner(
        [[_seg(3.0, 6.0, 1), _seg(0.5, 3.5, 4), _seg(6.0, 8.0, 4), _seg(8.5, 9.0, 1)]]
    )
    timelines = _provider(runtime, runner).diarize_timelines(_asset(tmp_path))

    assert [(s.start, s.end, s.speaker_id, s.overlap) for s in timelines.evidence] == [
        (0.5, 3.5, "SPEAKER_00", True),
        (3.0, 6.0, "SPEAKER_01", True),
        (6.0, 8.0, "SPEAKER_00", False),
        (8.5, 9.0, "SPEAKER_01", False),
    ]
    assert [(s.start, s.end, s.speaker_id) for s in timelines.exclusive] == [
        (0.5, 3.5, "SPEAKER_00"),
        (3.5, 6.0, "SPEAKER_01"),
        (6.0, 8.0, "SPEAKER_00"),
        (8.5, 9.0, "SPEAKER_01"),
    ]
    assert not any(segment.overlap for segment in timelines.exclusive)


def test_exclusive_timeline_never_overlaps_and_is_idempotent_to_input_order() -> None:
    turns = [Turn(0.0, 5.0, 2), Turn(1.0, 2.0, 1), Turn(4.0, 7.0, 1), Turn(6.5, 9.0, 3)]
    speakers = speaker_map(turns)
    forward = exclusive_segments(sorted(turns), speakers)
    backward = exclusive_segments(sorted(turns, reverse=True), speakers)

    assert forward == backward
    for current, following in zip(forward, forward[1:], strict=False):
        assert current.end <= following.start
    assert [(s.start, s.end, s.speaker_id) for s in forward] == [
        (0.0, 5.0, "SPEAKER_00"),
        (5.0, 7.0, "SPEAKER_01"),
        (7.0, 9.0, "SPEAKER_02"),
    ]


def test_same_speaker_fragments_are_merged() -> None:
    turns = parse_segments(
        _document([_seg(0.0, 1.0, 1), _seg(0.8, 2.0, 1), _seg(2.0, 3.0, 1), _seg(5, 6, 2)]),
        10.0,
    )
    assert turns == [Turn(0.0, 3.0, 1), Turn(5.0, 6.0, 2)]


def test_end_padding_is_clipped_to_the_audio_but_gross_overruns_are_rejected() -> None:
    assert parse_segments(_document([_seg(9.0, 10.3, 1)]), 10.0) == [Turn(9.0, 10.0, 1)]
    with pytest.raises(ProviderError) as failure:
        parse_segments(_document([_seg(9.0, 11.0, 1)]), 10.0)
    assert failure.value.code == "invalid_provider_output"


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"not json",
        b"\xff\xfe",
        b"[]",
        json.dumps({"segments": []}).encode(),
        json.dumps({"file": "x", "segments": [], "extra": 1}).encode(),
        json.dumps({"file": 1, "segments": []}).encode(),
        _document([]),
        _document([{"start": 0, "end": 1}]),
        _document([{"start": 0, "end": 1, "speaker": 1, "confidence": 1}]),
        _document([_seg(1.0, 1.0, 1)]),
        _document([_seg(2.0, 1.0, 1)]),
        _document([_seg(-0.1, 1.0, 1)]),
        _document([{"start": "0", "end": 1, "speaker": 1}]),
        _document([{"start": 0, "end": 1, "speaker": "1"}]),
        _document([{"start": 0, "end": 1, "speaker": True}]),
        _document([{"start": 0, "end": 1, "speaker": 1.0}]),
        _document([_seg(0, 1, 0)]),
        _document([_seg(0, 1, 9)]),
        b'{"file": "x", "segments": [{"start": NaN, "end": 1, "speaker": 1}]}',
        b'{"file": "x", "segments": [{"start": 0, "end": Infinity, "speaker": 1}]}',
        b" " * (8 * 1024 * 1024 + 1),
    ],
)
def test_malformed_output_is_rejected_with_a_fixed_code(payload: bytes) -> None:
    with pytest.raises(ProviderError) as failure:
        parse_segments(payload, 10.0)
    assert failure.value.code == "invalid_provider_output"


def test_two_speaker_gate_applies_to_the_product_path_only(
    runtime: tuple[Path, Path], tmp_path: Path
) -> None:
    three = [_seg(0, 1, 1), _seg(2, 3, 2), _seg(4, 5, 3)]
    provider = _provider(runtime, FakeRunner([three]))

    with pytest.raises(ProviderError) as failure:
        provider.diarize(_asset(tmp_path))
    assert failure.value.code == "unsupported_speaker_count"
    unconstrained = provider.diarize_unconstrained(_asset(tmp_path))
    assert {segment.speaker_id for segment in unconstrained.evidence} == {
        "SPEAKER_00",
        "SPEAKER_01",
        "SPEAKER_02",
    }


def test_separate_channels_fold_each_channel_into_one_speaker(
    runtime: tuple[Path, Path], tmp_path: Path
) -> None:
    runner = FakeRunner(
        [
            [_seg(0.0, 2.0, 1), _seg(1.5, 3.0, 2)],
            [_seg(2.5, 5.0, 3)],
        ]
    )
    timelines = _provider(runtime, runner).diarize_timelines(_asset(tmp_path, mode="separate"))

    assert [(s.start, s.end, s.speaker_id, s.overlap) for s in timelines.evidence] == [
        (0.0, 3.0, "SPEAKER_00", True),
        (2.5, 5.0, "SPEAKER_01", True),
    ]
    assert [(s.start, s.end, s.speaker_id) for s in timelines.exclusive] == [
        (0.0, 3.0, "SPEAKER_00"),
        (3.0, 5.0, "SPEAKER_01"),
    ]


@pytest.mark.parametrize(
    ("rate", "channels"),
    [(8_000, 1), (44_100, 1), (16_000, 2)],
)
def test_input_must_be_16khz_mono_pcm(
    runtime: tuple[Path, Path], tmp_path: Path, rate: int, channels: int
) -> None:
    runner = FakeRunner([[_seg(0, 1, 1), _seg(1, 2, 2)]])
    asset = _asset(tmp_path)
    _wav(Path(asset.normalized_paths[0]), rate=rate, channels=channels)

    with pytest.raises(ProviderError) as failure:
        _provider(runtime, runner).diarize(asset)
    assert failure.value.code == "invalid_audio_asset"
    assert len(runner.calls) == 1  # only the construction-time version probe


@pytest.mark.parametrize(
    "update",
    [
        {"duration_seconds": None},
        {"duration_seconds": 0.0},
        {"channel_mode": None},
        {"normalized_paths": ()},
        {"normalized_paths": ("",)},
    ],
)
def test_audio_asset_contract_is_enforced(
    runtime: tuple[Path, Path], tmp_path: Path, update: dict[str, object]
) -> None:
    asset = _asset(tmp_path).model_copy(update=update)
    with pytest.raises(ProviderError) as failure:
        _provider(runtime, FakeRunner()).diarize(asset)
    assert failure.value.code == "invalid_audio_asset"


def test_non_wav_normalized_input_is_rejected(runtime: tuple[Path, Path], tmp_path: Path) -> None:
    asset = _asset(tmp_path)
    Path(asset.normalized_paths[0]).write_bytes(b"not a wav " + PRIVATE_MARKER.encode())
    with pytest.raises(ProviderError) as failure:
        _provider(runtime, FakeRunner()).diarize(asset)
    assert failure.value.code == "invalid_audio_asset"


def test_missing_executable_fails_with_actionable_code(
    runtime: tuple[Path, Path], tmp_path: Path
) -> None:
    with pytest.raises(ProviderError) as failure:
        NemotronDiarizationProvider(
            executable_path=tmp_path / "absent" / "nemo-speech",
            model_path=runtime[1],
            runner=FakeRunner(),
        )
    assert failure.value.code == "local_runtime_missing"
    assert "absent" not in str(failure.value)
    assert "install" in str(failure.value)


def test_non_executable_runtime_is_missing(runtime: tuple[Path, Path]) -> None:
    runtime[0].chmod(0o644)
    with pytest.raises(ProviderError) as failure:
        _provider(runtime, FakeRunner())
    assert failure.value.code == "local_runtime_missing"


def test_unrecognised_runtime_version_is_missing(runtime: tuple[Path, Path]) -> None:
    with pytest.raises(ProviderError) as failure:
        _provider(runtime, FakeRunner(version=b"something-else 1.0\n"))
    assert failure.value.code == "local_runtime_missing"


@pytest.mark.parametrize("variant", ["absent", "symlink", "directory", "wrong-suffix"])
def test_missing_or_unsafe_model_fails_with_actionable_code(
    runtime: tuple[Path, Path], tmp_path: Path, variant: str
) -> None:
    model = runtime[1]
    if variant == "absent":
        target = tmp_path / "models" / "absent.gguf"
    elif variant == "symlink":
        target = tmp_path / "models" / "link.gguf"
        target.symlink_to(model)
    elif variant == "directory":
        target = tmp_path / "models" / "dir.gguf"
        target.mkdir()
    else:
        target = tmp_path / "models" / "model.bin"
        target.write_bytes(b"x")
    with pytest.raises(ProviderError) as failure:
        NemotronDiarizationProvider(
            executable_path=runtime[0], model_path=target, runner=FakeRunner()
        )
    assert failure.value.code == "local_model_missing"
    assert str(tmp_path) not in str(failure.value)


def test_model_removed_after_startup_is_reported_as_missing(
    runtime: tuple[Path, Path], tmp_path: Path
) -> None:
    runner = FakeRunner([[_seg(0, 1, 1), _seg(1, 2, 2)]])
    provider = _provider(runtime, runner)
    runtime[1].unlink()
    with pytest.raises(ProviderError) as failure:
        provider.diarize(_asset(tmp_path))
    assert failure.value.code == "local_model_missing"
    assert len(runner.calls) == 1


@pytest.mark.parametrize(
    ("outcome", "code"),
    [
        (
            subprocess.TimeoutExpired(cmd=["nemo-speech", PRIVATE_MARKER], timeout=1),
            "provider_timeout",
        ),
        (TimeoutError(PRIVATE_MARKER), "provider_timeout"),
        (FileNotFoundError(PRIVATE_MARKER), "local_runtime_missing"),
        (RuntimeError(PRIVATE_MARKER), "provider_unavailable"),
        (CommandResult(2, PRIVATE_MARKER.encode()), "provider_unavailable"),
        (CommandResult(0, PRIVATE_MARKER.encode()), "invalid_provider_output"),
    ],
)
def test_runtime_failures_are_sanitized(
    runtime: tuple[Path, Path], tmp_path: Path, outcome: object, code: str
) -> None:
    provider = _provider(runtime, FakeRunner([outcome]))
    with pytest.raises(ProviderError) as failure:
        provider.diarize(_asset(tmp_path))

    assert failure.value.code == code
    assert PRIVATE_MARKER not in str(failure.value)
    assert PRIVATE_MARKER not in repr(failure.value.args)
    assert failure.value.__cause__ is None
    assert failure.value.__context__ is None or failure.value.__suppress_context__


def test_invalid_timeout_is_refused(runtime: tuple[Path, Path]) -> None:
    for timeout in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ProviderError):
            _provider(runtime, FakeRunner(), timeout_seconds=timeout)


def test_diarization_opens_no_network_socket(
    runtime: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    runner = FakeRunner([[_seg(0, 1, 1), _seg(1, 2, 2)]])
    provider = _provider(runtime, runner)
    assert len(provider.diarize(_asset(tmp_path))) == 2


def test_default_runner_executes_a_real_local_command_with_timeout(tmp_path: Path) -> None:
    from voxdelta.providers.nemotron_diarization import CommandRunner

    script = tmp_path / "echo.sh"
    script.write_text('#!/bin/sh\necho "$NEMO_SPEECH_DIAR_CHUNK"\necho err >&2\n')
    script.chmod(0o755)
    result = CommandRunner()(
        [str(script)],
        env={"PATH": "/usr/bin:/bin", "NEMO_SPEECH_DIAR_CHUNK": "340"},
        timeout_seconds=10,
    )
    assert result == CommandResult(0, b"340\n")

    slow = tmp_path / "slow.sh"
    slow.write_text("#!/bin/sh\nsleep 5\n")
    slow.chmod(0o755)
    with pytest.raises(subprocess.TimeoutExpired):
        CommandRunner()([str(slow)], env={"PATH": "/usr/bin:/bin"}, timeout_seconds=0.2)
