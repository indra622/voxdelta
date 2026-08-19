from __future__ import annotations

import hashlib
import json
import subprocess
import wave
from array import array
from pathlib import Path
from typing import Any

import pytest

import voxdelta.audio.service as service_module
from voxdelta.audio.service import AudioRejected, AudioService

FIXTURES = Path(__file__).parents[1] / "fixtures"
MONO_FIXTURE = FIXTURES / "synthetic_65s.wav"
STEREO_FIXTURE = FIXTURES / "stereo_split_65s.wav"


def _probe_payload(*, duration: Any = "65.000000", channels: Any = 1) -> bytes:
    return json.dumps(
        {
            "streams": [{"codec_type": "audio", "channels": channels}],
            "format": {"duration": duration},
        }
    ).encode()


def _completed(stdout: bytes = b"") -> subprocess.CompletedProcess[list[str]]:
    return subprocess.CompletedProcess([], 0, stdout=stdout, stderr=b"")


def _write_stereo_silence(path: Path, *, seconds: int = 65) -> None:
    frame = b"\0\0\0\0"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(frame * 16_000 * seconds)


def _write_stereo_pattern(path: Path, pattern: tuple[tuple[int, int], ...]) -> None:
    frames = 16_000 * 65
    repetitions = frames // len(pattern)
    samples = array("h", (sample for frame in pattern for sample in frame)) * repetitions
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(samples.tobytes())


def _assert_mono_16khz(path: Path) -> None:
    with wave.open(str(path), "rb") as audio:
        assert audio.getnchannels() == 1
        assert audio.getsampwidth() == 2
        assert audio.getframerate() == 16_000


def test_ingest_normalizes_supported_audio(tmp_path: Path) -> None:
    service = AudioService(tmp_path / "jobs", min_seconds=60, max_seconds=3600)

    asset = service.ingest(MONO_FIXTURE, "j1")

    assert asset.duration_seconds == pytest.approx(65, abs=0.2)
    assert asset.channels == 1
    assert asset.channel_mode == "mixed"
    assert len(asset.normalized_paths) == 1
    normalized = Path(asset.normalized_paths[0])
    assert normalized.exists()
    assert normalized.parent == (tmp_path / "jobs" / "j1").resolve()
    _assert_mono_16khz(normalized)
    assert asset.sha256 == hashlib.sha256(MONO_FIXTURE.read_bytes()).hexdigest()


def test_ingest_prefers_distinct_stereo_channels(tmp_path: Path) -> None:
    service = AudioService(tmp_path / "jobs", min_seconds=60, max_seconds=3600)

    asset = service.ingest(STEREO_FIXTURE, "j1")

    assert asset.channel_mode == "separate"
    assert asset.channels == 2
    assert len(asset.normalized_paths) == 2
    assert {Path(path).name for path in asset.normalized_paths} == {"left.wav", "right.wav"}
    assert (tmp_path / "jobs" / "j1" / "mixed.wav").is_file()
    for path in map(Path, asset.normalized_paths):
        _assert_mono_16khz(path)


def test_ingest_rejects_unsupported_suffix_before_creating_a_job(tmp_path: Path) -> None:
    source = tmp_path / "private-call.txt"
    source.write_text("not audio", encoding="utf-8")

    with pytest.raises(AudioRejected, match=r"^unsupported extension$"):
        AudioService(tmp_path / "jobs", 60, 3600).ingest(source, "j1")

    assert not (tmp_path / "jobs").exists()


@pytest.mark.parametrize("preference", ["auto", "mixed"])
def test_ingest_mono_always_selects_mixed(preference: str, tmp_path: Path) -> None:
    asset = AudioService(tmp_path / "jobs", 60, 3600).ingest(
        MONO_FIXTURE,
        "j1",
        channel_preference=preference,  # type: ignore[arg-type]
    )

    assert asset.channel_mode == "mixed"
    assert len(asset.normalized_paths) == 1


def test_ingest_explicit_separate_rejects_mono_without_outputs(tmp_path: Path) -> None:
    with pytest.raises(AudioRejected, match=r"^separate channels requested for mono audio$"):
        AudioService(tmp_path / "jobs", 60, 3600).ingest(
            MONO_FIXTURE, "j1", channel_preference="separate"
        )

    assert not (tmp_path / "jobs").exists()


def test_ingest_explicit_mixed_overrides_distinct_stereo(tmp_path: Path) -> None:
    asset = AudioService(tmp_path / "jobs", 60, 3600).ingest(
        STEREO_FIXTURE, "j1", channel_preference="mixed"
    )

    assert asset.channel_mode == "mixed"
    assert [Path(path).name for path in asset.normalized_paths] == ["mixed.wav"]
    assert not (tmp_path / "jobs" / "j1" / "left.wav").exists()
    assert not (tmp_path / "jobs" / "j1" / "right.wav").exists()


def test_ingest_auto_treats_silent_zero_variance_stereo_as_mixed(tmp_path: Path) -> None:
    source = tmp_path / "silence.wav"
    _write_stereo_silence(source)

    asset = AudioService(tmp_path / "jobs", 60, 3600).ingest(source, "j1")

    assert asset.channel_mode == "mixed"
    assert [Path(path).name for path in asset.normalized_paths] == ["mixed.wav"]


@pytest.mark.parametrize(
    ("name", "pattern"),
    [
        ("rms-at-threshold", ((500, 500), (-500, 500), (500, -500), (-500, -500))),
        ("perfect-correlation", ((2000, 2000), (-2000, -2000))),
    ],
)
def test_ingest_auto_requires_both_rms_and_correlation_thresholds(
    name: str, pattern: tuple[tuple[int, int], ...], tmp_path: Path
) -> None:
    source = tmp_path / f"{name}.wav"
    _write_stereo_pattern(source, pattern)

    asset = AudioService(tmp_path / "jobs", 60, 3600).ingest(source, "j1")

    assert asset.channel_mode == "mixed"


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [
        (65, 3600),
        (60, 65),
        (65, 65),
    ],
)
def test_ingest_accepts_duration_at_inclusive_boundaries(
    minimum: int, maximum: int, tmp_path: Path
) -> None:
    asset = AudioService(tmp_path / "jobs", minimum, maximum).ingest(MONO_FIXTURE, "j1")

    assert asset.duration_seconds == 65


@pytest.mark.parametrize(
    ("minimum", "maximum", "message"),
    [
        (66, 3600, "audio is shorter than 66 seconds"),
        (60, 64, "audio exceeds 64 seconds"),
    ],
)
def test_ingest_rejects_duration_outside_configured_bounds(
    minimum: int, maximum: int, message: str, tmp_path: Path
) -> None:
    with pytest.raises(AudioRejected, match=rf"^{message}$"):
        AudioService(tmp_path / "jobs", minimum, maximum).ingest(MONO_FIXTURE, "j1")

    assert not (tmp_path / "jobs").exists()


@pytest.mark.parametrize("kind", ["missing", "directory", "symlink"])
def test_ingest_rejects_missing_non_file_and_symlink_uploads_safely(
    kind: str, tmp_path: Path
) -> None:
    source = tmp_path / "private-customer-name.wav"
    if kind == "directory":
        source.mkdir()
    elif kind == "symlink":
        source.symlink_to(MONO_FIXTURE)

    with pytest.raises(AudioRejected) as raised:
        AudioService(tmp_path / "jobs", 60, 3600).ingest(source, "j1")

    assert str(raised.value) == "audio is not decodable"
    assert str(source) not in str(raised.value)
    assert not (tmp_path / "jobs").exists()


@pytest.mark.parametrize(
    "probe_failure",
    [
        FileNotFoundError("ffprobe missing"),
        subprocess.TimeoutExpired(["ffprobe"], 15, stderr=b"private stderr"),
        subprocess.CalledProcessError(1, ["ffprobe"], stderr=b"private stderr"),
    ],
)
def test_ingest_maps_probe_process_failures_to_safe_decodability_error(
    probe_failure: BaseException, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_probe(*args: object, **kwargs: object) -> subprocess.CompletedProcess[list[str]]:
        raise probe_failure

    monkeypatch.setattr(service_module.subprocess, "run", fail_probe)

    with pytest.raises(AudioRejected) as raised:
        AudioService(tmp_path / "jobs", 60, 3600).ingest(MONO_FIXTURE, "j1")

    assert str(raised.value) == "audio is not decodable"
    assert "private" not in str(raised.value)
    assert str(MONO_FIXTURE) not in str(raised.value)


@pytest.mark.parametrize(
    "stdout",
    [
        b"not-json",
        b"{}",
        b'{"streams": {}, "format": {"duration": "65"}}',
    ],
)
def test_ingest_rejects_malformed_probe_json(
    stdout: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        service_module.subprocess,
        "run",
        lambda *args, **kwargs: _completed(stdout),
    )

    with pytest.raises(AudioRejected, match=r"^audio is not decodable$"):
        AudioService(tmp_path / "jobs", 60, 3600).ingest(MONO_FIXTURE, "j1")


@pytest.mark.parametrize("duration", [None, True, "unknown", float("nan"), float("inf")])
def test_ingest_rejects_non_numeric_or_non_finite_probe_duration(
    duration: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        service_module.subprocess,
        "run",
        lambda *args, **kwargs: _completed(_probe_payload(duration=duration)),
    )

    with pytest.raises(AudioRejected, match=r"^audio is not decodable$"):
        AudioService(tmp_path / "jobs", 60, 3600).ingest(MONO_FIXTURE, "j1")


@pytest.mark.parametrize("channels", [None, True, 0, -1, "2"])
def test_ingest_rejects_invalid_audio_channel_metadata(
    channels: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        service_module.subprocess,
        "run",
        lambda *args, **kwargs: _completed(_probe_payload(channels=channels)),
    )

    with pytest.raises(AudioRejected, match=r"^audio is not decodable$"):
        AudioService(tmp_path / "jobs", 60, 3600).ingest(MONO_FIXTURE, "j1")


@pytest.mark.parametrize(
    "streams",
    [
        [],
        [{"codec_type": "video"}],
        [
            {"codec_type": "audio", "channels": 1},
            {"codec_type": "audio", "channels": 1},
        ],
    ],
)
def test_ingest_requires_exactly_one_audio_stream(
    streams: list[dict[str, object]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = json.dumps({"streams": streams, "format": {"duration": "65"}}).encode()
    monkeypatch.setattr(
        service_module.subprocess,
        "run",
        lambda *args, **kwargs: _completed(payload),
    )

    with pytest.raises(AudioRejected, match=r"^audio is not decodable$"):
        AudioService(tmp_path / "jobs", 60, 3600).ingest(MONO_FIXTURE, "j1")


@pytest.mark.parametrize(
    "normalization_failure",
    [
        FileNotFoundError("ffmpeg missing"),
        subprocess.TimeoutExpired(["ffmpeg"], 300, stderr=b"private stderr"),
        subprocess.CalledProcessError(1, ["ffmpeg"], stderr=b"private stderr"),
    ],
)
def test_ingest_maps_normalization_failures_to_safe_error_and_cleans_partial_outputs(
    normalization_failure: BaseException, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def probe_then_fail(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[list[str]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _completed(_probe_payload())
        Path(command[-1]).write_bytes(b"partial audio containing a private customer name")
        raise normalization_failure

    monkeypatch.setattr(service_module.subprocess, "run", probe_then_fail)

    with pytest.raises(AudioRejected) as raised:
        AudioService(tmp_path / "jobs", 60, 3600).ingest(MONO_FIXTURE, "j1")

    assert str(raised.value) == "audio is not decodable"
    assert "private" not in str(raised.value)
    assert list((tmp_path / "jobs" / "j1").iterdir()) == []


def test_ingest_uses_argument_lists_and_bounded_subprocess_timeouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[list[str]]:
        calls.append((command, kwargs))
        if command[0] == "ffprobe":
            return _completed(_probe_payload())
        Path(command[-1]).write_bytes(b"normalized")
        return _completed()

    monkeypatch.setattr(service_module.subprocess, "run", fake_run)

    AudioService(tmp_path / "jobs", 60, 3600).ingest(MONO_FIXTURE, "j1")

    assert [command[0] for command, _ in calls] == ["ffprobe", "ffmpeg"]
    assert all(type(command) is list for command, _ in calls)
    assert all(kwargs.get("shell", False) is False for _, kwargs in calls)
    assert all(isinstance(kwargs.get("timeout"), (int, float)) for _, kwargs in calls)
    assert all(float(kwargs["timeout"]) > 0 for _, kwargs in calls)


def test_ingest_preserves_artifact_store_job_path_validation(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    (jobs / "j1").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="job directory"):
        AudioService(jobs, 60, 3600).ingest(MONO_FIXTURE, "j1")

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert list(outside.iterdir()) == [sentinel]


def test_ingest_replaces_output_symlink_without_touching_its_target(tmp_path: Path) -> None:
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"sentinel")
    job_dir = tmp_path / "jobs" / "j1"
    job_dir.mkdir(parents=True)
    output_link = job_dir / "mixed.wav"
    output_link.symlink_to(outside)

    asset = AudioService(tmp_path / "jobs", 60, 3600).ingest(MONO_FIXTURE, "j1")

    assert outside.read_bytes() == b"sentinel"
    assert not output_link.is_symlink()
    assert Path(asset.normalized_paths[0]) == output_link
