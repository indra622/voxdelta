"""Validate uploaded call audio and create job-scoped normalized WAV files."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
import wave
from array import array
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from voxdelta.domain.models import AudioAsset
from voxdelta.jobs.artifacts import ArtifactStore

ChannelPreference = Literal["auto", "mixed", "separate"]

_ALLOWED_EXTENSIONS = frozenset({".wav", ".mp3", ".m4a"})
_PROBE_TIMEOUT_SECONDS = 15
_NORMALIZE_TIMEOUT_SECONDS = 300
_HASH_CHUNK_BYTES = 1024 * 1024
_MIN_SEPARATE_RMS = 500.0
_MAX_SEPARATE_CORRELATION = 0.85
_DECODABILITY_ERROR = "audio is not decodable"


class AudioRejected(ValueError):
    """A safe, user-facing rejection raised at the audio ingestion boundary."""


def _is_link_like(metadata: os.stat_result) -> bool:
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(file_attributes & reparse_attribute)


def _regular_source_metadata(source: Path) -> os.stat_result:
    try:
        metadata = source.lstat()
    except OSError:
        raise AudioRejected(_DECODABILITY_ERROR) from None
    if _is_link_like(metadata) or not stat.S_ISREG(metadata.st_mode):
        raise AudioRejected(_DECODABILITY_ERROR)
    return metadata


def _source_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns)


def _require_unchanged_source(source: Path, expected: tuple[int, int, int, int]) -> None:
    if _source_identity(_regular_source_metadata(source)) != expected:
        raise AudioRejected(_DECODABILITY_ERROR)


def _probe(source: Path) -> tuple[float, int]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(source),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
        payload: Any = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, UnicodeDecodeError):
        raise AudioRejected(_DECODABILITY_ERROR) from None

    if not isinstance(payload, dict):
        raise AudioRejected(_DECODABILITY_ERROR)
    streams = payload.get("streams")
    audio_streams = (
        [
            stream
            for stream in streams
            if isinstance(stream, dict) and stream.get("codec_type") == "audio"
        ]
        if isinstance(streams, list)
        else []
    )
    if len(audio_streams) != 1:
        raise AudioRejected(_DECODABILITY_ERROR)

    channels = audio_streams[0].get("channels")
    if isinstance(channels, bool) or not isinstance(channels, int) or channels <= 0:
        raise AudioRejected(_DECODABILITY_ERROR)

    format_metadata = payload.get("format")
    raw_duration = format_metadata.get("duration") if isinstance(format_metadata, dict) else None
    if isinstance(raw_duration, bool) or not isinstance(raw_duration, (str, int, float)):
        raise AudioRejected(_DECODABILITY_ERROR)
    try:
        duration = float(raw_duration)
    except (TypeError, ValueError):
        raise AudioRejected(_DECODABILITY_ERROR) from None
    if not math.isfinite(duration):
        raise AudioRejected(_DECODABILITY_ERROR)
    return duration, channels


def _sha256(source: Path) -> str:
    digest = hashlib.sha256()
    try:
        with source.open("rb") as input_file:
            while chunk := input_file.read(_HASH_CHUNK_BYTES):
                digest.update(chunk)
    except OSError:
        raise AudioRejected(_DECODABILITY_ERROR) from None
    return digest.hexdigest()


def _temporary_wav(job_directory: Path, label: str) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=job_directory,
        prefix=f".{label}.",
        suffix=".wav",
    )
    os.close(descriptor)
    return Path(name)


def _run_ffmpeg(command: list[str]) -> None:
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            timeout=_NORMALIZE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        raise AudioRejected(_DECODABILITY_ERROR) from None


def _normalize_mixed(source: Path, target: Path) -> None:
    _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(target),
        ]
    )


def _normalize_stereo_channels(source: Path, left: Path, right: Path) -> None:
    _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-filter_complex",
            "[0:a]channelsplit=channel_layout=stereo[left][right]",
            "-map",
            "[left]",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(left),
            "-map",
            "[right]",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(right),
        ]
    )


def _read_pcm16_mono(path: Path) -> array[int]:
    try:
        with wave.open(str(path), "rb") as audio:
            if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
                raise AudioRejected(_DECODABILITY_ERROR)
            samples = array("h")
            samples.frombytes(audio.readframes(audio.getnframes()))
    except (EOFError, OSError, wave.Error):
        raise AudioRejected(_DECODABILITY_ERROR) from None
    if sys.byteorder == "big":
        samples.byteswap()
    if not samples:
        raise AudioRejected(_DECODABILITY_ERROR)
    return samples


def _channels_are_distinct(left_path: Path, right_path: Path) -> bool:
    left = _read_pcm16_mono(left_path)
    right = _read_pcm16_mono(right_path)
    if len(left) != len(right):
        raise AudioRejected(_DECODABILITY_ERROR)

    sample_count = len(left)
    left_sum = sum(left)
    right_sum = sum(right)
    left_square_sum = sum(sample * sample for sample in left)
    right_square_sum = sum(sample * sample for sample in right)
    left_rms = math.sqrt(left_square_sum / sample_count)
    right_rms = math.sqrt(right_square_sum / sample_count)
    if left_rms <= _MIN_SEPARATE_RMS or right_rms <= _MIN_SEPARATE_RMS:
        return False

    left_variance = left_square_sum - (left_sum * left_sum / sample_count)
    right_variance = right_square_sum - (right_sum * right_sum / sample_count)
    if left_variance <= 0 or right_variance <= 0:
        return False
    covariance = sum(a * b for a, b in zip(left, right, strict=True)) - (
        left_sum * right_sum / sample_count
    )
    correlation = covariance / math.sqrt(left_variance * right_variance)
    return abs(correlation) < _MAX_SEPARATE_CORRELATION


def _unlink_all(paths: Sequence[Path]) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


class AudioService:
    """Probe one local upload and normalize it beneath a validated job directory."""

    def __init__(self, jobs_root: Path, min_seconds: int, max_seconds: int) -> None:
        if min_seconds <= 0 or max_seconds <= 0 or min_seconds > max_seconds:
            raise ValueError("audio duration limits must be positive and ordered")
        self._store = ArtifactStore(jobs_root)
        self._min_seconds = min_seconds
        self._max_seconds = max_seconds

    def ingest(
        self,
        upload_path: Path,
        job_id: str,
        channel_preference: ChannelPreference = "auto",
    ) -> AudioAsset:
        """Validate an upload and return metadata for its selected normalized channel mode."""

        if upload_path.suffix.lower() not in _ALLOWED_EXTENSIONS:
            raise AudioRejected("unsupported extension")
        source_metadata = _regular_source_metadata(upload_path)
        source_identity = _source_identity(source_metadata)
        duration, channels = _probe(upload_path)
        _require_unchanged_source(upload_path, source_identity)

        if duration < self._min_seconds:
            raise AudioRejected(f"audio is shorter than {self._min_seconds} seconds")
        if duration > self._max_seconds:
            raise AudioRejected(f"audio exceeds {self._max_seconds} seconds")
        if channel_preference == "separate" and channels == 1:
            raise AudioRejected("separate channels requested for mono audio")

        sha256 = _sha256(upload_path)
        _require_unchanged_source(upload_path, source_identity)
        job_directory = self._store.job_dir(job_id)
        temporary_paths: list[Path] = []
        committed_paths: list[Path] = []

        try:
            mixed_temporary = _temporary_wav(job_directory, "mixed")
            temporary_paths.append(mixed_temporary)
            _normalize_mixed(upload_path, mixed_temporary)
            _require_unchanged_source(upload_path, source_identity)

            selected_mode: Literal["mixed", "separate"] = "mixed"
            left_temporary: Path | None = None
            right_temporary: Path | None = None
            if channels > 1 and channel_preference != "mixed":
                left_temporary = _temporary_wav(job_directory, "left")
                right_temporary = _temporary_wav(job_directory, "right")
                temporary_paths.extend((left_temporary, right_temporary))
                _normalize_stereo_channels(upload_path, left_temporary, right_temporary)
                _require_unchanged_source(upload_path, source_identity)
                if channel_preference == "separate" or _channels_are_distinct(
                    left_temporary, right_temporary
                ):
                    selected_mode = "separate"

            if self._store.job_dir(job_id) != job_directory:
                raise ValueError("job directory changed while audio was being normalized")

            mixed_target = job_directory / "mixed.wav"
            os.replace(mixed_temporary, mixed_target)
            committed_paths.append(mixed_target)
            normalized_paths: tuple[str, ...]
            if selected_mode == "separate":
                if left_temporary is None or right_temporary is None:
                    raise AudioRejected(_DECODABILITY_ERROR)
                left_target = job_directory / "left.wav"
                right_target = job_directory / "right.wav"
                os.replace(left_temporary, left_target)
                committed_paths.append(left_target)
                os.replace(right_temporary, right_target)
                committed_paths.append(right_target)
                normalized_paths = (str(left_target), str(right_target))
            else:
                _unlink_all((job_directory / "left.wav", job_directory / "right.wav"))
                normalized_paths = (str(mixed_target),)
        except AudioRejected:
            _unlink_all((*temporary_paths, *committed_paths))
            raise
        except (OSError, ValueError):
            _unlink_all((*temporary_paths, *committed_paths))
            raise AudioRejected(_DECODABILITY_ERROR) from None
        finally:
            _unlink_all(temporary_paths)

        return AudioAsset(
            source_name=upload_path.name,
            source_path=str(upload_path),
            normalized_paths=normalized_paths,
            channel_mode=selected_mode,
            duration_seconds=duration,
            channels=channels,
            sha256=sha256,
        )


__all__ = ["AudioRejected", "AudioService", "ChannelPreference"]
