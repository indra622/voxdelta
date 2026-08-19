"""Validate uploaded call audio and create job-scoped normalized WAV files."""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import wave
from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal
from uuid import uuid4

from voxdelta.domain.models import AudioAsset
from voxdelta.jobs.artifacts import ArtifactStore

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback keeps process ownership only
    fcntl = None  # type: ignore[assignment]

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


@dataclass(slots=True)
class PreparedAudio:
    """Private decoded media staged for one later job-scoped atomic publication."""

    workspace: Path
    source_name: str
    sha256: str
    duration_seconds: float
    channels: int
    channel_mode: Literal["mixed", "separate"]
    normalized_names: tuple[str, ...]
    lease_descriptor: int
    published: bool = False

    def discard(self) -> None:
        """Release the workspace lease and remove media that was never published."""

        if self.lease_descriptor >= 0:
            _release_workspace_lease(self.workspace, self.lease_descriptor)
            self.lease_descriptor = -1
        if not self.published and self.workspace.exists() and not self.workspace.is_symlink():
            shutil.rmtree(self.workspace, ignore_errors=True)


def _release_workspace_lease(workspace: Path, descriptor: int) -> None:
    try:
        (workspace / ".lease").unlink(missing_ok=True)
    except OSError:
        pass
    try:
        os.close(descriptor)
    except OSError:
        pass


def _create_ingest_workspace(parent: Path) -> tuple[Path, int]:
    workspace = Path(tempfile.mkdtemp(dir=parent, prefix=".ingest-"))
    lease = workspace / ".lease"
    flags = os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    descriptor = -1
    try:
        descriptor = os.open(lease, flags, 0o600)
        os.fchmod(descriptor, 0o600)
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        return workspace, descriptor
    except BaseException:
        if descriptor >= 0:
            _release_workspace_lease(workspace, descriptor)
        shutil.rmtree(workspace, ignore_errors=True)
        raise


def _is_link_like(metadata: os.stat_result) -> bool:
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(file_attributes & reparse_attribute)


def _source_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns)


def _open_trusted_source(source: Path) -> BinaryIO:
    """Open the final source component once without following it on POSIX.

    Windows lacks a portable ``O_NOFOLLOW`` equivalent. There we reject reparse points before
    opening and compare the path metadata with the opened descriptor as a practical fallback.
    """

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    before: os.stat_result | None = None
    if os.name == "posix":
        flags |= getattr(os, "O_NOFOLLOW", 0)
    else:
        try:
            before = source.lstat()
        except OSError:
            raise AudioRejected(_DECODABILITY_ERROR) from None
        if _is_link_like(before) or not stat.S_ISREG(before.st_mode):
            raise AudioRejected(_DECODABILITY_ERROR)

    try:
        descriptor = os.open(source, flags)
    except OSError:
        raise AudioRejected(_DECODABILITY_ERROR) from None
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise AudioRejected(_DECODABILITY_ERROR)
        if before is not None:
            try:
                after = source.lstat()
            except OSError:
                raise AudioRejected(_DECODABILITY_ERROR) from None
            if (
                _is_link_like(after)
                or _source_identity(before) != _source_identity(opened)
                or _source_identity(after) != _source_identity(opened)
            ):
                raise AudioRejected(_DECODABILITY_ERROR)
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def _stage_source(source_file: BinaryIO, workspace: Path, suffix: str) -> tuple[Path, str]:
    staged = workspace / f"source{suffix.lower()}"
    digest = hashlib.sha256()
    try:
        with staged.open("xb") as output:
            while chunk := source_file.read(_HASH_CHUNK_BYTES):
                output.write(chunk)
                digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())
    except OSError:
        raise AudioRejected(_DECODABILITY_ERROR) from None
    return staged, digest.hexdigest()


def _probe(source: Path) -> tuple[float, int, bool]:
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
    channel_layout = audio_streams[0].get("channel_layout")
    if channel_layout is not None and not isinstance(channel_layout, str):
        raise AudioRejected(_DECODABILITY_ERROR)
    is_stereo = channels == 2 and channel_layout in {None, "stereo"}

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
    return duration, channels, is_stereo


def _temporary_wav(job_directory: Path, label: str) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=job_directory,
        prefix=f".{label}.",
        suffix=".wav",
    )
    os.close(descriptor)
    return Path(name)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as file:
        os.fsync(file.fileno())


def _fsync_directory(directory: Path) -> None:
    if os.name != "posix":
        return
    unsupported = {
        errno.EBADF,
        errno.EINVAL,
        getattr(errno, "ENOTSUP", errno.EINVAL),
        getattr(errno, "EOPNOTSUPP", errno.EINVAL),
    }
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(directory, flags)
    except OSError as error:
        if error.errno in unsupported:
            return
        raise
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            if error.errno not in unsupported:
                raise
    finally:
        os.close(descriptor)


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


def _decoded_wav_duration(path: Path) -> float:
    """Return authoritative media time from the normalized PCM frame count."""

    try:
        with wave.open(str(path), "rb") as audio:
            if (
                audio.getnchannels() != 1
                or audio.getsampwidth() != 2
                or audio.getframerate() != 16_000
                or audio.getnframes() <= 0
            ):
                raise AudioRejected(_DECODABILITY_ERROR)
            duration = audio.getnframes() / audio.getframerate()
    except (EOFError, OSError, wave.Error, ZeroDivisionError):
        raise AudioRejected(_DECODABILITY_ERROR) from None
    if not math.isfinite(duration) or duration <= 0:
        raise AudioRejected(_DECODABILITY_ERROR)
    return duration


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
        self._generation_leases: dict[Path, int] = {}
        self._generation_leases_lock = threading.Lock()

    def _prepare(
        self,
        upload_path: Path,
        workspace_parent: Path,
        channel_preference: ChannelPreference = "auto",
        *,
        source_file: BinaryIO | None = None,
    ) -> PreparedAudio:
        """Decode one stable source snapshot without publishing durable job media."""

        if upload_path.suffix.lower() not in _ALLOWED_EXTENSIONS:
            raise AudioRejected("unsupported extension")
        workspace: Path | None = None
        lease_descriptor = -1
        temporary_paths: list[Path] = []
        prepared = False

        try:
            if source_file is None:
                with _open_trusted_source(upload_path) as opened_source:
                    workspace, lease_descriptor = _create_ingest_workspace(workspace_parent)
                    staged_source, sha256 = _stage_source(
                        opened_source, workspace, upload_path.suffix
                    )
            else:
                workspace, lease_descriptor = _create_ingest_workspace(workspace_parent)
                staged_source, sha256 = _stage_source(source_file, workspace, upload_path.suffix)

            _, channels, is_stereo = _probe(staged_source)
            if channel_preference == "separate" and channels == 1:
                raise AudioRejected("separate channels requested for mono audio")
            if channel_preference == "separate" and not is_stereo:
                raise AudioRejected(_DECODABILITY_ERROR)

            mixed_temporary = _temporary_wav(workspace, "mixed")
            temporary_paths.append(mixed_temporary)
            _normalize_mixed(staged_source, mixed_temporary)
            mixed_output = workspace / "mixed.wav"
            os.replace(mixed_temporary, mixed_output)
            duration = _decoded_wav_duration(mixed_output)
            if duration < self._min_seconds:
                raise AudioRejected(f"audio is shorter than {self._min_seconds} seconds")
            if duration > self._max_seconds:
                raise AudioRejected(f"audio exceeds {self._max_seconds} seconds")

            selected_mode: Literal["mixed", "separate"] = "mixed"
            left_temporary: Path | None = None
            right_temporary: Path | None = None
            if is_stereo and channel_preference != "mixed":
                left_temporary = _temporary_wav(workspace, "left")
                temporary_paths.append(left_temporary)
                right_temporary = _temporary_wav(workspace, "right")
                temporary_paths.append(right_temporary)
                _normalize_stereo_channels(staged_source, left_temporary, right_temporary)
                if channel_preference == "separate" or _channels_are_distinct(
                    left_temporary, right_temporary
                ):
                    selected_mode = "separate"

            normalized_names: tuple[str, ...]
            if selected_mode == "separate":
                if left_temporary is None or right_temporary is None:
                    raise AudioRejected(_DECODABILITY_ERROR)
                left_output = workspace / "left.wav"
                right_output = workspace / "right.wav"
                os.replace(left_temporary, left_output)
                os.replace(right_temporary, right_output)
                normalized_names = ("left.wav", "right.wav")
            else:
                normalized_names = ("mixed.wav",)

            staged_source.unlink()
            output_names = (
                ("mixed.wav", *normalized_names)
                if selected_mode == "separate"
                else normalized_names
            )
            for name in dict.fromkeys(output_names):
                _fsync_file(workspace / name)
            _fsync_directory(workspace)
            prepared = True
        except OSError:
            raise AudioRejected(_DECODABILITY_ERROR) from None
        finally:
            _unlink_all(temporary_paths)
            if workspace is not None and not prepared:
                if lease_descriptor >= 0:
                    _release_workspace_lease(workspace, lease_descriptor)
                shutil.rmtree(workspace, ignore_errors=True)
        if workspace is None:
            raise AudioRejected(_DECODABILITY_ERROR)
        return PreparedAudio(
            workspace=workspace,
            source_name=upload_path.name,
            sha256=sha256,
            duration_seconds=duration,
            channels=channels,
            channel_mode=selected_mode,
            normalized_names=normalized_names,
            lease_descriptor=lease_descriptor,
        )

    def preflight(
        self,
        upload_path: Path,
        channel_preference: ChannelPreference = "auto",
    ) -> PreparedAudio:
        """Fully decode an incoming upload before durable job or database admission."""

        return self._prepare(upload_path, upload_path.parent, channel_preference)

    def publish_prepared(
        self,
        prepared: PreparedAudio,
        upload_path: Path,
        job_id: str,
        *,
        hold_generation_lease: bool = False,
    ) -> AudioAsset:
        """Atomically publish a preflighted generation beneath one validated job."""

        job_directory = self._store.job_dir(job_id)
        workspace = prepared.workspace
        if workspace.is_symlink() or not workspace.is_dir():
            raise AudioRejected(_DECODABILITY_ERROR)
        generation = job_directory / f"audio-{uuid4().hex}"
        try:
            os.replace(workspace, generation)
            _fsync_directory(job_directory)
        except OSError:
            raise AudioRejected(_DECODABILITY_ERROR) from None
        prepared.workspace = generation
        prepared.published = True
        if hold_generation_lease:
            with self._generation_leases_lock:
                if generation in self._generation_leases:
                    raise RuntimeError("audio generation lease already exists")
                self._generation_leases[generation] = prepared.lease_descriptor
                prepared.lease_descriptor = -1
        else:
            prepared.discard()
        normalized_paths = tuple(str(generation / name) for name in prepared.normalized_names)

        return AudioAsset(
            source_name=upload_path.name,
            source_path=str(upload_path),
            normalized_paths=normalized_paths,
            channel_mode=prepared.channel_mode,
            duration_seconds=prepared.duration_seconds,
            channels=prepared.channels,
            sha256=prepared.sha256,
        )

    def ingest(
        self,
        upload_path: Path,
        job_id: str,
        channel_preference: ChannelPreference = "auto",
        *,
        hold_generation_lease: bool = False,
    ) -> AudioAsset:
        """Validate an upload and return metadata for its selected normalized channel mode."""

        if upload_path.suffix.lower() not in _ALLOWED_EXTENSIONS:
            raise AudioRejected("unsupported extension")
        with _open_trusted_source(upload_path) as source_file:
            job_directory = self._store.job_dir(job_id)
            prepared = self._prepare(
                upload_path,
                job_directory,
                channel_preference,
                source_file=source_file,
            )
        try:
            if self._store.job_dir(job_id) != job_directory:
                raise ValueError("job directory changed while audio was being normalized")
            return self.publish_prepared(
                prepared,
                upload_path,
                job_id,
                hold_generation_lease=hold_generation_lease,
            )
        finally:
            prepared.discard()

    def release_generation_lease(self, asset: AudioAsset) -> None:
        """Release a normalized generation only after its stage publication is resolved."""

        generation = Path(asset.normalized_paths[0]).parent
        with self._generation_leases_lock:
            descriptor = self._generation_leases.pop(generation, None)
        if descriptor is None:
            return
        _release_workspace_lease(generation, descriptor)
        if generation.is_dir():
            try:
                _fsync_directory(generation)
            except OSError:
                pass


__all__ = ["AudioRejected", "AudioService", "ChannelPreference", "PreparedAudio"]
