"""Opt-in, local-only NVIDIA Nemotron 3 Diarization through the NeMo-Speech.cpp CLI.

The adapter never downloads anything: it is constructed with an explicit local
``nemo-speech`` executable and an explicit local GGUF file, and runs the executable with a
scrubbed environment so that neither an ambient ``NEMO_SPEECH_*`` override nor a proxy
setting can change what runs. Passing an existing local GGUF path is the runtime's
documented no-network path; installing the runtime and pulling the model are separate,
documented setup steps.

Only the runtime's structured JSON output is parsed. Its ``file`` field (which echoes the
input path) is ignored, runtime stderr is discarded, and every failure is reported through
a fixed provider error code, so neither a path nor model detail reaches a user.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import wave
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Literal

from voxdelta.domain.models import AudioAsset, ProviderProvenance, SpeakerSegment
from voxdelta.providers.base import DiarizationTimelines, ProviderError

NEMOTRON_MODEL_ID = "nvidia/Nemotron-3-Diarization"
NEMOTRON_MODEL_NAME = "Nemotron-3-Diarization"
PROVIDER_NAME = "nemo-speech"

# The model card's "Offline" latency setting (30.4 s), in 80 ms encoder frames. It is
# passed field by field because the runtime's own ``v3-offline`` preset is a different,
# smaller geometry (a 264-frame chunk), and this is the configuration the A/B names.
OFFLINE_30_4S_GEOMETRY: Mapping[str, int] = {
    "chunk": 340,
    "right_context": 40,
    "left_context": 0,
    "fifo": 40,
    "spkcache": 264,
    "update_period": 300,
}
PRESET_NAME = "model-card-offline-30.4s"

REQUIRED_SAMPLE_RATE = 16_000
MAX_SPEAKERS = 8
# The runtime pads segment ends; anything further past the end of the audio is not a
# rounding artifact but a malformed result.
END_TOLERANCE_SECONDS = 0.5
MAX_OUTPUT_BYTES = 8 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 600.0
_VERSION_PATTERN = re.compile(r"^nemo-speech ([0-9A-Za-z.+-]{1,64})$")

Device = Literal["auto", "metal", "cpu"]


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int
    stdout: bytes


class CommandRunner:
    """Execute one local command; the only seam unit tests replace."""

    def __call__(
        self, argv: Sequence[str], *, env: Mapping[str, str], timeout_seconds: float
    ) -> CommandResult:
        completed = subprocess.run(  # noqa: S603 - fixed argv, never a shell
            list(argv),
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout_seconds,
            check=False,
        )
        return CommandResult(completed.returncode, completed.stdout)


Runner = Callable[..., CommandResult]


@dataclass(frozen=True, slots=True, order=True)
class Turn:
    start: float
    end: float
    label: int


def _validated_executable(path: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        raise ProviderError("local_runtime_missing") from None
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ProviderError("local_runtime_missing")
    return resolved


def _validated_model(path: Path) -> Path:
    try:
        if path.is_symlink():
            raise ProviderError("local_model_missing")
        resolved = path.resolve(strict=True)
    except ProviderError:
        raise
    except (OSError, RuntimeError):
        raise ProviderError("local_model_missing") from None
    if not resolved.is_file() or resolved.suffix != ".gguf":
        raise ProviderError("local_model_missing")
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError:
        raise ProviderError("local_model_missing") from None
    return digest.hexdigest()


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ProviderError("invalid_provider_output")
    converted = float(value)
    if not isfinite(converted):
        raise ProviderError("invalid_provider_output")
    return converted


def parse_segments(stdout: bytes, duration: float) -> list[Turn]:
    """Validate the runtime's JSON document and return bounded, merged speaker turns.

    Accepts exactly ``{"file": str, "segments": [{"start", "end", "speaker"}, ...]}``
    with 1-based integer speakers. Times are clipped to the audio; a turn that ends more
    than ``END_TOLERANCE_SECONDS`` past it, or has no positive extent, is rejected.
    """

    if len(stdout) > MAX_OUTPUT_BYTES:
        raise ProviderError("invalid_provider_output")
    try:
        document = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ProviderError("invalid_provider_output") from None
    if not isinstance(document, dict) or set(document) != {"file", "segments"}:
        raise ProviderError("invalid_provider_output")
    if not isinstance(document["file"], str) or not isinstance(document["segments"], list):
        raise ProviderError("invalid_provider_output")
    turns: list[Turn] = []
    for raw in document["segments"]:
        if not isinstance(raw, dict) or set(raw) != {"start", "end", "speaker"}:
            raise ProviderError("invalid_provider_output")
        speaker = raw["speaker"]
        if isinstance(speaker, bool) or not isinstance(speaker, int):
            raise ProviderError("invalid_provider_output")
        if not 1 <= speaker <= MAX_SPEAKERS:
            raise ProviderError("invalid_provider_output")
        start, end = _number(raw["start"]), _number(raw["end"])
        if start < 0 or end <= start or end > duration + END_TOLERANCE_SECONDS:
            raise ProviderError("invalid_provider_output")
        bounded_end = min(end, duration)
        if bounded_end <= start:
            # Wholly inside the tolerated tail: padding past the audio, not speech.
            continue
        turns.append(Turn(start, bounded_end, speaker))
    if not turns:
        raise ProviderError("invalid_provider_output")
    return _merge_same_speaker(turns)


def _merge_same_speaker(turns: Sequence[Turn]) -> list[Turn]:
    merged: list[Turn] = []
    for label in sorted({turn.label for turn in turns}):
        current: Turn | None = None
        for turn in sorted(item for item in turns if item.label == label):
            if current is not None and turn.start <= current.end:
                current = Turn(current.start, max(current.end, turn.end), label)
                continue
            if current is not None:
                merged.append(current)
            current = turn
        if current is not None:
            merged.append(current)
    return sorted(merged)


def speaker_map(turns: Sequence[Turn]) -> dict[int, str]:
    """Order runtime speakers by first activity so the mapping is reproducible."""

    labels = list(dict.fromkeys(turn.label for turn in sorted(turns)))
    return {label: f"SPEAKER_{index:02d}" for index, label in enumerate(labels)}


def evidence_segments(turns: Sequence[Turn], speakers: Mapping[int, str]) -> list[SpeakerSegment]:
    """Keep every turn, flagging those that overlap another speaker's turn."""

    return [
        SpeakerSegment(
            start=turn.start,
            end=turn.end,
            speaker_id=speakers[turn.label],
            overlap=any(
                other.label != turn.label and turn.start < other.end and other.start < turn.end
                for other in turns
            ),
            # The CLI exposes thresholded segments, not frame probabilities.
            confidence=1.0,
        )
        for turn in turns
    ]


def exclusive_segments(turns: Sequence[Turn], speakers: Mapping[int, str]) -> list[SpeakerSegment]:
    """Resolve overlap into one speaker per instant for transcript reconciliation.

    Where turns overlap, the speaker whose turn began first keeps the floor (ties go to
    the earlier-mapped speaker), and adjacent pieces of the same speaker are joined. With
    no probabilities exposed, this is a deterministic floor-holding rule, not a model
    decision.
    """

    order = {label: index for index, label in enumerate(speakers)}
    boundaries = sorted({point for turn in turns for point in (turn.start, turn.end)})
    pieces: list[Turn] = []
    for left, right in zip(boundaries, boundaries[1:], strict=False):
        active = [turn for turn in turns if turn.start <= left and right <= turn.end]
        if not active:
            continue
        holder = min(active, key=lambda turn: (turn.start, order[turn.label]))
        if pieces and pieces[-1].label == holder.label and pieces[-1].end == left:
            pieces[-1] = Turn(pieces[-1].start, right, holder.label)
        else:
            pieces.append(Turn(left, right, holder.label))
    return [
        SpeakerSegment(
            start=piece.start,
            end=piece.end,
            speaker_id=speakers[piece.label],
            overlap=False,
            confidence=1.0,
        )
        for piece in pieces
    ]


def _require_normalized_wav(path: str) -> None:
    """Hold the provider input invariant: 16 kHz, mono, 16-bit PCM WAV."""

    try:
        with wave.open(path, "rb") as source:
            valid = (
                source.getframerate() == REQUIRED_SAMPLE_RATE
                and source.getnchannels() == 1
                and source.getsampwidth() == 2
                and source.getcomptype() == "NONE"
                and source.getnframes() > 0
            )
    except (OSError, EOFError, wave.Error):
        raise ProviderError("invalid_audio_asset") from None
    if not valid:
        raise ProviderError("invalid_audio_asset")


class NemotronDiarizationProvider:
    """Run Nemotron 3 Diarization locally; the source audio never leaves the machine."""

    def __init__(
        self,
        *,
        executable_path: Path,
        model_path: Path,
        device: Device = "auto",
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        runner: Runner | None = None,
    ) -> None:
        if not isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ProviderError("provider_unavailable")
        self._executable = _validated_executable(executable_path)
        self._model = _validated_model(model_path)
        self._device: Device = device
        self._timeout = float(timeout_seconds)
        self._runner: Runner = runner or CommandRunner()
        self.model_sha256 = _file_sha256(self._model)
        self.runtime_version = self._runtime_version()
        self.provenance = ProviderProvenance(
            name=PROVIDER_NAME,
            model=NEMOTRON_MODEL_NAME,
            remote=False,
            revision=(
                f"nemo-speech-{self.runtime_version}+gguf-sha256-{self.model_sha256}+{PRESET_NAME}"
            ),
        )

    @property
    def configuration(self) -> dict[str, object]:
        """Path-free record of what runs, for evaluation artifacts."""

        return {
            "model_id": NEMOTRON_MODEL_ID,
            "model_file": self._model.name,
            "model_sha256": self.model_sha256,
            "runtime": PROVIDER_NAME,
            "runtime_version": self.runtime_version,
            "device": self._device,
            "preset": PRESET_NAME,
            "geometry_80ms_frames": dict(OFFLINE_30_4S_GEOMETRY),
            "segmentation": "runtime default (no threshold override)",
            "output_format": "json",
            "timeout_seconds": self._timeout,
        }

    def _environment(self) -> dict[str, str]:
        # Nothing is inherited: no proxy, no ambient NEMO_SPEECH_* override, no token.
        env = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}
        for key, value in OFFLINE_30_4S_GEOMETRY.items():
            env[f"NEMO_SPEECH_DIAR_{key.upper()}"] = str(value)
        return env

    def _execute(self, argv: Sequence[str]) -> bytes:
        try:
            result = self._runner(argv, env=self._environment(), timeout_seconds=self._timeout)
        except (subprocess.TimeoutExpired, TimeoutError):
            raise ProviderError("provider_timeout") from None
        except FileNotFoundError:
            raise ProviderError("local_runtime_missing") from None
        except Exception:
            raise ProviderError("provider_unavailable") from None
        if not isinstance(result, CommandResult) or not isinstance(result.stdout, bytes):
            raise ProviderError("provider_unavailable")
        if result.returncode != 0:
            raise ProviderError("provider_unavailable")
        return result.stdout

    def _runtime_version(self) -> str:
        stdout = self._execute([str(self._executable), "--version"])
        match = _VERSION_PATTERN.match(stdout.decode("utf-8", "replace").strip())
        if match is None:
            raise ProviderError("local_runtime_missing")
        return match.group(1)

    @staticmethod
    def _validate_asset(asset: AudioAsset) -> tuple[float, tuple[str, ...]]:
        duration = asset.duration_seconds
        if duration is None or not isfinite(duration) or duration <= 0:
            raise ProviderError("invalid_audio_asset")
        paths = asset.normalized_paths
        expected = {"mixed": 1, "separate": 2}.get(asset.channel_mode or "", 0)
        if len(paths) != expected or any(not path for path in paths):
            raise ProviderError("invalid_audio_asset")
        for path in paths:
            _require_normalized_wav(path)
        return duration, paths

    def _diarize_path(self, path: str, duration: float) -> list[Turn]:
        # Re-checked per call: a removed runtime or model is reported as such, not as a
        # generic failure of the run.
        _validated_executable(self._executable)
        _validated_model(self._model)
        argv = [
            str(self._executable),
            "diarize",
            path,
            "--model",
            str(self._model),
            "--device",
            self._device,
            "--format",
            "json",
        ]
        return parse_segments(self._execute(argv), duration)

    def diarize_unconstrained(self, asset: AudioAsset) -> DiarizationTimelines:
        """Diarize without the product's two-speaker gate, for evaluation only."""

        duration, paths = self._validate_asset(asset)
        if asset.channel_mode == "separate":
            return self._separate_timelines(paths, duration)
        turns = self._diarize_path(paths[0], duration)
        speakers = speaker_map(turns)
        return DiarizationTimelines(
            evidence=evidence_segments(turns, speakers),
            exclusive=exclusive_segments(turns, speakers),
        )

    def _separate_timelines(self, paths: tuple[str, ...], duration: float) -> DiarizationTimelines:
        # One speaker per channel by construction: any activity the model finds on a
        # channel is that channel's speaker, so cross-talk splits are folded back.
        turns: list[Turn] = []
        for index, path in enumerate(paths):
            channel = self._diarize_path(path, duration)
            turns.extend(
                _merge_same_speaker([Turn(turn.start, turn.end, index + 1) for turn in channel])
            )
        ordered = sorted(turns)
        speakers = {1: "SPEAKER_00", 2: "SPEAKER_01"}
        return DiarizationTimelines(
            evidence=evidence_segments(ordered, speakers),
            exclusive=exclusive_segments(ordered, speakers),
        )

    def diarize_timelines(self, asset: AudioAsset) -> DiarizationTimelines:
        timelines = self.diarize_unconstrained(asset)
        if len({segment.speaker_id for segment in timelines.evidence}) != 2:
            raise ProviderError("unsupported_speaker_count")
        return timelines

    def diarize(self, asset: AudioAsset) -> list[SpeakerSegment]:
        return self.diarize_timelines(asset).evidence

    def diarize_for_alignment(self, asset: AudioAsset) -> list[SpeakerSegment]:
        return self.diarize_timelines(asset).exclusive


__all__ = [
    "NEMOTRON_MODEL_ID",
    "OFFLINE_30_4S_GEOMETRY",
    "PRESET_NAME",
    "CommandResult",
    "CommandRunner",
    "NemotronDiarizationProvider",
    "Turn",
    "exclusive_segments",
    "evidence_segments",
    "parse_segments",
    "speaker_map",
]
