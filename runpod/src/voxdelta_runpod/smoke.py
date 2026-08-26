"""Offline reload smoke check for an immutable XLS-R release bundle.

The release model card promises that inference loads entirely from local files:
the fine-tuned weights from `checkpoint/` and the feature extractor / architecture
from `base-model/`. This module executes that promise. It re-verifies the bundle,
forces the Hugging Face runtime into offline mode, reloads the provider from the
bundle alone, and runs one deterministic synthetic clip through it. No release
audio, manifest, or item identity is read, and only aggregates are returned.

Post-hoc calibration is a separately versioned, release-bound artifact. It is verified
and exercised through the production calibrated-provider path instead of mutating this
immutable release.
"""

from __future__ import annotations

import math
import os
import shutil
import struct
import tempfile
import wave
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from voxdelta.domain.models import EmotionResult
from voxdelta.providers._emotion_runtime import Device

from voxdelta_runpod.config import CANONICAL_LABELS
from voxdelta_runpod.release import ReleaseManifest, verify_release_bundle

SMOKE_SAMPLE_RATE = 16_000
SMOKE_DURATION_SECONDS = 2
SMOKE_UTTERANCE_ID = "release-smoke-0001"

# Reloading must never reach the network, whatever a stale cache or environment
# would otherwise permit. The provider already passes ``local_files_only=True``;
# these variables close the remaining Hugging Face fallbacks.
OFFLINE_ENVIRONMENT: tuple[tuple[str, str], ...] = (
    ("HF_HUB_OFFLINE", "1"),
    ("TRANSFORMERS_OFFLINE", "1"),
    ("HF_DATASETS_OFFLINE", "1"),
    ("HF_HUB_DISABLE_TELEMETRY", "1"),
)


class SmokeError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class EmotionAnalyzer(Protocol):
    def analyze(self, utterance_id: str, audio_path: Path, transcript: str) -> EmotionResult: ...


ProviderFactory = Callable[[Path, Path, Device], EmotionAnalyzer]


@dataclass(frozen=True, slots=True)
class SmokeResult:
    release_id: str
    candidate_checkpoint_sha256: str
    device: Device
    top_label: str
    confidence: float
    probability_sum: float
    negative_intensity: float
    latency_ms: float


def apply_offline_environment(environment: MutableMapping[str, str] | None = None) -> None:
    """Pin the Hugging Face runtime to local files before any model load."""

    target = os.environ if environment is None else environment
    for name, value in OFFLINE_ENVIRONMENT:
        target[name] = value


def default_provider_factory(
    checkpoint: Path,
    base_model: Path,
    device: Device,
) -> EmotionAnalyzer:
    from voxdelta.providers.wav2vec_emotion import Wav2VecEmotionProvider

    return Wav2VecEmotionProvider(checkpoint, base_model_path=base_model, device=device)


def _samples() -> tuple[int, ...]:
    """A fixed, reproducible voiced-like waveform; no RNG, no recorded audio."""

    count = SMOKE_SAMPLE_RATE * SMOKE_DURATION_SECONDS
    values: list[int] = []
    for index in range(count):
        seconds = index / SMOKE_SAMPLE_RATE
        amplitude = (
            0.40 * math.sin(2 * math.pi * 220.0 * seconds)
            + 0.25 * math.sin(2 * math.pi * 440.0 * seconds + 0.5)
            + 0.10 * math.sin(2 * math.pi * 55.0 * seconds)
        )
        values.append(max(-32768, min(32767, int(amplitude * 12000.0))))
    return tuple(values)


def write_smoke_audio(path: Path) -> Path:
    """Publish the synthetic 16 kHz mono PCM16 clip with a private mode."""

    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with open(descriptor, "wb") as stream:
        with wave.open(stream, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(SMOKE_SAMPLE_RATE)
            samples = _samples()
            writer.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return path


def _checked(result: EmotionResult, manifest: ReleaseManifest) -> tuple[str, float]:
    probabilities = result.probabilities
    if set(probabilities) != set(CANONICAL_LABELS):
        raise SmokeError("smoke_invalid_output")
    total = math.fsum(probabilities.values())
    top_label, top_value = max(probabilities.items(), key=lambda item: (item[1], item[0]))
    if (
        result.utterance_id != SMOKE_UTTERANCE_ID
        or not math.isfinite(total)
        or abs(total - 1.0) > 1e-6
        or abs(result.confidence - top_value) > 1e-9
        or not math.isfinite(result.negative_intensity)
    ):
        raise SmokeError("smoke_invalid_output")
    if result.provider.revision != manifest.candidate_checkpoint_sha256:
        raise SmokeError("smoke_checkpoint_identity_mismatch")
    return top_label, total


def run_release_smoke(
    bundle: Path,
    *,
    device: Device = "cpu",
    verifier: Callable[[Path], ReleaseManifest] = verify_release_bundle,
    provider_factory: ProviderFactory = default_provider_factory,
    environment: MutableMapping[str, str] | None = None,
) -> SmokeResult:
    """Re-verify one release bundle and prove it still infers offline from itself."""

    root = Path(bundle).resolve()
    try:
        manifest = verifier(root)
    except Exception:
        raise SmokeError("release_verification_failed") from None

    apply_offline_environment(environment)
    # Resolve the temporary root: the trusted reader rejects symlinked path
    # components, and platform temporary directories are often symlinks.
    workspace = Path(tempfile.mkdtemp(prefix=".release-smoke-")).resolve()
    try:
        os.chmod(workspace, 0o700)
        audio = write_smoke_audio(workspace / "smoke.wav")
        try:
            provider = provider_factory(root / "checkpoint", root / "base-model", device)
        except Exception:
            raise SmokeError("release_reload_failed") from None
        try:
            result = provider.analyze(SMOKE_UTTERANCE_ID, audio, "")
        except Exception:
            raise SmokeError("smoke_inference_failed") from None
        finally:
            unload = getattr(provider, "unload", None)
            if callable(unload):
                try:
                    unload()
                except Exception:
                    pass
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    top_label, total = _checked(result, manifest)
    usage = result.usage
    return SmokeResult(
        release_id=manifest.release_id,
        candidate_checkpoint_sha256=manifest.candidate_checkpoint_sha256,
        device=device,
        top_label=top_label,
        confidence=result.confidence,
        probability_sum=total,
        negative_intensity=result.negative_intensity,
        latency_ms=0.0 if usage is None else usage.latency_ms,
    )


__all__ = [
    "OFFLINE_ENVIRONMENT",
    "SMOKE_SAMPLE_RATE",
    "SMOKE_UTTERANCE_ID",
    "EmotionAnalyzer",
    "ProviderFactory",
    "SmokeError",
    "SmokeResult",
    "apply_offline_environment",
    "default_provider_factory",
    "run_release_smoke",
    "write_smoke_audio",
]
