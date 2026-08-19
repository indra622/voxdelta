"""Shared safe runtime for local seven-emotion adapters."""

from __future__ import annotations

import json
import math
import sys
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Literal, Protocol, cast

from voxdelta.analysis.emotions import map_operational_state
from voxdelta.domain.models import EmotionLabel, EmotionResult, ProviderProvenance, ProviderUsage
from voxdelta.evaluation.emotion_training import (
    CANONICAL_LABELS,
    evaluation_windows,
    load_audio,
)
from voxdelta.evaluation.manifest import read_trusted_regular_file
from voxdelta.providers.base import ProviderError, ProviderErrorCode

Device = Literal["auto", "cpu", "cuda", "mps"]
Architecture = Literal["wav2vec-xls-r", "emotion2vec-plus"]
LOCAL_EMOTION_INFERENCE_LOCK = threading.Lock()
_ACTIVE_OWNER: int | None = None
_ACTIVE_UNLOAD: Callable[[], None] | None = None


class Predictor(Protocol):
    def predict(self, samples: tuple[float, ...], sample_rate: int) -> Sequence[float]: ...


@dataclass(frozen=True, slots=True)
class CheckpointInfo:
    path: Path
    architecture: Architecture
    model_id: str
    encoder_revision: str | None
    encoder_hash: str | None
    freeze_encoder: bool | None


def activate_candidate(owner: object, unload: Callable[[], None]) -> None:
    """Keep at most one large local emotion candidate resident."""

    global _ACTIVE_OWNER, _ACTIVE_UNLOAD
    owner_id = id(owner)
    if _ACTIVE_OWNER == owner_id:
        return
    previous = _ACTIVE_UNLOAD
    _ACTIVE_OWNER = owner_id
    _ACTIVE_UNLOAD = unload
    if previous is not None:
        try:
            previous()
        except Exception:
            pass


def prepare_candidate_load(owner: object) -> None:
    """Evict any prior candidate before a new model factory allocates memory."""

    global _ACTIVE_OWNER, _ACTIVE_UNLOAD
    if _ACTIVE_OWNER == id(owner):
        return
    previous = _ACTIVE_UNLOAD
    _ACTIVE_OWNER = None
    _ACTIVE_UNLOAD = None
    if previous is not None:
        try:
            previous()
        except Exception:
            pass


def release_candidate(owner: object) -> None:
    global _ACTIVE_OWNER, _ACTIVE_UNLOAD
    if _ACTIVE_OWNER == id(owner):
        _ACTIVE_OWNER = None
        _ACTIVE_UNLOAD = None


def _sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value.lower() == value
        and all(character in "0123456789abcdef" for character in value)
    )


def _invalid_json_constant(value: str) -> None:
    del value
    raise ValueError


def _load_strict_json(path: Path) -> object:
    raw = read_trusted_regular_file(path)
    if len(raw) > 1024 * 1024:
        raise ValueError
    return json.loads(raw, parse_constant=_invalid_json_constant)


def validate_checkpoint(
    checkpoint: str | Path,
    *,
    architecture: Architecture,
    model_id: str,
) -> CheckpointInfo:
    try:
        candidate = Path(checkpoint)
        if ".." in candidate.parts:
            raise ValueError
        current = Path(candidate.anchor) if candidate.is_absolute() else Path()
        for part in candidate.parts[1:] if candidate.is_absolute() else candidate.parts:
            current /= part
            if current.is_symlink():
                raise ValueError
        resolved = candidate.resolve(strict=True)
        if not resolved.is_dir() or resolved.is_symlink():
            raise ValueError
        required = ("config.json", "label_mapping.json", "metrics.json", "model.safetensors")
        if any(
            (resolved / name).is_symlink() or not (resolved / name).is_file() for name in required
        ):
            raise ValueError
        config = _load_strict_json(resolved / "config.json")
        mapping = _load_strict_json(resolved / "label_mapping.json")
        metrics = _load_strict_json(resolved / "metrics.json")
        weights = read_trusted_regular_file(resolved / "model.safetensors")
        expected_mapping = {str(index): label for index, label in enumerate(CANONICAL_LABELS)}
        expected_config_keys = {"schema_version", "architecture", "model_id", "labels"}
        if architecture == "emotion2vec-plus":
            expected_config_keys.update(
                {"embedding_size", "encoder_hash", "encoder_revision", "freeze_encoder"}
            )
        allowed_metric_keys = {
            "macro_f1",
            "validation_hash",
            "expected_calibration_error",
        }
        required_metric_keys = {"macro_f1", "validation_hash"}
        if (
            not isinstance(config, dict)
            or set(config) != expected_config_keys
            or config.get("schema_version") != "1"
            or config.get("architecture") != architecture
            or config.get("model_id") != model_id
            or config.get("labels") != list(CANONICAL_LABELS)
            or mapping != expected_mapping
            or not isinstance(metrics, dict)
            or not required_metric_keys.issubset(metrics)
            or not set(metrics).issubset(allowed_metric_keys)
            or not _sha(metrics.get("validation_hash"))
            or type(metrics.get("macro_f1")) is not float
            or not math.isfinite(metrics["macro_f1"])
            or not 0 <= metrics["macro_f1"] <= 1
            or (
                "expected_calibration_error" in metrics
                and (
                    type(metrics["expected_calibration_error"]) is not float
                    or not math.isfinite(metrics["expected_calibration_error"])
                    or not 0 <= metrics["expected_calibration_error"] <= 1
                )
            )
            or not weights
        ):
            raise ValueError
        encoder_revision: str | None = None
        encoder_hash: str | None = None
        freeze_encoder: bool | None = None
        if architecture == "emotion2vec-plus":
            embedding_size = config.get("embedding_size")
            if (
                not _sha(config.get("encoder_hash"))
                or config.get("encoder_revision") != "v2.0.4"
                or config.get("freeze_encoder") is not True
                or isinstance(embedding_size, bool)
                or not isinstance(embedding_size, int)
                or embedding_size <= 0
            ):
                raise ValueError
            encoder_revision = cast(str, config["encoder_revision"])
            encoder_hash = cast(str, config["encoder_hash"])
            freeze_encoder = True
        elif "encoder_hash" in config or "freeze_encoder" in config:
            raise ValueError
        return CheckpointInfo(
            resolved,
            architecture,
            model_id,
            encoder_revision,
            encoder_hash,
            freeze_encoder,
        )
    except Exception:
        raise ProviderError("invalid_local_checkpoint") from None


def default_hardware_probe() -> tuple[bool, bool]:
    torch = import_module("torch")
    return bool(torch.cuda.is_available()), bool(torch.backends.mps.is_available())


def select_device(requested: Device, probe: Callable[[], tuple[bool, bool]]) -> str:
    try:
        cuda, mps = probe()
    except Exception:
        raise ProviderError("provider_runtime_unsupported") from None
    if not isinstance(cuda, bool) or not isinstance(mps, bool):
        raise ProviderError("provider_runtime_unsupported")
    if requested == "auto":
        return "cuda" if cuda else "mps" if mps else "cpu"
    if requested == "cuda" and not cuda or requested == "mps" and not mps:
        raise ProviderError("provider_runtime_unsupported")
    return requested


def default_inference_context() -> AbstractContextManager[object]:
    return cast(AbstractContextManager[object], import_module("torch").inference_mode())


def rss_megabytes(raw_rss: int | float, platform: str) -> float:
    value = float(raw_rss)
    if not math.isfinite(value) or value < 0:
        raise ValueError("invalid rss")
    return value / (1024 * 1024) if platform == "darwin" else value / 1024


def default_rss_probe() -> float | None:
    try:
        if sys.platform == "win32":
            psutil = import_module("psutil")
            process = psutil.Process()
            return float(process.memory_info().rss) / (1024 * 1024)
        resource = import_module("resource")
        maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss_megabytes(maximum, sys.platform)
    except Exception:
        return None


def _probabilities(window_logits: Sequence[Sequence[float]]) -> dict[EmotionLabel, float]:
    if not window_logits:
        raise ProviderError("invalid_provider_output")
    normalized: list[tuple[float, ...]] = []
    for raw in window_logits:
        try:
            row = tuple(float(value) for value in raw)
        except (TypeError, ValueError, OverflowError):
            raise ProviderError("invalid_provider_output") from None
        if len(row) != len(CANONICAL_LABELS) or any(not math.isfinite(value) for value in row):
            raise ProviderError("invalid_provider_output")
        normalized.append(row)
    try:
        means = tuple(
            math.fsum(row[index] for row in normalized) / len(normalized)
            for index in range(len(CANONICAL_LABELS))
        )
        peak = max(means)
        exponentials = tuple(math.exp(value - peak) for value in means)
        denominator = math.fsum(exponentials)
    except (OverflowError, ValueError):
        raise ProviderError("invalid_provider_output") from None
    if not math.isfinite(denominator) or denominator <= 0:
        raise ProviderError("invalid_provider_output")
    values = tuple(value / denominator for value in exponentials)
    if any(not math.isfinite(value) for value in values):
        raise ProviderError("invalid_provider_output")
    return {label: values[index] for index, label in enumerate(CANONICAL_LABELS)}


def _safe_runtime_error(error: Exception) -> ProviderError:
    code: ProviderErrorCode
    if isinstance(error, TimeoutError) or "timeout" in type(error).__name__.lower():
        code = "provider_timeout"
    elif any(
        marker in str(error).lower()
        for marker in ("out of memory", "not implemented", "unsupported", "mps")
    ):
        code = "provider_runtime_unsupported"
    else:
        code = "provider_unavailable"
    return ProviderError(code)


def analyze_local_emotion(
    *,
    utterance_id: str,
    audio_path: Path,
    transcript: str,
    provenance: ProviderProvenance,
    load_predictor: Callable[[], Predictor],
    inference_context: Callable[[], AbstractContextManager[object]],
    clock: Callable[[], float] = time.perf_counter,
    rss_probe: Callable[[], float | None] = default_rss_probe,
) -> EmotionResult:
    del transcript
    if not utterance_id:
        raise ProviderError("invalid_provider_output")
    try:
        clip = load_audio(audio_path)
    except ValueError:
        raise ProviderError("invalid_audio_asset") from None
    start = clock()
    try:
        with LOCAL_EMOTION_INFERENCE_LOCK, inference_context():
            predictor = load_predictor()
            logits = [
                predictor.predict(window.samples, window.sample_rate)
                for window in evaluation_windows(clip)
            ]
            probabilities = _probabilities(logits)
    except ProviderError:
        raise
    except Exception as error:
        raise _safe_runtime_error(error) from None
    elapsed = clock() - start
    if not math.isfinite(elapsed) or elapsed < 0:
        elapsed = 0.0
    rss: float | None
    try:
        raw_rss = rss_probe()
        measured_rss = None if raw_rss is None else float(raw_rss)
        rss = (
            measured_rss
            if measured_rss is not None and math.isfinite(measured_rss) and measured_rss >= 0
            else None
        )
    except Exception:
        rss = None
    confidence = max(probabilities.values())
    negative_labels: tuple[EmotionLabel, ...] = ("anger", "disgust", "fear", "sadness")
    negative = math.fsum(probabilities[label] for label in negative_labels)
    return EmotionResult(
        utterance_id=utterance_id,
        probabilities=probabilities,
        operational_state=map_operational_state(probabilities, confidence),
        negative_intensity=negative,
        confidence=confidence,
        provider=provenance,
        usage=ProviderUsage(latency_ms=elapsed * 1000, peak_rss_mb=rss),
    )
