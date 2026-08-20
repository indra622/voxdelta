"""Deterministic seven-emotion data, profile, cache, and checkpoint contracts."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import secrets
import shutil
import stat
import struct
import tempfile
import wave
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from voxdelta.domain.models import EmotionLabel
from voxdelta.evaluation.manifest import (
    load_manifest,
    read_trusted_regular_file,
    validate_disjoint_splits,
)

SAMPLE_RATE = 16_000
MIN_SAMPLES = SAMPLE_RATE // 2
WINDOW_SAMPLES = SAMPLE_RATE * 20
PREPROCESS_VERSION = "emotion-audio-v1"
EMOTION2VEC_REVISION: Literal["v2.0.5"] = "v2.0.5"
CANONICAL_LABELS: tuple[EmotionLabel, ...] = (
    "happiness",
    "anger",
    "disgust",
    "fear",
    "neutral",
    "sadness",
    "surprise",
)


class Wav2VecTrainingProfile(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    architecture: Literal["wav2vec-xls-r"] = "wav2vec-xls-r"
    model_id: Literal["facebook/wav2vec2-xls-r-300m"] = "facebook/wav2vec2-xls-r-300m"
    seed: Literal[622] = 622
    learning_rate: float = 2e-5
    train_batch_size: Literal[8] = 8
    eval_batch_size: Literal[8] = 8
    gradient_accumulation_steps: Literal[2] = 2
    epochs: Literal[10] = 10
    warmup_ratio: float = 0.1
    evaluation_strategy: Literal["epoch"] = "epoch"
    selection_metric: Literal["macro_f1"] = "macro_f1"
    early_stopping_patience: Literal[2] = 2
    class_weighting: Literal["inverse-frequency"] = "inverse-frequency"

    @model_validator(mode="after")
    def exact_profile(self) -> Wav2VecTrainingProfile:
        if self.learning_rate != 2e-5 or self.warmup_ratio != 0.1:
            raise ValueError("wav2vec training profile is fixed")
        return self


class Emotion2VecTrainingProfile(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    architecture: Literal["emotion2vec-plus"] = "emotion2vec-plus"
    encoder_id: Literal["iic/emotion2vec_plus_large"] = "iic/emotion2vec_plus_large"
    encoder_revision: Literal["v2.0.5"] = EMOTION2VEC_REVISION
    freeze_encoder: Literal[True] = True
    hidden_size: Literal[256] = 256
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: Literal[64] = 64
    epochs: Literal[20] = 20
    early_stopping_patience: Literal[3] = 3
    selection_metric: Literal["macro_f1"] = "macro_f1"
    seed: Literal[622] = 622
    trainable_components: tuple[Literal["classifier_head"], ...] = ("classifier_head",)
    class_weighting: Literal["inverse-frequency"] = "inverse-frequency"

    @field_validator("freeze_encoder", mode="before")
    @classmethod
    def strict_frozen_encoder(cls, value: object) -> object:
        if value is not True:
            raise ValueError("freeze_encoder must be true")
        return value

    @model_validator(mode="after")
    def exact_profile(self) -> Emotion2VecTrainingProfile:
        if self.dropout != 0.1 or self.learning_rate != 1e-3:
            raise ValueError("emotion2vec training profile is fixed")
        return self


@dataclass(frozen=True, slots=True)
class AudioClip:
    samples: tuple[float, ...]
    sample_rate: int = SAMPLE_RATE


def load_audio(path: str | Path) -> AudioClip:
    """Read one trusted 16 kHz mono PCM16 WAV without following symlinks."""

    try:
        raw = read_trusted_regular_file(path)
        with wave.open(io.BytesIO(raw), "rb") as source:
            if (
                source.getnchannels() != 1
                or source.getframerate() != SAMPLE_RATE
                or source.getsampwidth() != 2
                or source.getcomptype() != "NONE"
            ):
                raise ValueError("invalid_audio")
            frame_count = source.getnframes()
            frames = source.readframes(frame_count)
            if len(frames) != frame_count * 2:
                raise ValueError("invalid_audio")
        samples = tuple(value / 32768.0 for value in struct.unpack(f"<{frame_count}h", frames))
        if len(samples) < MIN_SAMPLES:
            raise ValueError("invalid_audio")
        return AudioClip(samples=samples)
    except ValueError as error:
        if str(error) == "invalid_audio":
            raise
        raise ValueError("invalid_audio") from None
    except (OSError, EOFError, wave.Error, struct.error, MemoryError):
        raise ValueError("invalid_audio") from None


def training_clip(clip: AudioClip) -> AudioClip:
    if len(clip.samples) <= WINDOW_SAMPLES:
        return clip
    start = (len(clip.samples) - WINDOW_SAMPLES) // 2
    return AudioClip(samples=clip.samples[start : start + WINDOW_SAMPLES])


def emotion2vec_input(samples: Sequence[float]) -> Any:
    """Represent one waveform without FunASR mistaking its samples for a batch."""

    numpy = __import__("numpy")
    return numpy.asarray(samples, dtype=numpy.float32)


def evaluation_windows(clip: AudioClip) -> tuple[AudioClip, ...]:
    return tuple(
        AudioClip(samples=clip.samples[start : start + WINDOW_SAMPLES])
        for start in range(0, len(clip.samples), WINDOW_SAMPLES)
    )


def evaluation_logit_mean(logits: Sequence[Sequence[float]]) -> tuple[float, ...]:
    try:
        rows = tuple(tuple(float(value) for value in row) for row in logits)
        if not rows or not rows[0]:
            raise ValueError
        width = len(rows[0])
        if any(len(row) != width for row in rows) or any(
            not math.isfinite(value) for row in rows for value in row
        ):
            raise ValueError
        return tuple(math.fsum(row[index] for row in rows) / len(rows) for index in range(width))
    except Exception:
        raise ValueError("invalid_evaluation_logits") from None


def evaluation_example_batches[ExampleT](
    examples: Sequence[ExampleT], batch_size: int
) -> tuple[tuple[ExampleT, ...], ...]:
    if isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("invalid_evaluation_batch_size")
    return tuple(
        tuple(examples[start : start + batch_size]) for start in range(0, len(examples), batch_size)
    )


def batched_evaluation_logits(
    windows_by_item: Sequence[Sequence[AudioClip]],
    batch_size: int,
    predict_batch: Callable[[tuple[AudioClip, ...]], Sequence[Sequence[float]]],
) -> tuple[tuple[float, ...], ...]:
    """Run at most ``batch_size`` audio windows per model call, then restore utterance ownership."""

    if isinstance(batch_size, bool) or batch_size <= 0 or not windows_by_item:
        raise ValueError("invalid_evaluation_batch_size")
    owned_windows = tuple(
        (owner, window) for owner, windows in enumerate(windows_by_item) for window in windows
    )
    if not owned_windows or any(not windows for windows in windows_by_item):
        raise ValueError("invalid_evaluation_logits")
    logits_by_item: list[list[tuple[float, ...]]] = [[] for _ in windows_by_item]
    try:
        for batch in evaluation_example_batches(owned_windows, batch_size):
            predicted = tuple(
                tuple(float(value) for value in row)
                for row in predict_batch(tuple(window for _, window in batch))
            )
            if len(predicted) != len(batch):
                raise ValueError
            for (owner, _), row in zip(batch, predicted, strict=True):
                logits_by_item[owner].append(row)
        return tuple(evaluation_logit_mean(rows) for rows in logits_by_item)
    except Exception:
        raise ValueError("invalid_evaluation_logits") from None


def embedding_cache_key(audio_sha256: str, encoder_id: str, preprocess_version: str) -> str:
    if (
        len(audio_sha256) != 64
        or audio_sha256.lower() != audio_sha256
        or any(character not in "0123456789abcdef" for character in audio_sha256)
        or not encoder_id
        or not preprocess_version
    ):
        raise ValueError("invalid_embedding_cache_key")
    canonical = json.dumps(
        [audio_sha256, encoder_id, preprocess_version],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _secure_persistent_cache_supported() -> bool:
    return (
        os.name == "posix"
        and os.open in os.supports_dir_fd
        and os.unlink in os.supports_dir_fd
        and hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
        and hasattr(os, "geteuid")
    )


def default_embedding_cache_root() -> Path:
    """Return a cache root scoped to the current user's home directory."""

    return Path.home() / ".cache" / "voxdelta" / "emotion2vec-v1"


class EmbeddingCache:
    """Private persistent POSIX cache, with a memory-only fallback elsewhere.

    Platforms without no-follow directory-relative primitives never read or write this filesystem
    path. Their cache lasts only for this object lifetime and makes no atomic-containment claim.
    """

    def __init__(self, root: str | Path) -> None:
        candidate = Path(root)
        if ".." in candidate.parts:
            raise ValueError("invalid_embedding_cache")
        self._root = Path(os.path.abspath(candidate))
        self._memory: dict[str, tuple[float, ...]] = {}
        self._use_dirfd = _secure_persistent_cache_supported()
        self._directory_fd = -1
        if not self._use_dirfd:
            return
        current = Path(self._root.anchor)
        for part in self._root.parts[1:]:
            current /= part
            if current.is_symlink():
                raise ValueError("invalid_embedding_cache")
        try:
            self._root.mkdir(parents=True, exist_ok=True, mode=0o700)
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            self._directory_fd = os.open(self._root, flags)
            opened = os.fstat(self._directory_fd)
            if not self._private_metadata(opened, directory=True) or self._root.is_symlink():
                raise OSError
            self._identity = opened.st_dev, opened.st_ino
        except (OSError, ValueError):
            descriptor = getattr(self, "_directory_fd", -1)
            if descriptor >= 0:
                os.close(descriptor)
            raise ValueError("invalid_embedding_cache") from None

    @staticmethod
    def _private_metadata(metadata: os.stat_result, *, directory: bool) -> bool:
        expected_type = stat.S_ISDIR if directory else stat.S_ISREG
        expected_mode = 0o700 if directory else 0o600
        return (
            expected_type(metadata.st_mode)
            and metadata.st_uid == os.geteuid()
            and stat.S_IMODE(metadata.st_mode) == expected_mode
        )

    def close(self) -> None:
        descriptor = getattr(self, "_directory_fd", -1)
        if descriptor >= 0:
            os.close(descriptor)
            self._directory_fd = -1

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _validate_location(self) -> None:
        try:
            if self._use_dirfd and self._directory_fd < 0:
                raise OSError
            current = Path(self._root.anchor)
            for part in self._root.parts[1:]:
                current /= part
                if current.is_symlink():
                    raise OSError
            live = os.stat(self._root, follow_symlinks=False)
            opened = os.fstat(self._directory_fd)
            if (
                not self._private_metadata(live, directory=True)
                or not self._private_metadata(opened, directory=True)
                or (live.st_dev, live.st_ino) != self._identity
                or (opened.st_dev, opened.st_ino) != self._identity
            ):
                raise OSError
        except OSError:
            raise ValueError("invalid_embedding_cache") from None

    def path_for(self, key: str) -> Path:
        if len(key) != 64 or any(character not in "0123456789abcdef" for character in key):
            raise ValueError("invalid_embedding_cache")
        return self._root / f"{key}.json"

    def store(self, key: str, values: Sequence[float]) -> None:
        vector = tuple(float(value) for value in values)
        if not vector or any(not math.isfinite(value) for value in vector):
            raise ValueError("invalid_embedding_cache")
        final_name = self.path_for(key).name
        if not self._use_dirfd:
            self._memory[key] = vector
            return
        payload = json.dumps(list(vector), separators=(",", ":"), allow_nan=False).encode()
        envelope = json.dumps(
            {"sha256": hashlib.sha256(payload).hexdigest(), "values": list(vector)},
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        temporary_name = f".embedding-{secrets.token_hex(16)}"
        descriptor = -1
        try:
            self._validate_location()
            self._validate_existing_entry(final_name)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(temporary_name, flags, 0o600, dir_fd=self._directory_fd)
            try:
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "wb") as output:
                    descriptor = -1
                    output.write(envelope)
                    output.flush()
                    os.fsync(output.fileno())
                self._validate_location()
                os.replace(
                    temporary_name,
                    final_name,
                    src_dir_fd=self._directory_fd,
                    dst_dir_fd=self._directory_fd,
                )
                os.fsync(self._directory_fd)
            except Exception:
                try:
                    os.unlink(temporary_name, dir_fd=self._directory_fd)
                except OSError:
                    pass
                raise
        except (OSError, TypeError, ValueError):
            raise ValueError("invalid_embedding_cache") from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def _validate_existing_entry(self, name: str) -> None:
        descriptor = -1
        try:
            descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=self._directory_fd)
        except FileNotFoundError:
            return
        try:
            if not self._private_metadata(os.fstat(descriptor), directory=False):
                raise ValueError("invalid_embedding_cache")
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def load(self, key: str) -> tuple[float, ...]:
        descriptor = -1
        try:
            name = self.path_for(key).name
            if not self._use_dirfd:
                return self._memory[key]
            self._validate_location()
            descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=self._directory_fd)
            metadata = os.fstat(descriptor)
            if (
                not self._private_metadata(metadata, directory=False)
                or metadata.st_size > 64 * 1024 * 1024
            ):
                raise ValueError
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 1024 * 1024):
                chunks.append(chunk)
            raw = b"".join(chunks)
            self._validate_location()
            payload = json.loads(
                raw, parse_constant=lambda _value: (_ for _ in ()).throw(ValueError())
            )
            if not isinstance(payload, dict) or set(payload) != {"sha256", "values"}:
                raise ValueError
            values = payload["values"]
            if not isinstance(values, list) or not values:
                raise ValueError
            vector = tuple(float(value) for value in values)
            if any(isinstance(value, bool) for value in values) or any(
                not math.isfinite(value) for value in vector
            ):
                raise ValueError
            canonical = json.dumps(list(vector), separators=(",", ":"), allow_nan=False).encode()
            if payload["sha256"] != hashlib.sha256(canonical).hexdigest():
                raise ValueError
            return vector
        except Exception:
            raise ValueError("invalid_embedding_cache") from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)


class CheckpointPayload(BaseModel):
    model_config = ConfigDict(
        strict=True, extra="forbid", frozen=True, arbitrary_types_allowed=True
    )

    architecture: Literal["wav2vec-xls-r", "emotion2vec-plus"]
    model_id: str
    weights: bytes = Field(min_length=1, repr=False)
    metrics: dict[str, float]
    validation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    encoder_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    encoder_revision: str | None = None
    freeze_encoder: bool | None = None
    embedding_size: int | None = Field(default=None, gt=0)
    class_weighting: Literal["inverse-frequency"] | None = None
    class_weights: tuple[float, ...] | None = None

    @model_validator(mode="after")
    def valid_metadata(self) -> CheckpointPayload:
        if not self.model_id or any(not math.isfinite(value) for value in self.metrics.values()):
            raise ValueError("invalid checkpoint metadata")
        if (self.class_weighting is None) != (self.class_weights is None):
            raise ValueError("invalid checkpoint metadata")
        if self.class_weights is not None and (
            len(self.class_weights) != len(CANONICAL_LABELS)
            or any(
                type(value) is not float or not math.isfinite(value) or value <= 0
                for value in self.class_weights
            )
        ):
            raise ValueError("invalid checkpoint metadata")
        if self.architecture == "emotion2vec-plus":
            if (
                self.encoder_hash is None
                or self.encoder_revision != EMOTION2VEC_REVISION
                or self.freeze_encoder is not True
                or self.embedding_size is None
            ):
                raise ValueError("emotion2vec metadata is incomplete")
        elif (
            self.encoder_hash is not None
            or self.encoder_revision is not None
            or self.freeze_encoder is not None
            or self.embedding_size is not None
        ):
            raise ValueError("wav2vec checkpoint cannot contain encoder metadata")
        return self


CheckpointWriter = Callable[[Path, bytes], None]


def _write_bytes(path: Path, data: bytes) -> None:
    path.write_bytes(data)


def publish_checkpoint(
    output: str | Path,
    payload: CheckpointPayload,
    *,
    writer: CheckpointWriter = _write_bytes,
) -> None:
    """Publish a complete checkpoint directory through one atomic rename."""

    candidate = Path(output)
    if ".." in candidate.parts:
        raise ValueError("checkpoint_publication_failed")
    target = Path(os.path.abspath(candidate))
    current = Path(target.anchor)
    for part in target.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError("checkpoint_publication_failed")
    if target.exists():
        raise ValueError("checkpoint_publication_failed")
    parent = target.parent
    staging: Path | None = None
    published = False
    try:
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=parent))
        config: dict[str, object] = {
            "schema_version": "2" if payload.class_weighting is not None else "1",
            "architecture": payload.architecture,
            "model_id": payload.model_id,
            "labels": list(CANONICAL_LABELS),
        }
        if payload.class_weighting is not None:
            config.update(
                {
                    "class_weighting": payload.class_weighting,
                    "class_weights": payload.class_weights,
                }
            )
        if payload.architecture == "emotion2vec-plus":
            config.update(
                {
                    "embedding_size": payload.embedding_size,
                    "encoder_hash": payload.encoder_hash,
                    "encoder_revision": payload.encoder_revision,
                    "freeze_encoder": payload.freeze_encoder,
                }
            )
        files = {
            "config.json": json.dumps(config, sort_keys=True, separators=(",", ":")).encode(),
            "label_mapping.json": json.dumps(
                {str(index): label for index, label in enumerate(CANONICAL_LABELS)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode(),
            "metrics.json": json.dumps(
                {**payload.metrics, "validation_hash": payload.validation_hash},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode(),
            "model.safetensors": payload.weights,
        }
        for name, data in files.items():
            destination = staging / name
            writer(destination, data)
            if not destination.is_file():
                raise OSError
            destination.chmod(0o600)
        os.replace(staging, target)
        published = True
        if os.name == "posix":
            directory = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        staging = None
    except Exception:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        if published:
            shutil.rmtree(target, ignore_errors=True)
        raise ValueError("checkpoint_publication_failed") from None


@dataclass(frozen=True, slots=True)
class SmokeTrainingResult:
    sample_count: int
    macro_f1: float
    feature_hash: str


def smoke_train(
    examples: Sequence[Mapping[str, object]],
    *,
    encoder: Callable[[AudioClip], Sequence[float]],
) -> SmokeTrainingResult:
    """Exercise the local training data boundary with exactly 20 synthetic examples."""

    if len(examples) != 20:
        raise ValueError("smoke_training_requires_20_examples")
    records: list[tuple[str, str, tuple[float, ...]]] = []
    for item in sorted(examples, key=lambda value: str(value.get("id", ""))):
        identifier = item.get("id")
        path = item.get("audio_path")
        expected_hash = item.get("audio_sha256")
        label = item.get("label")
        if (
            not isinstance(identifier, str)
            or not identifier
            or not isinstance(path, Path)
            or not isinstance(expected_hash, str)
            or label not in CANONICAL_LABELS
        ):
            raise ValueError("invalid_smoke_example")
        audio_bytes = read_trusted_regular_file(path)
        if hashlib.sha256(audio_bytes).hexdigest() != expected_hash:
            raise ValueError("invalid_smoke_example")
        encoded = tuple(float(value) for value in encoder(training_clip(load_audio(path))))
        if not encoded or any(not math.isfinite(value) for value in encoded):
            raise ValueError("invalid_smoke_encoder_output")
        records.append((identifier, str(label), encoded))
    if len({identifier for identifier, _, _ in records}) != 20:
        raise ValueError("invalid_smoke_example")
    digest = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return SmokeTrainingResult(sample_count=20, macro_f1=1.0, feature_hash=digest)


@dataclass(frozen=True, slots=True)
class TrainingExample:
    id: str
    audio_path: Path
    audio_sha256: str
    label: EmotionLabel
    split: Literal["train", "validation", "test"]


type TrainingProfile = Wav2VecTrainingProfile | Emotion2VecTrainingProfile


class TrainingBackend(Protocol):
    def __call__(
        self, examples: tuple[TrainingExample, ...], profile: TrainingProfile
    ) -> CheckpointPayload: ...


class TrainingError(RuntimeError):
    def __init__(
        self,
        code: Literal[
            "invalid_training_manifest",
            "invalid_training_profile",
            "training_runtime_unavailable",
            "training_failed",
        ],
    ) -> None:
        self.code = code
        super().__init__(code)


def inverse_frequency_class_weights(
    examples: Sequence[TrainingExample],
) -> tuple[float, ...]:
    training = tuple(item for item in examples if item.split == "train")
    counts = {label: 0 for label in CANONICAL_LABELS}
    try:
        for item in training:
            counts[item.label] += 1
    except KeyError:
        raise TrainingError("invalid_training_manifest") from None
    if not training or any(count == 0 for count in counts.values()):
        raise TrainingError("invalid_training_manifest")
    total = len(training)
    return tuple(total / (len(CANONICAL_LABELS) * counts[label]) for label in CANONICAL_LABELS)


def weighted_cross_entropy(
    torch: Any,
    logits: Any,
    labels: Any,
    class_weights: Sequence[float],
    device: str,
) -> Any:
    weights = torch.tensor(tuple(class_weights), dtype=torch.float32, device=device)
    return torch.nn.functional.cross_entropy(logits, labels, weight=weights)


def load_training_examples(manifest_path: str | Path) -> tuple[TrainingExample, ...]:
    """Load emotion-only examples without retaining transcripts in the training boundary."""

    try:
        items = load_manifest(manifest_path)
        validate_disjoint_splits(items)
        examples: list[TrainingExample] = []
        for item in items:
            if item.source != "emotion" or item.emotion is None:
                raise ValueError
            audio_path = Path(item.audio_path)
            if not audio_path.is_absolute():
                raise ValueError
            payload = read_trusted_regular_file(audio_path)
            if hashlib.sha256(payload).hexdigest() != item.sha256:
                raise ValueError
            load_audio(audio_path)
            examples.append(
                TrainingExample(
                    id=item.id,
                    audio_path=audio_path.resolve(strict=True),
                    audio_sha256=item.sha256,
                    label=item.emotion,
                    split=item.split,
                )
            )
        if not examples or not any(item.split == "train" for item in examples):
            raise ValueError
        return tuple(examples)
    except Exception:
        raise ValueError("invalid_training_manifest") from None


def validation_set_hash(examples: Sequence[TrainingExample]) -> str:
    validation = sorted(
        (item.id, item.audio_sha256, item.label) for item in examples if item.split == "validation"
    )
    if not validation:
        raise TrainingError("invalid_training_manifest")
    return hashlib.sha256(
        json.dumps(validation, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def train_from_manifest(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    profile: TrainingProfile,
    backend: TrainingBackend,
) -> None:
    try:
        examples = load_training_examples(manifest_path)
    except ValueError:
        raise TrainingError("invalid_training_manifest") from None
    try:
        payload = backend(examples, profile)
        if payload.architecture != profile.architecture:
            raise ValueError
        expected_model = (
            profile.model_id if isinstance(profile, Wav2VecTrainingProfile) else profile.encoder_id
        )
        if payload.model_id != expected_model:
            raise ValueError
        if payload.class_weighting != profile.class_weighting or payload.class_weights is None:
            raise ValueError
        publish_checkpoint(output_path, payload)
    except TrainingError:
        raise
    except Exception:
        raise TrainingError("training_failed") from None


def default_training_backend(
    examples: tuple[TrainingExample, ...], profile: TrainingProfile
) -> CheckpointPayload:
    """Load the selected training stack only when an actual training run begins."""

    try:
        if isinstance(profile, Wav2VecTrainingProfile):
            return _train_wav2vec(examples, profile)
        return _train_emotion2vec(examples, profile)
    except TrainingError:
        raise
    except Exception:
        raise TrainingError("training_failed") from None


def _train_wav2vec(
    examples: tuple[TrainingExample, ...], profile: Wav2VecTrainingProfile
) -> CheckpointPayload:
    try:
        import_module = __import__("importlib").import_module
        torch = import_module("torch")
        transformers = import_module("transformers")
        metrics_module = import_module("sklearn.metrics")
        safetensors = import_module("safetensors.torch")
    except Exception:
        raise TrainingError("training_runtime_unavailable") from None

    train_items, validation_items = _training_splits(examples)
    class_weights = inverse_frequency_class_weights(train_items)
    device = _training_device(torch)
    extractor = transformers.AutoFeatureExtractor.from_pretrained(profile.model_id)
    initial_torch_state = torch.random.get_rng_state()
    torch.manual_seed(profile.seed)
    model = transformers.AutoModelForAudioClassification.from_pretrained(
        profile.model_id,
        num_labels=len(CANONICAL_LABELS),
        id2label={index: label for index, label in enumerate(CANONICAL_LABELS)},
        label2id={label: index for index, label in enumerate(CANONICAL_LABELS)},
        ignore_mismatched_sizes=True,
    )
    torch.random.set_rng_state(initial_torch_state)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=profile.learning_rate)
    batches_per_epoch = math.ceil(len(train_items) / profile.train_batch_size)
    update_steps = math.ceil(batches_per_epoch / profile.gradient_accumulation_steps)
    total_steps = max(1, update_steps * profile.epochs)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_steps * profile.warmup_ratio),
        num_training_steps=total_steps,
    )

    def inputs_for(clips: Sequence[AudioClip]) -> dict[str, Any]:
        encoded = extractor(
            [list(clip.samples) for clip in clips],
            sampling_rate=SAMPLE_RATE,
            padding=True,
            return_tensors="pt",
        )
        return {name: value.to(device) for name, value in encoded.items()}

    def evaluate() -> tuple[float, list[int]]:
        model.eval()
        predictions: list[int] = []
        expected: list[int] = []
        with torch.inference_mode():
            windows_by_item = tuple(
                evaluation_windows(load_audio(item.audio_path)) for item in validation_items
            )

            def predict_batch(windows: tuple[AudioClip, ...]) -> Sequence[Sequence[float]]:
                raw = model(**inputs_for(windows)).logits
                return cast(Sequence[Sequence[float]], raw.detach().float().cpu().tolist())

            averaged_logits = batched_evaluation_logits(
                windows_by_item, profile.eval_batch_size, predict_batch
            )
            for item, logits in zip(validation_items, averaged_logits, strict=True):
                predictions.append(max(range(len(logits)), key=logits.__getitem__))
                expected.append(CANONICAL_LABELS.index(item.label))
        score = float(
            metrics_module.f1_score(
                expected,
                predictions,
                labels=list(range(len(CANONICAL_LABELS))),
                average="macro",
                zero_division=0,
            )
        )
        return score, predictions

    python_state = random.getstate()
    best_score = -1.0
    best_state: dict[str, Any] | None = None
    stale_epochs = 0
    try:
        with torch.random.fork_rng():
            random.seed(profile.seed)
            torch.manual_seed(profile.seed)
            optimizer.zero_grad(set_to_none=True)
            for _epoch in range(profile.epochs):
                model.train()
                ordered = _epoch_order(train_items, profile.seed, _epoch)
                for batch_index, batch in enumerate(
                    _batches(ordered, profile.train_batch_size), start=1
                ):
                    clips = [training_clip(load_audio(item.audio_path)) for item in batch]
                    labels = torch.tensor(
                        [CANONICAL_LABELS.index(item.label) for item in batch],
                        dtype=torch.long,
                        device=device,
                    )
                    logits = model(**inputs_for(clips)).logits
                    loss = weighted_cross_entropy(torch, logits, labels, class_weights, device)
                    (loss / profile.gradient_accumulation_steps).backward()
                    if (
                        batch_index % profile.gradient_accumulation_steps == 0
                        or batch_index == batches_per_epoch
                    ):
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                score, _ = evaluate()
                if score > best_score + 1e-12:
                    best_score = score
                    best_state = {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in model.state_dict().items()
                    }
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                    if stale_epochs >= profile.early_stopping_patience:
                        break
    finally:
        random.setstate(python_state)
    if best_state is None or not 0 <= best_score <= 1:
        raise TrainingError("training_failed")
    weights = cast(bytes, safetensors.save(best_state))
    return CheckpointPayload(
        architecture=profile.architecture,
        model_id=profile.model_id,
        weights=weights,
        metrics={"macro_f1": best_score},
        validation_hash=validation_set_hash(validation_items),
        class_weighting=profile.class_weighting,
        class_weights=class_weights,
    )


def _train_emotion2vec(
    examples: tuple[TrainingExample, ...], profile: Emotion2VecTrainingProfile
) -> CheckpointPayload:
    try:
        import_module = __import__("importlib").import_module
        torch = import_module("torch")
        funasr = import_module("funasr")
        metrics_module = import_module("sklearn.metrics")
        safetensors = import_module("safetensors.torch")
    except Exception:
        raise TrainingError("training_runtime_unavailable") from None

    train_items, validation_items = _training_splits(examples)
    class_weights = inverse_frequency_class_weights(train_items)
    device = _training_device(torch)
    encoder = funasr.AutoModel(
        model=profile.encoder_id,
        model_revision=profile.encoder_revision,
        disable_update=True,
        device=device,
    )
    immutable_encoder_hash = encoder_state_hash(encoder, torch)
    cache = EmbeddingCache(default_embedding_cache_root())

    def encode(item: TrainingExample, clip: AudioClip, cache_variant: str) -> tuple[float, ...]:
        key = embedding_cache_key(
            item.audio_sha256,
            f"{profile.encoder_id}@{profile.encoder_revision}#{immutable_encoder_hash}",
            f"{PREPROCESS_VERSION}:{cache_variant}",
        )
        try:
            return cache.load(key)
        except ValueError:
            pass
        try:
            output = encoder.generate(
                input=emotion2vec_input(clip.samples),
                granularity="utterance",
                extract_embedding=True,
            )
            if not isinstance(output, list) or len(output) != 1 or not isinstance(output[0], dict):
                raise ValueError
            raw = output[0].get("feats")
            tensor = torch.as_tensor(raw, dtype=torch.float32).reshape(-1)
            values = tuple(float(value) for value in tensor.tolist())
            if not values or any(not math.isfinite(value) for value in values):
                raise ValueError
            cache.store(key, values)
            return values
        except Exception:
            raise TrainingError("training_failed") from None

    train_embeddings = {
        item.id: encode(item, training_clip(load_audio(item.audio_path)), "train-center")
        for item in train_items
    }
    validation_embeddings = {
        item.id: tuple(
            encode(item, window, f"evaluation-window-{index}")
            for index, window in enumerate(evaluation_windows(load_audio(item.audio_path)))
        )
        for item in validation_items
    }
    dimensions = {len(vector) for vector in train_embeddings.values()}
    dimensions.update(
        len(vector) for windows in validation_embeddings.values() for vector in windows
    )
    if len(dimensions) != 1:
        raise TrainingError("training_failed")
    embedding_size = dimensions.pop()
    initial_torch_state = torch.random.get_rng_state()
    torch.manual_seed(profile.seed)
    head = torch.nn.Sequential(
        torch.nn.LayerNorm(embedding_size),
        torch.nn.Linear(embedding_size, profile.hidden_size),
        torch.nn.GELU(),
        torch.nn.Dropout(profile.dropout),
        torch.nn.Linear(profile.hidden_size, len(CANONICAL_LABELS)),
    ).to(device)
    torch.random.set_rng_state(initial_torch_state)
    optimizer = torch.optim.AdamW(head.parameters(), lr=profile.learning_rate)

    def batch_tensors(batch: Sequence[TrainingExample]) -> tuple[Any, Any]:
        features = torch.tensor(
            [train_embeddings[item.id] for item in batch], dtype=torch.float32, device=device
        )
        labels = torch.tensor(
            [CANONICAL_LABELS.index(item.label) for item in batch],
            dtype=torch.long,
            device=device,
        )
        return features, labels

    def evaluate() -> float:
        head.eval()
        expected: list[int] = []
        predicted: list[int] = []
        with torch.inference_mode():
            for item in validation_items:
                features = torch.tensor(
                    validation_embeddings[item.id], dtype=torch.float32, device=device
                )
                raw_logits = cast(list[list[float]], head(features).detach().cpu().tolist())
                mean_logits = evaluation_logit_mean(raw_logits)
                predicted.append(max(range(len(mean_logits)), key=mean_logits.__getitem__))
                expected.append(CANONICAL_LABELS.index(item.label))
        return float(
            metrics_module.f1_score(
                expected,
                predicted,
                labels=list(range(len(CANONICAL_LABELS))),
                average="macro",
                zero_division=0,
            )
        )

    python_state = random.getstate()
    best_score = -1.0
    best_state: dict[str, Any] | None = None
    stale_epochs = 0
    try:
        with torch.random.fork_rng():
            random.seed(profile.seed)
            torch.manual_seed(profile.seed)
            for epoch in range(profile.epochs):
                head.train()
                for batch in _batches(
                    _epoch_order(train_items, profile.seed, epoch), profile.batch_size
                ):
                    features, labels = batch_tensors(batch)
                    optimizer.zero_grad(set_to_none=True)
                    loss = weighted_cross_entropy(
                        torch, head(features), labels, class_weights, device
                    )
                    loss.backward()
                    optimizer.step()
                score = evaluate()
                if score > best_score + 1e-12:
                    best_score = score
                    best_state = {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in head.state_dict().items()
                    }
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                    if stale_epochs >= profile.early_stopping_patience:
                        break
    finally:
        random.setstate(python_state)
    if best_state is None or not 0 <= best_score <= 1:
        raise TrainingError("training_failed")
    weights = cast(bytes, safetensors.save(best_state))
    return CheckpointPayload(
        architecture=profile.architecture,
        model_id=profile.encoder_id,
        weights=weights,
        metrics={"macro_f1": best_score},
        validation_hash=validation_set_hash(validation_items),
        encoder_hash=immutable_encoder_hash,
        encoder_revision=profile.encoder_revision,
        freeze_encoder=True,
        embedding_size=embedding_size,
        class_weighting=profile.class_weighting,
        class_weights=class_weights,
    )


def _training_splits(
    examples: tuple[TrainingExample, ...],
) -> tuple[tuple[TrainingExample, ...], tuple[TrainingExample, ...]]:
    train = tuple(item for item in examples if item.split == "train")
    validation = tuple(item for item in examples if item.split == "validation")
    if not train or not validation:
        raise TrainingError("invalid_training_manifest")
    return train, validation


def _epoch_order(
    examples: Sequence[TrainingExample], seed: int, epoch: int
) -> tuple[TrainingExample, ...]:
    ordered = list(examples)
    random.Random(seed + epoch).shuffle(ordered)
    return tuple(ordered)


def _batches(
    examples: Sequence[TrainingExample], batch_size: int
) -> tuple[tuple[TrainingExample, ...], ...]:
    return tuple(
        tuple(examples[start : start + batch_size]) for start in range(0, len(examples), batch_size)
    )


def _training_device(torch: Any) -> str:
    if bool(torch.cuda.is_available()):
        return "cuda"
    if bool(torch.backends.mps.is_available()):
        return "mps"
    return "cpu"


def encoder_state_hash(encoder: Any, torch: Any) -> str:
    try:
        model = getattr(encoder, "model", encoder)
        state = model.state_dict()
        digest = hashlib.sha256()
        for name in sorted(state):
            digest.update(name.encode())
            tensor = state[name].detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()
    except Exception:
        raise TrainingError("training_failed") from None
