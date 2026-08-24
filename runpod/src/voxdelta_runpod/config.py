"""Strict immutable configuration for the RunPod XLS-R experiment."""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from voxdelta.domain.models import EmotionLabel
from voxdelta.evaluation.manifest import read_trusted_regular_file
from voxdelta.evaluation.wav2vec_base import (
    WAV2VEC_MODEL_REVISION,
    WAV2VEC_WEIGHTS_SHA256,
)

CANONICAL_LABELS: tuple[EmotionLabel, ...] = (
    "happiness",
    "anger",
    "disgust",
    "fear",
    "neutral",
    "sadness",
    "surprise",
)
BASE_IMAGE: Literal[
    "docker.io/nvidia/cuda@sha256:9175fa92f96de35a8cfb9493f0dfcf9435c7a597e9d95ad41d2cae382a95e3f9"
] = "docker.io/nvidia/cuda@sha256:9175fa92f96de35a8cfb9493f0dfcf9435c7a597e9d95ad41d2cae382a95e3f9"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class ModelConfig(_FrozenModel):
    model_id: Literal["facebook/wav2vec2-xls-r-300m"] = "facebook/wav2vec2-xls-r-300m"
    revision: Literal["1a640f32ac3e39899438a2931f9924c02f080a54"] = WAV2VEC_MODEL_REVISION
    weights_sha256: Literal["d5e490574712ad0a6736923b9ed11d4cd51c78609c36205f704fc4e87b11d2e0"] = (
        WAV2VEC_WEIGHTS_SHA256
    )


class DataConfig(_FrozenModel):
    train_count: Literal[29476] = 29476
    validation_count: Literal[3569] = 3569
    nominal_test_count: Literal[3620] = 3620
    exposed_test_count: Literal[35] = 35
    final_holdout_count: Literal[3585] = 3585
    pilot_train_count: Literal[3500] = 3500
    pilot_validation_count: Literal[350] = 350

    @model_validator(mode="after")
    def consistent_holdout(self) -> DataConfig:
        if self.nominal_test_count - self.exposed_test_count != self.final_holdout_count:
            raise ValueError("invalid final holdout counts")
        return self


class RuntimeConfig(_FrozenModel):
    platform: Literal["linux/amd64"] = "linux/amd64"
    python_version: Literal["3.12"] = "3.12"
    base_image: Literal[
        "docker.io/nvidia/cuda@sha256:9175fa92f96de35a8cfb9493f0dfcf9435c7a597e9d95ad41d2cae382a95e3f9"
    ] = BASE_IMAGE
    gpu_model: Literal["NVIDIA A40"] = "NVIDIA A40"
    gpu_count: Literal[1] = 1
    gpu_memory_gb: Literal[48] = 48
    volume_root: Literal["/workspace/voxdelta"] = "/workspace/voxdelta"
    volume_disk_gb: Literal[100] = 100
    minimum_free_gb: Literal[80] = 80


class TrainingConfig(_FrozenModel):
    effective_batch_size: Literal[16] = 16
    micro_batch_candidates: tuple[Literal[8, 4, 2, 1], ...] = (8, 4, 2, 1)
    gradient_accumulation_candidates: tuple[Literal[2, 4, 8, 16], ...] = (2, 4, 8, 16)
    encoder_learning_rate: float = 1e-5
    head_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_clip_norm: float = 1.0
    pilot_max_epochs: Literal[5] = 5
    full_max_epochs: Literal[10] = 10
    early_stopping_patience: Literal[2] = 2
    sample_rate: Literal[16000] = 16000
    window_seconds: Literal[20] = 20

    @field_validator("micro_batch_candidates", "gradient_accumulation_candidates", mode="before")
    @classmethod
    def immutable_toml_arrays(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def exact_training_profile(self) -> TrainingConfig:
        pairs = tuple(
            zip(
                self.micro_batch_candidates,
                self.gradient_accumulation_candidates,
                strict=True,
            )
        )
        if (
            pairs != ((8, 2), (4, 4), (2, 8), (1, 16))
            or any(micro * accumulation != 16 for micro, accumulation in pairs)
            or self.encoder_learning_rate != 1e-5
            or self.head_learning_rate != 1e-4
            or self.weight_decay != 0.01
            or self.warmup_ratio != 0.1
            or self.gradient_clip_norm != 1.0
        ):
            raise ValueError("invalid training profile")
        return self


class GateConfig(_FrozenModel):
    pilot_macro_f1_min_exclusive: float = 0.15
    pilot_min_predicted_classes: Literal[6] = 6
    pilot_min_positive_label_f1: Literal[5] = 5
    full_macro_f1_min_exclusive: float = 0.2400352
    full_required_predicted_classes: Literal[7] = 7
    full_required_positive_label_f1: Literal[7] = 7

    @model_validator(mode="after")
    def exact_gates(self) -> GateConfig:
        numeric = (self.pilot_macro_f1_min_exclusive, self.full_macro_f1_min_exclusive)
        if (
            any(not math.isfinite(value) for value in numeric)
            or self.pilot_macro_f1_min_exclusive != 0.15
            or self.full_macro_f1_min_exclusive != 0.2400352
        ):
            raise ValueError("invalid experiment gates")
        return self


class ExperimentConfig(_FrozenModel):
    schema_version: Literal["1"] = "1"
    experiment_name: Literal["xls-r-300m-full-finetuning"] = "xls-r-300m-full-finetuning"
    seed: Literal[622] = 622
    canonical_labels: tuple[EmotionLabel, ...] = CANONICAL_LABELS
    model: ModelConfig
    data: DataConfig
    runtime: RuntimeConfig
    training: TrainingConfig
    gates: GateConfig

    @field_validator("canonical_labels", mode="before")
    @classmethod
    def immutable_labels(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def exact_labels(self) -> ExperimentConfig:
        if self.canonical_labels != CANONICAL_LABELS:
            raise ValueError("invalid canonical label order")
        return self

    def digest(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return hashlib.sha256(payload).hexdigest()


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Read an absolute trusted TOML file and return its immutable configuration."""

    if not path.is_absolute():
        raise ValueError("invalid_experiment_config")
    try:
        raw = read_trusted_regular_file(path)
        payload = tomllib.loads(raw.decode("utf-8"))
        return ExperimentConfig.model_validate(payload)
    except Exception:
        raise ValueError("invalid_experiment_config") from None


__all__ = [
    "BASE_IMAGE",
    "CANONICAL_LABELS",
    "DataConfig",
    "ExperimentConfig",
    "GateConfig",
    "ModelConfig",
    "RuntimeConfig",
    "TrainingConfig",
    "load_experiment_config",
]
