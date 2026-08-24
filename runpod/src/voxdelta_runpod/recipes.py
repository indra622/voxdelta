"""Deterministic pilot selection and mutually exclusive sampling/loss recipes."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator
from voxdelta.domain.models import EmotionLabel
from voxdelta.evaluation.manifest import DatasetItem, load_manifest

from voxdelta_runpod.config import CANONICAL_LABELS
from voxdelta_runpod.package import SanitizedRecord, opaque_item_key

RecipeName = Literal["pilot-a", "pilot-b"]
SamplerKind = Literal["class-balanced-with-replacement", "uniform-without-replacement"]

PILOT_B_COUNTS: dict[EmotionLabel, int] = {
    "anger": 661,
    "disgust": 201,
    "fear": 239,
    "happiness": 331,
    "neutral": 516,
    "sadness": 1_484,
    "surprise": 68,
}


class RecipeError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class RecipePlan(_FrozenModel):
    name: RecipeName
    scope: Literal["pilot", "full"]
    sampler: SamplerKind
    epoch_draws: int = Field(gt=0)
    loss_weights: tuple[float, ...]
    train: tuple[SanitizedRecord, ...]
    validation: tuple[SanitizedRecord, ...]
    train_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def valid_recipe(self) -> RecipePlan:
        if (
            len(self.loss_weights) != len(CANONICAL_LABELS)
            or any(not math.isfinite(weight) or weight <= 0 for weight in self.loss_weights)
            or len({item.item_key for item in self.train}) != len(self.train)
            or len({item.item_key for item in self.validation}) != len(self.validation)
            or {item.item_key for item in self.train} & {item.item_key for item in self.validation}
        ):
            raise ValueError("invalid recipe")
        uniform_loss = all(abs(weight - 1.0) <= 1e-12 for weight in self.loss_weights)
        if (self.sampler == "class-balanced-with-replacement") != uniform_loss:
            raise ValueError("sampling and loss weighting must remain exclusive")
        if self.sampler == "uniform-without-replacement" and self.epoch_draws != len(self.train):
            raise ValueError("invalid natural sampler")
        return self


class PilotPlans(_FrozenModel):
    pilot_a: RecipePlan
    pilot_b: RecipePlan

    @model_validator(mode="after")
    def shared_validation(self) -> PilotPlans:
        if (
            self.pilot_a.name != "pilot-a"
            or self.pilot_b.name != "pilot-b"
            or self.pilot_a.validation != self.pilot_b.validation
            or self.pilot_a.validation_manifest_sha256 != self.pilot_b.validation_manifest_sha256
        ):
            raise ValueError("pilots must share validation")
        return self


def _rank(item: DatasetItem, seed: int, purpose: str) -> tuple[str, str]:
    payload = f"{seed}\0{purpose}\0{item.sha256}".encode()
    return hashlib.sha256(payload).hexdigest(), item.sha256


def _records(items: list[DatasetItem]) -> tuple[SanitizedRecord, ...]:
    records = [
        SanitizedRecord(
            item_key=opaque_item_key(item),
            audio_path=f"audio/{opaque_item_key(item)}.wav",
            split=item.split,
            emotion=cast(EmotionLabel, item.emotion),
            audio_sha256=item.sha256,
        )
        for item in items
    ]
    if len({item.item_key for item in records}) != len(records):
        raise RecipeError("duplicate_recipe_item")
    return tuple(records)


def _digest(records: tuple[SanitizedRecord, ...]) -> str:
    payload = b"".join(
        (
            json.dumps(
                record.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
        for record in records
    )
    return hashlib.sha256(payload).hexdigest()


def square_root_class_weights(counts: dict[EmotionLabel, int]) -> tuple[float, ...]:
    if set(counts) != set(CANONICAL_LABELS) or any(
        isinstance(value, bool) or value <= 0 for value in counts.values()
    ):
        raise RecipeError("invalid_class_counts")
    total = sum(counts.values())
    raw = [math.sqrt(total / (len(CANONICAL_LABELS) * counts[label])) for label in CANONICAL_LABELS]
    mean = sum(raw) / len(raw)
    return tuple(value / mean for value in raw)


def _source_items(source_manifest: Path) -> list[DatasetItem]:
    if not source_manifest.is_absolute():
        raise RecipeError("invalid_recipe_manifest")
    try:
        items = load_manifest(source_manifest)
    except Exception:
        raise RecipeError("invalid_recipe_manifest") from None
    if any(item.source != "emotion" or item.emotion not in CANONICAL_LABELS for item in items):
        raise RecipeError("invalid_recipe_manifest")
    if len({item.sha256 for item in items}) != len(items):
        raise RecipeError("duplicate_recipe_item")
    return items


def build_pilot_plans(source_manifest: Path, *, seed: int = 622) -> PilotPlans:
    if isinstance(seed, bool) or seed <= 0:
        raise RecipeError("invalid_recipe_seed")
    items = _source_items(source_manifest)
    validation_items: list[DatasetItem] = []
    pilot_a_items: list[DatasetItem] = []
    pilot_b_items: list[DatasetItem] = []
    for label in CANONICAL_LABELS:
        train = [item for item in items if item.split == "train" and item.emotion == label]
        validation = [
            item for item in items if item.split == "validation" and item.emotion == label
        ]
        validation.sort(key=lambda item: _rank(item, seed, "shared-validation"))
        train_a = sorted(train, key=lambda item: _rank(item, seed, "pilot-a"))
        train_b = sorted(train, key=lambda item: _rank(item, seed, "pilot-b"))
        if len(validation) < 50 or len(train_a) < 500 or len(train_b) < PILOT_B_COUNTS[label]:
            raise RecipeError("invalid_recipe_counts")
        validation_items.extend(validation[:50])
        pilot_a_items.extend(train_a[:500])
        pilot_b_items.extend(train_b[: PILOT_B_COUNTS[label]])
    validation_records = _records(validation_items)
    pilot_a_records = _records(pilot_a_items)
    pilot_b_records = _records(pilot_b_items)
    validation_digest = _digest(validation_records)
    pilot_a = RecipePlan(
        name="pilot-a",
        scope="pilot",
        sampler="class-balanced-with-replacement",
        epoch_draws=3_500,
        loss_weights=(1.0,) * len(CANONICAL_LABELS),
        train=pilot_a_records,
        validation=validation_records,
        train_manifest_sha256=_digest(pilot_a_records),
        validation_manifest_sha256=validation_digest,
    )
    pilot_b_counts = Counter(item.emotion for item in pilot_b_records)
    pilot_b = RecipePlan(
        name="pilot-b",
        scope="pilot",
        sampler="uniform-without-replacement",
        epoch_draws=3_500,
        loss_weights=square_root_class_weights(
            {label: pilot_b_counts[label] for label in CANONICAL_LABELS}
        ),
        train=pilot_b_records,
        validation=validation_records,
        train_manifest_sha256=_digest(pilot_b_records),
        validation_manifest_sha256=validation_digest,
    )
    return PilotPlans(pilot_a=pilot_a, pilot_b=pilot_b)


def expand_full_recipe(
    source_manifest: Path,
    winner: RecipeName,
    *,
    seed: int = 622,
) -> RecipePlan:
    items = _source_items(source_manifest)
    train = sorted(
        (item for item in items if item.split == "train"),
        key=lambda item: _rank(item, seed, f"{winner}-full"),
    )
    validation = sorted(
        (item for item in items if item.split == "validation"),
        key=lambda item: _rank(item, seed, "full-validation"),
    )
    if len(train) != 29_476 or len(validation) != 3_569:
        raise RecipeError("invalid_full_recipe_counts")
    train_records = _records(train)
    validation_records = _records(validation)
    counts = Counter(item.emotion for item in train_records)
    if winner == "pilot-a":
        sampler: SamplerKind = "class-balanced-with-replacement"
        loss_weights = (1.0,) * len(CANONICAL_LABELS)
    elif winner == "pilot-b":
        sampler = "uniform-without-replacement"
        loss_weights = square_root_class_weights(
            {label: counts[label] for label in CANONICAL_LABELS}
        )
    else:
        raise RecipeError("invalid_recipe_winner")
    return RecipePlan(
        name=winner,
        scope="full",
        sampler=sampler,
        epoch_draws=len(train_records),
        loss_weights=loss_weights,
        train=train_records,
        validation=validation_records,
        train_manifest_sha256=_digest(train_records),
        validation_manifest_sha256=_digest(validation_records),
    )


__all__ = [
    "PILOT_B_COUNTS",
    "PilotPlans",
    "RecipeError",
    "RecipeName",
    "RecipePlan",
    "SamplerKind",
    "build_pilot_plans",
    "expand_full_recipe",
    "square_root_class_weights",
]
