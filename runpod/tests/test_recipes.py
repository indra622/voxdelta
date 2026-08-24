from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError
from voxdelta.evaluation.manifest import DatasetItem

from voxdelta_runpod.config import CANONICAL_LABELS
from voxdelta_runpod.recipes import (
    PILOT_B_COUNTS,
    RecipePlan,
    build_pilot_plans,
    expand_full_recipe,
    square_root_class_weights,
)

ACTUAL_MANIFEST = Path("/Volumes/nvme1/codes/voxdelta/data/manifests/emotion.jsonl")


def _manifest(path: Path, train_per_label: int = 1_500, validation_per_label: int = 60) -> Path:
    records: list[DatasetItem] = []
    index = 0
    for label in CANONICAL_LABELS:
        for split, count in (("train", train_per_label), ("validation", validation_per_label)):
            for _ in range(count):
                digest = hashlib.sha256(f"recipe-audio-{index}".encode()).hexdigest()
                records.append(
                    DatasetItem(
                        id=f"private-{index}",
                        call_id=f"call-{index}",
                        speaker_id=f"speaker-{index}",
                        audio_path=f"/private/{index}.wav",
                        transcript="private transcript",
                        split=split,
                        source="emotion",
                        emotion=label,
                        sha256=digest,
                    )
                )
                index += 1
    path.write_text("".join(item.model_dump_json() + "\n" for item in records))
    return path.resolve()


def test_pilot_plans_share_validation_and_freeze_exact_recipe_counts(tmp_path: Path) -> None:
    plans = build_pilot_plans(_manifest(tmp_path / "source.jsonl"))
    repeated = build_pilot_plans((tmp_path / "source.jsonl").resolve())

    assert plans == repeated
    assert len(plans.pilot_a.train) == 3_500
    assert len(plans.pilot_b.train) == 3_500
    assert len(plans.pilot_a.validation) == 350
    assert plans.pilot_a.validation == plans.pilot_b.validation
    assert Counter(item.emotion for item in plans.pilot_a.train) == Counter(
        {label: 500 for label in CANONICAL_LABELS}
    )
    assert Counter(item.emotion for item in plans.pilot_b.train) == Counter(PILOT_B_COUNTS)
    assert plans.pilot_a.sampler == "class-balanced-with-replacement"
    assert plans.pilot_a.loss_weights == (1.0,) * 7
    assert plans.pilot_b.sampler == "uniform-without-replacement"
    assert plans.pilot_b.loss_weights != (1.0,) * 7
    assert set(item.item_key for item in plans.pilot_a.train).isdisjoint(
        item.item_key for item in plans.pilot_a.validation
    )


def test_square_root_weights_are_mean_one_and_use_only_supplied_train_counts() -> None:
    weights = square_root_class_weights(PILOT_B_COUNTS)

    assert sum(weights) / len(weights) == pytest.approx(1.0)
    anger = weights[CANONICAL_LABELS.index("anger")]
    surprise = weights[CANONICAL_LABELS.index("surprise")]
    assert surprise > anger


def test_recipe_model_rejects_balanced_sampling_with_nonuniform_weights(
    tmp_path: Path,
) -> None:
    plan = build_pilot_plans(_manifest(tmp_path / "source.jsonl")).pilot_a
    payload = plan.model_dump()
    payload["loss_weights"] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0)

    with pytest.raises(ValidationError, match="sampling and loss weighting"):
        RecipePlan.model_validate(payload)


def test_actual_full_recipe_counts_and_digests_are_stable() -> None:
    pilot_a = expand_full_recipe(ACTUAL_MANIFEST.resolve(), "pilot-a")
    pilot_b = expand_full_recipe(ACTUAL_MANIFEST.resolve(), "pilot-b")

    assert len(pilot_a.train) == len(pilot_b.train) == 29_476
    assert len(pilot_a.validation) == len(pilot_b.validation) == 3_569
    assert pilot_a.sampler == "class-balanced-with-replacement"
    assert pilot_b.sampler == "uniform-without-replacement"
    assert pilot_a.train_manifest_sha256 != pilot_b.train_manifest_sha256
    assert pilot_a.validation_manifest_sha256 == pilot_b.validation_manifest_sha256
    assert pilot_a.loss_weights == (1.0,) * 7
    assert sum(pilot_b.loss_weights) / 7 == pytest.approx(1.0)
