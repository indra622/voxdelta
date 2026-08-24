from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from voxdelta_runpod.config import CANONICAL_LABELS, load_experiment_config
from voxdelta_runpod.package import SanitizedRecord
from voxdelta_runpod.recipes import RecipePlan
from voxdelta_runpod.training import (
    BATCH_PROFILES,
    TrainingRuntimeError,
    configure_full_model,
    deterministic_crop_start,
    deterministic_epoch_records,
    optimizer_parameter_groups,
    publish_provider_checkpoint,
    run_isolated_memory_probe,
    validate_cuda_runtime,
)

CONFIG = Path(__file__).parents[1] / "config" / "experiment.toml"


class FakeParameter:
    def __init__(self) -> None:
        self.requires_grad = False

    def requires_grad_(self, value: bool) -> FakeParameter:
        self.requires_grad = value
        return self


class FakeModel:
    def __init__(self) -> None:
        self.encoder = FakeParameter()
        self.projector = FakeParameter()
        self.classifier = FakeParameter()
        self.gradient_checkpointing = False

    def parameters(self) -> tuple[FakeParameter, ...]:
        return self.encoder, self.projector, self.classifier

    def named_parameters(self) -> tuple[tuple[str, FakeParameter], ...]:
        return (
            ("wav2vec2.encoder.layer.0.weight", self.encoder),
            ("projector.weight", self.projector),
            ("classifier.weight", self.classifier),
        )

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True


@dataclass
class FakeCompleted:
    returncode: int
    stdout: str
    stderr: str


def _record(index: int, label: str, split: str = "train") -> SanitizedRecord:
    key = f"{index:064x}"
    return SanitizedRecord(
        item_key=key,
        audio_path=f"audio/{key}.wav",
        split=split,
        emotion=label,
        audio_sha256=f"{index + 100:064x}",
    )


def _plan(sampler: str) -> RecipePlan:
    train = tuple(_record(index, label) for index, label in enumerate(CANONICAL_LABELS))
    validation = tuple(
        _record(index + 20, label, "validation") for index, label in enumerate(CANONICAL_LABELS)
    )
    return RecipePlan.model_validate(
        {
            "name": "pilot-a" if sampler.startswith("class") else "pilot-b",
            "scope": "pilot",
            "sampler": sampler,
            "epoch_draws": 14 if sampler.startswith("class") else 7,
            "loss_weights": (1.0,) * 7
            if sampler.startswith("class")
            else (0.9, 1.0, 1.1, 1.0, 1.0, 1.0, 1.0),
            "train": train,
            "validation": validation,
            "train_manifest_sha256": "a" * 64,
            "validation_manifest_sha256": "b" * 64,
        }
    )


def test_model_setup_trains_every_parameter_and_uses_exact_learning_rate_groups() -> None:
    model = FakeModel()
    config = load_experiment_config(CONFIG.resolve())

    configure_full_model(model)
    groups = optimizer_parameter_groups(model, config)

    assert model.gradient_checkpointing is True
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert groups[0]["params"] == [model.encoder]
    assert groups[0]["lr"] == 1e-5
    assert groups[1]["params"] == [model.projector, model.classifier]
    assert groups[1]["lr"] == 1e-4


def test_crop_and_sampler_orders_are_deterministic_per_seed_epoch_and_fingerprint() -> None:
    first = deterministic_crop_start(
        400_000, 320_000, seed=622, epoch=2, semantic_fingerprint="a" * 64
    )
    assert first == deterministic_crop_start(
        400_000, 320_000, seed=622, epoch=2, semantic_fingerprint="a" * 64
    )
    assert first != deterministic_crop_start(
        400_000, 320_000, seed=622, epoch=3, semantic_fingerprint="a" * 64
    )

    balanced = _plan("class-balanced-with-replacement")
    natural = _plan("uniform-without-replacement")
    assert deterministic_epoch_records(balanced, seed=622, epoch=0) == (
        deterministic_epoch_records(balanced, seed=622, epoch=0)
    )
    assert len(deterministic_epoch_records(balanced, seed=622, epoch=0)) == 14
    assert len(set(deterministic_epoch_records(natural, seed=622, epoch=0))) == 7


def test_memory_probe_uses_fresh_calls_and_falls_back_only_on_explicit_oom() -> None:
    calls: list[list[str]] = []

    def runner(command: list[str], **_kwargs: Any) -> FakeCompleted:
        calls.append(command)
        if "8" in command[-3:]:
            return FakeCompleted(75, "", "probe_oom\n")
        return FakeCompleted(0, "probe_ok\n", "")

    result = run_isolated_memory_probe(["python", "probe.py"], runner=runner)

    assert result.selected == BATCH_PROFILES[1]
    assert [attempt.status for attempt in result.attempts] == ["oom", "pass"]
    assert len(calls) == 2
    assert calls[0] is not calls[1]

    def bad_runner(_command: list[str], **_kwargs: Any) -> FakeCompleted:
        return FakeCompleted(1, "", "unexpected private error")

    with pytest.raises(TrainingRuntimeError, match="^memory_probe_failed$"):
        run_isolated_memory_probe(["python", "probe.py"], runner=bad_runner)


class FakeCuda:
    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def is_bf16_supported(self) -> bool:
        return True


class FakeTorch:
    cuda = FakeCuda()


def test_cuda_contract_requires_one_bf16_gpu() -> None:
    validate_cuda_runtime(FakeTorch())
    bad = FakeTorch()
    bad.cuda.device_count = lambda: 2  # type: ignore[method-assign]
    with pytest.raises(TrainingRuntimeError, match="^invalid_cuda_runtime$"):
        validate_cuda_runtime(bad)


def test_published_runpod_checkpoint_reloads_through_production_contract(tmp_path: Path) -> None:
    from voxdelta.providers._emotion_runtime import validate_checkpoint

    from voxdelta_runpod.training import EpochMetrics

    plan = _plan("uniform-without-replacement")
    output = publish_provider_checkpoint(
        (tmp_path / "checkpoint").resolve(),
        b"synthetic-safetensors",
        plan,
        EpochMetrics(
            completed_count=7,
            macro_f1=0.5,
            per_label_f1={label: 0.5 for label in CANONICAL_LABELS},
            predicted_class_count=7,
        ),
        load_experiment_config(CONFIG.resolve()),
    )

    info = validate_checkpoint(
        output,
        architecture="wav2vec-xls-r",
        model_id="facebook/wav2vec2-xls-r-300m",
    )
    assert info.path == output
    assert all(path.stat().st_mode & 0o077 == 0 for path in output.iterdir())
