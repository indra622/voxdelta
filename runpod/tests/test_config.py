from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from voxdelta_runpod.config import (
    BASE_IMAGE,
    CANONICAL_LABELS,
    ExperimentConfig,
    load_experiment_config,
)

CONFIG = Path(__file__).parents[1] / "config" / "experiment.toml"


def test_canonical_configuration_loads_with_stable_identity() -> None:
    config = load_experiment_config(CONFIG.resolve())

    assert config.canonical_labels == CANONICAL_LABELS
    assert config.runtime.base_image == BASE_IMAGE
    assert config.data.train_count == 29_476
    assert config.data.validation_count == 3_569
    assert config.data.final_holdout_count == 3_585
    assert config.digest() == load_experiment_config(CONFIG.resolve()).digest()
    assert len(config.digest()) == 64


def test_configuration_rejects_relative_path() -> None:
    with pytest.raises(ValueError, match="^invalid_experiment_config$"):
        load_experiment_config(Path("config/experiment.toml"))


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("runtime", "base_image", "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04"),
        ("runtime", "gpu_count", 2),
        ("data", "final_holdout_count", 3_620),
        ("training", "effective_batch_size", 8),
        ("gates", "pilot_macro_f1_min_exclusive", 0.14),
    ],
)
def test_configuration_rejects_changed_fixed_contract(
    section: str, field: str, value: object
) -> None:
    payload = tomllib.loads(CONFIG.read_text())
    payload[section][field] = value

    with pytest.raises(ValueError):
        ExperimentConfig.model_validate(payload)


def test_configuration_rejects_unknown_or_sensitive_fields() -> None:
    payload = tomllib.loads(CONFIG.read_text())
    payload["runtime"]["ssh_host"] = "example.invalid"
    payload["registry_token"] = "not-a-real-token"

    with pytest.raises(ValueError):
        ExperimentConfig.model_validate(payload)


def test_remote_roots_are_configurable_with_secure_defaults() -> None:
    config = load_experiment_config(CONFIG.resolve())

    assert config.runtime.volume_root == "/opt/voxdelta-run/runs"
    assert config.runtime.model_root == "/opt/voxdelta-run/models"

    relocated = ExperimentConfig.model_validate(
        {
            **config.model_dump(mode="json"),
            "runtime": {
                **config.runtime.model_dump(mode="json"),
                "volume_root": "/mnt/secure/voxdelta",
                "model_root": "/mnt/secure/models",
            },
        }
    )
    assert relocated.runtime.volume_root == "/mnt/secure/voxdelta"
    assert relocated.runtime.model_root == "/mnt/secure/models"
    assert relocated.digest() != config.digest()


@pytest.mark.parametrize(
    "value",
    [
        "relative/path",
        "/trailing/slash/",
        "/dots/../escape",
        "/has space",
        "/quote'injection",
        "/back`tick",
        "/dollar$sign",
        "/semi;colon",
        "",
        "/",
    ],
)
def test_remote_roots_reject_unsafe_values(value: str) -> None:
    config = load_experiment_config(CONFIG.resolve())
    for field in ("volume_root", "model_root"):
        with pytest.raises(ValueError):
            ExperimentConfig.model_validate(
                {
                    **config.model_dump(mode="json"),
                    "runtime": {**config.runtime.model_dump(mode="json"), field: value},
                }
            )
