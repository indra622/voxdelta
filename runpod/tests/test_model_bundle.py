from __future__ import annotations

import json
from pathlib import Path

import pytest

from voxdelta_runpod.model_bundle import (
    ModelBundleError,
    build_model_bundle,
    extract_model_bundle,
)


def _baseline(root: Path) -> Path:
    root.mkdir(mode=0o700)
    payloads = {
        "config.json": {
            "schema_version": "2",
            "architecture": "emotion2vec-plus",
            "model_id": "iic/emotion2vec_plus_large",
            "labels": [
                "happiness",
                "anger",
                "disgust",
                "fear",
                "neutral",
                "sadness",
                "surprise",
            ],
            "embedding_size": 1024,
            "encoder_hash": "a" * 64,
            "encoder_revision": "v2.0.5",
            "freeze_encoder": True,
            "class_weighting": "inverse-frequency",
            "class_weights": [1.0] * 7,
        },
        "label_mapping.json": {
            str(index): label
            for index, label in enumerate(
                ["happiness", "anger", "disgust", "fear", "neutral", "sadness", "surprise"]
            )
        },
        "metrics.json": {"macro_f1": 0.24, "validation_hash": "b" * 64},
    }
    for name, payload in payloads.items():
        path = root / name
        path.write_text(json.dumps(payload, separators=(",", ":")))
        path.chmod(0o600)
    weights = root / "model.safetensors"
    weights.write_bytes(b"synthetic-weights")
    weights.chmod(0o600)
    return root


def test_baseline_model_bundle_round_trip_is_private_and_idempotent(tmp_path: Path) -> None:
    source = _baseline((tmp_path / "source").resolve())
    bundle = build_model_bundle(
        source,
        (tmp_path / "bundle").resolve(),
        "emotion2vec-baseline",
    )
    target = (tmp_path / "target").resolve()
    extracted = extract_model_bundle(
        bundle / "emotion2vec-baseline.tar.zst",
        bundle / "emotion2vec-baseline.sidecar.json",
        target,
        expected_kind="emotion2vec-baseline",
    )
    assert extracted == target
    assert {path.name for path in target.iterdir()} == {
        "config.json",
        "label_mapping.json",
        "metrics.json",
        "model.safetensors",
    }
    assert all(path.stat().st_mode & 0o077 == 0 for path in (target, *target.iterdir()))
    assert (
        extract_model_bundle(
            bundle / "emotion2vec-baseline.tar.zst",
            bundle / "emotion2vec-baseline.sidecar.json",
            target,
            expected_kind="emotion2vec-baseline",
        )
        == target
    )


def test_model_bundle_rejects_tampered_archive_and_existing_mismatch(tmp_path: Path) -> None:
    source = _baseline((tmp_path / "source").resolve())
    bundle = build_model_bundle(
        source,
        (tmp_path / "bundle").resolve(),
        "emotion2vec-baseline",
    )
    archive = bundle / "emotion2vec-baseline.tar.zst"
    archive.write_bytes(archive.read_bytes() + b"tampered")
    with pytest.raises(ModelBundleError, match="model_bundle_digest_mismatch"):
        extract_model_bundle(
            archive,
            bundle / "emotion2vec-baseline.sidecar.json",
            (tmp_path / "target").resolve(),
            expected_kind="emotion2vec-baseline",
        )
