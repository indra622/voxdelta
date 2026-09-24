"""Shared synthetic fixtures for the promoted XLS-R release bundle."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from voxdelta.providers import release_bundle
from voxdelta.providers.release_bundle import PROMOTED_RELEASE_ID

RELEASE_LABELS = ("happiness", "anger", "disgust", "fear", "neutral", "sadness", "surprise")
RELEASE_MODEL_ID = "facebook/wav2vec2-xls-r-300m"
RELEASE_MODEL_REVISION = "1a640f32ac3e39899438a2931f9924c02f080a54"
PINNED_BASE_WEIGHTS_SHA256 = "d5e490574712ad0a6736923b9ed11d4cd51c78609c36205f704fc4e87b11d2e0"
SYNTHETIC_BASE_WEIGHTS = b"synthetic-base-weights"


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ReleaseBundleBuilder:
    """Write self-consistent synthetic release bundles for verification tests.

    The pinned base-model digest is rebound to synthetic bytes so tests can exercise
    every structural rule without a 1.2 GiB encoder.
    """

    labels: tuple[str, ...] = RELEASE_LABELS
    model_id: str = RELEASE_MODEL_ID
    model_revision: str = RELEASE_MODEL_REVISION
    base_weights: bytes = SYNTHETIC_BASE_WEIGHTS

    @property
    def base_weights_sha256(self) -> str:
        return hashlib.sha256(self.base_weights).hexdigest()

    def checkpoint_config(self, **updates: object) -> dict[str, object]:
        config: dict[str, object] = {
            "schema_version": "4",
            "architecture": "wav2vec-xls-r",
            "model_id": self.model_id,
            "labels": list(self.labels),
            "model_revision": self.model_revision,
            "base_model_sha256": self.base_weights_sha256,
            "class_weighting": "none",
            "class_weights": [1.0] * 7,
        }
        config.update(updates)
        return config

    def contents(
        self,
        *,
        label_mapping: Mapping[str, str] | None = None,
        checkpoint_config: Mapping[str, object] | None = None,
        base_weights: bytes | None = None,
    ) -> dict[str, bytes]:
        mapping = (
            dict(label_mapping)
            if label_mapping is not None
            else {str(index): label for index, label in enumerate(self.labels)}
        )
        config = (
            dict(checkpoint_config) if checkpoint_config is not None else self.checkpoint_config()
        )
        return {
            "MODEL_CARD.md": b"# synthetic release\n",
            "base-model/config.json": b'{"model_type":"wav2vec2"}',
            "base-model/preprocessor_config.json": b'{"sampling_rate":16000}',
            "base-model/pytorch_model.bin": (
                self.base_weights if base_weights is None else base_weights
            ),
            "checkpoint/config.json": json.dumps(config).encode("utf-8"),
            "checkpoint/label_mapping.json": json.dumps(mapping).encode("utf-8"),
            "checkpoint/metrics.json": json.dumps(
                {"macro_f1": 0.75, "validation_hash": "a" * 64}
            ).encode("utf-8"),
            "checkpoint/model.safetensors": b"synthetic-candidate-weights",
            "provenance/decision.json": b'{"decision":"xls-r-wins"}',
        }

    def build(
        self,
        root: Path,
        *,
        release_id: str = PROMOTED_RELEASE_ID,
        label_mapping: Mapping[str, str] | None = None,
        checkpoint_config: Mapping[str, object] | None = None,
        base_weights: bytes | None = None,
        manifest_labels: Sequence[str] | None = None,
        manifest_updates: Mapping[str, object] | None = None,
        manifest_drop: Sequence[str] = (),
        extra_files: Mapping[str, bytes] | None = None,
        omit_payloads: Sequence[str] = (),
    ) -> Path:
        root.mkdir(parents=True)
        files = self.contents(
            label_mapping=label_mapping,
            checkpoint_config=checkpoint_config,
            base_weights=base_weights,
        )
        for name in omit_payloads:
            files.pop(name, None)
        files.update(extra_files or {})
        for relative, data in files.items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

        payloads: list[dict[str, object]] = sorted(
            (
                {
                    "path": relative,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data),
                }
                for relative, data in files.items()
            ),
            key=lambda payload: str(payload["path"]),
        )
        checkpoint = root / "checkpoint"
        candidate = (
            release_bundle.checkpoint_tree_digest(checkpoint) if checkpoint.is_dir() else "c" * 64
        )
        manifest: dict[str, object] = {
            "schema_version": "1",
            "release_id": release_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "base_model_sha256": hashlib.sha256(
                self.base_weights if base_weights is None else base_weights
            ).hexdigest(),
            "candidate_checkpoint_sha256": candidate,
            "labels": list(self.labels if manifest_labels is None else manifest_labels),
            "final_sealed": True,
            "final_holdout_count": 3585,
            "decision": "xls-r-wins",
            "metrics": {
                "validation_macro_f1": 0.77,
                "validation_expected_calibration_error": 0.09,
                "final_macro_f1": 0.75,
                "final_expected_calibration_error": 0.10,
                "baseline_final_macro_f1": 0.23,
            },
            "provenance": {"capability_token": "b" * 64},
            "payloads": payloads,
            "bundle_tree_sha256": _canonical_sha256(payloads),
        }
        for key in manifest_drop:
            manifest.pop(key, None)
        manifest.update(manifest_updates or {})
        self.reseal(root, manifest)
        return root

    @staticmethod
    def reseal(root: Path, manifest: object) -> None:
        """Rewrite RELEASE.json and regenerate SHA256SUMS to match it."""

        raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        (root / "RELEASE.json").write_bytes(raw)
        lines = [f"{hashlib.sha256(raw).hexdigest()}  RELEASE.json"]
        payloads = manifest.get("payloads") if isinstance(manifest, dict) else None
        for payload in payloads if isinstance(payloads, list) else []:
            if isinstance(payload, dict):
                lines.append(f"{payload['sha256']}  {payload['path']}")
        (root / "SHA256SUMS").write_text(
            "".join(f"{line}\n" for line in sorted(lines, key=lambda line: line.split("  ", 1)[1])),
            encoding="utf-8",
        )


@pytest.fixture
def release_bundles(monkeypatch: pytest.MonkeyPatch) -> ReleaseBundleBuilder:
    """Bind the pinned encoder digest to synthetic bytes for the duration of one test."""

    builder = ReleaseBundleBuilder()
    monkeypatch.setattr(release_bundle, "WAV2VEC_WEIGHTS_SHA256", builder.base_weights_sha256)
    return builder
