from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from voxdelta.evaluation.wav2vec_base import (
    WAV2VEC_MODEL_REVISION,
    WAV2VEC_WEIGHTS_SHA256,
    PreparedWav2VecBase,
)
from voxdelta.providers.checkpoints import checkpoint_tree_digest

from voxdelta_runpod.config import CANONICAL_LABELS, load_experiment_config
from voxdelta_runpod.gates import AggregateReport
from voxdelta_runpod.release import (
    ReleaseError,
    ReleaseManifest,
    build_release_bundle,
    verify_release_bundle,
)

_CONFIG_TOML = Path(__file__).resolve().parents[1] / "config" / "experiment.toml"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, separators=(",", ":")))
    path.chmod(0o600)


def _checkpoint(root: Path) -> Path:
    root.mkdir(mode=0o700)
    _write_json(
        root / "config.json",
        {
            "schema_version": "4",
            "architecture": "wav2vec-xls-r",
            "model_id": "facebook/wav2vec2-xls-r-300m",
            "labels": list(CANONICAL_LABELS),
            "model_revision": WAV2VEC_MODEL_REVISION,
            "base_model_sha256": WAV2VEC_WEIGHTS_SHA256,
            "class_weighting": "none",
            "class_weights": [1.0] * 7,
        },
    )
    _write_json(
        root / "label_mapping.json",
        {str(index): label for index, label in enumerate(CANONICAL_LABELS)},
    )
    _write_json(
        root / "metrics.json",
        {"macro_f1": 0.7749, "validation_hash": "a" * 64, "expected_calibration_error": 0.0897},
    )
    weights = root / "model.safetensors"
    weights.write_bytes(b"synthetic-xls-r-weights")
    weights.chmod(0o600)
    return root


def _base_dir(root: Path) -> Path:
    root.mkdir(mode=0o700)
    for name, data in (
        ("config.json", b'{"model_type":"wav2vec2"}'),
        ("preprocessor_config.json", b'{"feature_size":1}'),
        ("pytorch_model.bin", b"synthetic-base-weights"),
    ):
        path = root / name
        path.write_bytes(data)
        path.chmod(0o600)
    return root


def _fake_base_validator(path: Path) -> PreparedWav2VecBase:
    # The real validate_wav2vec_base pins the base hashes; the stub asserts the
    # three files are present and returns the pinned identity constants.
    names = {entry.name for entry in Path(path).iterdir()}
    assert names == {"config.json", "preprocessor_config.json", "pytorch_model.bin"}
    return PreparedWav2VecBase(path=Path(path))


def _report(
    provider: str,
    checkpoint: str,
    *,
    macro_f1: float,
    ece: float,
    item_count: int,
    opened: int,
) -> dict[str, object]:
    matrix = [[0] * 7 for _ in range(7)]
    matrix[0][0] = item_count
    report: dict[str, object] = {
        "schema_version": "1",
        "provider": provider,
        "item_count": item_count,
        "completed_count": item_count,
        "macro_f1": macro_f1,
        "per_label_f1": {label: 0.5 for label in CANONICAL_LABELS},
        "confusion_matrix": matrix,
        "expected_calibration_error": ece,
        "predicted_class_count": 7,
        "latency_ms": 25.0,
        "elapsed_seconds": 100.0,
        "peak_cpu_rss_mb": 100.0,
        "checkpoint_sha256": checkpoint,
        "report_input_sha256": "b" * 64,
        "finite_training_state": True,
        "provider_reload_verified": True,
        "provenance_verified": True,
        "permissions_private": True,
        "privacy_verified": True,
        "opened_test_count": opened,
    }
    if provider == "wav2vec-xls-r":
        report["peak_cuda_allocated_mb"] = 100.0
        report["peak_cuda_reserved_mb"] = 200.0
    return report


def _frozen_candidate(digest: str) -> dict[str, str]:
    payload = {
        "schema_version": "1",
        "candidate_checkpoint_sha256": digest,
        "baseline_checkpoint_sha256": "c" * 64,
        "config_sha256": "d" * 64,
        "validation_report_sha256": "e" * 64,
        "baseline_config_sha256": "f" * 64,
        "metric_schema_sha256": "0" * 64,
        "decision_rule_sha256": "1" * 64,
        "holdout_identity_sha256": "2" * 64,
    }
    token = hashlib.sha256(
        b"voxdelta-final-capability-v1\0"
        + json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload["capability_token"] = token
    return payload


def _retoken_candidate(payload: dict[str, str]) -> dict[str, str]:
    token_payload = {key: value for key, value in payload.items() if key != "capability_token"}
    payload["capability_token"] = hashlib.sha256(
        b"voxdelta-final-capability-v1\0"
        + json.dumps(token_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def _rewrite_checksums(bundle: Path) -> None:
    sums = bundle / "SHA256SUMS"
    files = sorted(
        (path for path in bundle.rglob("*") if path.is_file() and path != sums),
        key=lambda path: path.relative_to(bundle).as_posix(),
    )
    sums.write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
            f"{path.relative_to(bundle).as_posix()}\n"
            for path in files
        )
    )
    sums.chmod(0o600)


class _Sources:
    def __init__(
        self,
        tmp_path: Path,
        *,
        final_opened: int | None = None,
    ) -> None:
        self.checkpoint = _checkpoint((tmp_path / "checkpoint").resolve())
        self.base = _base_dir((tmp_path / "base").resolve())
        self.digest = checkpoint_tree_digest(self.checkpoint)
        provenance = (tmp_path / "provenance").resolve()
        provenance.mkdir(mode=0o700)
        item_count = 3585
        self.validation = provenance / "validation-report.json"
        _write_json(
            self.validation,
            _report(
                "wav2vec-xls-r",
                self.digest,
                macro_f1=0.7734,
                ece=0.0897,
                item_count=3569,
                opened=0,
            ),
        )
        self.final = provenance / "final-report.json"
        _write_json(
            self.final,
            _report(
                "wav2vec-xls-r",
                self.digest,
                macro_f1=0.7570,
                ece=0.0986,
                item_count=item_count,
                opened=final_opened if final_opened is not None else item_count,
            ),
        )
        self.baseline = provenance / "baseline-report.json"
        _write_json(
            self.baseline,
            _report(
                "emotion2vec-plus",
                "9" * 64,
                macro_f1=0.2275,
                ece=0.4,
                item_count=item_count,
                opened=item_count,
            ),
        )
        self.decision = provenance / "decision.json"
        _write_json(
            self.decision,
            {"promote_xls_r": True, "selected_provider": "wav2vec-xls-r", "reason": "xls-r-wins"},
        )
        validation = AggregateReport.model_validate_json(self.validation.read_text())
        candidate = _frozen_candidate(self.digest)
        candidate.update(
            {
                "baseline_checkpoint_sha256": "9" * 64,
                "config_sha256": load_experiment_config(_CONFIG_TOML).digest(),
                "validation_report_sha256": validation.digest(),
            }
        )
        self.frozen = provenance / "frozen-candidate.json"
        _write_json(self.frozen, _retoken_candidate(candidate))

    def build(self, output: Path) -> Path:
        return build_release_bundle(
            checkpoint_dir=self.checkpoint,
            base_model_dir=self.base,
            frozen_candidate=self.frozen,
            validation_report=self.validation,
            final_report=self.final,
            baseline_report=self.baseline,
            decision=self.decision,
            experiment_config=_CONFIG_TOML,
            output_dir=output,
            release_id="xls-r-emotion-7class-v1",
            base_validator=_fake_base_validator,
        )


def test_release_round_trip_is_private_deterministic_and_reverifiable(tmp_path: Path) -> None:
    sources = _Sources(tmp_path)
    bundle = sources.build((tmp_path / "release").resolve())

    assert bundle.is_dir()
    assert {entry.name for entry in bundle.iterdir()} == {
        "RELEASE.json",
        "MODEL_CARD.md",
        "SHA256SUMS",
        "checkpoint",
        "base-model",
        "provenance",
    }
    assert all(
        not path.is_symlink() and path.stat().st_mode & 0o077 == 0
        for path in (bundle, *bundle.rglob("*"))
    )

    manifest = verify_release_bundle(bundle, base_validator=_fake_base_validator)
    assert isinstance(manifest, ReleaseManifest)
    assert manifest.candidate_checkpoint_sha256 == sources.digest
    assert manifest.labels == CANONICAL_LABELS
    assert manifest.final_sealed is True
    assert manifest.model_id == "facebook/wav2vec2-xls-r-300m"
    assert manifest.base_model_sha256 == WAV2VEC_WEIGHTS_SHA256
    assert manifest.metrics.final_macro_f1 == 0.7570
    assert manifest.metrics.baseline_final_macro_f1 == 0.2275

    # Determinism: a second build from identical sources yields the same tree digest.
    second = sources.build((tmp_path / "release-2").resolve())
    second_manifest = verify_release_bundle(second, base_validator=_fake_base_validator)
    assert second_manifest.bundle_tree_sha256 == manifest.bundle_tree_sha256
    assert (second / "RELEASE.json").read_bytes() == (bundle / "RELEASE.json").read_bytes()


def test_release_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    sources = _Sources(tmp_path)
    output = (tmp_path / "release").resolve()
    sources.build(output)
    with pytest.raises(ReleaseError, match="release_exists"):
        sources.build(output)


def test_release_verify_detects_tampered_payload(tmp_path: Path) -> None:
    sources = _Sources(tmp_path)
    bundle = sources.build((tmp_path / "release").resolve())
    weights = bundle / "checkpoint" / "model.safetensors"
    weights.write_bytes(weights.read_bytes() + b"tampered")
    with pytest.raises(ReleaseError):
        verify_release_bundle(bundle, base_validator=_fake_base_validator)


def test_release_verify_detects_manifest_tampering(tmp_path: Path) -> None:
    sources = _Sources(tmp_path)
    bundle = sources.build((tmp_path / "release").resolve())
    manifest_path = bundle / "RELEASE.json"
    payload = json.loads(manifest_path.read_text())
    payload["candidate_checkpoint_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload, separators=(",", ":")))
    with pytest.raises(ReleaseError):
        verify_release_bundle(bundle, base_validator=_fake_base_validator)


def test_release_rejects_candidate_checkpoint_mismatch(tmp_path: Path) -> None:
    sources = _Sources(tmp_path)
    _write_json(sources.frozen, _frozen_candidate("7" * 64))
    with pytest.raises(ReleaseError, match="candidate_identity_mismatch"):
        sources.build((tmp_path / "release").resolve())


def test_release_rejects_unsealed_final_report(tmp_path: Path) -> None:
    sources = _Sources(tmp_path, final_opened=0)
    with pytest.raises(ReleaseError, match="final_not_sealed"):
        sources.build((tmp_path / "release").resolve())


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("config_sha256", "3" * 64),
        ("validation_report_sha256", "4" * 64),
        ("baseline_checkpoint_sha256", "5" * 64),
    ),
)
def test_release_rejects_candidate_provenance_mismatch(
    tmp_path: Path, field: str, value: str
) -> None:
    sources = _Sources(tmp_path)
    payload = json.loads(sources.frozen.read_text())
    payload[field] = value
    _write_json(sources.frozen, _retoken_candidate(payload))

    with pytest.raises(ReleaseError, match="candidate_provenance_mismatch"):
        sources.build((tmp_path / "release").resolve())


def test_release_rejects_incomplete_or_mismatched_final_reports(tmp_path: Path) -> None:
    sources = _Sources(tmp_path)
    final = json.loads(sources.final.read_text())
    final["completed_count"] -= 1
    final["confusion_matrix"][0][0] -= 1
    _write_json(sources.final, final)

    with pytest.raises(ReleaseError, match="invalid_final_comparison"):
        sources.build((tmp_path / "release").resolve())

    second = tmp_path / "second"
    second.mkdir(mode=0o700)
    sources = _Sources(second)
    baseline = json.loads(sources.baseline.read_text())
    baseline["report_input_sha256"] = "6" * 64
    _write_json(sources.baseline, baseline)

    with pytest.raises(ReleaseError, match="invalid_final_comparison"):
        sources.build((tmp_path / "release-2").resolve())


def test_release_rejects_wrong_final_decision(tmp_path: Path) -> None:
    sources = _Sources(tmp_path)
    _write_json(
        sources.decision,
        {
            "promote_xls_r": False,
            "selected_provider": "emotion2vec-plus",
            "reason": "baseline-retained",
        },
    )
    with pytest.raises(ReleaseError, match="invalid_final_decision"):
        sources.build((tmp_path / "release").resolve())


def test_release_verify_rebinds_manifest_provenance(tmp_path: Path) -> None:
    sources = _Sources(tmp_path)
    bundle = sources.build((tmp_path / "release").resolve())
    manifest_path = bundle / "RELEASE.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["provenance"]["final_report_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")
    manifest_path.chmod(0o600)
    _rewrite_checksums(bundle)

    with pytest.raises(ReleaseError, match="release_provenance_failed"):
        verify_release_bundle(bundle, base_validator=_fake_base_validator)


def test_release_verify_rejects_symlinked_bundle_root(tmp_path: Path) -> None:
    sources = _Sources(tmp_path)
    bundle = sources.build((tmp_path / "release").resolve())
    alias = tmp_path / "release-alias"
    alias.symlink_to(bundle, target_is_directory=True)

    with pytest.raises(ReleaseError, match="invalid_release_bundle"):
        verify_release_bundle(alias, base_validator=_fake_base_validator)
