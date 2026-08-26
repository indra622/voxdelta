from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from conftest import RELEASE_LABELS, ReleaseBundleBuilder

from voxdelta.providers import release_bundle
from voxdelta.providers.release_bundle import (
    PROMOTED_RELEASE_ID,
    ReleaseBundleError,
    verify_release_bundle,
)

REORDERED_LABELS = ("anger", "happiness", "disgust", "fear", "neutral", "sadness", "surprise")


def test_valid_bundle_returns_bound_checkpoint_and_base_model_paths(
    tmp_path: Path, release_bundles: ReleaseBundleBuilder
) -> None:
    root = release_bundles.build(tmp_path / "release")

    verified = verify_release_bundle(root)

    assert verified.path == root
    assert verified.release_id == PROMOTED_RELEASE_ID
    assert verified.checkpoint_path == root / "checkpoint"
    assert verified.base_model_path == root / "base-model"
    assert verified.labels == RELEASE_LABELS
    assert verified.candidate_checkpoint_sha256 == release_bundle.checkpoint_tree_digest(
        root / "checkpoint"
    )


def test_verifier_accepts_a_string_bundle_path(
    tmp_path: Path, release_bundles: ReleaseBundleBuilder
) -> None:
    root = release_bundles.build(tmp_path / "release")

    assert verify_release_bundle(str(root)).path == root


def test_missing_bundle_is_rejected_without_disclosing_the_path(tmp_path: Path) -> None:
    absent = tmp_path / "private-release-sentinel"

    with pytest.raises(ReleaseBundleError) as raised:
        verify_release_bundle(absent)

    assert str(raised.value) == "invalid_release_bundle"
    assert str(absent) not in str(raised.value)


def test_relative_bundle_path_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, release_bundles: ReleaseBundleBuilder
) -> None:
    release_bundles.build(tmp_path / "release")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ReleaseBundleError):
        verify_release_bundle(Path("release"))


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlinks are required")
def test_symlinked_bundle_root_is_rejected(
    tmp_path: Path, release_bundles: ReleaseBundleBuilder
) -> None:
    root = release_bundles.build(tmp_path / "release")
    link = tmp_path / "linked-release"
    link.symlink_to(root, target_is_directory=True)

    with pytest.raises(ReleaseBundleError):
        verify_release_bundle(link)


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlinks are required")
def test_symlinked_payload_inside_the_bundle_is_rejected(
    tmp_path: Path, release_bundles: ReleaseBundleBuilder
) -> None:
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"synthetic-candidate-weights")
    root = release_bundles.build(tmp_path / "release")
    target = root / "checkpoint" / "model.safetensors"
    target.unlink()
    target.symlink_to(outside)

    with pytest.raises(ReleaseBundleError):
        verify_release_bundle(root)


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlinks are required")
def test_symlinked_directory_inside_the_bundle_is_rejected(
    tmp_path: Path, release_bundles: ReleaseBundleBuilder
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    root = release_bundles.build(tmp_path / "release")
    (root / "provenance" / "decision.json").unlink()
    (root / "provenance").rmdir()
    (root / "provenance").symlink_to(elsewhere, target_is_directory=True)

    with pytest.raises(ReleaseBundleError):
        verify_release_bundle(root)


def test_tampered_payload_bytes_are_rejected(
    tmp_path: Path, release_bundles: ReleaseBundleBuilder
) -> None:
    root = release_bundles.build(tmp_path / "release")
    (root / "checkpoint" / "model.safetensors").write_bytes(b"synthetic-candidate-weightt")

    with pytest.raises(ReleaseBundleError):
        verify_release_bundle(root)


def test_tampered_manifest_that_no_longer_matches_the_checksums_is_rejected(
    tmp_path: Path, release_bundles: ReleaseBundleBuilder
) -> None:
    root = release_bundles.build(tmp_path / "release")
    manifest = json.loads((root / "RELEASE.json").read_text(encoding="utf-8"))
    manifest["final_holdout_count"] = 1
    (root / "RELEASE.json").write_bytes(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )

    with pytest.raises(ReleaseBundleError):
        verify_release_bundle(root)


def test_resealed_forgery_of_a_payload_hash_is_still_rejected(
    tmp_path: Path, release_bundles: ReleaseBundleBuilder
) -> None:
    root = release_bundles.build(tmp_path / "release")
    forged = b"forged-candidate-weights"
    (root / "checkpoint" / "model.safetensors").write_bytes(forged)
    manifest = json.loads((root / "RELEASE.json").read_text(encoding="utf-8"))
    for payload in manifest["payloads"]:
        if payload["path"] == "checkpoint/model.safetensors":
            payload["sha256"] = hashlib.sha256(forged).hexdigest()
            payload["bytes"] = len(forged)
    release_bundles.reseal(root, manifest)

    with pytest.raises(ReleaseBundleError):
        verify_release_bundle(root)


def test_missing_checksums_file_is_rejected(
    tmp_path: Path, release_bundles: ReleaseBundleBuilder
) -> None:
    root = release_bundles.build(tmp_path / "release")
    (root / "SHA256SUMS").unlink()

    with pytest.raises(ReleaseBundleError):
        verify_release_bundle(root)


def test_missing_manifest_file_is_rejected(
    tmp_path: Path, release_bundles: ReleaseBundleBuilder
) -> None:
    root = release_bundles.build(tmp_path / "release")
    (root / "RELEASE.json").unlink()

    with pytest.raises(ReleaseBundleError):
        verify_release_bundle(root)


def test_malformed_manifest_json_is_rejected(
    tmp_path: Path, release_bundles: ReleaseBundleBuilder
) -> None:
    root = release_bundles.build(tmp_path / "release")
    raw = b"{not-json"
    (root / "RELEASE.json").write_bytes(raw)
    lines = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(raw).hexdigest()}  RELEASE.json\n"
            if line.endswith("  RELEASE.json")
            else f"{line}\n"
            for line in lines
        ),
        encoding="utf-8",
    )

    with pytest.raises(ReleaseBundleError):
        verify_release_bundle(root)


def test_malformed_checksums_line_is_rejected(
    tmp_path: Path, release_bundles: ReleaseBundleBuilder
) -> None:
    root = release_bundles.build(tmp_path / "release")
    existing = (root / "SHA256SUMS").read_text(encoding="utf-8")
    (root / "SHA256SUMS").write_text(existing + "not-a-checksum-line\n", encoding="utf-8")

    with pytest.raises(ReleaseBundleError):
        verify_release_bundle(root)


def test_manifest_that_is_not_an_object_is_rejected(
    tmp_path: Path, release_bundles: ReleaseBundleBuilder
) -> None:
    root = release_bundles.build(tmp_path / "release")
    release_bundles.reseal(root, ["not", "an", "object"])

    with pytest.raises(ReleaseBundleError):
        verify_release_bundle(root)


@pytest.mark.parametrize(
    "updates",
    [
        {"schema_version": "2"},
        {"model_id": "facebook/wav2vec2-large"},
        {"model_revision": "0" * 40},
        {"decision": "baseline-wins"},
        {"final_sealed": False},
        {"final_holdout_count": 0},
        {"final_holdout_count": True},
        {"candidate_checkpoint_sha256": "d" * 64},
        {"bundle_tree_sha256": "e" * 64},
    ],
)
def test_incompatible_manifest_pins_are_rejected(
    tmp_path: Path, release_bundles: ReleaseBundleBuilder, updates: dict[str, object]
) -> None:
    root = release_bundles.build(tmp_path / "release", manifest_updates=updates)

    with pytest.raises(ReleaseBundleError):
        verify_release_bundle(root)


def test_unknown_manifest_key_is_rejected(
    tmp_path: Path, release_bundles: ReleaseBundleBuilder
) -> None:
    root = release_bundles.build(tmp_path / "release", manifest_updates={"promoted_by": "operator"})

    with pytest.raises(ReleaseBundleError):
        verify_release_bundle(root)


def test_missing_manifest_key_is_rejected(
    tmp_path: Path, release_bundles: ReleaseBundleBuilder
) -> None:
    root = release_bundles.build(tmp_path / "release", manifest_drop=("decision",))

    with pytest.raises(ReleaseBundleError):
        verify_release_bundle(root)


def test_embedded_calibration_key_is_rejected_because_calibration_is_a_separate_artifact(
    tmp_path: Path, release_bundles: ReleaseBundleBuilder
) -> None:
    root = release_bundles.build(
        tmp_path / "release", manifest_updates={"calibration": {"bins": 15}}
    )

    with pytest.raises(ReleaseBundleError):
        verify_release_bundle(root)


def test_wrong_release_id_is_rejected(
    tmp_path: Path, release_bundles: ReleaseBundleBuilder
) -> None:
    root = release_bundles.build(tmp_path / "release", release_id="xls-r-emotion-7class-v2")

    with pytest.raises(ReleaseBundleError):
        verify_release_bundle(root)


def test_base_model_digest_that_is_not_the_pinned_encoder_is_rejected(
    tmp_path: Path, release_bundles: ReleaseBundleBuilder
) -> None:
    root = release_bundles.build(tmp_path / "release", base_weights=b"a-different-encoder")

    with pytest.raises(ReleaseBundleError):
        verify_release_bundle(root)


def test_manifest_labels_outside_the_fixed_seven_class_order_are_rejected(
    tmp_path: Path, release_bundles: ReleaseBundleBuilder
) -> None:
    root = release_bundles.build(tmp_path / "release", manifest_labels=REORDERED_LABELS)

    with pytest.raises(ReleaseBundleError):
        verify_release_bundle(root)


def test_checkpoint_label_mapping_outside_the_fixed_seven_class_order_is_rejected(
    tmp_path: Path, release_bundles: ReleaseBundleBuilder
) -> None:
    root = release_bundles.build(
        tmp_path / "release",
        label_mapping={str(index): label for index, label in enumerate(REORDERED_LABELS)},
    )

    with pytest.raises(ReleaseBundleError):
        verify_release_bundle(root)


def test_six_class_checkpoint_label_mapping_is_rejected(
    tmp_path: Path, release_bundles: ReleaseBundleBuilder
) -> None:
    root = release_bundles.build(
        tmp_path / "release",
        label_mapping={str(index): label for index, label in enumerate(RELEASE_LABELS[:6])},
    )

    with pytest.raises(ReleaseBundleError):
        verify_release_bundle(root)


def test_checkpoint_config_labels_outside_the_fixed_seven_class_order_are_rejected(
    tmp_path: Path, release_bundles: ReleaseBundleBuilder
) -> None:
    root = release_bundles.build(
        tmp_path / "release",
        checkpoint_config=release_bundles.checkpoint_config(labels=list(REORDERED_LABELS)),
    )

    with pytest.raises(ReleaseBundleError):
        verify_release_bundle(root)


def test_checkpoint_config_for_a_foreign_architecture_is_rejected(
    tmp_path: Path, release_bundles: ReleaseBundleBuilder
) -> None:
    root = release_bundles.build(
        tmp_path / "release",
        checkpoint_config=release_bundles.checkpoint_config(architecture="emotion2vec-plus"),
    )

    with pytest.raises(ReleaseBundleError):
        verify_release_bundle(root)


def test_checkpoint_config_pinned_to_a_foreign_encoder_revision_is_rejected(
    tmp_path: Path, release_bundles: ReleaseBundleBuilder
) -> None:
    root = release_bundles.build(
        tmp_path / "release",
        checkpoint_config=release_bundles.checkpoint_config(model_revision="0" * 40),
    )

    with pytest.raises(ReleaseBundleError):
        verify_release_bundle(root)


@pytest.mark.parametrize(
    "omitted",
    [
        "checkpoint/model.safetensors",
        "checkpoint/label_mapping.json",
        "checkpoint/metrics.json",
        "checkpoint/config.json",
        "base-model/pytorch_model.bin",
        "base-model/config.json",
        "base-model/preprocessor_config.json",
    ],
)
def test_bundle_missing_a_required_checkpoint_or_base_model_file_is_rejected(
    tmp_path: Path, release_bundles: ReleaseBundleBuilder, omitted: str
) -> None:
    root = release_bundles.build(tmp_path / "release", omit_payloads=(omitted,))

    with pytest.raises(ReleaseBundleError):
        verify_release_bundle(root)


def test_unattested_extra_file_in_the_bundle_is_rejected(
    tmp_path: Path, release_bundles: ReleaseBundleBuilder
) -> None:
    root = release_bundles.build(tmp_path / "release")
    (root / "checkpoint" / "smuggled.bin").write_bytes(b"smuggled")

    with pytest.raises(ReleaseBundleError):
        verify_release_bundle(root)


def test_attested_payload_that_is_absent_on_disk_is_rejected(
    tmp_path: Path, release_bundles: ReleaseBundleBuilder
) -> None:
    root = release_bundles.build(tmp_path / "release")
    (root / "MODEL_CARD.md").unlink()

    with pytest.raises(ReleaseBundleError):
        verify_release_bundle(root)


def test_checksums_entry_without_a_manifest_payload_is_rejected(
    tmp_path: Path, release_bundles: ReleaseBundleBuilder
) -> None:
    root = release_bundles.build(tmp_path / "release")
    smuggled = b"smuggled"
    (root / "checkpoint" / "smuggled.bin").write_bytes(smuggled)
    existing = (root / "SHA256SUMS").read_text(encoding="utf-8")
    (root / "SHA256SUMS").write_text(
        f"{existing}{hashlib.sha256(smuggled).hexdigest()}  checkpoint/smuggled.bin\n",
        encoding="utf-8",
    )

    with pytest.raises(ReleaseBundleError):
        verify_release_bundle(root)


def test_bundle_path_that_is_a_regular_file_is_rejected(tmp_path: Path) -> None:
    plain = tmp_path / "release-file"
    plain.write_bytes(b"not a bundle")

    with pytest.raises(ReleaseBundleError):
        verify_release_bundle(plain)


def test_manifest_payload_with_an_escaping_relative_path_is_rejected(
    tmp_path: Path, release_bundles: ReleaseBundleBuilder
) -> None:
    root = release_bundles.build(tmp_path / "release")
    manifest = json.loads((root / "RELEASE.json").read_text(encoding="utf-8"))
    manifest["payloads"][0]["path"] = "../escaped.bin"
    release_bundles.reseal(root, manifest)

    with pytest.raises(ReleaseBundleError):
        verify_release_bundle(root)
