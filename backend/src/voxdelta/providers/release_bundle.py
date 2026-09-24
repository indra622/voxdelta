"""Verify the promoted, immutable XLS-R emotion release bundle before local inference.

A release bundle is trusted only when its manifest, its checksum file, and the bytes on
disk agree with one another and with the pinned encoder identity and the fixed
seven-class label order. Verification is offline, follows no symlink, accepts no file the
manifest does not attest, and fails closed with one fixed, path-free error so a rejected
bundle never discloses a host path. Nothing here depends on the experiment-only
``voxdelta_runpod`` package.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from voxdelta.evaluation.emotion_training import CANONICAL_LABELS
from voxdelta.evaluation.manifest import read_trusted_regular_file
from voxdelta.evaluation.wav2vec_base import (
    WAV2VEC_MODEL_ID,
    WAV2VEC_MODEL_REVISION,
    WAV2VEC_WEIGHTS_SHA256,
)
from voxdelta.providers._attested_tree import (
    canonical_sha256,
    digest_and_size,
    is_positive_int,
    is_sha256,
    load_strict_json,
    parse_checksums,
    resolved_artifact_root,
    safe_relative,
    tree_files,
)
from voxdelta.providers.checkpoints import checkpoint_tree_digest

PROMOTED_RELEASE_ID = "xls-r-emotion-7class-v1"
RELEASE_SCHEMA_VERSION = "1"
RELEASE_DECISION = "xls-r-wins"
MANIFEST_NAME = "RELEASE.json"
CHECKSUMS_NAME = "SHA256SUMS"
CHECKPOINT_DIRECTORY = "checkpoint"
BASE_MODEL_DIRECTORY = "base-model"
CHECKPOINT_FILES = ("config.json", "label_mapping.json", "metrics.json", "model.safetensors")
BASE_MODEL_FILES = ("config.json", "preprocessor_config.json", "pytorch_model.bin")
BASE_MODEL_WEIGHTS = f"{BASE_MODEL_DIRECTORY}/pytorch_model.bin"
ARCHITECTURE = "wav2vec-xls-r"

_REQUIRED_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "release_id",
        "model_id",
        "model_revision",
        "base_model_sha256",
        "candidate_checkpoint_sha256",
        "labels",
        "final_sealed",
        "final_holdout_count",
        "decision",
        "metrics",
        "provenance",
        "payloads",
        "bundle_tree_sha256",
    }
)
_PAYLOAD_KEYS = frozenset({"path", "sha256", "bytes"})
_RESERVED_NAMES = frozenset({MANIFEST_NAME, CHECKSUMS_NAME})


class ReleaseBundleError(ValueError):
    """Fixed, path-free rejection of an untrusted or unrecognized release bundle."""

    def __init__(self) -> None:
        super().__init__("invalid_release_bundle")


@dataclass(frozen=True, slots=True)
class VerifiedRelease:
    """The verified identity and the two local model directories the provider may load."""

    path: Path
    release_id: str
    checkpoint_path: Path
    base_model_path: Path
    labels: tuple[str, ...]
    candidate_checkpoint_sha256: str
    bundle_tree_sha256: str


def _fail() -> ReleaseBundleError:
    return ReleaseBundleError()


def _safe_relative(value: object) -> str:
    try:
        return safe_relative(value, _RESERVED_NAMES)
    except ValueError:
        raise _fail() from None


def _load_strict_json(raw: bytes) -> object:
    try:
        return load_strict_json(raw)
    except (ValueError, UnicodeDecodeError):
        raise _fail() from None


def _resolved_bundle_root(path: str | Path) -> Path:
    try:
        return resolved_artifact_root(path)
    except (OSError, ValueError):
        raise _fail() from None


def _tree_files(root: Path) -> set[str]:
    try:
        return tree_files(root)
    except (OSError, ValueError):
        raise _fail() from None


def _digest_and_size(path: Path) -> tuple[str, int]:
    try:
        return digest_and_size(path)
    except (OSError, ValueError):
        raise _fail() from None


def _parse_checksums(raw: bytes) -> dict[str, str]:
    try:
        return parse_checksums(raw, manifest_name=MANIFEST_NAME, reserved=_RESERVED_NAMES)
    except (ValueError, UnicodeDecodeError):
        raise _fail() from None


def _manifest_payloads(manifest: dict[str, object]) -> list[dict[str, object]]:
    payloads = manifest["payloads"]
    if not isinstance(payloads, list) or not payloads:
        raise _fail()
    records: list[dict[str, object]] = []
    for payload in payloads:
        if not isinstance(payload, dict) or set(payload) != _PAYLOAD_KEYS:
            raise _fail()
        path = _safe_relative(payload["path"])
        if not is_sha256(payload["sha256"]) or not is_positive_int(payload["bytes"]):
            raise _fail()
        records.append({"path": path, "sha256": payload["sha256"], "bytes": payload["bytes"]})
    paths = [str(record["path"]) for record in records]
    if paths != sorted(paths) or len(set(paths)) != len(paths):
        raise _fail()
    if canonical_sha256(records) != manifest["bundle_tree_sha256"]:
        raise _fail()
    return records


def _check_manifest_identity(manifest: dict[str, object]) -> None:
    if set(manifest) != _REQUIRED_MANIFEST_KEYS:
        raise _fail()
    if (
        manifest["schema_version"] != RELEASE_SCHEMA_VERSION
        or manifest["release_id"] != PROMOTED_RELEASE_ID
        or manifest["model_id"] != WAV2VEC_MODEL_ID
        or manifest["model_revision"] != WAV2VEC_MODEL_REVISION
        or manifest["base_model_sha256"] != WAV2VEC_WEIGHTS_SHA256
        or manifest["labels"] != list(CANONICAL_LABELS)
        or manifest["final_sealed"] is not True
        or manifest["decision"] != RELEASE_DECISION
        or not is_positive_int(manifest["final_holdout_count"])
        or not is_sha256(manifest["candidate_checkpoint_sha256"])
        or not is_sha256(manifest["bundle_tree_sha256"])
        or not isinstance(manifest["metrics"], dict)
        or not isinstance(manifest["provenance"], dict)
    ):
        raise _fail()


def _check_checkpoint_metadata(checkpoint: Path) -> None:
    """Bind the checkpoint's own head declaration to the fixed seven-class order."""

    mapping = _load_strict_json(read_trusted_regular_file(checkpoint / "label_mapping.json"))
    if mapping != {str(index): label for index, label in enumerate(CANONICAL_LABELS)}:
        raise _fail()
    config = _load_strict_json(read_trusted_regular_file(checkpoint / "config.json"))
    if (
        not isinstance(config, dict)
        or config.get("architecture") != ARCHITECTURE
        or config.get("model_id") != WAV2VEC_MODEL_ID
        or config.get("model_revision") != WAV2VEC_MODEL_REVISION
        or config.get("base_model_sha256") != WAV2VEC_WEIGHTS_SHA256
        or config.get("labels") != list(CANONICAL_LABELS)
    ):
        raise _fail()


def verify_release_bundle(path: str | Path) -> VerifiedRelease:
    """Verify an immutable release bundle and return its bound local model directories.

    Raises ``ReleaseBundleError`` for every rejection, with no host path in the message.
    """

    try:
        root = _resolved_bundle_root(path)
        checksums = _parse_checksums(read_trusted_regular_file(root / CHECKSUMS_NAME))
        manifest_raw = read_trusted_regular_file(root / MANIFEST_NAME)
        if hashlib.sha256(manifest_raw).hexdigest() != checksums[MANIFEST_NAME]:
            raise _fail()

        manifest = _load_strict_json(manifest_raw)
        if not isinstance(manifest, dict):
            raise _fail()
        _check_manifest_identity(manifest)
        payloads = _manifest_payloads(manifest)
        attested = {str(record["path"]): record for record in payloads}

        if set(checksums) != set(attested) | {MANIFEST_NAME}:
            raise _fail()
        if _tree_files(root) != set(attested) | _RESERVED_NAMES:
            raise _fail()

        required = {f"{CHECKPOINT_DIRECTORY}/{name}" for name in CHECKPOINT_FILES}
        required |= {f"{BASE_MODEL_DIRECTORY}/{name}" for name in BASE_MODEL_FILES}
        if not required <= set(attested):
            raise _fail()
        if attested[BASE_MODEL_WEIGHTS]["sha256"] != manifest["base_model_sha256"]:
            raise _fail()

        for relative, record in sorted(attested.items()):
            digest, size = _digest_and_size(root / relative)
            if (
                digest != record["sha256"]
                or digest != checksums[relative]
                or size != record["bytes"]
            ):
                raise _fail()

        checkpoint = root / CHECKPOINT_DIRECTORY
        _check_checkpoint_metadata(checkpoint)
        if checkpoint_tree_digest(checkpoint) != manifest["candidate_checkpoint_sha256"]:
            raise _fail()

        return VerifiedRelease(
            path=root,
            release_id=PROMOTED_RELEASE_ID,
            checkpoint_path=checkpoint,
            base_model_path=root / BASE_MODEL_DIRECTORY,
            labels=tuple(CANONICAL_LABELS),
            candidate_checkpoint_sha256=str(manifest["candidate_checkpoint_sha256"]),
            bundle_tree_sha256=str(manifest["bundle_tree_sha256"]),
        )
    except ReleaseBundleError:
        raise
    except Exception:
        raise _fail() from None


__all__ = [
    "PROMOTED_RELEASE_ID",
    "ReleaseBundleError",
    "VerifiedRelease",
    "verify_release_bundle",
]
