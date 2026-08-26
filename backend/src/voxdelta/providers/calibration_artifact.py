"""Verify a post-hoc calibration artifact and bind it to one already-verified release.

The promoted release is immutable and carries no calibration of its own. A temperature
and abstain threshold are fitted afterwards, on the development split alone, and are
published as a separately versioned artifact that names the release it belongs to. This
module is the runtime gate for that artifact.

A calibration is trusted only when its manifest, its checksum file, and the bytes on disk
agree with one another; when the release identity it claims is exactly the release the
caller already verified; and when the fit it carries hashes to the binding digest it
declares. Verification is offline, follows no symlink, accepts no unattested file, and
fails closed with one fixed, path-free error, so a rejected artifact never discloses a
host path and there is never a fallback to uncalibrated scoring.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from voxdelta.evaluation.calibration import CALIBRATION_SCHEMA_VERSION, CalibrationSummary
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
from voxdelta.providers.release_bundle import VerifiedRelease

ARTIFACT_KIND = "post-hoc-calibration"
ARTIFACT_SCHEMA_VERSION = "1"
MANIFEST_NAME = "CALIBRATION.json"
CHECKSUMS_NAME = "SHA256SUMS"
CALIBRATION_METHOD = "temperature-scaling"
FITTED_SPLIT = "validation"
VALIDATION_PACKAGE_KIND = "train-validation"

_REQUIRED_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "calibration_id",
        "release",
        "validation",
        "fit",
        "binding_sha256",
        "payloads",
        "artifact_tree_sha256",
    }
)
_RELEASE_KEYS = frozenset(
    {
        "release_id",
        "release_manifest_sha256",
        "bundle_tree_sha256",
        "candidate_checkpoint_sha256",
        "base_model_sha256",
        "model_id",
        "model_revision",
        "labels",
    }
)
_VALIDATION_KEYS = frozenset(
    {
        "package_kind",
        "split",
        "archive_sha256",
        "sidecar_sha256",
        "packaged_manifest_sha256",
        "validation_items_sha256",
        "validation_item_count",
        "holdout_used",
    }
)
_PAYLOAD_KEYS = frozenset({"path", "sha256", "bytes"})
_RESERVED_NAMES = frozenset({MANIFEST_NAME, CHECKSUMS_NAME})
_CALIBRATION_ID_CHARACTERS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-")


class CalibrationArtifactError(ValueError):
    """Fixed, path-free rejection of an untrusted or mismatched calibration artifact."""

    def __init__(self) -> None:
        super().__init__("invalid_calibration_artifact")


@dataclass(frozen=True, slots=True)
class VerifiedCalibration:
    """A fit that has been proven to belong to one specific release and validation split."""

    path: Path
    calibration_id: str
    binding_sha256: str
    release_id: str
    bundle_tree_sha256: str
    candidate_checkpoint_sha256: str
    validation_items_sha256: str
    validation_item_count: int
    summary: CalibrationSummary

    @property
    def temperature(self) -> float:
        return self.summary.temperature

    @property
    def abstain_threshold(self) -> float:
        return self.summary.abstain_threshold


def _fail() -> CalibrationArtifactError:
    return CalibrationArtifactError()


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


def _resolved_root(path: str | Path) -> Path:
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


def binding_digest(
    release: dict[str, object],
    validation: dict[str, object],
    fit: dict[str, object],
) -> str:
    """The single digest that ties one fit to one release and one validation split."""

    return canonical_sha256({"release": release, "validation": validation, "fit": fit})


def _check_calibration_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not 3 <= len(value) <= 96
        or set(value) - _CALIBRATION_ID_CHARACTERS
        or value.startswith("-")
        or value.endswith("-")
    ):
        raise _fail()
    return value


def _check_release_binding(payload: object, release: VerifiedRelease) -> dict[str, object]:
    """The artifact must name the exact release the caller already verified."""

    if not isinstance(payload, dict) or set(payload) != _RELEASE_KEYS:
        raise _fail()
    if (
        payload["release_id"] != release.release_id
        or payload["bundle_tree_sha256"] != release.bundle_tree_sha256
        or payload["candidate_checkpoint_sha256"] != release.candidate_checkpoint_sha256
        or payload["base_model_sha256"] != WAV2VEC_WEIGHTS_SHA256
        or payload["model_id"] != WAV2VEC_MODEL_ID
        or payload["model_revision"] != WAV2VEC_MODEL_REVISION
        or payload["labels"] != list(CANONICAL_LABELS)
        or not is_sha256(payload["release_manifest_sha256"])
    ):
        raise _fail()
    manifest_digest, _ = _digest_and_size(release.path / "RELEASE.json")
    if manifest_digest != payload["release_manifest_sha256"]:
        raise _fail()
    return payload


def _check_validation_source(payload: object) -> dict[str, object]:
    """The fit must declare a development-split source that never opened the holdout."""

    if not isinstance(payload, dict) or set(payload) != _VALIDATION_KEYS:
        raise _fail()
    if (
        payload["package_kind"] != VALIDATION_PACKAGE_KIND
        or payload["split"] != FITTED_SPLIT
        or payload["holdout_used"] is not False
        or not is_positive_int(payload["validation_item_count"])
        or not is_sha256(payload["archive_sha256"])
        or not is_sha256(payload["sidecar_sha256"])
        or not is_sha256(payload["packaged_manifest_sha256"])
        or not is_sha256(payload["validation_items_sha256"])
    ):
        raise _fail()
    return payload


def _check_fit(payload: object, validation: dict[str, object]) -> CalibrationSummary:
    if not isinstance(payload, dict):
        raise _fail()
    try:
        summary = CalibrationSummary.model_validate(payload)
    except ValueError:
        raise _fail() from None
    if (
        summary.schema_version != CALIBRATION_SCHEMA_VERSION
        or summary.method != CALIBRATION_METHOD
        or summary.fitted_item_count != validation["validation_item_count"]
    ):
        raise _fail()
    return summary


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
    if canonical_sha256(records) != manifest["artifact_tree_sha256"]:
        raise _fail()
    return records


def verify_calibration_artifact(
    path: str | Path,
    *,
    release: VerifiedRelease,
) -> VerifiedCalibration:
    """Verify a calibration artifact against the release it claims to calibrate.

    The release must already have been verified: a calibration is meaningless on its own,
    and accepting one without its model would be exactly the mismatch this guards against.
    Raises ``CalibrationArtifactError`` for every rejection, with no host path in it.
    """

    try:
        root = _resolved_root(path)
        checksums = _parse_checksums(read_trusted_regular_file(root / CHECKSUMS_NAME))
        manifest_raw = read_trusted_regular_file(root / MANIFEST_NAME)
        if hashlib.sha256(manifest_raw).hexdigest() != checksums[MANIFEST_NAME]:
            raise _fail()

        manifest = _load_strict_json(manifest_raw)
        if not isinstance(manifest, dict) or set(manifest) != _REQUIRED_MANIFEST_KEYS:
            raise _fail()
        if (
            manifest["schema_version"] != ARTIFACT_SCHEMA_VERSION
            or manifest["artifact_kind"] != ARTIFACT_KIND
            or not is_sha256(manifest["binding_sha256"])
            or not is_sha256(manifest["artifact_tree_sha256"])
        ):
            raise _fail()
        calibration_id = _check_calibration_id(manifest["calibration_id"])

        bound_release = _check_release_binding(manifest["release"], release)
        validation = _check_validation_source(manifest["validation"])
        summary = _check_fit(manifest["fit"], validation)
        if (
            binding_digest(bound_release, validation, summary.model_dump(mode="json"))
            != (manifest["binding_sha256"])
        ):
            raise _fail()

        payloads = _manifest_payloads(manifest)
        attested = {str(record["path"]): record for record in payloads}
        if set(checksums) != set(attested) | {MANIFEST_NAME}:
            raise _fail()
        if _tree_files(root) != set(attested) | _RESERVED_NAMES:
            raise _fail()
        for relative, record in sorted(attested.items()):
            digest, size = _digest_and_size(root / relative)
            if (
                digest != record["sha256"]
                or digest != checksums[relative]
                or size != record["bytes"]
            ):
                raise _fail()

        return VerifiedCalibration(
            path=root,
            calibration_id=calibration_id,
            binding_sha256=str(manifest["binding_sha256"]),
            release_id=str(bound_release["release_id"]),
            bundle_tree_sha256=str(bound_release["bundle_tree_sha256"]),
            candidate_checkpoint_sha256=str(bound_release["candidate_checkpoint_sha256"]),
            validation_items_sha256=str(validation["validation_items_sha256"]),
            validation_item_count=int(str(validation["validation_item_count"])),
            summary=summary,
        )
    except CalibrationArtifactError:
        raise
    except Exception:
        raise _fail() from None


__all__ = [
    "ARTIFACT_KIND",
    "ARTIFACT_SCHEMA_VERSION",
    "MANIFEST_NAME",
    "CalibrationArtifactError",
    "VerifiedCalibration",
    "binding_digest",
    "verify_calibration_artifact",
]
