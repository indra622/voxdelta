"""Publish a post-hoc calibration as its own immutable, release-bound artifact.

The promoted release is frozen: its candidate selection, its sealed final-holdout
comparison, and its byte-for-byte contents are the record of a decision that was made
once and cannot be remade. A temperature and abstain threshold fitted afterwards is a
*different* claim about the same frozen model, so it is published beside the release
rather than inside it. Nothing here reads, rewrites, or re-derives the frozen candidate
provenance, and building a calibration never requires a new candidate freeze.

The artifact binds, in one digest, the release identity it calibrates, the validation
split it was fitted on, and the fit itself. A consumer that has verified the release can
therefore prove the calibration belongs to it; a calibration fitted on a different model
or a different validation package cannot be substituted unnoticed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from voxdelta.evaluation.calibration import CalibrationSummary
from voxdelta.evaluation.manifest import read_trusted_regular_file
from voxdelta.evaluation.wav2vec_base import (
    WAV2VEC_MODEL_REVISION,
    WAV2VEC_WEIGHTS_SHA256,
)
from voxdelta.providers._attested_tree import canonical_sha256
from voxdelta.providers.calibration_artifact import (
    CalibrationArtifactError,
    VerifiedCalibration,
    binding_digest,
    verify_calibration_artifact,
)
from voxdelta.providers.release_bundle import verify_release_bundle

from voxdelta_runpod.config import CANONICAL_LABELS

MANIFEST_NAME = "CALIBRATION.json"
CHECKSUMS_NAME = "SHA256SUMS"
CARD_NAME = "CALIBRATION_CARD.md"
_RESERVED_NAMES = frozenset({MANIFEST_NAME, CHECKSUMS_NAME})
_SHA256_HEX = r"^[0-9a-f]{64}$"


class CalibrationReleaseError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class CalibrationPayload(_FrozenModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256_HEX)
    bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def safe_relative_path(self) -> CalibrationPayload:
        parts = self.path.split("/")
        if (
            self.path.startswith("/")
            or self.path != Path(self.path).as_posix()
            or "" in parts
            or "." in parts
            or ".." in parts
            or self.path in _RESERVED_NAMES
        ):
            raise ValueError("invalid calibration payload path")
        return self


class CalibrationReleaseBinding(_FrozenModel):
    """The exact release this calibration is valid for, and for no other."""

    release_id: str = Field(min_length=1)
    release_manifest_sha256: str = Field(pattern=_SHA256_HEX)
    bundle_tree_sha256: str = Field(pattern=_SHA256_HEX)
    candidate_checkpoint_sha256: str = Field(pattern=_SHA256_HEX)
    base_model_sha256: Literal[
        "d5e490574712ad0a6736923b9ed11d4cd51c78609c36205f704fc4e87b11d2e0"
    ] = WAV2VEC_WEIGHTS_SHA256
    model_id: Literal["facebook/wav2vec2-xls-r-300m"] = "facebook/wav2vec2-xls-r-300m"
    model_revision: Literal["1a640f32ac3e39899438a2931f9924c02f080a54"] = WAV2VEC_MODEL_REVISION
    labels: tuple[str, ...]

    @model_validator(mode="after")
    def fixed_label_order(self) -> CalibrationReleaseBinding:
        if tuple(self.labels) != CANONICAL_LABELS:
            raise ValueError("invalid calibration label order")
        return self


class CalibrationValidationSource(_FrozenModel):
    """The development split the fit consumed; the sealed holdout is not reachable here."""

    package_kind: Literal["train-validation"] = "train-validation"
    split: Literal["validation"] = "validation"
    archive_sha256: str = Field(pattern=_SHA256_HEX)
    sidecar_sha256: str = Field(pattern=_SHA256_HEX)
    packaged_manifest_sha256: str = Field(pattern=_SHA256_HEX)
    validation_items_sha256: str = Field(pattern=_SHA256_HEX)
    validation_item_count: int = Field(gt=0)
    holdout_used: Literal[False] = False


class CalibrationManifest(_FrozenModel):
    schema_version: Literal["1"] = "1"
    artifact_kind: Literal["post-hoc-calibration"] = "post-hoc-calibration"
    calibration_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,95}$")
    release: CalibrationReleaseBinding
    validation: CalibrationValidationSource
    fit: CalibrationSummary
    binding_sha256: str = Field(pattern=_SHA256_HEX)
    payloads: tuple[CalibrationPayload, ...]
    artifact_tree_sha256: str = Field(pattern=_SHA256_HEX)

    @model_validator(mode="after")
    def consistent_manifest(self) -> CalibrationManifest:
        paths = [payload.path for payload in self.payloads]
        if (
            not self.payloads
            or paths != sorted(paths)
            or len(set(paths)) != len(paths)
            or _tree_digest(self.payloads) != self.artifact_tree_sha256
            or self.fit.fitted_item_count != self.validation.validation_item_count
            or binding_digest(
                self.release.model_dump(mode="json"),
                self.validation.model_dump(mode="json"),
                self.fit.model_dump(mode="json"),
            )
            != self.binding_sha256
        ):
            raise ValueError("invalid calibration manifest")
        return self


def _tree_digest(payloads: tuple[CalibrationPayload, ...]) -> str:
    return canonical_sha256(
        [
            {"path": payload.path, "sha256": payload.sha256, "bytes": payload.bytes}
            for payload in sorted(payloads, key=lambda payload: payload.path)
        ]
    )


def validation_items_digest(records: object) -> str:
    """A host-independent identity for exactly which validation items were fitted on.

    Derived from the shipped package's own opaque item keys and reference labels, so it
    is reproducible from the package alone and reveals nothing the package sidecar's
    manifest digest does not already commit to.
    """

    if not isinstance(records, list | tuple) or not records:
        raise CalibrationReleaseError("invalid_validation_records")
    rows = []
    for record in records:
        key = getattr(record, "item_key", None)
        emotion = getattr(record, "emotion", None)
        split = getattr(record, "split", None)
        if not isinstance(key, str) or not isinstance(emotion, str) or split != "validation":
            raise CalibrationReleaseError("invalid_validation_records")
        rows.append({"item_key": key, "emotion": emotion})
    rows.sort(key=lambda row: row["item_key"])
    if len({row["item_key"] for row in rows}) != len(rows):
        raise CalibrationReleaseError("invalid_validation_records")
    return canonical_sha256(rows)


def _write_private(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(payload)


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    )


def _payloads(root: Path) -> tuple[CalibrationPayload, ...]:
    entries = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and path.relative_to(root).as_posix() not in _RESERVED_NAMES
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    return tuple(
        CalibrationPayload(
            path=path.relative_to(root).as_posix(),
            sha256=hashlib.sha256(data).hexdigest(),
            bytes=len(data),
        )
        for path, data in ((path, read_trusted_regular_file(path)) for path in entries)
    )


def _write_checksums(root: Path) -> None:
    lines = sorted(
        f"{hashlib.sha256(read_trusted_regular_file(path)).hexdigest()}  "
        f"{path.relative_to(root).as_posix()}"
        for path in root.rglob("*")
        if path.is_file() and path.name != CHECKSUMS_NAME
    )
    _write_private(root / CHECKSUMS_NAME, ("\n".join(lines) + "\n").encode())


def build_calibration_artifact(
    *,
    release_dir: Path,
    output_dir: Path,
    calibration_id: str,
    fit: CalibrationSummary,
    validation: CalibrationValidationSource,
) -> Path:
    """Assemble one immutable calibration artifact beside an already-frozen release.

    The release is re-verified but never opened for writing, and the output path must not
    already exist: a calibration is published, never amended in place.
    """

    output = Path(output_dir)
    if not output.is_absolute() or ".." in output.parts:
        raise CalibrationReleaseError("invalid_calibration_target")
    staging = output.with_name(f".{output.name}.staging")
    if output.exists() or output.is_symlink() or staging.exists() or staging.is_symlink():
        raise CalibrationReleaseError("calibration_exists")

    try:
        release = verify_release_bundle(release_dir)
    except Exception:
        raise CalibrationReleaseError("invalid_release_bundle") from None
    if fit.fitted_item_count != validation.validation_item_count:
        raise CalibrationReleaseError("calibration_item_count_mismatch")

    binding = CalibrationReleaseBinding(
        release_id=release.release_id,
        release_manifest_sha256=hashlib.sha256(
            read_trusted_regular_file(release.path / "RELEASE.json")
        ).hexdigest(),
        bundle_tree_sha256=release.bundle_tree_sha256,
        candidate_checkpoint_sha256=release.candidate_checkpoint_sha256,
        labels=CANONICAL_LABELS,
    )

    try:
        staging.mkdir(mode=0o700, parents=True)
        _write_private(
            staging / CARD_NAME,
            _render_calibration_card(
                calibration_id=calibration_id, binding=binding, validation=validation, fit=fit
            ).encode(),
        )
        payloads = _payloads(staging)
        manifest = CalibrationManifest(
            calibration_id=calibration_id,
            release=binding,
            validation=validation,
            fit=fit,
            binding_sha256=binding_digest(
                binding.model_dump(mode="json"),
                validation.model_dump(mode="json"),
                fit.model_dump(mode="json"),
            ),
            payloads=payloads,
            artifact_tree_sha256=_tree_digest(payloads),
        )
        _write_private(staging / MANIFEST_NAME, _canonical_bytes(manifest.model_dump(mode="json")))
        _write_checksums(staging)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def verify_calibration_release(
    calibration_dir: Path,
    release_dir: Path,
) -> VerifiedCalibration:
    """Re-verify a published calibration through the same gate the runtime uses."""

    try:
        release = verify_release_bundle(release_dir)
    except Exception:
        raise CalibrationReleaseError("invalid_release_bundle") from None
    try:
        return verify_calibration_artifact(calibration_dir, release=release)
    except CalibrationArtifactError:
        raise CalibrationReleaseError("invalid_calibration_artifact") from None


def _render_calibration_card(
    *,
    calibration_id: str,
    binding: CalibrationReleaseBinding,
    validation: CalibrationValidationSource,
    fit: CalibrationSummary,
) -> str:
    return f"""# VoxDelta XLS-R Emotion Calibration — `{calibration_id}`

Post-hoc calibration and abstention for the immutable release
`{binding.release_id}`. This artifact does not contain, replace, or modify that release;
it is published beside it and names it cryptographically.

## What this calibrates

- Release: `{binding.release_id}`
- Release manifest SHA-256: `{binding.release_manifest_sha256}`
- Release bundle tree SHA-256: `{binding.bundle_tree_sha256}`
- Fine-tuned checkpoint identity: `{binding.candidate_checkpoint_sha256}`
- Base model: `{binding.model_id}` @ `{binding.model_revision}`
- Fixed 7-class order: {", ".join(f"{index}={label}" for index, label in enumerate(binding.labels))}

A consumer that has verified the release can check every value above against it. A
calibration fitted for a different checkpoint cannot be substituted unnoticed.

## What it was fitted on

- Split: `{validation.split}` only — {validation.validation_item_count} items.
- The sealed final holdout was **not** opened: `holdout_used = {validation.holdout_used}`.
  The package this fit consumed contains only train and validation members, so holdout
  audio is not reachable from the fitting path at all.
- Training package archive SHA-256: `{validation.archive_sha256}`
- Packaged manifest SHA-256: `{validation.packaged_manifest_sha256}`
- Validation item-set SHA-256: `{validation.validation_items_sha256}`

## The fit

- Method: `{fit.method}`, schema `{fit.schema_version}`, {fit.ece_bin_count}-bin ECE.
- Temperature: `{fit.temperature:.17g}`
- Abstain threshold on the calibrated top-class probability: `{fit.abstain_threshold:.17g}`
- Target coverage: {fit.target_coverage:.4f} — achieved {fit.achieved_coverage:.6f}
- Accuracy on answered items: {fit.accuracy_at_coverage:.6f}
- Validation ECE before scaling: {fit.pre_temperature_ece:.6f}
- Validation ECE after scaling: {fit.post_temperature_ece:.6f}

## How to use it

Rescale before reading any probability as confidence:
`p_calibrated = softmax(log(p_raw) / {fit.temperature:.17g})`, then abstain when
`max(p_calibrated) < {fit.abstain_threshold:.17g}`. `apply_temperature` in
`voxdelta.evaluation.calibration` implements exactly this, and
`voxdelta.providers.calibrated_emotion` applies it in the production provider path.

An abstained result reports the pre-existing `uncertain` operational state, so consumers
that already handle low-confidence emotion need no change.

## Limits

- The residual ECE of {fit.post_temperature_ece:.6f} is what a single temperature could
  not remove. Treat it as the floor on how far these probabilities can be trusted.
- Coverage and accuracy above describe the development split. They are an estimate for
  new audio, not a guarantee, and they degrade under domain shift.
- The sealed final-holdout metrics in the release describe the **raw** scores. They are
  not corrected by this temperature, and the holdout is consumed and cannot be rerun.
"""


__all__ = [
    "CARD_NAME",
    "CHECKSUMS_NAME",
    "MANIFEST_NAME",
    "CalibrationManifest",
    "CalibrationPayload",
    "CalibrationReleaseBinding",
    "CalibrationReleaseError",
    "CalibrationValidationSource",
    "build_calibration_artifact",
    "validation_items_digest",
    "verify_calibration_release",
]
