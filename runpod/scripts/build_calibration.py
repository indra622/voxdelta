"""Fit a post-hoc calibration on the validation split and publish it beside a release.

Runs entirely offline on local hardware. The promoted release is re-verified through the
production verifier, the shipped training package's own manifest supplies the validation
identity, and inference runs only over items whose split is `validation`. The sealed
final holdout is not an input to this script and is not reachable from the package it
reads.

Per-item distributions are held in a caller-supplied cache so an interrupted pass
resumes; the cache is an input to the fit, never part of the published artifact, and the
artifact retains no per-item score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from voxdelta.evaluation.calibration import CalibrationError, fit_temperature_scaling
from voxdelta.evaluation.emotion_training import CANONICAL_LABELS
from voxdelta.evaluation.manifest import DatasetItem, load_manifest, read_trusted_regular_file
from voxdelta.providers._emotion_runtime import Device
from voxdelta.providers.release_bundle import verify_release_bundle

from voxdelta_runpod.calibration_release import (
    CalibrationReleaseError,
    CalibrationValidationSource,
    build_calibration_artifact,
    validation_items_digest,
)
from voxdelta_runpod.package import PackageSidecar, SanitizedRecord
from voxdelta_runpod.workflow import load_packaged_manifest

PROGRESS_INTERVAL = 100
OFFLINE_ENVIRONMENT: tuple[tuple[str, str], ...] = (
    ("HF_HUB_OFFLINE", "1"),
    ("TRANSFORMERS_OFFLINE", "1"),
    ("HF_DATASETS_OFFLINE", "1"),
    ("HF_HUB_DISABLE_TELEMETRY", "1"),
)


def apply_offline_environment() -> None:
    for name, value in OFFLINE_ENVIRONMENT:
        os.environ[name] = value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--packaged-manifest", required=True, type=Path)
    parser.add_argument("--sidecar", required=True, type=Path)
    parser.add_argument("--evaluation-manifest", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration-id", required=True)
    parser.add_argument("--target-coverage", required=True, type=float)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    return parser


def _validation_items(evaluation_manifest: Path) -> list[DatasetItem]:
    items = load_manifest(evaluation_manifest)
    if not items or any(item.split != "validation" or item.emotion is None for item in items):
        raise CalibrationReleaseError("evaluation_manifest_is_not_validation_only")
    return items


def _validate_validation_identity(
    items: Sequence[DatasetItem],
    records: Sequence[SanitizedRecord],
) -> None:
    """Bind the host evaluation paths to the exact packaged validation identities."""

    ordered_records = sorted(records, key=lambda record: record.item_key)
    if len(items) != len(ordered_records):
        raise CalibrationReleaseError("validation_identity_mismatch")
    for item, record in zip(items, ordered_records, strict=True):
        if (
            item.split != "validation"
            or record.split != "validation"
            or item.id != record.item_key
            or item.emotion != record.emotion
            or item.sha256 != record.audio_sha256
            or Path(item.audio_path).name != f"{record.item_key}.wav"
        ):
            raise CalibrationReleaseError("validation_identity_mismatch")
        try:
            digest = hashlib.sha256(read_trusted_regular_file(item.audio_path)).hexdigest()
        except Exception:
            raise CalibrationReleaseError("validation_audio_mismatch") from None
        if digest != item.sha256:
            raise CalibrationReleaseError("validation_audio_mismatch")


def _load_cache(cache: Path, items: Sequence[DatasetItem]) -> list[list[float]]:
    """Read the resumable per-item cache, rejecting any row that is not in step."""

    if not cache.exists():
        return []
    try:
        if cache.stat().st_mode & 0o077:
            raise CalibrationReleaseError("calibration_cache_not_private")
    except CalibrationReleaseError:
        raise
    except OSError:
        raise CalibrationReleaseError("calibration_cache_out_of_step") from None
    rows: list[list[float]] = []
    for index, line in enumerate(read_trusted_regular_file(cache).decode().splitlines()):
        if not line.strip():
            continue
        try:
            record = json.loads(
                line,
                parse_constant=lambda _value: (_ for _ in ()).throw(ValueError),
            )
        except (json.JSONDecodeError, ValueError):
            raise CalibrationReleaseError("calibration_cache_out_of_step") from None
        if (
            not isinstance(record, dict)
            or set(record) != {"i", "item_id", "audio_sha256", "expected", "p"}
            or index >= len(items)
            or type(record["i"]) is not int
            or record["i"] != index
            or record["item_id"] != items[index].id
            or record["audio_sha256"] != items[index].sha256
            or record["expected"] != items[index].emotion
            or not isinstance(record["p"], list)
            or len(record["p"]) != len(CANONICAL_LABELS)
            or any(type(value) not in (int, float) for value in record["p"])
        ):
            raise CalibrationReleaseError("calibration_cache_out_of_step")
        values = [float(value) for value in record["p"]]
        if (
            any(not math.isfinite(value) or not 0 <= value <= 1 for value in values)
            or abs(math.fsum(values) - 1.0) > 1e-6
        ):
            raise CalibrationReleaseError("calibration_cache_out_of_step")
        rows.append(values)
    return rows


def _complete_cache(
    cache: Path,
    items: Sequence[DatasetItem],
    rows: list[list[float]],
    *,
    release_path: Path,
    checkpoint: Path,
    base_model: Path,
    device: Device,
) -> list[list[float]]:
    """Run inference for whatever the cache is still missing, appending as it goes."""

    if len(rows) >= len(items):
        return rows
    from voxdelta.providers.wav2vec_emotion import Wav2VecEmotionProvider

    del release_path
    provider = Wav2VecEmotionProvider(checkpoint, base_model_path=base_model, device=device)
    descriptor = os.open(cache, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        with os.fdopen(descriptor, "w") as writer:
            for index in range(len(rows), len(items)):
                item = items[index]
                outcome = provider.analyze(item.id, Path(item.audio_path), "")
                values = [outcome.probabilities[label] for label in CANONICAL_LABELS]
                writer.write(
                    json.dumps(
                        {
                            "i": index,
                            "item_id": item.id,
                            "audio_sha256": item.sha256,
                            "expected": item.emotion,
                            "p": values,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
                rows.append(values)
                if (index + 1) % PROGRESS_INTERVAL == 0 or index + 1 == len(items):
                    writer.flush()
                    os.fsync(writer.fileno())
                    print(f"inference_progress {index + 1}/{len(items)}", flush=True)
    finally:
        provider.unload()
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    apply_offline_environment()
    try:
        release = verify_release_bundle(arguments.release.absolute())
        sidecar = PackageSidecar.model_validate_json(
            read_trusted_regular_file(arguments.sidecar.absolute())
        )
        packaged = load_packaged_manifest(arguments.packaged_manifest.absolute())
        validation_records = tuple(record for record in packaged if record.split == "validation")
        if any(record.split == "test" for record in packaged):
            raise CalibrationReleaseError("packaged_manifest_contains_test_split")
        if len(validation_records) != sidecar.split_counts.get("validation"):
            raise CalibrationReleaseError("validation_count_mismatch")

        items = _validation_items(arguments.evaluation_manifest.absolute())
        _validate_validation_identity(items, validation_records)

        cache = arguments.cache.absolute()
        rows = _complete_cache(
            cache,
            items,
            _load_cache(cache, items),
            release_path=release.path,
            checkpoint=release.checkpoint_path,
            base_model=release.base_model_path,
            device=cast(Device, arguments.device),
        )
        fit = fit_temperature_scaling(
            [dict(zip(CANONICAL_LABELS, row, strict=True)) for row in rows],
            [item.emotion for item in items],  # type: ignore[misc]
            target_coverage=arguments.target_coverage,
        )
        source = CalibrationValidationSource(
            archive_sha256=sidecar.archive_sha256,
            sidecar_sha256=hashlib.sha256(
                read_trusted_regular_file(arguments.sidecar.absolute())
            ).hexdigest(),
            packaged_manifest_sha256=sidecar.manifest_sha256,
            validation_items_sha256=validation_items_digest(list(validation_records)),
            validation_item_count=len(validation_records),
        )
        output = build_calibration_artifact(
            release_dir=release.path,
            output_dir=arguments.output.absolute(),
            calibration_id=arguments.calibration_id,
            fit=fit,
            validation=source,
        )
    except (CalibrationError, CalibrationReleaseError, OSError, ValueError) as error:
        print(f"calibration_build_failed {getattr(error, 'code', type(error).__name__)}")
        return 2
    print("calibration_built")
    print(f"calibration_path {output}")
    print(f"temperature {fit.temperature!r}")
    print(f"abstain_threshold {fit.abstain_threshold!r}")
    print(f"achieved_coverage {fit.achieved_coverage!r}")
    print(f"accuracy_at_coverage {fit.accuracy_at_coverage!r}")
    print(f"pre_temperature_ece {fit.pre_temperature_ece!r}")
    print(f"post_temperature_ece {fit.post_temperature_ece!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
