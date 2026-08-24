"""Verify remote GPU, disk, base model, transfer digests, and safe extraction."""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Sequence
from pathlib import Path

from voxdelta.evaluation.wav2vec_base import validate_wav2vec_base

from voxdelta_runpod.config import load_experiment_config
from voxdelta_runpod.package import extract_verified_package
from voxdelta_runpod.training import validate_cuda_runtime


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--sidecar", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--base-model", type=Path, default=Path("/workspace/models/xls-r-300m"))
    arguments = parser.parse_args(argv)
    try:
        import torch

        config = load_experiment_config(arguments.config.resolve())
        validate_cuda_runtime(torch)
        gpu_name = str(torch.cuda.get_device_name(0))
        if config.runtime.gpu_model not in gpu_name:
            raise ValueError
        free = shutil.disk_usage(arguments.data_root.resolve().parent).free
        if free < config.runtime.minimum_free_gb * 1024**3:
            raise ValueError
        prepared = validate_wav2vec_base(arguments.base_model.resolve())
        if prepared.weights_sha256 != config.model.weights_sha256:
            raise ValueError
        extract_verified_package(
            arguments.archive.resolve(),
            arguments.sidecar.resolve(),
            arguments.data_root.resolve(),
        )
    except Exception:
        print("remote_preflight_error")
        return 2
    print("remote_preflight_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
