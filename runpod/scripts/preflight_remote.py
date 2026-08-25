"""Verify remote GPU, disk, base model, transfer digests, and safe extraction."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from voxdelta.evaluation.manifest import read_trusted_regular_file
from voxdelta.evaluation.wav2vec_base import validate_wav2vec_base

from voxdelta_runpod.config import load_experiment_config
from voxdelta_runpod.ledger import append_transition, initialize_ledger, load_ledger
from voxdelta_runpod.package import PackageSidecar, extract_verified_package
from voxdelta_runpod.recipes import build_pilot_plans_from_package
from voxdelta_runpod.training import validate_cuda_runtime
from voxdelta_runpod.workflow import (
    RuntimeEnvironment,
    build_run_identity,
    canonical_sha256,
    digest_file,
    load_run_identity,
    load_runtime_environment,
    publish_model,
)


def _driver_version() -> str:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    versions = {line.strip() for line in completed.stdout.splitlines() if line.strip()}
    if len(versions) != 1:
        raise ValueError
    return versions.pop()


def _runtime_environment(
    torch: Any, remote_root: Path, identity_sha256: str
) -> RuntimeEnvironment:
    cuda = torch.cuda
    cudnn = torch.backends.cudnn.version()
    cuda_version = torch.version.cuda
    if not isinstance(cudnn, int) or not isinstance(cuda_version, str):
        raise ValueError
    disk = shutil.disk_usage(remote_root)
    properties = cuda.get_device_properties(0)
    return RuntimeEnvironment(
        python_version=platform.python_version(),
        torch_version=str(torch.__version__),
        transformers_version=importlib.metadata.version("transformers"),
        cuda_version=cuda_version,
        cudnn_version=cudnn,
        driver_version=_driver_version(),
        gpu_name=str(cuda.get_device_name(0)),
        gpu_count=int(cuda.device_count()),
        gpu_total_memory_bytes=int(properties.total_memory),
        gpu_capability=tuple(cuda.get_device_capability(0)),
        bf16_supported=bool(cuda.is_bf16_supported()),
        disk_total_bytes=disk.total,
        disk_free_bytes=disk.free,
        run_identity_sha256=identity_sha256,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--sidecar", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--container-sha256", required=True)
    parser.add_argument("--base-model", type=Path, default=None)
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
        base_model = arguments.base_model or Path(config.runtime.model_root) / "xls-r-300m"
        prepared = validate_wav2vec_base(base_model.resolve())
        if prepared.weights_sha256 != config.model.weights_sha256:
            raise ValueError
        remote_root = arguments.data_root.resolve().parent
        state_root = remote_root / "state"
        ledger_root = remote_root / "ledger"
        completion_marker = state_root / "preflight-complete.json"
        environment_path = state_root / "environment.json"
        if completion_marker.exists():
            marker_identity = load_run_identity(completion_marker)
            sidecar = PackageSidecar.model_validate_json(
                read_trusted_regular_file(arguments.sidecar.resolve())
            )
            if (
                load_run_identity(state_root / "run-identity.json") != marker_identity
                or load_ledger(ledger_root)[-1].identity != marker_identity
                or marker_identity.config_sha256 != config.digest()
                or marker_identity.container_sha256 != arguments.container_sha256
                or marker_identity.code_sha256 != os.environ.get("VOXDELTA_CODE_SHA256", "")
                or marker_identity.manifest_sha256 != sidecar.manifest_sha256
                or digest_file(arguments.data_root.resolve() / "manifest.jsonl")
                != sidecar.manifest_sha256
                or load_runtime_environment(environment_path).run_identity_sha256
                != canonical_sha256(marker_identity.model_dump(mode="json"))
            ):
                raise ValueError
            print("remote_preflight_ok")
            return 0
        if ledger_root.exists() and load_ledger(ledger_root)[-1].to_stage != "preflight":
            raise ValueError
        for partial in (arguments.data_root.resolve(), state_root, ledger_root):
            if partial.is_symlink() or (partial.exists() and not partial.is_dir()):
                raise ValueError
            if partial.exists():
                shutil.rmtree(partial, ignore_errors=False)
        data_root = extract_verified_package(
            arguments.archive.resolve(),
            arguments.sidecar.resolve(),
            arguments.data_root.resolve(),
        )
        sidecar = PackageSidecar.model_validate_json(
            read_trusted_regular_file(arguments.sidecar.resolve())
        )
        manifest = data_root / "manifest.jsonl"
        if digest_file(manifest) != sidecar.manifest_sha256:
            raise ValueError
        plans = build_pilot_plans_from_package(manifest, seed=config.seed)
        identity = build_run_identity(
            config,
            sidecar,
            plans,
            code_sha256=os.environ.get("VOXDELTA_CODE_SHA256", ""),
            container_sha256=arguments.container_sha256,
        )
        publish_model(state_root / "run-identity.json", identity)
        publish_model(
            environment_path,
            _runtime_environment(
                torch,
                remote_root,
                canonical_sha256(identity.model_dump(mode="json")),
            ),
        )
        initialize_ledger(ledger_root, identity)
        append_transition(ledger_root, "preflight", identity)
        publish_model(completion_marker, identity)
    except Exception:
        print("remote_preflight_error")
        return 2
    print("remote_preflight_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
