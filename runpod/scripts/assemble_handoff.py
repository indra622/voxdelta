"""Atomically assemble verified image, model, and training packets into one handoff."""

from __future__ import annotations

import argparse
import os
import shutil
from collections.abc import Sequence
from pathlib import Path

from voxdelta.evaluation.manifest import read_trusted_regular_file

from voxdelta_runpod.commands import render_operator_commands, update_checksums

IMAGE_FILES = frozenset(
    {
        "voxdelta-runpod.oci.tar.zst",
        "image-digest.txt",
        "image-build.json",
        "image-sbom.spdx.json",
        "push-image.sh",
    }
)
TRAINING_FILES = frozenset({"train-validation.tar.zst", "train-validation.sidecar.json"})
BASE_MODEL_FILES = frozenset({"xls-r-base.tar.zst", "xls-r-base.sidecar.json"})


def _trusted_files(root: Path, expected: frozenset[str]) -> dict[str, Path]:
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError
    files = {
        path.name: path
        for path in root.iterdir()
        if path.is_file() and not path.is_symlink() and path.name != "SHA256SUMS"
    }
    if set(files) != expected:
        raise ValueError
    for path in files.values():
        read_trusted_regular_file(path)
    return files


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--training-root", required=True, type=Path)
    parser.add_argument("--base-model-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    arguments = parser.parse_args(argv)
    output = arguments.output.resolve()
    staging = output.with_name(f".{output.name}.staging")
    try:
        image = _trusted_files(arguments.image_root.resolve(), IMAGE_FILES)
        training = _trusted_files(arguments.training_root.resolve(), TRAINING_FILES)
        base_model = _trusted_files(arguments.base_model_root.resolve(), BASE_MODEL_FILES)
        if output.exists() or output.is_symlink() or staging.exists() or staging.is_symlink():
            raise ValueError
        staging.mkdir(mode=0o700, parents=True)
        for name, source in sorted({**image, **training, **base_model}.items()):
            destination = staging / name
            try:
                os.link(source, destination)
            except OSError:
                shutil.copyfile(source, destination)
                destination.chmod(source.stat(follow_symlinks=False).st_mode & 0o700)
        scripts = render_operator_commands(staging, run_id=arguments.run_id)
        files = tuple(
            path for path in staging.iterdir() if path.is_file() and path.name != "SHA256SUMS"
        )
        update_checksums(staging, (*files, *tuple(path for path in scripts if path not in files)))
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        print("handoff_assembly_error")
        return 2
    print("handoff_ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
