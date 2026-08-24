from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from voxdelta_runpod.config import BASE_IMAGE
from voxdelta_runpod.image import ImageHandoffError, inspect_oci_archive, publish_image_handoff

GIT_COMMIT = "a" * 40
LOCK_SHA256 = "b" * 64
RUNPOD = Path(__file__).parents[1]


def _digest(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _oci_archive(path: Path, *, unsafe_history: bool = False) -> Path:
    blobs: dict[str, bytes] = {}

    def blob(payload: bytes) -> str:
        digest = _digest(payload)
        blobs[digest] = payload
        return digest

    config_payload = _json(
        {
            "architecture": "amd64",
            "os": "linux",
            "config": {
                "Labels": {
                    "org.opencontainers.image.revision": GIT_COMMIT,
                    "io.voxdelta.runpod.lock-sha256": LOCK_SHA256,
                    "io.voxdelta.runpod.platform": "linux/amd64",
                }
            },
            "history": [
                {"created_by": "RUN safe build"},
                {"created_by": "RUN --mount=type=secret token"} if unsafe_history else {},
            ],
        }
    )
    config_digest = blob(config_payload)
    image_manifest = _json(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": {"digest": config_digest},
            "layers": [],
        }
    )
    image_digest = blob(image_manifest)
    sbom_statement = _json(
        {
            "_type": "https://in-toto.io/Statement/v0.1",
            "predicateType": "https://spdx.dev/Document",
            "predicate": {"spdxVersion": "SPDX-2.3", "name": "voxdelta-runpod"},
        }
    )
    sbom_digest = blob(sbom_statement)
    attestation_manifest = _json(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": {"digest": blob(b"{}")},
            "layers": [{"digest": sbom_digest}],
        }
    )
    attestation_digest = blob(attestation_manifest)
    index = _json(
        {
            "schemaVersion": 2,
            "manifests": [
                {
                    "digest": image_digest,
                    "platform": {"os": "linux", "architecture": "amd64"},
                },
                {
                    "digest": attestation_digest,
                    "platform": {"os": "unknown", "architecture": "unknown"},
                    "annotations": {"vnd.docker.reference.type": "attestation-manifest"},
                },
            ],
        }
    )

    def add(stream: tarfile.TarFile, name: str, payload: bytes) -> None:
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        info.mode = 0o600
        stream.addfile(info, io.BytesIO(payload))

    with tarfile.open(path, "w:") as stream:
        add(stream, "index.json", index)
        add(stream, "oci-layout", b'{"imageLayoutVersion":"1.0.0"}')
        for digest, payload in sorted(blobs.items()):
            add(stream, f"blobs/sha256/{digest.removeprefix('sha256:')}", payload)
    return path


def test_inspect_oci_archive_verifies_manifest_labels_and_spdx(tmp_path: Path) -> None:
    evidence = inspect_oci_archive(
        _oci_archive(tmp_path / "image.tar").resolve(),
        expected_git_commit=GIT_COMMIT,
        expected_lock_sha256=LOCK_SHA256,
    )

    assert evidence.manifest_digest.startswith("sha256:")
    assert evidence.config_digest.startswith("sha256:")
    assert evidence.git_commit == GIT_COMMIT
    assert evidence.lock_sha256 == LOCK_SHA256
    assert evidence.sbom == {"spdxVersion": "SPDX-2.3", "name": "voxdelta-runpod"}


def test_inspect_oci_archive_rejects_sensitive_build_history(tmp_path: Path) -> None:
    with pytest.raises(ImageHandoffError, match="^unsafe_image_history$"):
        inspect_oci_archive(
            _oci_archive(tmp_path / "image.tar", unsafe_history=True).resolve(),
            expected_git_commit=GIT_COMMIT,
            expected_lock_sha256=LOCK_SHA256,
        )


def test_publish_image_handoff_is_private_complete_and_non_overwriting(tmp_path: Path) -> None:
    archive = _oci_archive(tmp_path / "image.tar").resolve()
    output_root = (tmp_path / "dist").resolve()
    target = publish_image_handoff(
        archive,
        output_root,
        run_id="run-622",
        git_commit=GIT_COMMIT,
        lock_sha256=LOCK_SHA256,
        created_at=datetime(2026, 8, 24, 5, 0, tzinfo=UTC),
    )

    assert {path.name for path in target.iterdir()} == {
        "SHA256SUMS",
        "image-build.json",
        "image-digest.txt",
        "image-sbom.spdx.json",
        "push-image.sh",
        "voxdelta-runpod.oci.tar.zst",
    }
    assert target.stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o077 == 0 for path in target.iterdir())
    assert json.loads((target / "image-build.json").read_text()) == {
        "base_image": BASE_IMAGE,
        "config_digest": inspect_oci_archive(
            archive,
            expected_git_commit=GIT_COMMIT,
            expected_lock_sha256=LOCK_SHA256,
        ).config_digest,
        "created_at": "2026-08-24T05:00:00Z",
        "git_commit": GIT_COMMIT,
        "manifest_digest": (target / "image-digest.txt").read_text().strip(),
        "platform": "linux/amd64",
        "run_id": "run-622",
        "runpod_lock_sha256": LOCK_SHA256,
        "schema_version": "1",
    }
    assert (
        subprocess.run(["bash", "-n", str(target / "push-image.sh")], check=False).returncode == 0
    )
    completed = subprocess.run(
        ["shasum", "-a", "256", "-c", "SHA256SUMS"],
        cwd=target,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    push_script = (target / "push-image.sh").read_text()
    assert "image-reference.txt" in push_script
    assert "password" not in push_script.lower()
    assert "token" not in push_script.lower()

    with pytest.raises(ImageHandoffError, match="^handoff_exists$"):
        publish_image_handoff(
            archive,
            output_root,
            run_id="run-622",
            git_commit=GIT_COMMIT,
            lock_sha256=LOCK_SHA256,
        )


def test_docker_context_is_an_explicit_code_only_allowlist() -> None:
    ignore = (RUNPOD / "docker" / "Dockerfile.dockerignore").read_text().splitlines()
    assert ignore[0] == "**"
    assert "!backend/src/**" in ignore
    assert "!runpod/src/**" in ignore
    assert "!runpod/scripts/**" in ignore
    assert not any("data" in line or "runtime" in line or ".env" in line for line in ignore)

    dockerfile = (RUNPOD / "docker" / "Dockerfile").read_text()
    assert dockerfile.count("@sha256:") == 2
    assert "COPY ." not in dockerfile
    assert "--mount=type=secret" not in dockerfile
    assert "/Users/" not in dockerfile and "/Volumes/" not in dockerfile


def test_build_and_push_scripts_have_valid_shell_syntax() -> None:
    build_script = RUNPOD / "scripts" / "build_image.sh"
    completed = subprocess.run(
        ["bash", "-n", str(build_script)], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert os.access(build_script, os.X_OK)
