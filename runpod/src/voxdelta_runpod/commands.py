"""Render stage-scoped, credential-free operator shell packets."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from voxdelta.evaluation.manifest import read_trusted_regular_file

from voxdelta_runpod.config import validated_remote_root


class CommandPacketError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_VOLUME_ROOT = "__VOLUME_ROOT__"
_MODEL_ROOT = "__MODEL_ROOT__"
DEFAULT_VOLUME_ROOT = "/opt/voxdelta-run/runs"
DEFAULT_MODEL_ROOT = "/opt/voxdelta-run/models"


def _header() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# -ne 3 ]]; then echo "usage: $0 <ssh-user> <ssh-host> <ssh-port>" >&2; exit 64; fi
user="$1"; host="$2"; port="$3"
root="$(cd "$(dirname "$0")" && pwd -P)"
remote="__VOLUME_ROOT__/${VOXDELTA_RUN_ID:?set VOXDELTA_RUN_ID}"
ssh_args=(-p "$port" -o BatchMode=yes -o StrictHostKeyChecking=accept-new)
rsync_ssh="ssh -p $port -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
"""


def _scripts() -> dict[str, str]:
    return {
        "01-preflight-and-pilot.sh": _header()
        + """
cd "$root"; shasum -a 256 -c SHA256SUMS
image_digest="$(tr -d '\n' < image-digest.txt)"
[[ "$image_digest" =~ ^sha256:[0-9a-f]{64}$ ]] || { echo image_digest_error >&2; exit 65; }
container_sha256="${image_digest#sha256:}"
ssh "${ssh_args[@]}" "$user@$host" \
  "install -d -m 700 '$remote/incoming' '$remote/incoming/models' '$remote/results'"
ssh "${ssh_args[@]}" "$user@$host" \
  "/opt/voxdelta/runpod/.venv/bin/python /opt/voxdelta/runpod/scripts/probe_filesystem.py \
  --path '$remote' --path '$remote/incoming' --path '$remote/incoming/models' \
  --path '$remote/results' --path '__MODEL_ROOT__' --min-free-gb 100"
rsync --archive --no-owner --no-group --partial --chmod=F600,D700 -e "$rsync_ssh" \
  train-validation.tar.zst train-validation.sidecar.json "$user@$host:$remote/incoming/"
rsync --archive --no-owner --no-group --partial --chmod=F600,D700 -e "$rsync_ssh" \
  xls-r-base.tar.zst xls-r-base.sidecar.json "$user@$host:$remote/incoming/models/"
ssh "${ssh_args[@]}" "$user@$host" \
  "/opt/voxdelta/runpod/.venv/bin/python /opt/voxdelta/runpod/scripts/extract_model_bundle.py \
  xls-r-base --archive '$remote/incoming/models/xls-r-base.tar.zst' \
  --sidecar '$remote/incoming/models/xls-r-base.sidecar.json' \
  --target __MODEL_ROOT__/xls-r-300m"
ssh "${ssh_args[@]}" "$user@$host" \
  "/opt/voxdelta/runpod/.venv/bin/python /opt/voxdelta/runpod/scripts/preflight_remote.py \
  --config /opt/voxdelta/runpod/config/experiment.toml \
  --archive '$remote/incoming/train-validation.tar.zst' \
  --sidecar '$remote/incoming/train-validation.sidecar.json' --data-root '$remote/data' \
  --container-sha256 '$container_sha256'"
ssh "${ssh_args[@]}" "$user@$host" \
  "/opt/voxdelta/runpod/.venv/bin/python /opt/voxdelta/runpod/scripts/run_remote_stage.py \
  launch pilot --config /opt/voxdelta/runpod/config/experiment.toml --root '$remote' && \
  /opt/voxdelta/runpod/.venv/bin/python /opt/voxdelta/runpod/scripts/run_remote_stage.py \
  wait pilot --root '$remote'"
install -d -m 700 "$root/results" "$root/results/pilots"
rsync --archive --no-owner --no-group --partial --chmod=F600,D700 -e "$rsync_ssh" \
  "$user@$host:$remote/results/pilots/" "$root/results/pilots/"
""",
        "02-full-or-resume.sh": _header()
        + """
ssh "${ssh_args[@]}" "$user@$host" \
  "/opt/voxdelta/runpod/.venv/bin/python /opt/voxdelta/runpod/scripts/run_remote_stage.py \
  launch full-or-resume --config /opt/voxdelta/runpod/config/experiment.toml --root '$remote' && \
  /opt/voxdelta/runpod/.venv/bin/python /opt/voxdelta/runpod/scripts/run_remote_stage.py \
  wait full-or-resume --root '$remote'"
""",
        "03-download-results.sh": _header()
        + """
install -d -m 700 "$root/results/full"
install -d -m 700 "$root/evidence/full"
install -d -m 700 "$root/evidence/full/results/pilots"
install -d -m 700 "$root/evidence/full/state/checkpoints"
install -d -m 700 "$root/evidence/full/logs"
rsync --archive --no-owner --no-group --partial --chmod=F600,D700 -e "$rsync_ssh" \
  "$user@$host:$remote/results/full/" "$root/results/full/"
rsync --archive --no-owner --no-group --partial --chmod=F600,D700 -e "$rsync_ssh" \
  "$user@$host:$remote/results/pilots/" "$root/evidence/full/results/pilots/"
rsync --archive --no-owner --no-group --partial --chmod=F600,D700 -e "$rsync_ssh" \
  "$user@$host:$remote/state/checkpoints/" "$root/evidence/full/state/checkpoints/"
for artifact in batch-profile.json environment.json preflight-complete.json run-identity.json; do
  rsync --archive --no-owner --no-group --partial --chmod=F600,D700 -e "$rsync_ssh" \
    "$user@$host:$remote/state/$artifact" "$root/evidence/full/state/$artifact"
done
rsync --archive --no-owner --no-group --partial --chmod=F600,D700 -e "$rsync_ssh" \
  "$user@$host:$remote/logs/" "$root/evidence/full/logs/"
rsync --archive --no-owner --no-group --partial --chmod=F600,D700 -e "$rsync_ssh" \
  "$user@$host:$remote/ledger/" "$root/evidence/full/ledger/"
install -d -m 700 "$root/archive"
(
  cd "$root"
  while IFS= read -r file; do shasum -a 256 "$file"; done \
    < <(find results/full evidence/full -type f ! -name SHA256SUMS.local | LC_ALL=C sort) \
    > evidence/full/SHA256SUMS.local
  chmod 600 evidence/full/SHA256SUMS.local
  tar -cf - results/full evidence/full | zstd -3 --threads=1 --quiet \
    -o archive/full-retrieval.tar.zst
  chmod 600 archive/full-retrieval.tar.zst
  shasum -a 256 archive/full-retrieval.tar.zst \
    > archive/full-retrieval.tar.zst.sha256
  chmod 600 archive/full-retrieval.tar.zst.sha256
  zstd --test --quiet archive/full-retrieval.tar.zst
  shasum -a 256 -c archive/full-retrieval.tar.zst.sha256
)
""",
        "04-final-once.sh": _header()
        + """
cd "$root"; shasum -a 256 -c SHA256SUMS
ssh "${ssh_args[@]}" "$user@$host" \
  "install -d -m 700 '$remote/incoming/final' '$remote/incoming/models'"
ssh "${ssh_args[@]}" "$user@$host" \
  "/opt/voxdelta/runpod/.venv/bin/python /opt/voxdelta/runpod/scripts/probe_filesystem.py \
  --path '$remote/incoming/final' --path '$remote/incoming/models' --path '__MODEL_ROOT__'"
rsync --archive --no-owner --no-group --partial --chmod=F600,D700 -e "$rsync_ssh" \
  final-holdout.tar.zst final-holdout.sidecar.json frozen-candidate.json \
  "$user@$host:$remote/incoming/final/"
rsync --archive --no-owner --no-group --partial --chmod=F600,D700 -e "$rsync_ssh" \
  emotion2vec-baseline.tar.zst emotion2vec-baseline.sidecar.json \
  "$user@$host:$remote/incoming/models/"
ssh "${ssh_args[@]}" "$user@$host" \
  "/opt/voxdelta/runpod/.venv/bin/python /opt/voxdelta/runpod/scripts/extract_model_bundle.py \
  emotion2vec-baseline \
  --archive '$remote/incoming/models/emotion2vec-baseline.tar.zst' \
  --sidecar '$remote/incoming/models/emotion2vec-baseline.sidecar.json' \
  --target __MODEL_ROOT__/emotion2vec-plus"
ssh "${ssh_args[@]}" "$user@$host" \
  "/opt/voxdelta/runpod/.venv/bin/python /opt/voxdelta/runpod/scripts/evaluate_final.py \
  --config /opt/voxdelta/runpod/config/experiment.toml --root '$remote'"
""",
        "05-download-final.sh": _header()
        + """
install -d -m 700 "$root/results/final"
install -d -m 700 "$root/evidence/final"
install -d -m 700 "$root/evidence/final/state"
install -d -m 700 "$root/evidence/final/logs"
rsync --archive --no-owner --no-group --partial --chmod=F600,D700 -e "$rsync_ssh" \
  "$user@$host:$remote/results/final/" "$root/results/final/"
rsync --archive --no-owner --no-group --partial --chmod=F600,D700 -e "$rsync_ssh" \
  "$user@$host:$remote/ledger/" "$root/evidence/final/ledger/"
rsync --archive --no-owner --no-group --partial --chmod=F600,D700 -e "$rsync_ssh" \
  "$user@$host:$remote/state/run-identity.json" "$root/evidence/final/state/run-identity.json"
rsync --archive --no-owner --no-group --partial --chmod=F600,D700 -e "$rsync_ssh" \
  "$user@$host:$remote/state/final-consumed.json" "$root/evidence/final/state/final-consumed.json"
rsync --archive --no-owner --no-group --partial --chmod=F600,D700 -e "$rsync_ssh" \
  "$user@$host:$remote/logs/" "$root/evidence/final/logs/"
install -d -m 700 "$root/archive"
(
  cd "$root"
  while IFS= read -r file; do shasum -a 256 "$file"; done \
    < <(find results/final evidence/final -type f ! -name SHA256SUMS.local | LC_ALL=C sort) \
    > evidence/final/SHA256SUMS.local
  chmod 600 evidence/final/SHA256SUMS.local
  tar -cf - results/final evidence/final | zstd -3 --threads=1 --quiet \
    -o archive/final-retrieval.tar.zst
  chmod 600 archive/final-retrieval.tar.zst
  shasum -a 256 archive/final-retrieval.tar.zst \
    > archive/final-retrieval.tar.zst.sha256
  chmod 600 archive/final-retrieval.tar.zst.sha256
  zstd --test --quiet archive/final-retrieval.tar.zst
  shasum -a 256 -c archive/final-retrieval.tar.zst.sha256
)
""",
    }


def render_operator_commands(
    output: Path,
    *,
    run_id: str,
    final: bool = False,
    volume_root: str = DEFAULT_VOLUME_ROOT,
    model_root: str = DEFAULT_MODEL_ROOT,
) -> tuple[Path, ...]:
    if not output.is_absolute() or output.is_symlink() or not _RUN_ID.fullmatch(run_id):
        raise CommandPacketError("invalid_command_target")
    try:
        volume_root = validated_remote_root(volume_root)
        model_root = validated_remote_root(model_root)
    except ValueError as error:
        raise CommandPacketError("invalid_command_target") from error
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    selected = (
        ("04-final-once.sh", "05-download-final.sh")
        if final
        else (
            "01-preflight-and-pilot.sh",
            "02-full-or-resume.sh",
            "03-download-results.sh",
        )
    )
    rendered: list[Path] = []
    for name in selected:
        path = output / name
        if path.exists() or path.is_symlink():
            raise CommandPacketError("command_packet_exists")
        payload = (
            _scripts()[name]
            .replace("${VOXDELTA_RUN_ID:?set VOXDELTA_RUN_ID}", run_id)
            .replace(_VOLUME_ROOT, volume_root)
            .replace(_MODEL_ROOT, model_root)
            .encode()
        )
        with path.open("xb") as stream:
            os.fchmod(stream.fileno(), 0o700)
            stream.write(payload)
        rendered.append(path)
    return tuple(rendered)


def update_checksums(root: Path, files: tuple[Path, ...]) -> None:
    sums = root / "SHA256SUMS"
    existing = sums.read_text().splitlines() if sums.exists() else []
    names = {path.name for path in files}
    retained = [line for line in existing if line.split("  ", 1)[-1] not in names]
    additions = [
        f"{hashlib.sha256(read_trusted_regular_file(path)).hexdigest()}  {path.name}"
        for path in files
    ]
    payload = "\n".join(sorted([*retained, *additions])) + "\n"
    temporary = root / ".SHA256SUMS.staging"
    temporary.write_text(payload)
    temporary.chmod(0o600)
    os.replace(temporary, sums)


__all__ = [
    "DEFAULT_MODEL_ROOT",
    "DEFAULT_VOLUME_ROOT",
    "CommandPacketError",
    "render_operator_commands",
    "update_checksums",
]
