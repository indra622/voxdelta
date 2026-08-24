"""Render stage-scoped, credential-free operator shell packets."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from voxdelta.evaluation.manifest import read_trusted_regular_file


class CommandPacketError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def _header() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# -ne 3 ]]; then echo "usage: $0 <ssh-user> <ssh-host> <ssh-port>" >&2; exit 64; fi
user="$1"; host="$2"; port="$3"
root="$(cd "$(dirname "$0")" && pwd -P)"
remote="/workspace/voxdelta/${VOXDELTA_RUN_ID:?set VOXDELTA_RUN_ID}"
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
ssh "${ssh_args[@]}" "$user@$host" "install -d -m 700 '$remote/incoming' '$remote/results'"
rsync --archive --partial --chmod=F600,D700 -e "$rsync_ssh" \
  train-validation.tar.zst train-validation.sidecar.json "$user@$host:$remote/incoming/"
ssh "${ssh_args[@]}" "$user@$host" \
  "/opt/voxdelta/runpod/.venv/bin/python /opt/voxdelta/runpod/scripts/preflight_remote.py \
  --config /opt/voxdelta/runpod/config/experiment.toml \
  --archive '$remote/incoming/train-validation.tar.zst' \
  --sidecar '$remote/incoming/train-validation.sidecar.json' --data-root '$remote/data' \
  --container-sha256 '$container_sha256'"
ssh "${ssh_args[@]}" "$user@$host" \
  "/opt/voxdelta/runpod/.venv/bin/python /opt/voxdelta/runpod/scripts/run_experiment.py \
  pilot --config /opt/voxdelta/runpod/config/experiment.toml --root '$remote'"
rsync --archive --partial -e "$rsync_ssh" \
  "$user@$host:$remote/results/pilots/" "$root/results/pilots/"
""",
        "02-full-or-resume.sh": _header()
        + """
ssh "${ssh_args[@]}" "$user@$host" \
  "/opt/voxdelta/runpod/.venv/bin/python /opt/voxdelta/runpod/scripts/run_experiment.py \
  full-or-resume --config /opt/voxdelta/runpod/config/experiment.toml --root '$remote'"
""",
        "03-download-results.sh": _header()
        + """
install -d -m 700 "$root/results/full"
install -d -m 700 "$root/evidence/full"
rsync --archive --partial --chmod=F600,D700 -e "$rsync_ssh" \
  "$user@$host:$remote/results/full/" "$root/results/full/"
rsync --archive --partial --chmod=F600,D700 -e "$rsync_ssh" \
  "$user@$host:$remote/ledger/" "$root/evidence/full/ledger/"
rsync --archive --partial --chmod=F600,D700 -e "$rsync_ssh" \
  "$user@$host:$remote/state/run-identity.json" "$root/evidence/full/run-identity.json"
""",
        "04-final-once.sh": _header()
        + """
cd "$root"; shasum -a 256 -c SHA256SUMS
ssh "${ssh_args[@]}" "$user@$host" "install -d -m 700 '$remote/incoming/final'"
rsync --archive --partial --chmod=F600,D700 -e "$rsync_ssh" \
  final-holdout.tar.zst final-holdout.sidecar.json frozen-candidate.json \
  "$user@$host:$remote/incoming/final/"
ssh "${ssh_args[@]}" "$user@$host" \
  "/opt/voxdelta/runpod/.venv/bin/python /opt/voxdelta/runpod/scripts/evaluate_final.py \
  --config /opt/voxdelta/runpod/config/experiment.toml --root '$remote'"
""",
        "05-download-final.sh": _header()
        + """
install -d -m 700 "$root/results/final"
install -d -m 700 "$root/evidence/final"
rsync --archive --partial --chmod=F600,D700 -e "$rsync_ssh" \
  "$user@$host:$remote/results/final/" "$root/results/final/"
rsync --archive --partial --chmod=F600,D700 -e "$rsync_ssh" \
  "$user@$host:$remote/ledger/" "$root/evidence/final/ledger/"
rsync --archive --partial --chmod=F600,D700 -e "$rsync_ssh" \
  "$user@$host:$remote/state/run-identity.json" "$root/evidence/final/run-identity.json"
""",
    }


def render_operator_commands(output: Path, *, run_id: str, final: bool = False) -> tuple[Path, ...]:
    if not output.is_absolute() or output.is_symlink() or not _RUN_ID.fullmatch(run_id):
        raise CommandPacketError("invalid_command_target")
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
            _scripts()[name].replace("${VOXDELTA_RUN_ID:?set VOXDELTA_RUN_ID}", run_id).encode()
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


__all__ = ["CommandPacketError", "render_operator_commands", "update_checksums"]
