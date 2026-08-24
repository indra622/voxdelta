#!/usr/bin/env bash
set -euo pipefail
umask 077

repo="$(cd "$(dirname "$0")/../.." && pwd -P)"
runpod="$repo/runpod"
for command in docker git shasum zstd; do
  command -v "$command" >/dev/null 2>&1 || { echo "missing command: $command" >&2; exit 69; }
done

if [[ $# -ne 1 || ! "$1" =~ ^[a-z0-9][a-z0-9-]{0,63}$ ]]; then
  echo "usage: build_image.sh <run-id>" >&2
  exit 64
fi
if [[ -n "$(git -C "$repo" status --porcelain --untracked-files=all)" ]]; then
  echo "image build requires a clean worktree" >&2
  exit 65
fi

git_commit="$(git -C "$repo" rev-parse HEAD)"
lock_sha="$(shasum -a 256 "$runpod/uv.lock" | awk '{print $1}')"
temporary="$(mktemp -d "${TMPDIR:-/tmp}/voxdelta-build.XXXXXX")"
trap 'rm -rf -- "$temporary"' EXIT
archive="$temporary/voxdelta-runpod.oci.tar"

docker buildx build \
  --file "$runpod/docker/Dockerfile" \
  --platform linux/amd64 \
  --build-arg "VOXDELTA_GIT_COMMIT=$git_commit" \
  --build-arg "VOXDELTA_RUNPOD_LOCK_SHA256=$lock_sha" \
  --provenance=mode=max \
  --sbom=true \
  --output "type=oci,dest=$archive" \
  "$repo"

uv run --project "$runpod" python "$runpod/scripts/render_image_handoff.py" \
  --oci-archive "$archive" \
  --output-root "$runpod/dist" \
  --run-id "$1" \
  --git-commit "$git_commit" \
  --lock "$runpod/uv.lock"
