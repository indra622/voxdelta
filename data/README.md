# Local data layout

VoxDelta expects licensed datasets and generated evaluation artifacts in these local-only paths:

```text
data/raw/aihub/consultation/
data/raw/aihub/emotion/
data/manifests/
data/gold/
data/jobs/
data/models/
data/benchmarks/
```

Raw data, derived audio clips, manifests containing local audio paths, gold annotations, job
artifacts, downloaded models, and benchmark outputs are ignored by Git. Raw data and derived audio
clips are not redistributed. Confirm that your AI Hub license permits each intended local use before
placing data here.

## Build manifests

From `backend/`, generate deterministic call-level train, validation, and test manifests:

```bash
uv run python scripts/build_aihub_manifests.py \
  --consultation-root ../data/raw/aihub/consultation \
  --emotion-root ../data/raw/aihub/emotion \
  --output-root ../data/manifests
```

The builder requires adjacent audio and JSON files with the same filename stem. It stores absolute
local audio paths and SHA-256 digests in `train.jsonl`, `validation.jsonl`, and `test.jsonl`.

## Validate audio hashes

From `backend/`, verify every manifest entry without printing transcripts:

```bash
uv run python - <<'PY'
import hashlib
from pathlib import Path

from voxdelta.evaluation.manifest import load_manifest, read_trusted_regular_file

for manifest in sorted(Path("../data/manifests").glob("*.jsonl")):
    for item in load_manifest(manifest):
        digest = hashlib.sha256(read_trusted_regular_file(item.audio_path)).hexdigest()
        if digest != item.sha256:
            raise SystemExit(f"hash mismatch for item {item.id}")
print("manifest hashes verified")
PY
```
