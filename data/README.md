# Local data layout

VoxDelta expects licensed datasets and generated evaluation artifacts in these local-only paths:

```text
data/raw/aihub/consultation/
data/raw/aihub/emotion/
data/derived/aihub/emotion/
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

## Prepare AI Hub dataset 263

AI Hub's **감정 분류를 위한 대화 음성 데이터셋** (`dataSetSn=263`) is delivered as
three annual WAV ZIP archives plus three CP949 CSV files, not as adjacent WAV+JSON pairs. Preserve
those six source files unchanged under a local source directory, then normalize them from `backend/`:

```bash
uv run python scripts/prepare_aihub_emotion.py \
  --source-root '../data/raw/감정 분류를 위한 대화 음성 데이터셋' \
  --output-root ../data/derived/aihub/emotion \
  --max-missing-audio 16 \
  --max-orphan-audio 16
```

The two mismatch allowances are explicit for the verified 4차년도 release, which contains 16 CSV
IDs without matching WAV basenames and 16 WAV basenames without CSV rows. The importer reads ZIP64
with Python instead of macOS `unzip`, verifies the official 48 kHz mono PCM16 input contract, and
resamples accepted audio to the training boundary's 16 kHz mono PCM16 format with SoXR HQ. It
requires at least three matching votes among the five emotion annotators and writes only
high-consensus items under `emotion/pairs/`. It never copies transcripts: the normalized JSON uses
an empty transcript, and `import-report.json` contains only opaque IDs and counts. Source archives
are opened read-only and the completed output is published atomically.

Dataset 263 does not expose real call or speaker identities. The normalized metadata therefore uses
an item-scoped synthetic grouping key for both fields. Hash-based train/validation/test assignment is
deterministic, but it must not be described as speaker-disjoint. Use the separately adjudicated
call-center gold set for the final domain evaluation.

## Build manifests

From `backend/`, generate deterministic call-level train, validation, and test manifests:

```bash
uv run python scripts/build_aihub_manifests.py \
  --consultation-root ../data/raw/aihub/consultation \
  --emotion-root ../data/derived/aihub/emotion/pairs \
  --output-root ../data/manifests
```

For an emotion-only build before consultation data is normalized, pass an existing empty local
directory as `--consultation-root`.

The builder requires adjacent audio and JSON files with the same filename stem. The preparation step
above creates that layout for dataset 263. The builder stores absolute local audio paths, source
identity, and SHA-256 digests in `train.jsonl`, `validation.jsonl`, and `test.jsonl`. It also writes
sorted source-specific `consultation.jsonl` and `emotion.jsonl` manifests; every item in those files
retains its deterministic split.

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
