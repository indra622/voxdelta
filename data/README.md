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

## Run a real-data emotion smoke experiment

Before a full 36,665-item training run, exercise the real model boundary on a deterministic balanced
subset. From `backend/`, resolve the canonical checkout once so every generated output path is
absolute:

```bash
VOXDELTA_ROOT="$(cd .. && pwd -P)"

uv run python scripts/build_emotion_smoke_manifest.py \
  --manifest "$VOXDELTA_ROOT/data/manifests/emotion.jsonl" \
  --output "$VOXDELTA_ROOT/data/manifests/emotion-smoke.jsonl"

uv run python scripts/train_emotion.py \
  --manifest "$VOXDELTA_ROOT/data/manifests/emotion-smoke.jsonl" \
  --output "$VOXDELTA_ROOT/data/models/emotion2vec-plus-large-smoke" \
  --architecture emotion2vec-plus \
  --base-model iic/emotion2vec_plus_large \
  --freeze-encoder \
  --seed 622

uv run python scripts/evaluate_emotion_checkpoint.py \
  --manifest "$VOXDELTA_ROOT/data/manifests/emotion-smoke.jsonl" \
  --checkpoint "$VOXDELTA_ROOT/data/models/emotion2vec-plus-large-smoke" \
  --output "$VOXDELTA_ROOT/data/benchmarks/emotion2vec-plus-large-smoke.json" \
  --architecture emotion2vec-plus \
  --device auto
```

The default smoke manifest contains 20 train, 5 validation, and 5 test items per emotion: 210 items
total. The aggregate report contains macro-F1, per-label F1, a canonical-order confusion matrix,
10-bin expected calibration error, latency, and RSS. It does not contain transcripts, item IDs,
audio paths, raw probabilities, or per-item predictions.

This smoke run proves model download, real-audio preprocessing, training, checkpoint publication,
production-provider loading, and held-out inference. Its small-sample quality is not final model
evidence. The earlier emotion2vec smoke evaluated the 35 smoke-test members, so those items are
already exposed members of the nominal 3,620-item test split. A future final-quality comparison must
exclude those 35 or freeze a new untouched holdout from the remaining 3,585 items.

## Prepare and run the controlled XLS-R smoke

The XLS-R path never resolves a floating Hugging Face model ID during training or inference. From
the implementation worktree's `backend/`, point every ignored artifact at the canonical NVMe data
tree explicitly:

```bash
VOXDELTA_DATA=/Volumes/nvme1/codes/voxdelta/data

uv run python scripts/prepare_wav2vec_base.py \
  --output "$VOXDELTA_DATA/models/base/wav2vec2-xls-r-300m-1a640f3"

uv run python scripts/train_emotion.py \
  --manifest "$VOXDELTA_DATA/manifests/emotion-smoke.jsonl" \
  --output "$VOXDELTA_DATA/models/wav2vec-xls-r-300m-smoke" \
  --architecture wav2vec-xls-r \
  --base-model facebook/wav2vec2-xls-r-300m \
  --base-model-path "$VOXDELTA_DATA/models/base/wav2vec2-xls-r-300m-1a640f3" \
  --micro-batch-size 4 \
  --seed 622

uv run python scripts/evaluate_emotion_checkpoint.py \
  --manifest "$VOXDELTA_DATA/manifests/emotion-smoke.jsonl" \
  --checkpoint "$VOXDELTA_DATA/models/wav2vec-xls-r-300m-smoke" \
  --base-model-path "$VOXDELTA_DATA/models/base/wav2vec2-xls-r-300m-1a640f3" \
  --output "$VOXDELTA_DATA/benchmarks/wav2vec-xls-r-300m-smoke-validation.json" \
  --architecture wav2vec-xls-r \
  --device mps \
  --split validation
```

The base preparation accepts only revision
`1a640f32ac3e39899438a2931f9924c02f080a54` and verifies the exact configuration,
preprocessor, and 1,269,737,156-byte weight file before atomically publishing private local files.
The initial micro-batch is 4 with gradient accumulation 4. Retry with micro-batch 2, then 1, only
after an observed MPS out-of-memory failure and confirmation that no final checkpoint directory was
published. Do not disable PyTorch's MPS high-watermark safety control.

This controlled smoke trains on 140 items and selects on 35 validation items. Evaluation also uses
those 35 validation items as readiness evidence; it does not evaluate any test-split item and is not
an unbiased final quality estimate.
