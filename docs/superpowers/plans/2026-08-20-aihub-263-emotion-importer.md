# AI Hub 263 Emotion Importer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the licensed AI Hub dataset 263 ZIP+CP949-CSV release into local, privacy-minimized WAV+JSON pairs that the existing VoxDelta manifest builder can consume.

**Architecture:** Add a focused importer module and CLI that read each annual CSV/ZIP pair without mutating the source archives, accept only rows with at least three matching emotion votes, and write sharded local pairs through a staging directory. The importer records ambiguous votes and unmatched IDs in a transcript-free report; generated metadata uses per-item synthetic grouping identifiers because AI Hub 263 does not provide call or speaker identity.

**Tech Stack:** Python 3.12 standard library (`csv`, `zipfile`, `tempfile`, `wave`, `json`), NumPy, SoXR HQ resampling, Pydantic manifest contracts, pytest, Ruff, strict mypy, uv.

## Global Constraints

- Preserve all source ZIP and CSV bytes exactly; never call `ZipFile.extract` or `extractall`.
- Decode official metadata as CP949 and reject malformed headers, duplicate IDs, unsupported labels, duplicate archive stems, symlinks, and unsafe output paths.
- Require at least three of five annotators to agree; do not manufacture a hard label for lower-consensus rows.
- Require official 48 kHz mono PCM16 source WAVs and normalize accepted items to 16 kHz mono PCM16 before publication.
- Permit the observed 4차년도 mismatch only through an explicit CLI allowance of 16 missing and 16 orphan IDs; report every skipped opaque ID without transcripts.
- Never copy transcripts into normalized metadata, reports, stdout, stderr, Git, or tests; emit an empty transcript because emotion training consumes audio and labels only.
- Synthetic `call_id` and `speaker_id` values are per-item grouping keys, not real identities; document that dataset-263 splits cannot claim speaker-disjointness.
- Keep source data, normalized pairs, manifests, and reports under Git-ignored `data/*` paths.

---

### Task 1: Parse and classify official AI Hub 263 metadata

**Files:**
- Create: `backend/src/voxdelta/evaluation/aihub_emotion.py`
- Create: `backend/tests/evaluation/test_aihub_emotion.py`

**Interfaces:**
- Consumes: trusted CP949 CSV bytes and a ZIP member-name sequence.
- Produces: `EmotionImportRow`, `EmotionImportPlan`, and `plan_emotion_import(csv_path: Path, zip_path: Path, *, max_missing_audio: int, max_orphan_audio: int) -> EmotionImportPlan`.

- [ ] **Step 1: Write failing parser tests**

```python
def test_plan_accepts_three_of_five_votes_and_omits_transcript(tmp_path: Path) -> None:
    csv_path, zip_path = _write_official_fixture(
        tmp_path,
        rows=[_row("item-a", ["angry", "angry", "angry", "neutral", "sadness"])],
    )
    plan = plan_emotion_import(csv_path, zip_path, max_missing_audio=0, max_orphan_audio=0)
    assert [(row.item_id, row.emotion) for row in plan.accepted] == [("item-a", "anger")]
    assert plan.accepted[0].transcript == ""


def test_plan_quarantines_votes_without_three_person_consensus(tmp_path: Path) -> None:
    csv_path, zip_path = _write_official_fixture(
        tmp_path,
        rows=[_row("item-a", ["angry", "angry", "neutral", "neutral", "sadness"])],
    )
    plan = plan_emotion_import(csv_path, zip_path, max_missing_audio=0, max_orphan_audio=0)
    assert plan.accepted == ()
    assert plan.ambiguous_ids == ("item-a",)
```

- [ ] **Step 2: Run the parser tests and verify RED**

Run: `cd backend && uv run pytest tests/evaluation/test_aihub_emotion.py -v`

Expected: collection fails because `voxdelta.evaluation.aihub_emotion` does not exist.

- [ ] **Step 3: Implement strict CP949 parsing and vote consensus**

```python
@dataclass(frozen=True)
class EmotionImportRow:
    item_id: str
    emotion: EmotionLabel
    transcript: str = ""


def _consensus(votes: tuple[str, str, str, str, str]) -> EmotionLabel | None:
    counts = Counter(_normalize_emotion(vote) for vote in votes)
    label, count = counts.most_common(1)[0]
    return label if count >= 3 else None
```

Parse exactly the 15 official columns, require five non-empty supported emotion votes, index ZIP members by WAV basename, classify ambiguous/missing/orphan IDs, and fail if mismatch counts exceed the explicit allowances.

- [ ] **Step 4: Add failing edge-case tests**

Cover malformed CP949, wrong headers, duplicate CSV IDs, duplicate ZIP stems, non-WAV members, unsupported emotions, symlinked source paths, allowance overflow, and transcript-free exception messages.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `cd backend && uv run pytest tests/evaluation/test_aihub_emotion.py -v`

Expected: all importer parser tests pass with no warnings.

- [ ] **Step 6: Commit the parser**

```bash
git add backend/src/voxdelta/evaluation/aihub_emotion.py backend/tests/evaluation/test_aihub_emotion.py
git commit -m "feat: parse AI Hub 263 emotion data"
```

### Task 2: Extract accepted WAVs atomically and write a safe import report

**Files:**
- Modify: `backend/src/voxdelta/evaluation/aihub_emotion.py`
- Modify: `backend/tests/evaluation/test_aihub_emotion.py`

**Interfaces:**
- Consumes: `EmotionImportPlan` objects from Task 1, official 48 kHz mono PCM16 WAV members, and a destination directory that does not exist.
- Produces: `import_emotion_dataset(source_root: Path, output_root: Path, *, max_missing_audio: int, max_orphan_audio: int) -> EmotionImportReport`, 16 kHz mono PCM16 sharded pairs under `output_root/pairs/<id-prefix>/`, and `output_root/import-report.json`.

- [ ] **Step 1: Write failing atomic-import tests**

```python
def test_import_writes_sharded_pairs_and_transcript_free_report(tmp_path: Path) -> None:
    source = _write_three_release_fixture(tmp_path / "source")
    output = tmp_path / "normalized"
    report = import_emotion_dataset(
        source, output, max_missing_audio=0, max_orphan_audio=0
    )
    metadata = json.loads((output / "pairs" / "it" / "item-a.json").read_text())
    assert metadata["transcript"] == ""
    assert metadata["emotion"] == "anger"
    assert metadata["call_id"] == "aihub-263-item:item-a"
    assert report.accepted == 3
    assert "PRIVATE_TRANSCRIPT_SENTINEL" not in (output / "import-report.json").read_text()
```

- [ ] **Step 2: Run the atomic-import test and verify RED**

Run: `cd backend && uv run pytest tests/evaluation/test_aihub_emotion.py::test_import_writes_sharded_pairs_and_transcript_free_report -v`

Expected: fails because `import_emotion_dataset` is not implemented.

- [ ] **Step 3: Implement streaming extraction and atomic publication**

```python
with archive.open(member, "r") as source, temporary_audio.open("wb") as target:
    while chunk := source.read(1024 * 1024):
        digest.update(chunk)
        target.write(chunk)
os.replace(temporary_audio, audio_path)
```

Create a same-filesystem staging directory, decode and validate the official mono PCM16 WAV headers, resample 48 kHz samples to 16 kHz with SoXR HQ, write mode-0600 WAV and JSON files, fsync files, publish the completed directory with `os.replace`, and remove only the importer-owned staging directory after a failure. Include annual counts, label counts, ambiguous IDs, missing IDs, orphan IDs, and source file sizes in the report without transcripts.

- [ ] **Step 4: Add failure-boundary tests**

Test Python-readable forced-ZIP64 input, corrupt member CRC, pre-existing output, symlinked output ancestors, interrupted writes, invalid annual filename sets, and no partial published output after failure.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `cd backend && uv run pytest tests/evaluation/test_aihub_emotion.py -v`

Expected: all parser and extraction tests pass with no partial directories left behind.

- [ ] **Step 6: Commit safe extraction**

```bash
git add backend/src/voxdelta/evaluation/aihub_emotion.py backend/tests/evaluation/test_aihub_emotion.py
git commit -m "feat: normalize AI Hub 263 emotion audio"
```

### Task 3: Add the CLI and connect it to manifest generation

**Files:**
- Create: `backend/scripts/prepare_aihub_emotion.py`
- Modify: `backend/tests/evaluation/test_aihub_emotion.py`
- Modify: `backend/tests/evaluation/test_manifest.py`
- Modify: `data/README.md`

**Interfaces:**
- Consumes: `--source-root`, `--output-root`, `--max-missing-audio`, and `--max-orphan-audio`.
- Produces: sanitized exit status 0/2 and normalized pairs accepted by `build_aihub_manifests.py --emotion-root <output>/pairs`.

- [ ] **Step 1: Write failing CLI and end-to-end builder tests**

```python
result = _run_preparer(source, normalized)
assert result.returncode == 0
assert result.stdout == ""
assert result.stderr == ""

build = _run_builder(empty_consultation, normalized / "pairs", manifests)
assert build.returncode == 0
assert {item.source for item in load_manifest(manifests / "emotion.jsonl")} == {"emotion"}
```

- [ ] **Step 2: Run the CLI tests and verify RED**

Run: `cd backend && uv run pytest tests/evaluation/test_aihub_emotion.py tests/evaluation/test_manifest.py -v`

Expected: fails because `scripts/prepare_aihub_emotion.py` does not exist.

- [ ] **Step 3: Implement the sanitized CLI**

```python
try:
    import_emotion_dataset(
        arguments.source_root,
        arguments.output_root,
        max_missing_audio=arguments.max_missing_audio,
        max_orphan_audio=arguments.max_orphan_audio,
    )
except (OSError, ValueError):
    print("emotion import failed", file=sys.stderr)
    return 2
return 0
```

Reject negative allowances in argparse without echoing user paths or transcripts.

- [ ] **Step 4: Document the exact local workflow and limitation**

Document source root, normalized output, explicit 16/16 mismatch allowance, Python ZIP64 handling, manifest command, transcript minimization, synthetic item grouping, and the inability to claim speaker-disjoint splits for AI Hub 263.

- [ ] **Step 5: Run CLI, manifest, lint, and type checks**

Run:

```bash
cd backend
uv run pytest tests/evaluation/test_aihub_emotion.py tests/evaluation/test_manifest.py -v
uv run ruff check scripts/prepare_aihub_emotion.py src/voxdelta/evaluation/aihub_emotion.py tests/evaluation/test_aihub_emotion.py tests/evaluation/test_manifest.py
uv run ruff format --check scripts/prepare_aihub_emotion.py src/voxdelta/evaluation/aihub_emotion.py tests/evaluation/test_aihub_emotion.py tests/evaluation/test_manifest.py
uv run mypy src scripts
```

Expected: all commands exit 0.

- [ ] **Step 6: Commit the CLI and documentation**

```bash
git add backend/scripts/prepare_aihub_emotion.py backend/tests/evaluation/test_aihub_emotion.py backend/tests/evaluation/test_manifest.py data/README.md
git commit -m "docs: add AI Hub 263 import workflow"
```

### Task 4: Normalize the real licensed dataset and verify the complete repository

**Files:**
- Generate ignored local data: `data/derived/aihub/emotion/`
- Generate ignored manifests: `data/manifests/`

**Interfaces:**
- Consumes: `data/raw/감정 분류를 위한 대화 음성 데이터셋/` with explicit 16/16 mismatch allowance.
- Produces: 36,680 or fewer high-consensus normalized pairs after removing any accepted rows whose audio is absent, a transcript-free report, and deterministic manifests.

- [ ] **Step 1: Run the real importer**

```bash
cd backend
uv run python scripts/prepare_aihub_emotion.py \
  --source-root '../data/raw/감정 분류를 위한 대화 음성 데이터셋' \
  --output-root ../data/derived/aihub/emotion \
  --max-missing-audio 16 \
  --max-orphan-audio 16
```

Expected: exit 0, source archive mtimes and sizes unchanged, and a published `import-report.json`.

- [ ] **Step 2: Build manifests from normalized pairs**

```bash
mkdir -p ../data/derived/aihub/empty-consultation
uv run python scripts/build_aihub_manifests.py \
  --consultation-root ../data/derived/aihub/empty-consultation \
  --emotion-root ../data/derived/aihub/emotion/pairs \
  --output-root ../data/manifests
```

Expected: exit 0 and non-empty train, validation, test, and emotion manifests.

- [ ] **Step 3: Verify report totals, manifests, and hashes**

Run a transcript-free verification script that checks accepted + ambiguous + missing equals 43,991 CSV rows, manifest count equals accepted count, label values are the canonical seven, all audio paths are under the normalized output, and every manifest SHA-256 matches its WAV.

- [ ] **Step 4: Run the full quality gate**

```bash
cd backend
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy src scripts
uv lock --check
git diff --check
```

Expected: all commands exit 0; the platform-specific Windows test may remain skipped on macOS.

- [ ] **Step 5: Commit any final test-only corrections**

```bash
git add backend data/README.md docs/superpowers/plans/2026-08-20-aihub-263-emotion-importer.md
git commit -m "test: verify AI Hub 263 import workflow"
```

Skip this commit when the worktree is already clean after the previous commits.
