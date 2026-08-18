# VoxDelta Models, Data, and Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace deterministic fake providers with reproducible local and frontier implementations, build licensed-data manifests and a small gold set, and produce a benchmark that compares quality, latency, cost, calibration, and failures.

**Architecture:** Local diarization uses pyannote `speaker-diarization-community-1` constrained to two speakers, local ASR uses faster-whisper `large-v3-turbo`, and the local emotion baseline fine-tunes Wav2Vec2-XLS-R on the AI Hub seven-emotion data. The frontier path uses Gemini `gemini-3.6-flash` audio understanding with structured output. All providers adapt to the canonical core contracts, while a separate evaluation package owns splits, metrics, calibration, and benchmark artifacts.

**Tech Stack:** Python 3.12, PyTorch, torchaudio, pyannote.audio 4, faster-whisper, Transformers, Datasets, Accelerate, scikit-learn, pandas, jiwer, pyannote.metrics, google-genai, Pydantic, pytest

## Global Constraints

- AI Hub data is downloaded manually under the user's account and remains outside Git.
- Every call and speaker belongs to exactly one of train, validation, or test.
- The local/API comparison uses identical gold examples and records dataset hashes and model versions.
- Local providers must work without network access after gated models and checkpoints are downloaded.
- Gemini raw-audio transmission requires per-job disclosure and `GEMINI_API_KEY`; uploaded remote files are deleted in a `finally` block.
- API structured output must contain all seven emotion probabilities and they must sum to one after validation.
- Provider failure never silently falls back to neutral or to a different model.
- Benchmark code records failures as failures and excludes them from quality denominators only while reporting completion rate separately.
- No report uses causal language about agent responses.

## Official implementation references

- Gemini audio understanding and structured output: `https://ai.google.dev/gemini-api/docs/audio`
- pyannote community diarization: `https://huggingface.co/pyannote/speaker-diarization-community-1`
- faster-whisper model names and API: `https://github.com/SYSTRAN/faster-whisper`

---

## Task 1: Dataset manifests, split integrity, and gold-label schema

**Files:**
- Create: `backend/src/voxdelta/evaluation/manifest.py`
- Create: `backend/src/voxdelta/evaluation/gold.py`
- Create: `backend/scripts/build_aihub_manifests.py`
- Create: `backend/src/voxdelta/evaluation/aihub_fields.py`
- Create: `backend/tests/evaluation/test_manifest.py`
- Create: `backend/tests/evaluation/test_gold.py`
- Create: `data/README.md`

**Interfaces:**
- Produces: `DatasetItem`, `GoldUtteranceLabel`, `GoldTransitionLabel`, `load_manifest`, `validate_disjoint_splits`, and JSONL manifests.
- Consumes: user-provided AI Hub consultation-speech and emotion-dataset directories.

- [ ] **Step 1: Write failing schema and leakage tests**

```python
def test_split_validator_rejects_same_speaker_in_train_and_test() -> None:
    train = [DatasetItem(id="a", call_id="c1", speaker_id="s1", audio_path="a.wav", transcript="", split="train")]
    test = [DatasetItem(id="b", call_id="c2", speaker_id="s1", audio_path="b.wav", transcript="", split="test")]
    with pytest.raises(ValueError, match="speaker leakage: s1"):
        validate_disjoint_splits(train + test)


def test_gold_intensity_uses_five_point_rubric() -> None:
    with pytest.raises(ValidationError):
        GoldUtteranceLabel(item_id="u1", annotator="a1", operational_state="dissatisfied", negative_intensity=6)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd backend && uv run pytest tests/evaluation/test_manifest.py tests/evaluation/test_gold.py -v`
Expected: FAIL because evaluation schemas do not exist.

- [ ] **Step 3: Define exact schemas**

```python
class DatasetItem(BaseModel):
    id: str
    call_id: str
    speaker_id: str
    audio_path: str
    transcript: str
    split: Literal["train", "validation", "test"]
    emotion: EmotionLabel | None = None
    start: float | None = None
    end: float | None = None
    sha256: str


class GoldUtteranceLabel(BaseModel):
    item_id: str
    annotator: str
    operational_state: Literal["satisfied", "stable", "dissatisfied", "escalated", "uncertain"]
    negative_intensity: int = Field(ge=1, le=5)


class GoldTransitionLabel(BaseModel):
    previous_customer_id: str
    agent_id: str
    next_customer_id: str
    annotator: str
    classification: Literal["recovery", "stable", "worsening"]
    response_strategy: Literal["apology", "empathy", "clarification", "information", "solution", "policy_refusal", "greeting_closing", "other"]
```

- [ ] **Step 4: Implement deterministic manifest building**

The script accepts `--consultation-root`, `--emotion-root`, and `--output-root`. Recursively pair audio with adjacent JSON metadata by filename stem. Resolve canonical fields using these ordered aliases: ID=`id,wav_id,audio_id`; call=`call_id,conversation_id,dialogue_id`; speaker=`speaker_id,speaker,talker_id`; transcript=`transcript,text,sentence,발화문`; emotion=`emotion,emotion_label,label,감정`. Search nested JSON dictionaries depth-first, reject multiple differing matches for one canonical field, and emit a diagnostic listing available key paths when no alias matches. Normalize paths to absolute paths, compute SHA-256, sort by item ID, and split by stable SHA-256 of call ID into 80% train, 10% validation, and 10% test. Then assert that no speaker crosses splits. Abort on missing audio/transcript pairs, duplicate IDs, or emotion labels outside the seven-label mapping.

- [ ] **Step 5: Document data placement and licensing**

`data/README.md` must specify:

```text
data/raw/aihub/consultation/
data/raw/aihub/emotion/
data/manifests/
data/gold/
data/jobs/
data/models/
data/benchmarks/
```

State that raw data and derived audio clips are not redistributed, and include commands for manifest generation and hash validation.

- [ ] **Step 6: Run tests and commit**

Run: `cd backend && uv run pytest tests/evaluation/test_manifest.py tests/evaluation/test_gold.py -v && uv run ruff check . && uv run mypy src`
Expected: schema, leakage, sorting, and bad-pair tests pass.

```bash
git add backend/src/voxdelta/evaluation backend/scripts/build_aihub_manifests.py backend/tests/evaluation data/README.md
git commit -m "feat: define licensed dataset manifests"
```

## Task 2: Local pyannote diarization provider

**Files:**
- Create: `backend/src/voxdelta/providers/pyannote_diarization.py`
- Create: `backend/tests/providers/test_pyannote_diarization.py`
- Modify: `backend/pyproject.toml`

**Interfaces:**
- Implements: `DiarizationProvider.diarize(asset) -> list[SpeakerSegment]`.
- Produces provenance: `name="pyannote"`, `model="speaker-diarization-community-1"`, `remote=False`.

- [ ] **Step 1: Write a failing adapter test with a fake pipeline**

Inject a callable returning segments with two speakers and one overlap. For mixed audio, assert `min_speakers=1` and `max_speakers=4` are passed, exactly two detected labels normalize to `SPEAKER_00/01`, valid overlaps remain flagged, and exclusive diarization is used for transcript alignment output. A one- or three-speaker result raises `unsupported_speaker_count`. For separate-channel audio, assert each channel is processed with `num_speakers=1` and relabeled deterministically by channel.

- [ ] **Step 2: Run test to verify failure**

Run: `cd backend && uv run pytest tests/providers/test_pyannote_diarization.py -v`
Expected: FAIL because the adapter does not exist.

- [ ] **Step 3: Add and configure pyannote**

Run: `cd backend && uv add 'pyannote.audio>=4,<5'`

Implement lazy loading:

```python
self.pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=os.environ["HUGGINGFACE_TOKEN"],
)
output = self.pipeline(str(audio_path), min_speakers=1, max_speakers=4)
```

Use `output.speaker_diarization` for overlap-aware evidence and expose `output.exclusive_speaker_diarization` through an adapter helper for transcription alignment. Reject mixed-channel output unless exactly two speakers are detected. When `asset.channel_mode == "separate"`, run each normalized channel with `num_speakers=1`, relabel channel 0 as `SPEAKER_00` and channel 1 as `SPEAKER_01`, then merge by timestamp. Never initialize the model at module import time.

- [ ] **Step 4: Add credential and offline behavior tests**

Missing `HUGGINGFACE_TOKEN` during first download raises provider error code `missing_huggingface_token`. A configured local model path bypasses the token and sets provenance model to the resolved checkpoint directory name.

- [ ] **Step 5: Run tests and commit**

Run: `cd backend && uv run pytest tests/providers/test_pyannote_diarization.py -v && uv run ruff check . && uv run mypy src`
Expected: injected tests pass without downloading model weights.

```bash
git add backend/pyproject.toml backend/uv.lock backend/src/voxdelta/providers/pyannote_diarization.py backend/tests/providers/test_pyannote_diarization.py
git commit -m "feat: add local speaker diarization"
```

## Task 3: Local faster-whisper Korean transcription provider

**Files:**
- Create: `backend/src/voxdelta/providers/faster_whisper_asr.py`
- Create: `backend/tests/providers/test_faster_whisper_asr.py`
- Modify: `backend/pyproject.toml`

**Interfaces:**
- Implements: `TranscriptionProvider.transcribe(asset, segments) -> list[Utterance]`.
- Produces provenance: `name="faster-whisper"`, `model="large-v3-turbo"`, `remote=False`.

- [ ] **Step 1: Write failing timestamp-alignment tests**

Inject ASR words at 0.0–0.8, 1.0–1.5, and 2.0–2.8 seconds plus exclusive diarization. Assert words group into utterances by speaker, text joins without duplicate spaces, Korean language is forced, and words outside any speech segment are omitted with a warning count.

- [ ] **Step 2: Run test to verify failure**

Run: `cd backend && uv run pytest tests/providers/test_faster_whisper_asr.py -v`
Expected: FAIL because the ASR adapter does not exist.

- [ ] **Step 3: Add faster-whisper and implement lazy model loading**

Run: `cd backend && uv add 'faster-whisper>=1.1,<2'`

Use:

```python
WhisperModel(
    "large-v3-turbo",
    device=settings.whisper_device,
    compute_type=settings.whisper_compute_type,
)
```

Call `transcribe(language="ko", word_timestamps=True, vad_filter=False, beam_size=5)`. VAD is disabled here because pyannote segments are authoritative. For mixed mode, align by maximum temporal intersection and reject zero-overlap words. For separate mode, transcribe each channel independently and assign its fixed channel speaker before merging all words chronologically.

- [ ] **Step 4: Add MPS/CPU configuration behavior**

The default on this macOS project is `device="cpu"`, `compute_type="int8"`. CUDA users may set `VOXDELTA_WHISPER_DEVICE=cuda` and `VOXDELTA_WHISPER_COMPUTE_TYPE=float16`. Unsupported device values fail during settings validation, not model invocation.

- [ ] **Step 5: Run tests and commit**

Run: `cd backend && uv run pytest tests/providers/test_faster_whisper_asr.py -v && uv run ruff check . && uv run mypy src`
Expected: alignment/configuration tests pass without downloading weights.

```bash
git add backend
git commit -m "feat: add local Korean transcription"
```

## Task 4: Local seven-emotion Wav2Vec2-XLS-R baseline

**Files:**
- Create: `backend/src/voxdelta/providers/wav2vec_emotion.py`
- Create: `backend/src/voxdelta/evaluation/emotion_training.py`
- Create: `backend/scripts/train_emotion.py`
- Create: `backend/tests/providers/test_wav2vec_emotion.py`
- Create: `backend/tests/evaluation/test_emotion_training.py`
- Modify: `backend/pyproject.toml`

**Interfaces:**
- Implements: `EmotionProvider.analyze(utterance_id, audio_path, transcript) -> EmotionResult`.
- Produces checkpoint directory containing `config.json`, weights, `label_mapping.json`, `metrics.json`, and validation-set hash.

- [ ] **Step 1: Write failing label-order and inference tests**

Assert label order is exactly `[happiness, anger, disgust, fear, neutral, sadness, surprise]`, softmax sums to one, model logits map to the correct names, confidence is max probability, and operational mapping uses the shared analysis function.

- [ ] **Step 2: Run tests to verify failure**

Run: `cd backend && uv run pytest tests/providers/test_wav2vec_emotion.py tests/evaluation/test_emotion_training.py -v`
Expected: FAIL because local emotion modules do not exist.

- [ ] **Step 3: Add ML dependencies**

Run:

```bash
cd backend
uv add torch torchaudio 'transformers>=4.55,<5' 'datasets>=4,<5' 'accelerate>=1.10,<2' 'scikit-learn>=1.7,<2' 'pandas>=2.3,<3' joblib
```

- [ ] **Step 4: Implement dataset and collator**

Load 16 kHz mono arrays from the emotion manifest, use `AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")`, and pad with a collator that returns `input_values`, `attention_mask`, and integer labels. Reject clips under 0.5 seconds and crop clips over 20 seconds from the center during training only; evaluation uses deterministic non-overlapping 20-second windows with mean logits.

- [ ] **Step 5: Implement deterministic training CLI**

Use `AutoModelForAudioClassification.from_pretrained` with `num_labels=7`, exact `id2label/label2id`, seed 622, learning rate `2e-5`, train batch size 8, eval batch size 8, gradient accumulation 2, 10 epochs, warmup ratio 0.1, evaluation each epoch, macro-F1 model selection, and early stopping patience 2. The CLI is:

```bash
uv run python scripts/train_emotion.py \
  --manifest ../data/manifests/emotion.jsonl \
  --output ../data/models/wav2vec-xls-r-emotion \
  --base-model facebook/wav2vec2-xls-r-300m \
  --seed 622
```

- [ ] **Step 6: Implement inference adapter**

Lazy-load feature extractor/model, select CUDA then MPS then CPU when `device=auto`, wrap inference in `torch.inference_mode()`, aggregate long-clip window logits before softmax, and return canonical `EmotionResult` with measured `ProviderUsage.latency_ms`. Transcript is accepted by the interface but recorded as unused evidence for this acoustic baseline.

- [ ] **Step 7: Run unit and 20-example smoke training tests**

Run: `cd backend && uv run pytest tests/providers/test_wav2vec_emotion.py tests/evaluation/test_emotion_training.py -v`
Expected: mapping/inference tests pass and the smoke trainer writes all required checkpoint metadata.

- [ ] **Step 8: Commit local emotion baseline**

```bash
git add backend
git commit -m "feat: train local seven-emotion baseline"
```

## Task 5: Gemini frontier audio-emotion provider

**Files:**
- Create: `backend/src/voxdelta/providers/gemini_emotion.py`
- Create: `backend/tests/providers/test_gemini_emotion.py`
- Modify: `backend/pyproject.toml`

**Interfaces:**
- Implements: `EmotionProvider.analyze` with remote audio and text evidence.
- Produces provenance: `name="gemini"`, `model="gemini-3.6-flash"`, `remote=True`, `transmits=("audio", "text")`.

- [ ] **Step 1: Write failing structured-output tests with a fake GenAI client**

Assert the adapter uploads the audio slice, sends the Korean transcript and seven-label rubric, validates the response, records token/cost metadata, and deletes the remote file whether the request succeeds or raises. Invalid distributions and missing labels raise `provider_invalid_output`.

- [ ] **Step 2: Run test to verify failure**

Run: `cd backend && uv run pytest tests/providers/test_gemini_emotion.py -v`
Expected: FAIL because the Gemini adapter does not exist.

- [ ] **Step 3: Add SDK and define response schema**

Run: `cd backend && uv add 'google-genai>=1,<2'`

```python
class GeminiEmotionOutput(BaseModel):
    happiness: float = Field(ge=0, le=1)
    anger: float = Field(ge=0, le=1)
    disgust: float = Field(ge=0, le=1)
    fear: float = Field(ge=0, le=1)
    neutral: float = Field(ge=0, le=1)
    sadness: float = Field(ge=0, le=1)
    surprise: float = Field(ge=0, le=1)
    negative_intensity_raw: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
```

- [ ] **Step 4: Implement Gemini Interactions API call**

Initialize `genai.Client(api_key=os.environ["GEMINI_API_KEY"])`, upload clips larger than the inline threshold, and call model `gemini-3.6-flash` with temperature 0 and structured response format. The prompt states the exact Korean call-center context, labels, probability rules, uncertainty rule, and that transcript content and vocal expression are both evidence. Normalize probabilities only when their sum differs from 1 by at most 0.02; reject larger deviations.

Always execute remote file deletion in `finally`. Convert SDK rate limit, timeout, authentication, and safety-block exceptions into distinct provider error codes. Store latency, usage counts, calculated cost, and remote-file deletion status in `EmotionResult.usage`.

- [ ] **Step 5: Run adapter tests and an opt-in live smoke test**

Run: `cd backend && uv run pytest tests/providers/test_gemini_emotion.py -v`
Expected: fake-client tests pass without credentials.

When `GEMINI_API_KEY` is present, run a marked test: `uv run pytest -m live tests/providers/test_gemini_emotion.py -v`
Expected: one licensed/generated audio slice returns all seven probabilities and its remote file is deleted.

- [ ] **Step 6: Commit frontier provider**

```bash
git add backend
git commit -m "feat: add Gemini audio emotion provider"
```

## Task 6: Response-strategy and summary providers

**Files:**
- Create: `backend/src/voxdelta/providers/gemini_text.py`
- Create: `backend/tests/providers/test_gemini_text.py`

**Interfaces:**
- Implements: `ResponseStrategyProvider.classify` and `ReportSummaryProvider.summarize`.
- Consumes: transcript context and canonical report JSON; transmits text only.

- [ ] **Step 1: Write failing strategy and banned-copy tests**

Assert structured labels map to the eight strategies, multiple labels choose the highest-confidence primary, and summary output containing Korean or English causal banned phrases is rejected and regenerated once with a stricter prompt. A second violation returns a deterministic template summary instead of unsafe copy.

- [ ] **Step 2: Implement structured text calls**

Use `gemini-3.6-flash`, temperature 0, and Pydantic schemas. Send at most the previous customer, target agent, and next customer utterance for strategy classification. For report summary, send aggregate scores and selected triplets rather than full raw transcript. Provenance declares `transmits=("text",)`.

- [ ] **Step 3: Add deterministic local fallback template**

The fallback reports start/end operational states, peak negative interval, counts of recovery/worsening segments, and valid coverage. It uses “직후 변화와 연관된 응답” and never names an agent response as a cause.

- [ ] **Step 4: Run tests and commit**

Run: `cd backend && uv run pytest tests/providers/test_gemini_text.py -v && uv run ruff check . && uv run mypy src`
Expected: structured output, fallback, and banned-copy tests pass.

```bash
git add backend/src/voxdelta/providers/gemini_text.py backend/tests/providers/test_gemini_text.py
git commit -m "feat: classify agent response strategies"
```

## Task 7: Gold annotation, calibration, and agreement

**Files:**
- Create: `backend/src/voxdelta/evaluation/annotation.py`
- Create: `backend/src/voxdelta/evaluation/calibration.py`
- Create: `backend/scripts/build_gold_set.py`
- Create: `backend/tests/evaluation/test_annotation.py`
- Create: `backend/tests/evaluation/test_calibration.py`

**Interfaces:**
- Produces: adjudicated gold JSONL, Cohen's kappa, weighted kappa, `IntensityCalibrator.fit/predict/save/load`.
- Consumes: two-annotator label CSVs and provider raw intensity scores.

- [ ] **Step 1: Write failing agreement and calibrator tests**

Assert identical labels produce kappa 1.0, unresolved disagreements block gold export, intensity 1 maps to 0.0 and 5 maps to 1.0 before fitting, isotonic predictions stay within [0,1], and serialized calibrators reproduce predictions exactly.

- [ ] **Step 2: Implement adjudication workflow**

`build_gold_set.py` requires at least 200 utterances from 20 calls and 60 transition triplets, exactly two independent annotators per item, and an adjudication row for every disagreement. Export separate utterance and transition JSONL plus `agreement.json` containing counts, kappa values, and source hashes.

- [ ] **Step 3: Implement provider-specific calibration**

Use `IsotonicRegression(out_of_bounds="clip")` fit on validation only. Store provider/model, validation hash, sample count, and scikit-learn version alongside the joblib model. Refuse to load when provider/model metadata differs.

- [ ] **Step 4: Run tests and commit**

Run: `cd backend && uv run pytest tests/evaluation/test_annotation.py tests/evaluation/test_calibration.py -v`
Expected: agreement, adjudication, leakage, and serialization tests pass.

```bash
git add backend
git commit -m "feat: calibrate call emotion intensity"
```

## Task 8: Reproducible benchmark runner

**Files:**
- Create: `backend/src/voxdelta/evaluation/metrics.py`
- Create: `backend/src/voxdelta/evaluation/benchmark.py`
- Create: `backend/scripts/run_benchmark.py`
- Create: `backend/configs/providers/local.yaml`
- Create: `backend/configs/providers/gemini.yaml`
- Create: `backend/tests/evaluation/test_metrics.py`
- Create: `backend/tests/evaluation/test_benchmark.py`

**Interfaces:**
- Produces: versioned `BenchmarkSummary` and per-item JSONL consumed by `/api/comparisons/latest`.
- Consumes: gold test split, provider factories, calibrators, timing and usage metadata.

- [ ] **Step 1: Write failing metric tests**

Use hand-calculated fixtures to assert macro-F1, per-class recall, expected calibration error with 10 equal-width bins, intensity MAE, Spearman correlation, transition macro-F1, completion rate, mean/p95 latency, and cost per audio minute.

- [ ] **Step 2: Implement metrics without hidden filtering**

Require explicit `valid` flags. Quality metrics use completed valid predictions; summary includes total, valid, failed, and uncertain counts. Return `null` for metrics with no support instead of zero. Store confusion matrices with fixed label order.

- [ ] **Step 3: Implement benchmark CLI**

Create `backend/configs/providers/local.yaml`:

```yaml
diarization:
  provider: pyannote
  model: speaker-diarization-community-1
transcription:
  provider: faster-whisper
  model: large-v3-turbo
emotion:
  provider: wav2vec-xls-r
  checkpoint: ../data/models/wav2vec-xls-r-emotion
```

Create `backend/configs/providers/gemini.yaml`:

```yaml
diarization:
  provider: pyannote
  model: speaker-diarization-community-1
transcription:
  provider: faster-whisper
  model: large-v3-turbo
emotion:
  provider: gemini
  model: gemini-3.6-flash
  api_key_env: GEMINI_API_KEY
  input_usd_per_million_tokens: 1.50
  output_usd_per_million_tokens: 7.50
  pricing_checked_on: 2026-08-18
  pricing_url: https://ai.google.dev/gemini-api/docs/pricing
```

No credential value is stored in either file. The adapter reads pricing from this versioned configuration rather than hardcoding it; benchmark provenance records the rate and check date used for each calculation.

```bash
uv run python scripts/run_benchmark.py \
  --gold ../data/gold/adjudicated \
  --local-config configs/providers/local.yaml \
  --api-config configs/providers/gemini.yaml \
  --output ../data/benchmarks/latest
```

The runner hashes all input manifests and configurations, rejects non-test items, resumes per-item outputs after interruption, and writes `summary.v1.json`, `items.v1.jsonl`, `environment.json`, and `run.log` with transcripts redacted.

- [ ] **Step 4: Add threshold sensitivity**

Evaluate transition thresholds from 0.10 through 0.40 in 0.05 increments while keeping 0.20 as the canonical displayed result. Store the sensitivity curve in the summary.

- [ ] **Step 5: Run tests and commit**

Run: `cd backend && uv run pytest tests/evaluation/test_metrics.py tests/evaluation/test_benchmark.py -v && uv run ruff check . && uv run mypy src`
Expected: hand-calculated metrics and resumability tests pass.

```bash
git add backend
git commit -m "feat: benchmark local and frontier models"
```

## Task 9: Real-provider acceptance and model handoff

**Files:**
- Create: `backend/tests/integration/test_real_pipeline.py`
- Create: `backend/MODELS.md`
- Modify: `backend/README.md`
- Modify: `docs/superpowers/plans/2026-08-18-voxdelta-models-evaluation.md`

**Interfaces:**
- Consumes: licensed/generated Korean customer-agent call, all real providers, pipeline runner, dashboard comparison endpoint.
- Produces: documented model setup and a complete benchmark artifact.

- [ ] **Step 1: Add marked real-provider integration tests**

The local test runs pyannote, faster-whisper, and the trained Wav2Vec2 checkpoint on one generated call. The API test runs Gemini on the same customer slices. Both assert exactly two diarized speakers, confirmed customer/agent roles, seven emotion labels, valid calibrated intensity, at least one transition, and provenance metadata.

- [ ] **Step 2: Run local acceptance**

Run: `cd backend && uv run pytest -m "integration and not live" tests/integration/test_real_pipeline.py -v`
Expected: local pipeline completes without network after model cache warm-up.

- [ ] **Step 3: Run opt-in API acceptance**

Run: `cd backend && GEMINI_API_KEY=... uv run pytest -m "integration and live" tests/integration/test_real_pipeline.py -v`
Expected: Gemini results validate, usage/cost metadata is recorded, and uploaded remote files are deleted.

- [ ] **Step 4: Run benchmark and dashboard contract check**

Run the benchmark CLI, start the API, then `curl -s http://127.0.0.1:8765/api/comparisons/latest | jq -e '.providers | length == 2'`.
Expected: command exits 0 and both provider summaries use the same dataset hash.

- [ ] **Step 5: Document model setup and limitations**

`backend/MODELS.md` documents gated pyannote access, local checkpoint paths, Whisper device configuration, Gemini disclosure/deletion behavior, AI Hub licensing, model versions, calibration hashes, Korean/domain limitations, and commands for every benchmark.

- [ ] **Step 6: Run full repository verification**

```bash
cd backend && uv run pytest -q && uv run ruff check . && uv run ruff format --check . && uv run mypy src
cd ../frontend && npm test && npm run build && npx playwright test
cd .. && git diff --check && git status --short
```

Expected: all unmarked tests and static checks pass; worktree contains only documentation/checklist changes.

- [ ] **Step 7: Commit model/evaluation completion**

```bash
git add backend docs/superpowers/plans/2026-08-18-voxdelta-models-evaluation.md
git commit -m "docs: complete model evaluation handoff"
```
