# VoxDelta Models, Data, and Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace deterministic fake providers with reproducible stable, modern local, and frontier implementations, build licensed-data manifests and a small gold set, and select defaults through benchmarks of quality, latency, memory, cost, calibration, and failures.

**Architecture:** pyannote `speaker-diarization-community-1`, faster-whisper `large-v3-turbo`, and Wav2Vec2-XLS-R form stable local baselines. Qwen3-ASR `1.7B` plus its `0.6B` forced aligner and emotion2vec+ large form the modern local profile, while pyannote `precision-2` is an optional remote diarization benchmark and Gemini `gemini-3.6-flash` is the frontier emotion path. All providers adapt to canonical contracts; benchmark code freezes upstream artifacts for fair component comparisons and selects a local default without loading candidates concurrently.

**Tech Stack:** Python 3.12, PyTorch, torchaudio, pyannote.audio 4.0.7, faster-whisper 1.2.1, qwen-asr 0.0.6, FunASR 1.4.2, psutil 7.2, Transformers, Datasets, Accelerate, scikit-learn, pandas, jiwer, pyannote.metrics, google-genai, Pydantic, pytest

## Global Constraints

- AI Hub data is downloaded manually under the user's account and remains outside Git.
- Every call and speaker belongs to exactly one of train, validation, or test.
- The local/API comparison uses identical gold examples and records dataset hashes and model versions.
- Local providers must work without network access after gated models and checkpoints are downloaded.
- Stable and modern local candidates run sequentially; no benchmark keeps both ASR models or both emotion encoders resident at once.
- The target Apple Silicon 24 GB profile rejects a model that exceeds 18 GiB peak resident memory or fails with unsupported MPS operations.
- An unavailable modern provider is recorded as unavailable and never triggers an implicit fallback.
- Precision-2 raw-audio transmission requires per-job disclosure and `PYANNOTEAI_API_KEY`; it is optional and never selected by an offline run.
- Gemini raw-audio transmission requires per-job disclosure and `GEMINI_API_KEY`; uploaded remote files are deleted in a `finally` block.
- API structured output must contain all seven emotion probabilities and they must sum to one after validation.
- Provider failure never silently falls back to neutral or to a different model.
- Benchmark code records failures as failures and excludes them from quality denominators only while reporting completion rate separately.
- No report uses causal language about agent responses.

## Official implementation references

- Gemini audio understanding and structured output: `https://ai.google.dev/gemini-api/docs/audio`
- pyannote community diarization: `https://huggingface.co/pyannote/speaker-diarization-community-1`
- faster-whisper model names and API: `https://github.com/SYSTRAN/faster-whisper`
- pyannote Community/Precision comparison: `https://github.com/pyannote/pyannote-audio`
- Qwen3-ASR and forced alignment: `https://github.com/QwenLM/Qwen3-ASR`
- emotion2vec+ checkpoints and labels: `https://github.com/ddlBoJack/emotion2vec`

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

## Task 2: Local and optional frontier pyannote diarization

**Files:**
- Create: `backend/src/voxdelta/providers/pyannote_diarization.py`
- Create: `backend/src/voxdelta/providers/pyannote_precision.py`
- Create: `backend/tests/providers/test_pyannote_diarization.py`
- Create: `backend/tests/providers/test_pyannote_precision.py`
- Modify: `backend/pyproject.toml`

**Interfaces:**
- Both implement: `DiarizationProvider.diarize(asset) -> list[SpeakerSegment]`.
- Local provenance: `name="pyannote"`, `model="speaker-diarization-community-1"`, `remote=False`.
- Optional provenance: `name="pyannote"`, `model="speaker-diarization-precision-2"`, `remote=True`, `transmits=("audio",)`.

- [ ] **Step 1: Write a failing adapter test with a fake pipeline**

Inject a callable returning segments with two speakers and one overlap. For mixed audio, assert `min_speakers=1` and `max_speakers=4` are passed, exactly two detected labels normalize to `SPEAKER_00/01`, valid overlaps remain flagged, and exclusive diarization is used for transcript alignment output. A one- or three-speaker result raises `unsupported_speaker_count`. For separate-channel audio, assert each channel is processed with `num_speakers=1` and relabeled deterministically by channel.

- [ ] **Step 2: Run test to verify failure**

Run: `cd backend && uv run pytest tests/providers/test_pyannote_diarization.py -v`
Expected: FAIL because the adapter does not exist.

- [ ] **Step 3: Add and configure pyannote**

Run: `cd backend && uv add 'pyannote.audio>=4.0.7,<5'`

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

- [ ] **Step 5: Write the failing optional Precision-2 contract test**

Inject a fake pyannoteAI pipeline and assert the adapter selects `speaker-diarization-precision-2`, declares remote audio transmission and its retention URL, normalizes exactly two labels, and rejects one or three labels with `unsupported_speaker_count`. Missing `PYANNOTEAI_API_KEY` raises `missing_pyannote_api_key` before reading audio. A service timeout raises `provider_timeout` and does not invoke Community-1 automatically.

- [ ] **Step 6: Implement the optional Precision-2 adapter**

Load `Pipeline.from_pretrained("pyannote/speaker-diarization-precision-2", token=os.environ["PYANNOTEAI_API_KEY"])` lazily only when the explicit `pyannote_precision` profile is selected. Preserve remote provenance and the configured retention-policy URL. Do not expose this profile as a fallback from Community-1 and do not use it in offline tests.

- [ ] **Step 7: Run tests and commit**

Run: `cd backend && uv run pytest tests/providers/test_pyannote_diarization.py tests/providers/test_pyannote_precision.py -v && uv run ruff check . && uv run mypy src`
Expected: injected local and remote tests pass without downloading weights or transmitting audio.

```bash
git add backend/pyproject.toml backend/uv.lock backend/src/voxdelta/providers/pyannote_diarization.py backend/src/voxdelta/providers/pyannote_precision.py backend/tests/providers/test_pyannote_diarization.py backend/tests/providers/test_pyannote_precision.py
git commit -m "feat: add selectable speaker diarization providers"
```

## Task 3: Stable and modern Korean transcription providers

**Files:**
- Create: `backend/src/voxdelta/providers/faster_whisper_asr.py`
- Create: `backend/src/voxdelta/providers/qwen3_asr.py`
- Create: `backend/src/voxdelta/evaluation/selection.py`
- Create: `backend/tests/providers/test_faster_whisper_asr.py`
- Create: `backend/tests/providers/test_qwen3_asr.py`
- Create: `backend/tests/evaluation/test_selection.py`
- Modify: `backend/pyproject.toml`

**Interfaces:**
- Both implement: `TranscriptionProvider.transcribe(asset, segments) -> list[Utterance]`.
- Stable provenance: `name="faster-whisper"`, `model="large-v3-turbo"`, `remote=False`.
- Modern provenance: `name="qwen3-asr"`, `model="Qwen3-ASR-1.7B"`, `remote=False`; the explicit low-memory profile uses `Qwen3-ASR-0.6B`.
- Produces: `CandidateMetrics`, `SelectionDecision`, `select_asr_candidate`, and `select_emotion_candidate` for later benchmark tasks.

- [ ] **Step 1: Write failing timestamp-alignment tests**

For faster-whisper, inject ASR words at 0.0–0.8, 1.0–1.5, and 2.0–2.8 seconds plus exclusive diarization. Assert words group into utterances by speaker, text joins without duplicate spaces, Korean is forced, and words outside speech are omitted with a warning count. For Qwen3-ASR, inject a fake model and forced aligner; assert model IDs are `Qwen/Qwen3-ASR-1.7B` and `Qwen/Qwen3-ForcedAligner-0.6B`, `language="Korean"` and timestamps are requested, timestamped tokens map through the same speaker-alignment helper, and the 0.6B ASR profile is only selected explicitly. In `test_selection.py`, add hand-calculated cases proving Qwen wins at a 0.01 CER improvement, loses a near-tie when slower than 2.0x, and is unavailable above 18,432 MiB or below 95% completion.

- [ ] **Step 2: Run test to verify failure**

Run: `cd backend && uv run pytest tests/providers/test_faster_whisper_asr.py tests/providers/test_qwen3_asr.py -v`
Expected: FAIL because the ASR adapters do not exist.

- [ ] **Step 3: Add faster-whisper and implement lazy model loading**

Run: `cd backend && uv add 'faster-whisper>=1.2.1,<2'`

Use:

```python
WhisperModel(
    "large-v3-turbo",
    device=settings.whisper_device,
    compute_type=settings.whisper_compute_type,
)
```

Call `transcribe(language="ko", word_timestamps=True, vad_filter=False, beam_size=5)`. VAD is disabled here because pyannote segments are authoritative. For mixed mode, align by maximum temporal intersection and reject zero-overlap words. For separate mode, transcribe each channel independently and assign its fixed channel speaker before merging all words chronologically.

- [ ] **Step 4: Add Qwen3-ASR and implement lazy model loading**

Run: `cd backend && uv add 'qwen-asr>=0.0.6,<0.1'`

Create `Qwen3AsrProvider` with constructor-injected model factories for tests. The default profile loads `Qwen/Qwen3-ASR-1.7B` and `Qwen/Qwen3-ForcedAligner-0.6B`; the low-memory profile changes only the ASR ID to `Qwen/Qwen3-ASR-0.6B`. Request Korean transcription and timestamps, convert aligner units to the shared word structure, and reuse the exact mixed/separate speaker assignment rules from faster-whisper. Load nothing at module import time.

- [ ] **Step 5: Add explicit Apple Silicon and CPU behavior**

For faster-whisper the target macOS profile remains `device="cpu"`, `compute_type="int8"`; CUDA may use `float16`. For Qwen3-ASR, `device="auto"` chooses CUDA/bfloat16, then MPS/float16, then CPU/float32. Before downloading weights, validate the configured model ID and device. During the first opt-in smoke run, convert an unsupported MPS operator or memory allocation failure to `provider_runtime_unsupported`; record it as candidate unavailability and never invoke faster-whisper from inside the Qwen adapter. Limit local provider concurrency to one.

- [ ] **Step 6: Add the ASR selection gate**

Implement `select_asr_candidate(candidates: list[CandidateMetrics]) -> SelectionDecision` in `evaluation/selection.py`. Select among completed ASR candidates only after at least 95% completion. Select Qwen3-ASR when Korean CER improves by at least 0.01 absolute. When the CER difference is below 0.01, select Qwen3-ASR only if its CER is no worse and median latency is no more than 2.0 times faster-whisper; otherwise keep faster-whisper. Reject either result if peak resident memory exceeds 18 GiB and persist the reason in `candidate_status`.

Define the shared selection contracts exactly:

```python
class CandidateMetrics(BaseModel):
    candidate_id: str
    task: Literal["diarization", "asr", "emotion"]
    provider: str
    model: str
    completion_rate: float = Field(ge=0, le=1)
    median_latency_ms: float | None = Field(default=None, ge=0)
    peak_rss_mb: float | None = Field(default=None, ge=0)
    der: float | None = Field(default=None, ge=0)
    cer: float | None = Field(default=None, ge=0)
    macro_f1: float | None = Field(default=None, ge=0, le=1)
    expected_calibration_error: float | None = Field(default=None, ge=0, le=1)
    unavailable_reason: str | None = None


class SelectionDecision(BaseModel):
    selected_candidate_id: str | None
    selected_provider: str | None
    selected_model: str | None
    candidate_status: dict[str, Literal["eligible", "unavailable", "rejected"]]
    reasons: dict[str, str]
```

- [ ] **Step 7: Run tests and commit**

Run: `cd backend && uv run pytest tests/providers/test_faster_whisper_asr.py tests/providers/test_qwen3_asr.py tests/evaluation/test_selection.py -v && uv run ruff check . && uv run mypy src`
Expected: alignment, explicit-device, unavailable-candidate, and selection-gate tests pass without downloading weights.

```bash
git add backend
git commit -m "feat: compare local Korean transcription providers"
```

## Task 4: Stable and modern local seven-emotion providers

**Files:**
- Create: `backend/src/voxdelta/providers/wav2vec_emotion.py`
- Create: `backend/src/voxdelta/providers/emotion2vec_emotion.py`
- Create: `backend/src/voxdelta/evaluation/emotion_training.py`
- Create: `backend/scripts/train_emotion.py`
- Create: `backend/tests/providers/test_wav2vec_emotion.py`
- Create: `backend/tests/providers/test_emotion2vec_emotion.py`
- Create: `backend/tests/evaluation/test_emotion_training.py`
- Modify: `backend/src/voxdelta/evaluation/selection.py`
- Modify: `backend/tests/evaluation/test_selection.py`
- Modify: `backend/pyproject.toml`

**Interfaces:**
- Both implement: `EmotionProvider.analyze(utterance_id, audio_path, transcript) -> EmotionResult`.
- Produces separate XLS-R baseline and emotion2vec+ classifier checkpoint directories containing `config.json`, weights, `label_mapping.json`, `metrics.json`, and validation-set hash.

- [ ] **Step 1: Write failing label-order and inference tests**

For both adapters assert label order is exactly `[happiness, anger, disgust, fear, neutral, sadness, surprise]`, softmax sums to one, logits map to the correct names, confidence is max probability, and operational mapping uses the shared analysis function. For emotion2vec+ inject a fake encoder, assert `iic/emotion2vec_plus_large` is selected, only the seven-class head is trainable in the default profile, and the encoder is not initialized at import time. Extend `test_selection.py` with cases proving emotion2vec+ wins at a 0.01 macro-F1 improvement, loses when ECE regresses by more than 0.02, uses lower ECE then lower median latency inside the ±0.01 band, and is unavailable above 18,432 MiB or below 95% completion.

- [ ] **Step 2: Run tests to verify failure**

Run: `cd backend && uv run pytest tests/providers/test_wav2vec_emotion.py tests/providers/test_emotion2vec_emotion.py tests/evaluation/test_emotion_training.py -v`
Expected: FAIL because local emotion modules do not exist.

- [ ] **Step 3: Add ML dependencies**

Run:

```bash
cd backend
uv add torch torchaudio 'transformers>=4.55,<5' 'datasets>=4,<5' 'accelerate>=1.10,<2' 'scikit-learn>=1.7,<2' 'pandas>=2.3,<3' joblib
uv add 'funasr>=1.4.2,<2'
```

- [ ] **Step 4: Implement dataset and collator**

Load 16 kHz mono arrays from the emotion manifest. The XLS-R path uses `AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")`; the emotion2vec+ path extracts one utterance embedding from `iic/emotion2vec_plus_large` through an injected encoder interface. Reject clips under 0.5 seconds and crop clips over 20 seconds from the center during training only; evaluation uses deterministic non-overlapping 20-second windows and mean logits. Cache embeddings under a directory keyed by audio SHA-256, encoder ID, and preprocessing version.

- [ ] **Step 5: Implement the deterministic XLS-R baseline training profile**

Use `AutoModelForAudioClassification.from_pretrained` with `num_labels=7`, exact `id2label/label2id`, seed 622, learning rate `2e-5`, train batch size 8, eval batch size 8, gradient accumulation 2, 10 epochs, warmup ratio 0.1, evaluation each epoch, macro-F1 model selection, and early stopping patience 2. The CLI is:

```bash
uv run python scripts/train_emotion.py \
  --manifest ../data/manifests/emotion.jsonl \
  --output ../data/models/wav2vec-xls-r-emotion \
  --base-model facebook/wav2vec2-xls-r-300m \
  --seed 622
```

- [ ] **Step 6: Implement the emotion2vec+ large training profile**

Add `--architecture emotion2vec-plus --base-model iic/emotion2vec_plus_large --freeze-encoder`. Train a two-layer head `embedding -> LayerNorm -> Linear(hidden, 256) -> GELU -> Dropout(0.1) -> Linear(256, 7)` for 20 epochs with AdamW at `1e-3`, batch size 64, early stopping patience 3, macro-F1 selection, and seed 622. Save the encoder ID and immutable encoder hash beside the head. Do not remap the pretrained nine-class scores into seven labels; learn the seven-class head from AI Hub labels.

```bash
uv run python scripts/train_emotion.py \
  --manifest ../data/manifests/emotion.jsonl \
  --output ../data/models/emotion2vec-plus-large-emotion \
  --architecture emotion2vec-plus \
  --base-model iic/emotion2vec_plus_large \
  --freeze-encoder \
  --seed 622
```

- [ ] **Step 7: Implement both inference adapters**

Lazy-load each feature extractor/encoder and classifier, select CUDA then MPS then CPU when `device=auto`, wrap inference in `torch.inference_mode()`, aggregate long-clip window logits before softmax, and return canonical `EmotionResult` with latency and peak resident-memory metadata. Enforce concurrency one and unload the previous candidate before loading the next benchmark candidate. Transcript is accepted by the interface but recorded as unused acoustic evidence.

- [ ] **Step 8: Add the emotion selection gate and run smoke tests**

Add `select_emotion_candidate(candidates: list[CandidateMetrics]) -> SelectionDecision` to `evaluation/selection.py`. Require 95% completion and reject a candidate whose expected calibration error is worse than the alternative by more than 0.02. Select emotion2vec+ when its macro-F1 improves by at least 0.01. Within ±0.01 macro-F1, select lower expected calibration error, then lower median latency. Reject any candidate above 18 GiB peak resident memory and persist the failure reason.

Run: `cd backend && uv run pytest tests/providers/test_wav2vec_emotion.py tests/providers/test_emotion2vec_emotion.py tests/evaluation/test_emotion_training.py tests/evaluation/test_selection.py -v`
Expected: both mapping/inference paths, 20-example smoke trainers, memory accounting, and selection-gate tests pass without downloading weights in unit tests.

- [ ] **Step 9: Commit the local emotion portfolio**

```bash
git add backend
git commit -m "feat: compare local seven-emotion providers"
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
- Create: `backend/configs/providers/stable-local.yaml`
- Create: `backend/configs/providers/modern-local.yaml`
- Create: `backend/configs/providers/gemini-emotion.yaml`
- Create: `backend/configs/providers/precision-diarization.yaml`
- Create: `backend/tests/evaluation/test_metrics.py`
- Create: `backend/tests/evaluation/test_benchmark.py`

**Interfaces:**
- Produces: versioned `BenchmarkSummary`, `selection.v1.json`, frozen upstream artifacts, and per-item JSONL consumed by `/api/comparisons/latest`.
- Consumes: gold test split, provider factories, `CandidateMetrics`, selection functions, calibrators, timing, peak-memory, and usage metadata.

- [ ] **Step 1: Write failing metric tests**

Use hand-calculated fixtures to assert macro-F1, per-class recall, expected calibration error with 10 equal-width bins, intensity MAE, Spearman correlation, transition macro-F1, completion rate, mean/p50/p95 latency, peak resident memory, and cost per audio minute. Assert the public comparison contains exactly the selected local emotion candidate and Gemini even though baseline candidate metrics remain in run metadata, and that mismatched `frozen_input_hash` values fail validation.

- [ ] **Step 2: Implement metrics without hidden filtering**

Require explicit `valid` flags. Quality metrics use completed valid predictions; summary includes total, valid, failed, and uncertain counts. Return `null` for metrics with no support instead of zero. Store confusion matrices with fixed label order.

Define the public benchmark contracts in `evaluation/benchmark.py`:

```python
class ProviderBenchmark(BaseModel):
    provider: str
    model: str
    frozen_input_hash: str
    macro_f1: float = Field(ge=0, le=1)
    expected_calibration_error: float = Field(ge=0, le=1)
    mean_latency_ms: float = Field(ge=0)
    median_latency_ms: float = Field(ge=0)
    p95_latency_ms: float = Field(ge=0)
    peak_rss_mb: float | None = Field(default=None, ge=0)
    cost_per_audio_minute_usd: float | None = Field(default=None, ge=0)
    failed_count: int = Field(ge=0)
    uncertain_rate: float = Field(ge=0, le=1)


class BenchmarkCandidate(CandidateMetrics):
    status: Literal["eligible", "unavailable", "rejected"]
    selection_reason: str


class BenchmarkSummary(BaseModel):
    providers: tuple[ProviderBenchmark, ProviderBenchmark]
    candidates: list[BenchmarkCandidate]
    frozen_input_hash: str
    dataset_hash: str
    schema_version: str = "1"

    @model_validator(mode="after")
    def matching_frozen_inputs(self) -> "BenchmarkSummary":
        if any(item.frozen_input_hash != self.frozen_input_hash for item in self.providers):
            raise ValueError("public providers must share frozen inputs")
        return self
```

- [ ] **Step 3: Implement benchmark CLI**

Run: `cd backend && uv add 'psutil>=7.2.2,<8'`

Create `backend/configs/providers/stable-local.yaml`:

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
runtime:
  concurrency: 1
  max_peak_rss_mb: 18432
```

Create `backend/configs/providers/modern-local.yaml`:

```yaml
diarization:
  provider: pyannote
  model: speaker-diarization-community-1
transcription:
  provider: qwen3-asr
  model: Qwen/Qwen3-ASR-1.7B
  forced_aligner: Qwen/Qwen3-ForcedAligner-0.6B
emotion:
  provider: emotion2vec-plus
  encoder: iic/emotion2vec_plus_large
  checkpoint: ../data/models/emotion2vec-plus-large-emotion
runtime:
  concurrency: 1
  max_peak_rss_mb: 18432
```

Create `backend/configs/providers/gemini-emotion.yaml`:

```yaml
emotion:
  provider: gemini
  model: gemini-3.6-flash
  api_key_env: GEMINI_API_KEY
  input_usd_per_million_tokens: 1.50
  output_usd_per_million_tokens: 7.50
  pricing_checked_on: 2026-08-18
  pricing_url: https://ai.google.dev/gemini-api/docs/pricing
```

Create `backend/configs/providers/precision-diarization.yaml`:

```yaml
diarization:
  provider: pyannote_precision
  model: speaker-diarization-precision-2
  api_key_env: PYANNOTEAI_API_KEY
  retention_policy_url: https://docs.pyannote.ai/data-retention
  optional: true
```

No credential value is stored in any configuration file. The adapter reads pricing from this versioned configuration rather than hardcoding it; benchmark provenance records the rate and check date used for each calculation.

```bash
uv run python scripts/run_benchmark.py \
  --gold ../data/gold/adjudicated \
  --stable-config configs/providers/stable-local.yaml \
  --modern-config configs/providers/modern-local.yaml \
  --api-config configs/providers/gemini-emotion.yaml \
  --optional-diarization-config configs/providers/precision-diarization.yaml \
  --output ../data/benchmarks/latest
```

The runner hashes all input manifests and configurations and rejects non-test items. Run each candidate in a fresh spawned worker process so only one large local model is resident. The parent polls the worker and all descendants through `psutil.Process(pid).memory_info().rss` and records the maximum summed RSS in MiB. First benchmark Community-1 and optionally Precision-2. Then benchmark both ASR candidates against the same gold segments, write the ASR selection decision, and freeze `speaker_segments.v1.json`, `utterances.v1.json`, and customer audio-slice hashes. Finally benchmark XLS-R, emotion2vec+, and Gemini against those identical frozen utterances and slices, then write the emotion selection decision.

The public `BenchmarkSummary.providers` contains only the selected local emotion provider and Gemini, with the same `frozen_input_hash`. `BenchmarkSummary.candidates` retains every stable/modern/optional result and unavailable reason. The runner resumes per-item outputs after interruption and writes `summary.v1.json`, `selection.v1.json`, `items.v1.jsonl`, `environment.json`, and `run.log` with transcripts redacted.

- [ ] **Step 4: Add threshold sensitivity**

Evaluate transition thresholds from 0.10 through 0.40 in 0.05 increments while keeping 0.20 as the canonical displayed result. Store the sensitivity curve in the summary.

- [ ] **Step 5: Run tests and commit**

Run: `cd backend && uv run pytest tests/evaluation/test_metrics.py tests/evaluation/test_benchmark.py -v && uv run ruff check . && uv run mypy src`
Expected: hand-calculated metrics, selection, frozen-input fairness, sequential-worker memory, and resumability tests pass.

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

The stable local test runs Community-1, faster-whisper, and the trained XLS-R checkpoint on one generated call. The modern local test runs Community-1, Qwen3-ASR 1.7B with forced alignment, and the emotion2vec+ classifier on the same call. The Gemini test consumes the exact frozen customer audio slices and transcript artifact selected by the local benchmark. All assert exactly two diarized speakers, confirmed customer/agent roles, seven emotion labels, valid calibrated intensity, at least one transition, provenance, latency, and peak-memory metadata. An optional Precision-2 live test compares diarization only and never changes the local default implicitly.

- [ ] **Step 2: Run local acceptance**

Run: `cd backend && uv run pytest -m "integration and not live" tests/integration/test_real_pipeline.py -v`
Expected: both stable and modern local pipelines complete sequentially without network after model caches warm up, or the modern profile records an explicit `provider_runtime_unsupported` decision while the stable path completes.

- [ ] **Step 3: Run opt-in API acceptance**

Run Gemini: `cd backend && GEMINI_API_KEY=... uv run pytest -m "integration and live and gemini" tests/integration/test_real_pipeline.py -v`
Expected: Gemini results validate on frozen inputs, usage/cost metadata is recorded, and uploaded remote files are deleted.

Run optional diarization: `cd backend && PYANNOTEAI_API_KEY=... uv run pytest -m "integration and live and pyannote_precision" tests/integration/test_real_pipeline.py -v`
Expected: Precision-2 returns exactly two speakers or a documented unsupported-speaker result, and remote provenance includes the data-retention URL.

- [ ] **Step 4: Run benchmark and dashboard contract check**

Run the benchmark CLI, assert `selection.v1.json` contains explicit ASR and emotion decisions, start the API, then `curl -s http://127.0.0.1:8765/api/comparisons/latest | jq -e '(.providers | length == 2) and (.candidates | length >= 4)'`.
Expected: commands exit 0; the public providers are the selected local emotion model and Gemini with the same frozen-input hash, while baseline and modern candidate metrics remain available.

- [ ] **Step 5: Document model setup and limitations**

`backend/MODELS.md` documents gated pyannote access, Community-1 versus optional Precision-2, faster-whisper and Qwen3-ASR profiles, Qwen forced alignment, XLS-R and emotion2vec+ checkpoints, Apple Silicon device behavior, sequential loading, 18 GiB memory gate, Gemini disclosure/deletion behavior, AI Hub licensing, model versions, calibration hashes, selection reasons, Korean/domain limitations, and commands for every benchmark.

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
