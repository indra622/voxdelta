# VoxDelta Core Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a tested local FastAPI vertical slice that accepts a two-party call, checkpoints deterministic fake-model stages, pauses for customer/agent confirmation, computes emotion transitions, and returns a canonical report JSON.

**Architecture:** Canonical Pydantic contracts sit at the center. A SQLite job repository stores status while an atomic filesystem artifact store keeps versioned stage JSON; a stage runner calls provider protocols and invalidates only downstream results. FastAPI exposes the runner without containing business logic.

**Tech Stack:** Python 3.12, uv, FastAPI, Pydantic 2, pydantic-settings, SQLite, orjson, python-multipart, pytest, pytest-asyncio, HTTPX, Ruff, mypy, FFmpeg/ffprobe

## Global Constraints

- Input contains exactly one customer and one agent.
- Do not call real model APIs in this plan; deterministic fake providers prove the contracts first.
- Store runtime files under `data/jobs/<job_id>` and metadata in `data/voxdelta.sqlite3`.
- Never log credentials, full transcripts, or provider raw payloads.
- All JSON writes are atomic and all stage outputs include schema and model versions.
- A role-confirmation gate must stop downstream emotion stages until exactly one customer and one agent are assigned.
- Recovery is `delta <= -0.20`; worsening is `delta >= +0.20`.

---

## Task 1: Backend foundation and canonical domain contracts

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/src/voxdelta/__init__.py`
- Create: `backend/src/voxdelta/api/__init__.py`
- Create: `backend/src/voxdelta/analysis/__init__.py`
- Create: `backend/src/voxdelta/audio/__init__.py`
- Create: `backend/src/voxdelta/domain/__init__.py`
- Create: `backend/src/voxdelta/evaluation/__init__.py`
- Create: `backend/src/voxdelta/jobs/__init__.py`
- Create: `backend/src/voxdelta/pipeline/__init__.py`
- Create: `backend/src/voxdelta/providers/__init__.py`
- Create: `backend/src/voxdelta/reports/__init__.py`
- Create: `backend/src/voxdelta/config.py`
- Create: `backend/src/voxdelta/domain/models.py`
- Create: `backend/tests/domain/test_models.py`
- Create: `.gitignore`

**Interfaces:**
- Produces: `Settings`, `Role`, `StageName`, `StageStatus`, `ProviderProvenance`, `ProviderUsage`, `AudioAsset`, `SpeakerSegment`, `Utterance`, `EmotionResult`, `ResponseStrategyResult`, `EmotionTransition`, `CallSummary`, and `AnalysisReport`.
- Consumes: no earlier application code.

- [ ] **Step 1: Create the backend package and dependency manifest**

Create `backend/pyproject.toml`:

```toml
[project]
name = "voxdelta"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
  "fastapi>=0.116,<1",
  "httpx>=0.28,<1",
  "orjson>=3.10,<4",
  "pydantic>=2.11,<3",
  "pydantic-settings>=2.10,<3",
  "python-multipart>=0.0.20,<1",
  "uvicorn[standard]>=0.35,<1",
]

[dependency-groups]
dev = [
  "mypy>=1.17,<2",
  "pytest>=8.4,<9",
  "pytest-asyncio>=1.1,<2",
  "ruff>=0.12,<1",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/voxdelta"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.mypy]
python_version = "3.12"
strict = true
packages = ["voxdelta"]
```

Create `.gitignore` at the repository root:

```gitignore
.DS_Store
.env
.venv/
__pycache__/
.pytest_cache/
.mypy_cache/
.ruff_cache/
backend/.coverage
frontend/node_modules/
frontend/dist/
data/*
!data/.gitkeep
!data/README.md
*.wav
*.mp3
*.m4a
*.pdf
```

Run: `cd backend && uv sync --dev`
Expected: lockfile created and dependencies installed without resolution errors.

- [ ] **Step 2: Write failing contract tests**

Create `backend/tests/domain/test_models.py`:

```python
from pydantic import ValidationError
import pytest

from voxdelta.domain.models import EmotionResult, Role, Utterance


def test_utterance_rejects_non_positive_interval() -> None:
    with pytest.raises(ValidationError):
        Utterance(
            id="u1", start=2.0, end=1.0, speaker_id="SPEAKER_00",
            role=Role.CUSTOMER, transcript="안녕하세요",
        )


def test_emotion_probabilities_must_sum_to_one() -> None:
    with pytest.raises(ValidationError):
        EmotionResult(
            utterance_id="u1",
            probabilities={
                "happiness": 0.1, "anger": 0.1, "disgust": 0.1,
                "fear": 0.1, "neutral": 0.1, "sadness": 0.1,
                "surprise": 0.1,
            },
            operational_state="stable", negative_intensity=0.2,
            confidence=0.8, provider={"name": "fake", "model": "v1", "remote": False},
        )


def test_emotion_probabilities_must_be_bounded() -> None:
    with pytest.raises(ValidationError):
        EmotionResult(
            utterance_id="u1",
            probabilities={
                "happiness": 1.1, "anger": -0.1, "disgust": 0.0,
                "fear": 0.0, "neutral": 0.0, "sadness": 0.0,
                "surprise": 0.0,
            },
            operational_state="stable", negative_intensity=0.2,
            confidence=0.8, provider={"name": "fake", "model": "v1", "remote": False},
        )
```

- [ ] **Step 3: Run the tests to verify failure**

Run: `cd backend && uv run pytest tests/domain/test_models.py -v`
Expected: FAIL during import because `voxdelta.domain.models` does not exist.

- [ ] **Step 4: Implement settings and complete domain models**

Create `backend/src/voxdelta/config.py`:

```python
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    data_root: Path = Path("../data")
    database_path: Path = Path("../data/voxdelta.sqlite3")
    max_audio_seconds: int = 3600
    min_audio_seconds: int = 60
    model_config = SettingsConfigDict(env_prefix="VOXDELTA_", env_file=".env")
```

Create `backend/src/voxdelta/domain/models.py`:

```python
from enum import StrEnum
from typing import Literal
from pydantic import BaseModel, Field, model_validator

EmotionLabel = Literal["happiness", "anger", "disgust", "fear", "neutral", "sadness", "surprise"]
OperationalState = Literal["satisfied", "stable", "dissatisfied", "escalated", "uncertain"]
TransitionClass = Literal["recovery", "stable", "worsening"]


class Role(StrEnum):
    CUSTOMER = "customer"
    AGENT = "agent"
    UNKNOWN = "unknown"


class StageName(StrEnum):
    NORMALIZE = "normalize"
    DIARIZE = "diarize"
    TRANSCRIBE = "transcribe"
    CONFIRM_ROLES = "confirm_roles"
    EMOTION = "emotion"
    RESPONSE_STRATEGY = "response_strategy"
    TRANSITIONS = "transitions"
    REPORT = "report"


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ProviderProvenance(BaseModel):
    name: str
    model: str
    remote: bool
    transmits: tuple[Literal["audio", "text", "features"], ...] = ()
    retention_policy_url: str | None = None
    schema_version: str = "1"


class ProviderUsage(BaseModel):
    latency_ms: float = Field(ge=0)
    input_units: int | None = Field(default=None, ge=0)
    output_units: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    remote_file_deleted: bool | None = None


class AudioAsset(BaseModel):
    source_name: str
    source_path: str
    normalized_paths: tuple[str, ...] = ()
    channel_mode: Literal["mixed", "separate"] | None = None
    duration_seconds: float | None = None
    channels: int | None = None
    sha256: str


class SpeakerSegment(BaseModel):
    start: float = Field(ge=0)
    end: float
    speaker_id: str
    overlap: bool = False
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def valid_interval(self) -> "SpeakerSegment":
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self


class Utterance(SpeakerSegment):
    id: str
    role: Role = Role.UNKNOWN
    transcript: str


class EmotionResult(BaseModel):
    utterance_id: str
    probabilities: dict[EmotionLabel, float]
    operational_state: OperationalState
    negative_intensity: float = Field(ge=0, le=1)
    smoothed_negative_intensity: float | None = Field(default=None, ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    provider: ProviderProvenance
    usage: ProviderUsage | None = None

    @model_validator(mode="after")
    def valid_distribution(self) -> "EmotionResult":
        expected = {"happiness", "anger", "disgust", "fear", "neutral", "sadness", "surprise"}
        if set(self.probabilities) != expected:
            raise ValueError("all seven emotion labels are required")
        if any(value < 0 or value > 1 for value in self.probabilities.values()):
            raise ValueError("emotion probabilities must be between zero and one")
        if abs(sum(self.probabilities.values()) - 1.0) > 1e-6:
            raise ValueError("emotion probabilities must sum to one")
        return self


class ResponseStrategyResult(BaseModel):
    utterance_id: str
    primary: Literal["apology", "empathy", "clarification", "information", "solution", "policy_refusal", "greeting_closing", "other"]
    secondary: tuple[str, ...] = ()
    confidence: float = Field(ge=0, le=1)
    provider: ProviderProvenance


class EmotionTransition(BaseModel):
    previous_customer_id: str
    agent_id: str
    next_customer_id: str
    delta: float = Field(ge=-1, le=1)
    classification: TransitionClass


class CallSummary(BaseModel):
    start_state: OperationalState
    end_state: OperationalState
    peak_customer_utterance_id: str
    overall_delta: float = Field(ge=-1, le=1)
    valid_coverage: float = Field(ge=0, le=1)
    recovery_count: int = Field(ge=0)
    worsening_count: int = Field(ge=0)
    narrative: str | None = None


class AnalysisReport(BaseModel):
    job_id: str
    summary: CallSummary
    utterances: list[Utterance]
    emotions: list[EmotionResult]
    strategies: list[ResponseStrategyResult]
    transitions: list[EmotionTransition]
    warnings: list[str] = Field(default_factory=list)
    schema_version: str = "1"
```

Create every empty package `__init__.py` listed in this task so later task imports work without implicit namespace assumptions.

- [ ] **Step 5: Run contract, lint, and type checks**

Run: `cd backend && uv run pytest tests/domain/test_models.py -v && uv run ruff check . && uv run mypy src`
Expected: 3 tests pass; Ruff and mypy exit 0.

- [ ] **Step 6: Commit the foundation**

```bash
git add .gitignore backend
git commit -m "feat: define backend domain contracts"
```

## Task 2: Atomic artifact store and SQLite job repository

**Files:**
- Create: `backend/src/voxdelta/jobs/artifacts.py`
- Create: `backend/src/voxdelta/jobs/repository.py`
- Create: `backend/tests/jobs/test_artifacts.py`
- Create: `backend/tests/jobs/test_repository.py`

**Interfaces:**
- Consumes: `StageName`, `StageStatus`.
- Produces: `ArtifactStore.write_model(job_id, stage, value) -> Path`, `ArtifactStore.read_model(job_id, stage, model_type)`, `ArtifactStore.delete_job(job_id)`, `JobRepository.create_job(source_name, diagnostic_capture=False) -> str`, `JobRepository.set_stage(...)`, `JobRepository.get_job(job_id)`, and `JobRepository.delete_job(job_id)`.

- [ ] **Step 1: Write failing persistence tests**

Create tests that assert atomic JSON round-tripping and persistent stage state:

```python
from pathlib import Path
from voxdelta.domain.models import AnalysisReport, StageName, StageStatus
from voxdelta.jobs.artifacts import ArtifactStore
from voxdelta.jobs.repository import JobRepository


def test_artifact_round_trip(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "jobs")
    report = AnalysisReport(
        job_id="j1",
        summary={"start_state": "stable", "end_state": "stable", "peak_customer_utterance_id": "u1", "overall_delta": 0.0, "valid_coverage": 1.0, "recovery_count": 0, "worsening_count": 0},
        utterances=[], emotions=[], strategies=[], transitions=[],
    )
    path = store.write_model("j1", StageName.REPORT, report)
    assert path.name == "report.v1.json"
    assert store.read_model("j1", StageName.REPORT, AnalysisReport) == report


def test_job_stage_survives_repository_reopen(tmp_path: Path) -> None:
    db = tmp_path / "voxdelta.sqlite3"
    repo = JobRepository(db)
    job_id = repo.create_job("sample.wav")
    repo.set_stage(job_id, StageName.NORMALIZE, StageStatus.COMPLETED, "normalize.v1.json")
    reopened = JobRepository(db)
    assert reopened.get_job(job_id)["stages"]["normalize"]["status"] == "completed"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd backend && uv run pytest tests/jobs -v`
Expected: FAIL because `voxdelta.jobs` modules do not exist.

- [ ] **Step 3: Implement atomic artifacts**

Create `backend/src/voxdelta/jobs/artifacts.py`:

```python
from pathlib import Path
from typing import TypeVar
import os
import orjson
from pydantic import BaseModel
from voxdelta.domain.models import StageName

T = TypeVar("T", bound=BaseModel)


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def job_dir(self, job_id: str) -> Path:
        path = self.root / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_model(self, job_id: str, stage: StageName, value: BaseModel) -> Path:
        target = self.job_dir(job_id) / f"{stage.value}.v1.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(orjson.dumps(value.model_dump(mode="json"), option=orjson.OPT_INDENT_2))
        os.replace(temporary, target)
        return target

    def read_model(self, job_id: str, stage: StageName, model_type: type[T]) -> T:
        path = self.job_dir(job_id) / f"{stage.value}.v1.json"
        return model_type.model_validate_json(path.read_bytes())
```

- [ ] **Step 4: Implement the SQLite repository**

Create `backend/src/voxdelta/jobs/repository.py` with schema initialization and parameterized queries:

```python
from datetime import UTC, datetime
from pathlib import Path
import json, sqlite3, uuid
from voxdelta.domain.models import StageName, StageStatus


class JobRepository:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY, source_name TEXT NOT NULL,
              status TEXT NOT NULL, diagnostic_capture INTEGER NOT NULL DEFAULT 0,
              created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS stages (
              job_id TEXT NOT NULL, stage TEXT NOT NULL, status TEXT NOT NULL,
              artifact_path TEXT, error_json TEXT,
              PRIMARY KEY (job_id, stage), FOREIGN KEY (job_id) REFERENCES jobs(id)
            );
            """)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def create_job(self, source_name: str, diagnostic_capture: bool = False) -> str:
        job_id, now = uuid.uuid4().hex, datetime.now(UTC).isoformat()
        with self._connect() as db:
            db.execute("INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?)", (job_id, source_name, "pending", int(diagnostic_capture), now, now))
            db.executemany(
                "INSERT INTO stages(job_id, stage, status) VALUES (?, ?, ?)",
                [(job_id, stage.value, StageStatus.PENDING.value) for stage in StageName],
            )
        return job_id

    def set_stage(self, job_id: str, stage: StageName, status: StageStatus, artifact_path: str | None = None, error: dict[str, str] | None = None) -> None:
        now = datetime.now(UTC).isoformat()
        with self._connect() as db:
            db.execute("UPDATE stages SET status=?, artifact_path=?, error_json=? WHERE job_id=? AND stage=?", (status.value, artifact_path, json.dumps(error) if error else None, job_id, stage.value))
            db.execute("UPDATE jobs SET updated_at=? WHERE id=?", (now, job_id))

    def get_job(self, job_id: str) -> dict[str, object]:
        with self._connect() as db:
            job = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if job is None:
                raise KeyError(job_id)
            stages = db.execute("SELECT * FROM stages WHERE job_id=? ORDER BY rowid", (job_id,)).fetchall()
        return {**dict(job), "stages": {row["stage"]: dict(row) for row in stages}}
```

Add `delete_job` methods that reject IDs containing path separators, delete database rows in a transaction, and remove only the resolved `<jobs_root>/<job_id>` directory after verifying its parent is exactly the configured jobs root. Tests must prove deleting `j1` cannot delete sibling `j10` or the jobs root.

- [ ] **Step 5: Run persistence tests**

Run: `cd backend && uv run pytest tests/jobs -v && uv run ruff check . && uv run mypy src`
Expected: all persistence tests pass; static checks exit 0.

- [ ] **Step 6: Commit persistence**

```bash
git add backend/src/voxdelta/jobs backend/tests/jobs
git commit -m "feat: persist jobs and stage artifacts"
```

## Task 3: Audio validation and normalization boundary

**Files:**
- Create: `backend/src/voxdelta/audio/service.py`
- Create: `backend/tests/audio/test_service.py`
- Create: `backend/tests/fixtures/synthetic_65s.wav`
- Create: `backend/tests/fixtures/stereo_split_65s.wav`

**Interfaces:**
- Produces: `AudioService.ingest(upload_path: Path, job_id: str, channel_preference: Literal["auto", "mixed", "separate"] = "auto") -> AudioAsset`.
- Consumes: `Settings`, `ArtifactStore.job_dir`.

- [ ] **Step 1: Generate a deterministic licensed-free fixture**

Run:

```bash
mkdir -p backend/tests/fixtures
ffmpeg -f lavfi -i "sine=frequency=440:duration=65" -ar 16000 -ac 1 backend/tests/fixtures/synthetic_65s.wav
ffmpeg -f lavfi -i "sine=frequency=440:duration=32.5" -f lavfi -i "anullsrc=r=16000:cl=mono:d=32.5" -f lavfi -i "anullsrc=r=16000:cl=mono:d=32.5" -f lavfi -i "sine=frequency=660:duration=32.5" -filter_complex "[0:a][1:a]concat=n=2:v=0:a=1[left];[2:a][3:a]concat=n=2:v=0:a=1[right];[left][right]amerge=inputs=2" -ar 16000 -ac 2 backend/tests/fixtures/stereo_split_65s.wav
```

Expected: a 16 kHz mono WAV lasting 65 seconds.

- [ ] **Step 2: Write failing validation tests**

```python
from pathlib import Path
import pytest
from voxdelta.audio.service import AudioRejected, AudioService


def test_ingest_normalizes_supported_audio(tmp_path: Path) -> None:
    service = AudioService(tmp_path / "jobs", min_seconds=60, max_seconds=3600)
    asset = service.ingest(Path("tests/fixtures/synthetic_65s.wav"), "j1")
    assert asset.duration_seconds == pytest.approx(65, abs=0.2)
    assert asset.channels == 1
    assert asset.channel_mode == "mixed"
    assert all(Path(path).exists() for path in asset.normalized_paths)


def test_ingest_prefers_distinct_stereo_channels(tmp_path: Path) -> None:
    service = AudioService(tmp_path / "jobs", min_seconds=60, max_seconds=3600)
    asset = service.ingest(Path("tests/fixtures/stereo_split_65s.wav"), "j1")
    assert asset.channel_mode == "separate"
    assert len(asset.normalized_paths) == 2


def test_ingest_rejects_unsupported_suffix(tmp_path: Path) -> None:
    source = tmp_path / "call.txt"
    source.write_text("not audio")
    with pytest.raises(AudioRejected, match="unsupported extension"):
        AudioService(tmp_path / "jobs", 60, 3600).ingest(source, "j1")
```

- [ ] **Step 3: Run the tests to verify failure**

Run: `cd backend && uv run pytest tests/audio/test_service.py -v`
Expected: FAIL because `voxdelta.audio.service` does not exist.

- [ ] **Step 4: Implement ffprobe validation and ffmpeg normalization**

Create `AudioService` that allows `.wav`, `.mp3`, and `.m4a`, probes JSON with `ffprobe -v error -show_streams -show_format -of json`, enforces duration bounds, computes SHA-256, and always produces a 16 kHz mono mix. For stereo input, also extract left/right mono candidates with FFmpeg `channelsplit`. Read the PCM candidates with the standard-library `wave` and `array` modules; `auto` selects `separate` only when each channel RMS exceeds 500 and absolute Pearson correlation is below 0.85, otherwise it selects `mixed`. Explicit `mixed` or `separate` preferences override the heuristic after validating the requested channels exist. The mono mix command is:

```python
subprocess.run([
    "ffmpeg", "-y", "-i", str(source), "-vn", "-ar", "16000", "-ac", "1",
    "-c:a", "pcm_s16le", str(normalized),
], check=True, capture_output=True)
```

Return an `AudioAsset` containing the original channel count, selected channel mode, and one or two normalized paths. Raise `AudioRejected` with one of `unsupported extension`, `audio is not decodable`, `audio is shorter than 60 seconds`, `audio exceeds 3600 seconds`, or `separate channels requested for mono audio`; never include ffmpeg stderr in the public exception message.

- [ ] **Step 5: Run audio tests and checks**

Run: `cd backend && uv run pytest tests/audio -v && uv run ruff check . && uv run mypy src`
Expected: 3 tests pass and static checks exit 0.

- [ ] **Step 6: Commit audio ingestion**

```bash
git add backend/src/voxdelta/audio backend/tests/audio backend/tests/fixtures
git commit -m "feat: validate and normalize call audio"
```

## Task 4: Provider protocols and deterministic fake providers

**Files:**
- Create: `backend/src/voxdelta/providers/base.py`
- Create: `backend/src/voxdelta/providers/fake.py`
- Create: `backend/tests/providers/test_fake.py`

**Interfaces:**
- Produces: `DiarizationProvider.diarize`, `TranscriptionProvider.transcribe`, `EmotionProvider.analyze`, `ResponseStrategyProvider.classify`, `ReportSummaryProvider.summarize`.
- Consumes: canonical domain models from Task 1.

- [ ] **Step 1: Write failing provider contract tests**

```python
from pathlib import Path
from voxdelta.domain.models import AudioAsset
from voxdelta.providers.fake import FakeDiarizationProvider, FakeEmotionProvider


def test_fake_diarizer_returns_exactly_two_speakers() -> None:
    asset = AudioAsset(source_name="call.wav", source_path="call.wav", normalized_paths=("call.wav",), channel_mode="mixed", duration_seconds=65, channels=1, sha256="0" * 64)
    segments = FakeDiarizationProvider().diarize(asset)
    assert {segment.speaker_id for segment in segments} == {"SPEAKER_00", "SPEAKER_01"}


def test_fake_emotion_is_deterministic() -> None:
    provider = FakeEmotionProvider()
    first = provider.analyze("u1", Path("slice.wav"), "정말 화가 납니다")
    second = provider.analyze("u1", Path("slice.wav"), "정말 화가 납니다")
    assert first == second
    assert first.provider.remote is False
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd backend && uv run pytest tests/providers/test_fake.py -v`
Expected: FAIL because provider modules do not exist.

- [ ] **Step 3: Define protocols**

Use `typing.Protocol` and exact signatures:

```python
class DiarizationProvider(Protocol):
    provenance: ProviderProvenance
    def diarize(self, asset: AudioAsset) -> list[SpeakerSegment]: ...

class TranscriptionProvider(Protocol):
    provenance: ProviderProvenance
    def transcribe(self, asset: AudioAsset, segments: list[SpeakerSegment]) -> list[Utterance]: ...

class EmotionProvider(Protocol):
    provenance: ProviderProvenance
    def analyze(self, utterance_id: str, audio_path: Path, transcript: str) -> EmotionResult: ...

class ResponseStrategyProvider(Protocol):
    provenance: ProviderProvenance
    def classify(self, utterance: Utterance, context: list[Utterance]) -> ResponseStrategyResult: ...

class ReportSummaryProvider(Protocol):
    provenance: ProviderProvenance
    def summarize(self, report: AnalysisReport) -> str: ...
```

- [ ] **Step 4: Implement fake providers**

Implement fixed alternating segments and transcript lines. Hash `utterance_id + transcript` with SHA-256, convert the first eight bytes to a deterministic seed, generate seven positive values, normalize to one, derive `negative_intensity`, and select the operational state. Fake response strategy uses transcript keywords for apology, solution, and policy refusal and otherwise returns `information`.

- [ ] **Step 5: Run provider tests**

Run: `cd backend && uv run pytest tests/providers -v && uv run ruff check . && uv run mypy src`
Expected: all provider tests pass and static checks exit 0.

- [ ] **Step 6: Commit provider contracts**

```bash
git add backend/src/voxdelta/providers backend/tests/providers
git commit -m "feat: define model provider boundaries"
```

## Task 5: Operational states and emotion-transition analysis

**Files:**
- Create: `backend/src/voxdelta/analysis/emotions.py`
- Create: `backend/src/voxdelta/analysis/transitions.py`
- Create: `backend/src/voxdelta/analysis/roles.py`
- Create: `backend/src/voxdelta/analysis/summary.py`
- Create: `backend/tests/analysis/test_emotions.py`
- Create: `backend/tests/analysis/test_transitions.py`
- Create: `backend/tests/analysis/test_roles.py`
- Create: `backend/tests/analysis/test_summary.py`

**Interfaces:**
- Produces: `map_operational_state(probabilities, confidence)`, `median_smooth(results)`, `suggest_roles(utterances) -> dict[str, Role] | None`, `build_transitions(utterances, emotions) -> list[EmotionTransition]`, and `build_call_summary(customer_utterances, emotions, transitions) -> CallSummary`.
- Consumes: `Utterance`, `EmotionResult`, `EmotionTransition`, and confirmed `Role` values.

- [ ] **Step 1: Write failing state-mapping and threshold tests**

```python
def test_dominant_surprise_becomes_uncertain() -> None:
    probabilities = {"happiness": .05, "anger": .05, "disgust": .05, "fear": .05, "neutral": .1, "sadness": .05, "surprise": .65}
    assert map_operational_state(probabilities, confidence=.9) == "uncertain"


def test_transition_thresholds_are_inclusive() -> None:
    assert classify_delta(-.20) == "recovery"
    assert classify_delta(.20) == "worsening"
    assert classify_delta(.19) == "stable"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd backend && uv run pytest tests/analysis -v`
Expected: FAIL because analysis modules do not exist.

- [ ] **Step 3: Implement deterministic mapping and smoothing**

Implement `map_operational_state` with confidence threshold `0.55`. Aggregate sadness, disgust, and fear for `dissatisfied`; use happiness, neutral, and anger for the other named states. When surprise is the largest single probability or confidence is below `0.55`, return `uncertain`. Implement a centered median over the current customer result and its nearest previous/next customer results, preserving edge windows of two values.

- [ ] **Step 4: Implement triplet construction**

Walk utterances by time and accept only adjacent `customer -> agent -> customer` triples with emotion results on both customer turns. Calculate the smoothed next-minus-previous intensity, round to four decimals, classify with inclusive ±0.20 thresholds, and return transitions. Skip triples containing `unknown` roles rather than inferring them.

Implement `suggest_roles` by scoring utterances in the first 30 seconds. Add one agent point for each distinct cue among `상담원`, `고객센터`, `무엇을 도와`, `도와드리`, and an opening self-introduction ending in `입니다`. Return a proposal only when exactly two speakers exist and one speaker leads by at least one point; otherwise return `None`. Role confirmation remains mandatory even when a proposal exists.

Implement `build_call_summary` from chronologically ordered valid customer results. Use the first and last operational state, the maximum smoothed intensity as the peak, last-minus-first intensity as overall delta, valid emotion count divided by customer utterance count as coverage, and transition-class counts. Raise `InsufficientEmotionCoverage` when fewer than three valid results or coverage is below 0.50.

- [ ] **Step 5: Run analysis tests and checks**

Run: `cd backend && uv run pytest tests/analysis -v && uv run ruff check . && uv run mypy src`
Expected: state mapping, smoothing, triplet, and boundary tests pass.

- [ ] **Step 6: Commit analysis logic**

```bash
git add backend/src/voxdelta/analysis backend/tests/analysis
git commit -m "feat: calculate customer emotion transitions"
```

## Task 6: Resumable pipeline runner and role gate

**Files:**
- Create: `backend/src/voxdelta/pipeline/runner.py`
- Create: `backend/src/voxdelta/pipeline/stages.py`
- Create: `backend/src/voxdelta/jobs/logging.py`
- Create: `backend/tests/pipeline/test_runner.py`

**Interfaces:**
- Produces: `PipelineRunner.run_until_pause(job_id)`, `PipelineRunner.confirm_roles(job_id, mapping)`, and `PipelineRunner.retry(job_id, stage)`.
- Consumes: artifact store, repository, audio service, providers, and analysis functions.

- [ ] **Step 1: Write failing pause, resume, and invalidation tests**

Create a test that seeds a normalized fixture, runs the fake pipeline, asserts `confirm_roles` is `paused`, confirms `{"SPEAKER_00": "customer", "SPEAKER_01": "agent"}`, resumes to a completed report, then retries `transcribe` and asserts downstream emotion/report artifacts are removed while normalize/diarize remain.

- [ ] **Step 2: Run the pipeline test to verify failure**

Run: `cd backend && uv run pytest tests/pipeline/test_runner.py -v`
Expected: FAIL because `PipelineRunner` does not exist.

- [ ] **Step 3: Define ordered stages and dependencies**

In `stages.py`, define the exact order from `NORMALIZE` through `REPORT` and a downstream map. `CONFIRM_ROLES` is a pause stage. Cache keys are SHA-256 of stage name, upstream artifact checksums, provider name/model, and a canonical JSON configuration object.

- [ ] **Step 4: Implement the runner**

For every stage: mark running, calculate or load its cache key, call the stage handler, atomically write the artifact, and mark completed. On a known validation error mark failed with `{code, message}`. On an unexpected exception record only the exception class plus a generic message and re-raise for local logs. At the role stage, persist candidate utterances and mark paused until exactly one customer and one agent mapping is supplied.

Write one JSON log line per stage event with job ID, stage, event, duration, provider/model, and public error code. Recursively replace values of keys matching `key`, `token`, `authorization`, `transcript`, and `payload` with `[REDACTED]`. When and only when `diagnostic_capture` is true, write provider request/response diagnostics to `<job>/diagnostics/` with mode `0600`; never write credentials even in diagnostic mode.

- [ ] **Step 5: Implement retry invalidation**

Delete only artifact files and database references for the selected stage and every downstream stage. Reset those stage rows to pending. Never delete the source upload when retrying. A retry of `normalize` may overwrite the normalized copy but not the original uploaded file.

- [ ] **Step 6: Run pipeline tests and all backend tests**

Run: `cd backend && uv run pytest -v && uv run ruff check . && uv run mypy src`
Expected: all tests pass; Ruff and mypy exit 0.

- [ ] **Step 7: Commit the pipeline**

```bash
git add backend/src/voxdelta/pipeline backend/tests/pipeline
git commit -m "feat: add resumable analysis pipeline"
```

## Task 7: FastAPI vertical slice

**Files:**
- Create: `backend/src/voxdelta/api/app.py`
- Create: `backend/src/voxdelta/api/dependencies.py`
- Create: `backend/src/voxdelta/api/schemas.py`
- Create: `backend/tests/api/test_jobs.py`

**Interfaces:**
- Produces: `GET /api/config/providers`, `POST /api/jobs`, `GET /api/jobs/{job_id}`, `GET /api/jobs/{job_id}/audio`, `DELETE /api/jobs/{job_id}`, `POST /api/jobs/{job_id}/roles`, `POST /api/jobs/{job_id}/retry`, and `GET /api/jobs/{job_id}/report`.
- Consumes: `PipelineRunner`, `JobRepository`, `ArtifactStore`, and canonical report models.

- [ ] **Step 1: Write failing API tests**

Use `httpx.AsyncClient` with ASGI transport. Test provider config lists each stage's provenance, transmitted content, and retention-policy URL; multipart upload returns HTTP 202 and a job ID; the `diagnostic_capture=false` form value persists; job status eventually pauses at role confirmation; an invalid role mapping returns 422; the valid mapping resumes the fake pipeline; report returns canonical JSON; deletion removes database/artifacts and returns 204; and a missing job returns 404 without a stack trace. Also test that the audio route returns normalized audio, honors `Range: bytes=0-1023` with HTTP 206 and `Content-Range`, returns 416 for an unsatisfiable range, and never exposes an absolute filesystem path in headers or JSON.

- [ ] **Step 2: Run tests to verify failure**

Run: `cd backend && uv run pytest tests/api/test_jobs.py -v`
Expected: FAIL because the API app does not exist.

- [ ] **Step 3: Define request and response schemas**

```python
class RoleConfirmation(BaseModel):
    mapping: dict[str, Role]

    @model_validator(mode="after")
    def one_customer_one_agent(self) -> "RoleConfirmation":
        if sorted(self.mapping.values()) != [Role.AGENT, Role.CUSTOMER]:
            raise ValueError("mapping must contain one customer and one agent")
        return self


class RetryRequest(BaseModel):
    stage: StageName
```

- [ ] **Step 4: Implement routes and local background execution**

Save uploads with a generated job ID before scheduling `PipelineRunner.run_until_pause` through FastAPI `BackgroundTasks`. Return 202 with `{job_id, status_url}`. Keep handlers thin: translate `KeyError` to 404, `AudioRejected` to 422, and role/pipeline state conflicts to 409. Return report JSON from the artifact store only when report stage is completed.

Implement `GET /api/jobs/{job_id}/audio` as a `StreamingResponse` over the job's normalized mixed preview. Parse a single byte range, respond with `Accept-Ranges: bytes`, `Content-Length`, and `Content-Range` when partial, and stream in bounded chunks. Resolve the path exclusively through `ArtifactStore`; reject traversal, never accept a client-supplied path, and never include the host path in a response.

Before report completion require at least three valid customer emotion results and at least 50% valid coverage across customer turns. Otherwise fail the report stage with public code `insufficient_emotion_coverage`. `DELETE` calls the repository and artifact-store guarded deletion methods and returns 204.

- [ ] **Step 5: Run the full backend verification**

Run:

```bash
cd backend
uv run pytest -v
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run uvicorn voxdelta.api.app:app --host 127.0.0.1 --port 8765
```

Expected: tests/static checks pass; Uvicorn starts at `http://127.0.0.1:8765`; `GET /docs` returns 200. Stop the server with Ctrl-C after the smoke check.

- [ ] **Step 6: Commit the vertical slice**

```bash
git add backend/src/voxdelta/api backend/tests/api backend/uv.lock
git commit -m "feat: expose local analysis job API"
```

## Task 8: Core-plan acceptance and documentation

**Files:**
- Create: `backend/README.md`
- Create: `data/.gitkeep`
- Modify: `docs/superpowers/plans/2026-08-18-voxdelta-core-pipeline.md`

**Interfaces:**
- Produces: reproducible setup and smoke-test instructions for the dashboard plan.
- Consumes: all core-plan commands and endpoints.

- [ ] **Step 1: Write the backend README**

Document Python 3.12 installation with `uv python install 3.12`, `uv sync --dev`, FFmpeg check, environment variables, `uv run uvicorn`, test commands, every core endpoint, runtime file locations, and the fake-provider limitation. State that no real audio leaves the machine in the core vertical slice.

- [ ] **Step 2: Run an end-to-end smoke test**

Start the API, upload `backend/tests/fixtures/synthetic_65s.wav`, poll status, confirm roles, fetch the audio with a byte-range request, fetch the report, and assert with `jq` that `.transitions` exists and `.utterances | length > 0`.

- [ ] **Step 3: Run final checks**

Run: `cd backend && uv run pytest -q && uv run ruff check . && uv run ruff format --check . && uv run mypy src && cd .. && git diff --check && git status --short`
Expected: all checks pass; Git shows only README, `.gitkeep`, and plan checkbox changes.

- [ ] **Step 4: Mark completed plan checkboxes and commit**

```bash
git add backend/README.md data/.gitkeep docs/superpowers/plans/2026-08-18-voxdelta-core-pipeline.md
git commit -m "docs: complete core pipeline handoff"
```
