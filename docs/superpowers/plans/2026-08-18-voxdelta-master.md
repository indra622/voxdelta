# VoxDelta MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local-first web application that analyzes recorded Korean calls containing exactly one customer and one agent, traces the customer's emotion, highlights recovery and worsening segments, and compares a local model with a frontier API model.

**Architecture:** A React/TypeScript frontend talks only to a local FastAPI backend. The backend owns a resumable stage pipeline, local SQLite metadata, versioned filesystem artifacts, and model-provider adapters; actual model integrations are added after a deterministic fake-provider vertical slice proves the contracts. The work is split into three independently reviewable plans so the core pipeline, user interface, and model/evaluation work can each be accepted or rejected on their own.

**Tech Stack:** Python 3.12, uv, FastAPI, Pydantic 2, SQLite, pytest, React, TypeScript, Vite, Vitest, Playwright, pyannote.audio `community-1` with optional `precision-2`, faster-whisper `large-v3-turbo`, Qwen3-ASR `1.7B` with `0.6B` forced alignment, Wav2Vec2-XLS-R 300M, emotion2vec+ large, Google GenAI SDK with `gemini-3.6-flash`, FFmpeg 8+

## Global Constraints

- Input is a recorded Korean customer-service call with exactly two primary speakers: one customer and one agent.
- MVP inputs are WAV, MP3, and M4A files between 1 and 60 minutes; the primary demo range is 5–15 minutes.
- The app runs locally; public deployment, accounts, multi-tenancy, and live streaming are out of scope.
- The user must be able to confirm or swap customer and agent roles before emotion analysis continues.
- Internal emotion output uses happiness, anger, disgust, fear, neutral, sadness, and surprise.
- Business-facing output uses satisfied, stable, dissatisfied, escalated, and uncertain states plus a calibrated negative-intensity score from 0.0 to 1.0.
- `delta <= -0.20` is recovery, `delta >= +0.20` is worsening, and all other valid transitions are stable.
- Remote audio processing is allowed only after the UI identifies the provider and transmitted content for that job.
- The report may state temporal association but must not claim that an agent response caused an emotion change.
- Raw datasets, call audio, credentials, generated reports, and provider payloads are never committed to Git.
- Every implementation task follows TDD and ends in a focused commit.
- Local model candidates run sequentially on the target Apple Silicon 24 GB machine; high-throughput batch serving is out of scope.
- A modern local provider never silently falls back when unavailable; the benchmark records the failure and keeps the stable profile selected.

---

## Plan sequence

1. [Core pipeline and local API](2026-08-18-voxdelta-core-pipeline.md)
   - Produces domain contracts, local storage, deterministic fake providers, resumable stages, role confirmation, transition logic, and a tested FastAPI vertical slice.
2. [Dashboard and reports](2026-08-18-voxdelta-dashboard.md)
   - Consumes the core API and produces upload/progress, role confirmation, synchronized analysis, highlighted transitions, comparison UI, and report export.
3. [Models, data, and evaluation](2026-08-18-voxdelta-models-evaluation.md)
   - Replaces fake providers with stable and modern pyannote/ASR/emotion candidates plus Gemini; adds AI Hub manifests, local model-selection gates, gold-label workflow, benchmarks, and final acceptance tests.

## Design-spec coverage

- Two-speaker validation, stereo-channel handling, audio normalization, role suggestion/confirmation, retry, deletion, artifact safety, confidence thresholds, and canonical report contracts are implemented in core Tasks 1–8.
- Upload disclosure, diagnostic-capture consent, stage progress, role correction, synchronized audio/transcript inspection, uncertainty display, transition evidence, provider comparison, and HTML/PDF export are implemented in dashboard Tasks 1–7.
- AI Hub ingestion, speaker-count rejection, overlap/low-confidence handling, pyannote Community/Precision, faster-whisper/Qwen3-ASR, XLS-R/emotion2vec+, Gemini adapters, calibration, annotator agreement, runtime/memory/cost metrics, and model-selection evaluation are implemented in model/evaluation Tasks 1–9.
- Causal claims remain prohibited in all user-facing copies; transitions are temporal triplets computed only from valid customer-agent-customer sequences.

## Locked repository structure

```text
voxdelta/
├── backend/
│   ├── pyproject.toml
│   ├── src/voxdelta/
│   │   ├── api/                 # FastAPI routes and request/response schemas
│   │   ├── audio/               # Validation, probing, normalization, slicing
│   │   ├── domain/              # Canonical data contracts and enums
│   │   ├── jobs/                # SQLite repository and artifact store
│   │   ├── pipeline/            # Stages, runner, cache, retry/invalidation
│   │   ├── providers/           # Provider protocols and concrete adapters
│   │   ├── analysis/            # State mapping, calibration, transitions
│   │   ├── reports/             # Shared report JSON and export rendering
│   │   └── evaluation/          # Metrics and benchmark runner
│   ├── tests/
│   └── scripts/                 # Dataset and training entry points
├── frontend/
│   ├── src/
│   │   ├── api/                 # Typed local API client
│   │   ├── features/jobs/       # Upload, progress, retry
│   │   ├── features/roles/      # Customer/agent confirmation
│   │   ├── features/analysis/   # Player, transcript, timeline, transitions
│   │   ├── features/compare/    # Local/API benchmark comparison
│   │   └── test/                # Shared browser-test utilities
│   └── tests/
├── data/                        # Gitignored runtime and dataset mount points
├── docs/superpowers/specs/
└── docs/superpowers/plans/
```

## Completion definition

The MVP is complete when all three subplans pass their documented checks, a licensed or generated 5–15 minute Korean customer-agent call completes end to end, the user can correct roles and retry a failed stage, stable and modern local candidates have reproducible selection results, the selected local and Gemini emotion providers are benchmarked on identical frozen inputs, the dashboard and exported report show the same transitions, and `git status --short` is empty.
