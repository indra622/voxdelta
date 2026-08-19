# Task 6 Report: Resumable Pipeline Runner and Role Gate

## Scope

- Base commit: `32aa2e4663889c33f50c2123116df562395a7ffc`
- Branch: `feature/core-pipeline`
- Focused commit message: `feat: add resumable analysis pipeline`
- No network, real model, credential, remote-provider, merge, or push operation was used.

Task 6 adds typed versioned stage artifacts, deterministic cache identities, resumable
fake-provider orchestration, an explicit role-confirmation pause, transactional retry
invalidation, structured redacted logging, and opt-in private diagnostics. The existing
repository and artifact store gained only the cache/reset/hash/exact-delete operations
needed by the runner.

## Contract Decisions

### Versioned artifacts and cache identity

- Every stage JSON is a Pydantic stage artifact with top-level `schema_version`,
  `cache_key`, ordered `upstream_hashes`, and `provider`. List-valued results are fields
  of typed envelopes rather than raw JSON arrays.
- Cache keys are SHA-256 over the stage name, ordered upstream artifact content hashes,
  provider name/model when present, and canonical JSON configuration using sorted keys
  and UTF-8 encoding. Non-SHA-256 upstream identifiers and non-finite/unsupported config
  values are rejected.
- Configuration keys containing `key`, `token`, or `authorization` are omitted
  recursively before hashing, so secret rotation neither enters nor perturbs cache
  identity.
- A completed stage is skipped only when its database path/cache reference, typed
  artifact, provider provenance, upstream hashes, recomputed cache key, and stage-level
  semantics validate. Missing, corrupt, mismatched, or role-bypass artifacts reset the
  selected stage and every downstream row before exact stage JSON deletion and
  recomputation.

### Role gate and stage composition

- Diarization and transcription finish before `confirm_roles` atomically publishes an
  unconfirmed candidate artifact and sets the stage/job to `paused`.
- Confirmation accepts exactly the two observed speaker IDs and exactly one customer
  plus one agent. Missing, extra, unknown, duplicate-role, and `unknown` role mappings
  are rejected while the candidate remains paused.
- Confirmed utterances are atomically published before the database row becomes
  completed. No emotion stage can execute from an unconfirmed or forged completed
  candidate.
- Emotion runs only for customer turns and is median-smoothed in deterministic customer
  chronology. Strategy classification runs only for agent turns in adjacent
  customer-agent-customer triples. Transition and report artifacts reuse those aligned
  typed results.

### Retry, concurrency, and failure safety

- Repository initialization adds nullable `cache_key` with a safe `ALTER TABLE` migration
  for existing databases. Multi-stage reset and stage claim operations use SQLite
  `BEGIN IMMEDIATE` transactions.
- Per-job reentrant locks make same-runner calls idempotent; transactional pending-stage
  claims prevent duplicate execution by multiple runner instances. A competing worker
  observes `running` rather than executing the provider again.
- `retry(stage)` resets the selected stage plus every downstream database reference,
  deletes only their exact `*.v1.json` files, and resumes to the next role pause or
  completion. Upstream artifacts, source uploads, normalized audio generations, logs,
  and diagnostics are untouched. Normalize retry reads the source but never deletes it.
- Known validation failures store stable public `{code, message}` data and return a
  failed job. Unexpected exceptions store only the exception class and a generic public
  message, set failed state, and re-raise. Handled failures never leave a stage running.

### Logging and diagnostics

- Each stage event is one JSONL object containing timestamp, job ID, stage, event,
  duration, provider, model, public error code, and bounded metadata.
- Redaction recursively replaces values under case-insensitive keys containing `key`,
  `token`, `authorization`, `transcript`, or `payload`, including dictionaries nested in
  lists. Unsupported and non-finite values serialize as a fixed marker; exception/user
  text is never passed to event records.
- Diagnostic writes require an explicit enabled flag, remain under the validated job's
  `diagnostics` directory, use sanitized UUID-suffixed names, redact credentials and
  payload text, publish atomically, and use mode `0600`. Fake providers create none.

## TDD Evidence

### Initial RED

The pipeline, logging, repository migration/reset, and artifact-boundary tests were
written before production implementation:

```text
uv run pytest tests/pipeline tests/jobs/test_logging.py \
  tests/jobs/test_repository.py tests/jobs/test_artifacts.py -q

ERROR tests/pipeline/test_runner.py
ModuleNotFoundError: No module named 'voxdelta.pipeline.runner'
ERROR tests/pipeline/test_stages.py
ModuleNotFoundError: No module named 'voxdelta.pipeline.stages'
ERROR tests/jobs/test_logging.py
ModuleNotFoundError: No module named 'voxdelta.jobs.logging'
```

After the first green pass, a narrower cache-boundary test was added and observed to
fail because arbitrary upstream strings were accepted:

```text
uv run pytest \
  tests/pipeline/test_stages.py::test_cache_key_rejects_non_sha256_upstream_identifiers -q

FAILED: DID NOT RAISE ValueError
```

### Focused GREEN

```text
uv run pytest tests/pipeline/test_runner.py -q
14 passed

uv run pytest tests/pipeline/test_stages.py -q
3 passed

uv run pytest tests/pipeline/test_stages.py tests/jobs/test_logging.py \
  tests/jobs/test_repository.py tests/jobs/test_artifacts.py -q
106 passed
```

The focused coverage includes end-to-end pause/resume/report completion, exact mapping
validation, role bypass, customer-only emotion, relevant-agent strategy alignment,
idempotency and concurrency, corrupt completed artifacts, retry boundaries, source and
normalized-file preservation, invalid retry input, safe known/unexpected failure state,
canonical secret-free cache keys, database migration/reset, exact artifact deletion,
recursive JSONL redaction, safe serialization, and diagnostic opt-in/mode/path behavior.

## Verification

```text
cd backend && uv run pytest -v
339 passed in 3.41s

cd backend && uv run ruff check .
All checks passed!

cd backend && uv run ruff format --check .
43 files already formatted

cd backend && uv run mypy src
Success: no issues found in 26 source files

git diff --check
exit code 0
```

The task is not merged or pushed; the worktree remains on `feature/core-pipeline`.
