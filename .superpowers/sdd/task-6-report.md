# Task 6 Report: Resumable Pipeline Runner and Role Gate

## Scope

- Base commit: `32aa2e4663889c33f50c2123116df562395a7ffc`
- Branch: `feature/core-pipeline`
- Initial implementation: `3da5ce3efc9ec2a9514ce4a71e63440a1121a8de`
- Review hardening: `ceb1e702d0a3290bfbcc9bf97dba606a2d6e45fa`
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
- Credential-bearing configuration fields are rejected rather than silently omitted.
  CamelCase, punctuation, and case are normalized before exact credential-name checks,
  while ordinary behavior fields such as `monkey_count` and `tokenizer` remain part of
  the cache identity.
- A completed stage is skipped only when its database path/cache reference, typed
  artifact, recorded artifact digest, provider provenance, upstream hashes, recomputed
  cache key, and complete stage semantics validate. Missing, corrupt, reordered,
  misaligned, provenance-mismatched, or role-bypass artifacts reset the selected stage
  and every downstream row before exact stage JSON deletion and recomputation.
- Runner configuration is a strict Pydantic model. The only currently supported override
  is normalize channel preference; unknown or unimplemented behavior-changing options are
  rejected instead of being accepted but ignored.

### Role gate and stage composition

- Diarization and transcription finish before `confirm_roles` atomically publishes an
  unconfirmed candidate artifact and sets the stage/job to `paused`.
- Confirmation accepts exactly the two observed speaker IDs and exactly one customer
  plus one agent. Missing, extra, unknown, duplicate-role, and `unknown` role mappings
  are rejected while the candidate remains paused.
- Confirmation revalidates the candidate digest and the transcription artifact, then
  reconstructs confirmed utterances from transcription rather than trusting candidate
  text. A persisted `role_confirmed` marker is required before downstream execution, and
  administrative `set_stage` calls cannot manufacture paused/completed role rows.
- Confirmed utterances are fenced and atomically published before the database row becomes
  completed. No emotion stage can execute from an unconfirmed or forged completed
  candidate.
- Emotion runs only for customer turns and is median-smoothed in deterministic customer
  chronology. Each provider request receives a private, exact-frame mono PCM WAV slice
  for that customer utterance; temporary slices are mode `0600` and are removed after
  synchronous provider return or failure. Strategy classification runs only for agent
  turns in adjacent customer-agent-customer triples. Transition and report artifacts
  reuse those aligned typed results.

### Retry, concurrency, and failure safety

- Repository initialization serializes additive schema migration under SQLite
  `BEGIN IMMEDIATE` and rechecks columns while holding the lock. Multi-stage reset,
  claim, failure, and publication operations are transactional.
- Per-job reentrant locks make same-runner calls idempotent; transactional pending-stage
  claims prevent duplicate execution by multiple runner instances. Claims carry a token,
  generation, and timestamped lease. A live claim cannot be stolen; an expired claim can
  be recovered with a new generation.
- Providers write only durable temporary artifacts. Fixed artifact replacement occurs
  inside a repository compare-and-swap publication transaction after the claim token and
  generation are rechecked. An invalidation or newer worker fences a stale worker from
  publishing, changing state, or overwriting the current artifact.
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
- JSONL appends use checked full-write loops plus a per-path process lock. POSIX systems
  additionally use `flock` for cross-process serialization.
- Redaction recursively replaces values under case-insensitive keys containing `key`,
  `token`, `authorization`, `transcript`, or `payload`, including dictionaries nested in
  lists. Unsupported and non-finite values serialize as a fixed marker; exception/user
  text is never passed to event records.
- Provider request/response diagnostics are emitted only when the persisted job
  `diagnostic_capture` flag is enabled. They contain bounded structural metadata, never
  transcript text, audio bytes, provider payloads, paths, or credentials. Diagnostic
  files remain under the validated job's `diagnostics` directory, use sanitized
  UUID-suffixed names, redact credentials and payload text, publish atomically, and use
  mode `0600`.

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
canonical credential-rejecting cache keys, database migration/reset, exact artifact
deletion, recursive JSONL redaction, safe serialization, and diagnostic
opt-in/mode/path behavior.

### Review RED → GREEN

The review follow-up first added focused regressions for each boundary. The observed RED
failures included missing migration/claim columns and clock injection; role-stage bypass,
missing confirmation marker, and tampered candidate acceptance; six cache-key failures;
five strict-config failures; four hostile-provider semantic failures; whole-call audio
being reused instead of per-utterance slices; no runner diagnostics when enabled; JSONL
interleaving under forced partial writes; and an expired running claim remaining stuck.

The implementation commit `ceb1e702d0a3290bfbcc9bf97dba606a2d6e45fa`
made those regressions green with generation-fenced publication, strict semantic/config
validation, transcription-bound role confirmation, exact WAV slicing, opt-in diagnostic
wiring, and serialized full JSONL writes.

## Verification

```text
cd backend && uv run pytest tests/pipeline tests/jobs -q
133 passed

cd backend && uv run pytest -q
366 passed

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

The only residual platform caveat is that cross-process JSONL serialization uses
`fcntl.flock` on POSIX. On platforms without `fcntl`, the documented fallback is
process-wide per-path locking; checked full writes still apply.
