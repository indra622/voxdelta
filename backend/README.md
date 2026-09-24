# VoxDelta backend

This directory contains the local FastAPI core for VoxDelta. The current vertical slice validates
and normalizes an uploaded call, runs deterministic fake providers, pauses for customer/agent role
confirmation, resumes the analysis, and serves the canonical report and normalized audio.

The core assumes every accepted call has exactly two primary speakers: one customer and one agent.
It is not a real diarization, transcription, emotion, response-strategy, or summarization system
yet. The fake providers produce deterministic contract-test data and must not be interpreted as
analysis of the recording's actual speech or emotion.

## Setup

Install [uv](https://docs.astral.sh/uv/), FFmpeg, `curl`, and `jq`, then run from the repository
root:

```bash
uv python install 3.12
cd backend
uv sync --locked --dev
ffmpeg -version
ffprobe -version
jq --version
```

Python is constrained to 3.12 by `pyproject.toml`. `uv sync --locked --dev` reproduces the
checked-in `uv.lock` without changing it. Use `uv sync --dev` only when intentionally resolving
dependency changes, then review and commit the resulting lock-file update. `uv lock --check`
verifies that the lock file still matches the project metadata.

Create a private local environment file before configuring future model providers:

```bash
cp backend/.env.example backend/.env
chmod 0600 backend/.env
```

If you are already in `backend/`, use `cp .env.example .env` instead. Never put real credential
values in `.env.example`, logs, diagnostics, Git, or commands that may be recorded in shell
history.

The fake core needs no network access and no credentials. The separate readiness checker reports
only `configured` or `missing` and is for later provider work:

```bash
cd backend
uv run python scripts/check_credentials.py
uv run python scripts/check_credentials.py --profile comparison
```

The `local` readiness profile requires `HUGGINGFACE_TOKEN` for future local model downloads. The
`comparison` profile additionally requires `GEMINI_API_KEY`. `PYANNOTEAI_API_KEY` is optional
in both profiles; the planned local pyannote Community-1 provider uses the Hugging Face token,
while a pyannoteAI key would be relevant only to an explicitly selected optional remote benchmark.
A missing-token exit code from this checker does not prevent the present fake core from running.

Application settings are read from `VOXDELTA_` process-environment variables. Uvicorn's
`--env-file` option loads the private file into that environment:

- `VOXDELTA_DATA_ROOT`: artifact root parent; default is repository `data/`.
- `VOXDELTA_DATABASE_PATH`: SQLite path; default is `data/voxdelta.sqlite3`.
- `VOXDELTA_MIN_AUDIO_SECONDS`: minimum decoded duration; default `60`.
- `VOXDELTA_MAX_AUDIO_SECONDS`: maximum decoded duration; default `3600`.
- `VOXDELTA_MAX_UPLOAD_BYTES`: maximum raw uploaded file size; default `1073741824` (1 GiB).
- `VOXDELTA_ADMISSION_RECONCILIATION_LEASE_SECONDS`: stale admission recovery lease; default
  `300`.
- `VOXDELTA_MAX_ACTIVE_JOBS`: maximum incomplete jobs admitted at once; default `8`.
- `VOXDELTA_ANNOTATION_ROOT`: private silver/gold annotation root; default is
  `<data root>/annotations`. Artifacts under it hold verbatim transcript, so it is kept out
  of benchmark output and served only through the capability-fenced review routes.
- `VOXDELTA_DIARIZATION_PROVIDER`: `fake` (default), `pyannote-community` for the local
  Community-1 pipeline, `nemotron-3-local` for NVIDIA Nemotron 3 Diarization through a
  locally installed NeMo-Speech.cpp CLI, or `pyannoteai-precision` for the managed pyannoteAI
  API. Only `pyannoteai-precision` sends audio off this machine, and it is never selected
  implicitly.
- `VOXDELTA_API_CAPABILITY_TOKEN`: per-launch local API capability. Supply a fresh high-entropy
  value of at least 32 visible HTTP-header ASCII characters (`!` through `~`) in the process
  environment; it is held as a secret in memory and is never logged or persisted by VoxDelta.

The upload middleware separately caps the complete multipart request at the configured raw-file
limit plus 64 KiB of multipart overhead. The stored file itself may never exceed the exact
raw-file limit.

Start the local-only server from `backend/`:

```bash
export VOXDELTA_API_CAPABILITY_TOKEN="$(uv run python -c 'import secrets; print(secrets.token_urlsafe(32))')"
uv run uvicorn voxdelta.api.app:app \
  --host 127.0.0.1 \
  --port 8000 \
  --env-file .env
```

Open <http://127.0.0.1:8000/docs> for the generated OpenAPI interface. Binding to `127.0.0.1`
keeps the development service off external network interfaces. Every `/api/jobs...` request must
also send `X-VoxDelta-Token: $VOXDELTA_API_CAPABILITY_TOKEN`. Strict local `Host` and `Origin`
checks block DNS-rebinding and drive-by browser requests; the custom token header is intentionally
not a CORS-simple request header. In Swagger UI, select **Authorize** and enter the same token.
`GET /api/config/providers`, `/docs`, and `/openapi.json` contain no job data and do not require
the capability.

## Local diarization through Nemotron 3 (opt-in)

`nemotron-3-local` runs `nvidia/Nemotron-3-Diarization` on this machine by executing a local
`nemo-speech` binary with an explicit local GGUF; it never downloads a runtime or model and is
reported as `remote: false`. It needs:

```bash
export VOXDELTA_DIARIZATION_PROVIDER=nemotron-3-local
export VOXDELTA_NEMOTRON_EXECUTABLE_PATH=/absolute/path/to/nemo-speech
export VOXDELTA_NEMOTRON_MODEL_PATH=/absolute/path/to/Nemotron-3-Diarization.q8_0.gguf
export VOXDELTA_NEMOTRON_DEVICE=metal            # auto | metal | cpu
export VOXDELTA_NEMOTRON_TIMEOUT_SECONDS=600     # per file, 0 < t <= 3600
```

A missing or non-executable runtime, or a missing, symlinked or non-GGUF model, refuses startup
(`provider_configuration_invalid`, with `local_runtime_missing` / `local_model_missing` as the
operator-facing provider code) instead of falling back to another provider. The adapter always
uses the model card's 30.4 s offline streaming geometry. Installing the runtime and pulling the
model are separate setup steps: see `docs/nemotron-3-local-setup.md`.

## Remote diarization through pyannoteAI

Every other provider runs locally. Selecting `pyannoteai-precision` is the one configuration
that transmits call audio to a third party, so it is opt-in by name and fails closed rather
than falling back to a local provider.

```bash
export VOXDELTA_DIARIZATION_PROVIDER=pyannoteai-precision
```

Startup additionally requires `PYANNOTEAI_API_KEY` in `backend/.env`; without it the provider
is refused before any request is built, so nothing is uploaded. The key is read at the moment
each request is signed, is sent only as an `Authorization: Bearer` header, and never appears
in provenance, logs, exceptions, or error messages.

Per job the provider uploads the normalized WAV to pyannoteAI temporary storage, submits
`POST /v1/diarize` with `transcription` disabled and `exclusive` enabled, polls
`GET /v1/jobs/{id}` until the job settles, and maps `diarization` and `exclusiveDiarization`
onto the same overlap-aware and alignment timelines the local provider returns. No transcript
is requested and none is returned. Separate-channel audio is diarized one channel at a time
with `numSpeakers=1`, which means two uploads and two jobs for that input class.

`GET /api/config/providers` reports this stage as `remote: true` with `transmits: ["audio"]`,
the pyannoteAI data-retention URL, and `retention_window_hours: 48`, so a consent prompt can
name the window instead of linking to a policy nobody opens. Read that retention policy before
sending anything you do not own: uploaded media sits in pyannoteAI temporary storage for up to
that long and the public API exposes no deletion endpoint.

`retention_window_hours` is optional and stays `null` for every local provider; it is the
longest window a provider states it may hold transmitted input for, never a guarantee that the
input is gone sooner.

## HTTP API

All job errors use a stable public `detail` payload and omit host paths, credentials, transcripts,
provider payloads, and internal cache/claim metadata. Framework-level request validation failures
return HTTP 422 as
`{"detail":{"code":"invalid_request","message":"The request is invalid."}}` without echoing the
invalid body.

- `GET /api/config/providers` returns all eight pipeline stages with public provider provenance,
  transmitted-content declarations, and retention-policy URLs. Fake provider stages are local,
  transmit nothing, and have no remote retention policy; non-provider stages have null provenance.
- `POST /api/jobs` accepts multipart `file` (`.wav`, `.mp3`, or `.m4a`) and optional boolean
  `diagnostic_capture` (default `false`). It streams and privately stores a non-empty file,
  fully decodes and validates the staged upload before durable job/database admission, reuses that
  normalized generation, schedules local processing, and returns HTTP 202 with `job_id` and
  `status_url`. Unsupported suffixes, empty/corrupt media, and decoded durations outside the
  configured interval return 422 without a durable job; size violations return 413 and a full
  incomplete-job quota returns 429. A 202 response does not mean analysis is complete.
- `GET /api/jobs/{job_id}` returns public job status, the eight stage statuses, sanitized stage
  errors, timestamps, and the diagnostic flag. While `confirm_roles` is paused it also returns the
  two observed speaker IDs and, when the conservative heuristic has one, an unconfirmed suggested
  mapping.
- `POST /api/jobs/{job_id}/roles` accepts
  `{"mapping":{"speaker-a":"customer","speaker-b":"agent"}}`. The keys must exactly match the two
  observed IDs and the values must contain one customer and one agent. It publishes the
  confirmation, resumes the local pipeline synchronously, and returns the updated job. Confirming
  outside the paused role gate returns 409; an invalid mapping returns 422.
- `POST /api/jobs/{job_id}/retry` accepts one of the eight stages as `{"stage":"report"}`. It
  removes the selected stage's JSON artifact and every downstream stage artifact/reference,
  preserves the original upload and unaffected upstream work, reruns from that point, and returns
  the updated job. A retry can pause again if it invalidates role confirmation.
- `GET /api/jobs/{job_id}/report` returns the canonical report JSON only after the report stage is
  complete; otherwise it returns 409. The report contains the summary, ordered utterances,
  customer-emotion results, relevant agent strategies, transition triplets, warnings, and schema
  version.
- `GET /api/jobs/{job_id}/audio` streams the hash-validated normalized mixed WAV preview. With no
  range it returns 200; one valid `Range: bytes=...` request returns 206 with `Accept-Ranges`,
  `Content-Length`, and `Content-Range`. Open-ended and suffix ranges are supported. Invalid,
  multiple, or unsatisfiable byte ranges return 416 with `Content-Range: bytes */<size>`.
- `DELETE /api/jobs/{job_id}` fences in-flight work, removes that job's database row and job
  directory, and returns 204. An interrupted local deletion returns a retryable 409; retry the same
  DELETE. A missing job returns 404.

### Silver annotation review

Silver drafts hold verbatim transcript, so these three routes sit behind the same per-launch
capability, host, and origin fence as `/api/jobs`. They read the private annotation root
(`VOXDELTA_ANNOTATION_ROOT`, default `data/annotations/`); they never write silver and never call a
remote annotation service.

- `GET /api/annotations` lists every silver draft this machine holds as counts and digests only,
  plus `unreadable_count` for drafts that exist but could not be parsed. No transcript is in this
  response, so choosing what to open reveals no speech.
- `GET /api/annotations/{conversation_id}` returns one draft in full: the reviewable turns with
  transcript, the declared speakers, the model's notes, the emotion labels the backend will accept,
  and the warnings that make the draft provisional. `review_required` is always present;
  `salvaged_dropped_turns` carries the count and per-rule breakdown of turns validation rejected,
  which are absent from the draft and were never recovered. A conversation id that is not a single
  name matching `[A-Za-z0-9][A-Za-z0-9_-]{0,63}` returns 422; an absent draft returns 404; a stored
  file that is not a readable silver artifact returns 409.
- `POST /api/annotations/{conversation_id}/gold` promotes one draft through
  `voxdelta.annotation.store.promote` and returns 201 with the gold digest, its parent silver
  digest, and `silver_unmodified`, which reports that silver was byte-identical after the write.
  The body needs `reviewer`, `acknowledged: true`, and the reviewer's `turns`; an optional
  `review_note` is recorded with the sign-off. A blank reviewer (`reviewer_required`), a missing
  acknowledgement (`review_acknowledgement_required`), no turns (`corrected_turns_required`), an
  emotion outside the allowed set (`invalid_turn_emotion`), a range that does not start at or after
  zero and end after it starts (`invalid_turn_interval`), a confidence outside 0 to 1
  (`invalid_turn_confidence`), and a blank speaker or transcript each return 422 and write nothing.
  A second promotion returns 409 `gold_already_exists`; gold is written once and is immutable.
  Refusals name the turn's position and never quote its transcript.

## Runtime files and privacy

With default settings, SQLite metadata is in `data/voxdelta.sqlite3` and each job is under
`data/jobs/<job_id>/`. A job directory can contain the private source upload, a generated
`audio-<id>/` directory with normalized WAV media, versioned stage JSON (`<stage>.v1.json`), a
redacted `pipeline.jsonl`, the canonical report artifact, and optional diagnostics.

Private annotation artifacts live under `data/annotations/<conversation_id>/`. `silver.json` is a
model's provisional draft and is never rewritten: it stays `review_state: review_required` for its
whole life, and the review API only reads it. `gold.json` is what a named reviewer asserted, is
written exactly once, and carries its parent silver's digest plus its own; a second promotion is
refused and an edited gold file fails `verify_gold`. Both files hold verbatim transcript, so they
must not be copied into logs, benchmark output, or anything shared.

The runtime parent is private and SQLite database/journal/WAL files are forced to mode `0600` on
POSIX even under a permissive umask. Startup reschedules pristine pending jobs and expired running
claims after the claim lease; live claims are never stolen. A successfully published normalize
retry garbage-collects obsolete `audio-*` generations. POSIX readers retain their already-open
descriptor; Windows may defer one generation's cleanup until a later retry after readers close.
Active decode/generation workspaces carry private PID/owner records and use `flock` where
available; Windows checks process state through non-destructive Win32 query handles. This preserves
live owners and lets abandoned work be collected on other platforms. Pre-lease `.ingest-*` upgrade
residue is retained for a one-hour safety grace before cleanup.

`data/jobs/.incoming/`, `.locks/`, and `.deleted/` are private control directories used for
durable upload admission, cross-process job coordination, and deletion tombstones. Empty control
directories, lock files, or tombstones can remain after a request or deletion. They are trusted
same-user local state protected by operating-system permissions, not an authentication or
cryptographic tamper-resistance boundary; do not edit them by hand.

Diagnostics are disabled by default and enabled per job only with `diagnostic_capture=true`.
Diagnostic payloads are recursively redacted, each file is mode `0600` on POSIX, and the
directory is private. Even so, enable capture only when needed and treat all runtime files as
sensitive. Pipeline logs never intentionally contain credential values, full transcripts, or raw
provider payloads.

No real audio, transcript, feature, report, or credential leaves the machine in this fake core.
There are no model-network calls and all provider disclosures report local execution with empty
transmission sets. Later explicitly configured remote providers have different privacy and
retention behavior; local deletion cannot imply deletion by a future remote provider.

All runtime media, SQLite files, diagnostics, generated reports, local `.env` files, and raw
datasets are Git-ignored. `data/.gitkeep` only preserves the empty runtime parent directory.

## Reproducible smoke test

The following Bash walkthrough uses the generated 65-second fixture and isolated temporary data
and database paths. It does not read or modify the repository's normal `data/` state. The role
payload uses a valid API suggestion only when it exactly covers both observed speakers; otherwise
it assigns the sorted IDs without assuming response order. For the deterministic fake provider,
that fallback is `SPEAKER_00=customer` and `SPEAKER_01=agent`, but the assertions still verify
the observed candidate before posting it.

Run from `backend/` with no other server on port 8765:

```bash
set -euo pipefail

SMOKE_ROOT="$(mktemp -d /tmp/voxdelta-smoke.XXXXXX)"
mkdir -p "$SMOKE_ROOT/data"
SMOKE_TOKEN="$(uv run python -c 'import secrets; print(secrets.token_urlsafe(32))')"
SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID"
    wait "$SERVER_PID" || true
  fi
  rm -rf -- "$SMOKE_ROOT"
}
trap cleanup EXIT

VOXDELTA_DATA_ROOT="$SMOKE_ROOT/data" \
VOXDELTA_DATABASE_PATH="$SMOKE_ROOT/data/voxdelta.sqlite3" \
VOXDELTA_API_CAPABILITY_TOKEN="$SMOKE_TOKEN" \
uv run uvicorn voxdelta.api.app:app \
  --host 127.0.0.1 \
  --port 8765 \
  >"$SMOKE_ROOT/server.log" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 100); do
  if curl --connect-timeout 1 --max-time 2 -fsS \
    http://127.0.0.1:8765/api/config/providers \
    >"$SMOKE_ROOT/providers.json"; then
    break
  fi
  sleep 0.1
done
jq -e '.stages | length == 8' "$SMOKE_ROOT/providers.json" >/dev/null

UPLOAD_CODE="$(curl --connect-timeout 2 --max-time 60 -sS \
  -o "$SMOKE_ROOT/created.json" \
  -w '%{http_code}' \
  -F 'file=@tests/fixtures/synthetic_65s.wav;type=audio/wav' \
  -F 'diagnostic_capture=false' \
  -H "X-VoxDelta-Token: $SMOKE_TOKEN" \
  http://127.0.0.1:8765/api/jobs)"
test "$UPLOAD_CODE" = 202
JOB_ID="$(jq -er '.job_id' "$SMOKE_ROOT/created.json")"

for _ in $(seq 1 100); do
  curl --connect-timeout 2 --max-time 10 -fsS \
    -H "X-VoxDelta-Token: $SMOKE_TOKEN" \
    "http://127.0.0.1:8765/api/jobs/$JOB_ID" \
    >"$SMOKE_ROOT/paused.json"
  if jq -e \
    '.status == "paused" and .stages.confirm_roles.status == "paused"' \
    "$SMOKE_ROOT/paused.json" >/dev/null; then
    break
  fi
  sleep 0.1
done
jq -e \
  '.status == "paused"
   and .stages.confirm_roles.status == "paused"
   and (.role_candidate.speakers | length == 2)' \
  "$SMOKE_ROOT/paused.json" >/dev/null

jq '
  .role_candidate as $candidate
  | ($candidate.speakers | sort) as $speakers
  | if (($candidate.suggested_mapping | type) == "object")
       and (($candidate.suggested_mapping | keys | sort) == $speakers)
       and (($candidate.suggested_mapping | [.[]] | sort) == ["agent", "customer"])
    then {mapping: $candidate.suggested_mapping}
    else {mapping: {($speakers[0]): "customer", ($speakers[1]): "agent"}}
    end
' "$SMOKE_ROOT/paused.json" >"$SMOKE_ROOT/roles.json"
jq -e \
  '(.mapping | keys | sort) == ($observed | sort)
   and (.mapping | [.[]] | sort) == ["agent", "customer"]' \
  --argjson observed "$(jq '.role_candidate.speakers' "$SMOKE_ROOT/paused.json")" \
  "$SMOKE_ROOT/roles.json" >/dev/null

ROLE_CODE="$(curl --connect-timeout 2 --max-time 60 -sS \
  -o "$SMOKE_ROOT/role-response.json" \
  -w '%{http_code}' \
  -H 'Content-Type: application/json' \
  -H "X-VoxDelta-Token: $SMOKE_TOKEN" \
  --data-binary @"$SMOKE_ROOT/roles.json" \
  "http://127.0.0.1:8765/api/jobs/$JOB_ID/roles")"
test "$ROLE_CODE" = 200

for _ in $(seq 1 100); do
  curl --connect-timeout 2 --max-time 10 -fsS \
    -H "X-VoxDelta-Token: $SMOKE_TOKEN" \
    "http://127.0.0.1:8765/api/jobs/$JOB_ID" \
    >"$SMOKE_ROOT/completed.json"
  if jq -e '.status == "completed" and .stages.report.status == "completed"' \
    "$SMOKE_ROOT/completed.json" >/dev/null; then
    break
  fi
  sleep 0.1
done
jq -e '.status == "completed" and .stages.report.status == "completed"' \
  "$SMOKE_ROOT/completed.json" >/dev/null

RANGE_CODE="$(curl --connect-timeout 2 --max-time 30 -sS \
  -D "$SMOKE_ROOT/range.headers" \
  -o "$SMOKE_ROOT/range.bin" \
  -w '%{http_code}' \
  -H 'Range: bytes=0-1023' \
  -H "X-VoxDelta-Token: $SMOKE_TOKEN" \
  "http://127.0.0.1:8765/api/jobs/$JOB_ID/audio")"
test "$RANGE_CODE" = 206
grep -Eiq '^content-range: bytes 0-1023/[0-9]+' "$SMOKE_ROOT/range.headers"
test "$(wc -c <"$SMOKE_ROOT/range.bin" | tr -d ' ')" = 1024

curl --connect-timeout 2 --max-time 30 -fsS \
  -H "X-VoxDelta-Token: $SMOKE_TOKEN" \
  "http://127.0.0.1:8765/api/jobs/$JOB_ID/report" \
  >"$SMOKE_ROOT/report.json"
jq -e 'has("transitions") and (.utterances | length > 0)' \
  "$SMOKE_ROOT/report.json" >/dev/null

DELETE_CODE="$(curl --connect-timeout 2 --max-time 30 -sS \
  -o /dev/null \
  -w '%{http_code}' \
  -X DELETE \
  -H "X-VoxDelta-Token: $SMOKE_TOKEN" \
  "http://127.0.0.1:8765/api/jobs/$JOB_ID")"
test "$DELETE_CODE" = 204

kill "$SERVER_PID"
wait "$SERVER_PID" || true
if kill -0 "$SERVER_PID" 2>/dev/null; then
  exit 1
fi
SERVER_PID=""
```

## Verification

From `backend/`:

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy src scripts
uv lock --check
```
