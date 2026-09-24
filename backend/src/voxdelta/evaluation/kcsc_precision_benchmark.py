"""Comparative pyannoteAI Precision-2 benchmark on the derived KCSC evaluation set.

This is the only KCSC benchmark that sends audio off the machine, so the budget it is
allowed to spend is written down in code rather than left to the caller's discipline:
three conversations, frozen by identifier, one diarization job each, no retries.

The ceiling is enforced at the HTTP boundary by :class:`LedgerClient` rather than by the
loop that calls the provider. A counter in the loop only bounds the calls the loop knows
it is making; a counter in the transport bounds every call, including a retry the
provider might one day perform internally. The ledger also refuses any request that is
not one of the four endpoints this workflow needs, so a future provider change that adds
a fetch cannot quietly widen what leaves the host.

Two artifacts come out of a run, deliberately separated:

* a **local evaluation record** holding remote job identifiers, per-call hosts, and
  timings — operational evidence that stays on this machine, and
* a **public aggregate report** with the same metrics as the local Community-1 benchmark
  and no job identifiers at all.

Neither carries audio or transcript. The ledger stores status codes, hosts, and endpoint
kinds; it never retains a request or response body.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from voxdelta.evaluation.kcsc_diarization_benchmark import (
    BenchmarkReport,
    Diarizer,
    KcscBenchmarkError,
    file_sha256,
    load_manifest,
    run_benchmark,
    select_conversations,
    verify_inputs,
    write_report,
)
from voxdelta.providers.pyannote_precision import (
    PRECISION_MODEL_ID,
    PYANNOTE_DATA_RETENTION_URL,
    PYANNOTE_MEDIA_RETENTION_HOURS,
    HttpClient,
    HttpResponse,
)

SCHEMA_VERSION = "1"

#: Frozen in code, not taken from the caller. The approved scope names exactly these three
#: conversations; a typo in an argument must not be able to send a fourth file anywhere.
FROZEN_CONVERSATIONS: tuple[str, ...] = ("A0051_S0001_0", "A0055_S0006_0", "A6000_S0005_0")

MAX_DIARIZATION_JOBS = len(FROZEN_CONVERSATIONS)
MAX_JOBS_PER_CONVERSATION = 1

MEDIA_INPUT_PATH = "/v1/media/input"
DIARIZE_PATH = "/v1/diarize"
JOBS_PREFIX = "/v1/jobs/"


class PrecisionBudgetExceeded(KcscBenchmarkError):
    """Raised when a run would exceed the approved external-call ceiling."""


class UnexpectedRemoteCall(KcscBenchmarkError):
    """Raised when a request is aimed at an endpoint this workflow does not use."""


@dataclass(frozen=True, slots=True)
class RemoteCall:
    """One outbound request, reduced to what an audit needs and nothing more."""

    kind: str
    method: str
    host: str
    status_code: int
    elapsed_seconds: float


@dataclass
class CallLedger:
    """Counts and classifies outbound calls, and refuses to exceed the approved budget."""

    max_jobs: int = MAX_DIARIZATION_JOBS
    calls: list[RemoteCall] = field(default_factory=list)
    job_ids: list[str] = field(default_factory=list)
    job_statuses: list[str] = field(default_factory=list)
    uploads: int = 0
    submissions: int = 0
    _upload_hosts: set[str] = field(default_factory=set)

    def classify(self, method: str, url: str) -> str:
        """Name the endpoint, failing closed on anything outside the four we use."""

        split = urlsplit(url)
        path = split.path
        if method == "POST" and path == MEDIA_INPUT_PATH:
            return "media_input"
        if method == "POST" and path == DIARIZE_PATH:
            return "diarize"
        if method == "GET" and path.startswith(JOBS_PREFIX):
            return "job_poll"
        if method == "PUT" and split.hostname in self._upload_hosts:
            return "upload"
        raise UnexpectedRemoteCall(f"refused unexpected remote call: {method} {split.hostname}")

    def allow_upload_host(self, url: str) -> None:
        host = urlsplit(url).hostname
        if host:
            self._upload_hosts.add(host)

    def reserve(self, kind: str) -> None:
        """Charge one call against the budget before it is issued."""

        if kind == "diarize":
            if self.submissions >= self.max_jobs:
                raise PrecisionBudgetExceeded(
                    f"refused diarization job {self.submissions + 1}: ceiling is {self.max_jobs}"
                )
            self.submissions += 1
        elif kind == "upload":
            if self.uploads >= self.max_jobs:
                raise PrecisionBudgetExceeded(
                    f"refused upload {self.uploads + 1}: ceiling is {self.max_jobs}"
                )
            self.uploads += 1

    def record(self, call: RemoteCall) -> None:
        self.calls.append(call)

    def note_job(self, job_id: str) -> None:
        if job_id not in self.job_ids:
            self.job_ids.append(job_id)

    def note_status(self, status: str) -> None:
        self.job_statuses.append(status)

    @property
    def hosts(self) -> tuple[str, ...]:
        return tuple(sorted({call.host for call in self.calls}))

    def counts(self) -> dict[str, int]:
        counted: dict[str, int] = {}
        for call in self.calls:
            counted[call.kind] = counted.get(call.kind, 0) + 1
        return counted


class LedgerClient:
    """An ``HttpClient`` that charges every request against a :class:`CallLedger`.

    Response bodies are inspected for exactly two scalars — the job identifier and the
    job status — and are otherwise never read, copied, or retained.
    """

    def __init__(
        self,
        inner: HttpClient,
        ledger: CallLedger,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._inner = inner
        self._ledger = ledger
        self._clock = clock

    def request(
        self,
        method: str,
        url: str,
        *,
        json: Any = None,
        content: bytes | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        kind = self._ledger.classify(method, url)
        self._ledger.reserve(kind)
        started = self._clock()
        response = self._inner.request(method, url, json=json, content=content, headers=headers)
        elapsed = self._clock() - started
        self._ledger.record(
            RemoteCall(
                kind=kind,
                method=method,
                host=urlsplit(url).hostname or "",
                status_code=int(response.status_code),
                elapsed_seconds=round(elapsed, 3),
            )
        )
        self._observe(kind, response)
        return response

    def _observe(self, kind: str, response: HttpResponse) -> None:
        """Extract only the two scalars the audit record needs."""

        if kind not in ("media_input", "diarize", "job_poll"):
            return
        try:
            body = response.json()
        except Exception:
            return
        if not isinstance(body, dict):
            return
        if kind == "media_input":
            destination = body.get("url")
            if isinstance(destination, str):
                self._ledger.allow_upload_host(destination)
            return
        if kind == "diarize":
            job_id = body.get("jobId")
            if isinstance(job_id, str) and job_id:
                self._ledger.note_job(job_id)
            return
        status = body.get("status")
        if isinstance(status, str) and status:
            self._ledger.note_status(status)

    def close(self) -> None:
        self._inner.close()


def preflight(
    derived_root: Path, conversation_ids: Sequence[str] = FROZEN_CONVERSATIONS
) -> dict[str, object]:
    """Verify every checksum before anything is submitted.

    Scoring re-verifies each conversation as it goes, but that check happens immediately
    before that one file is diarized — by which time earlier files have already been
    uploaded. Verifying the whole set up front means a corrupt third file stops the run
    before the first byte leaves the host.
    """

    manifest = load_manifest(derived_root)
    entries = select_conversations(manifest, conversation_ids)
    verified: list[dict[str, object]] = []
    for entry in entries:
        audio_path, reference_path = verify_inputs(entry, derived_root=derived_root)
        outputs = entry["outputs"]
        assert isinstance(outputs, dict)
        verified.append(
            {
                "conversation_id": entry["conversation_id"],
                "audio_sha256": outputs["audio_sha256"],
                "reference_sha256": outputs["reference_sha256"],
                "audio_bytes": audio_path.stat().st_size,
                "duration_seconds": entry.get("derived_duration_seconds"),
                "reference_path": reference_path.name,
            }
        )
    source = manifest.get("source")
    return {
        "conversations": verified,
        "planned_diarization_jobs": len(verified),
        "source_revision": (source or {}).get("revision") if isinstance(source, dict) else None,
    }


#: The corpus rights holder named on the upstream dataset card.
RIGHTS_HOLDER = "Beijing Magic Data Technology Co., Ltd."

#: Named once so the public report, the local record, and the operator prompt cannot
#: drift apart in how they describe what leaving the host actually costs.
EXTERNAL_PROCESSING_CAVEAT = (
    "Audio for these conversations was uploaded to the managed pyannoteAI API and "
    "processed on third-party infrastructure. pyannoteAI states that Media API input may "
    f"be held in temporary storage for up to {PYANNOTE_MEDIA_RETENTION_HOURS} hours and "
    "the public API exposes no deletion endpoint, so the upload cannot be recalled. "
    "Transcription was disabled, so no transcript was requested, produced, or returned. "
    "LICENSING IS UNRESOLVED: third-party processing rights for this corpus have NOT been "
    f"confirmed with the rights holder ({RIGHTS_HOLDER}). The upstream licence permits "
    "academic research use only and prohibits reproduction or distribution without the "
    "rights holder's permission; whether this upload was permitted is undetermined. The "
    "transfer proceeded solely on a limited operator override covering exactly the three "
    "derived files named in this artifact, one job each. Derived audio must not be "
    "redistributed, and this result must not be treated as evidence that the transfer "
    "was licensed."
)

#: What the operator actually authorized. There is deliberately no value meaning "rights
#: confirmed": that claim requires evidence this workflow has never been given, and a
#: boolean flag makes it far too easy to assert by accident.
ExternalAuthorization = Literal["user_limited_override"]

AUTHORIZATION_SCOPE = (
    "Operator override limited to exactly 3 fixed derived KCSC conversations, one "
    "diarization job each, no retries. Not a general permission to transmit this corpus."
)


def _authorization_block(authorization: ExternalAuthorization) -> dict[str, object]:
    """The authorization facts, stated identically in both artifacts."""

    return {
        "rights_holder": RIGHTS_HOLDER,
        "external_processing_authorization": authorization,
        "rights_confirmed": False,
        "third_party_processing_rights": "unknown",
        "authorization_scope": AUTHORIZATION_SCOPE,
        "corpus_use": "academic research only; redistribution prohibited",
    }


def _disclosure(ledger: CallLedger, *, authorization: ExternalAuthorization) -> dict[str, object]:
    return {
        "location": "remote",
        "processor": "pyannoteAI",
        "transmitted": ["audio"],
        "transcription_requested": False,
        "retention_window_hours": PYANNOTE_MEDIA_RETENTION_HOURS,
        "retention_policy_url": PYANNOTE_DATA_RETENTION_URL,
        "diarization_jobs_submitted": ledger.submissions,
        "caveat": EXTERNAL_PROCESSING_CAVEAT,
        **_authorization_block(authorization),
    }


def write_precision_report(
    report: BenchmarkReport,
    path: Path,
    ledger: CallLedger,
    *,
    authorization: ExternalAuthorization,
) -> str:
    """Write the public aggregate report: same metrics as the local benchmark, no job ids.

    ``authorization`` is required and has no default: what permitted the transfer is a
    fact about the run, and an artifact that omits it would read as though the question
    had never come up.
    """

    return write_report(report, path, disclosure=_disclosure(ledger, authorization=authorization))


def write_evaluation_record(
    path: Path,
    *,
    report: BenchmarkReport,
    ledger: CallLedger,
    preflight_summary: Mapping[str, object],
    started_at: str,
    total_elapsed_seconds: float,
    authorization: ExternalAuthorization,
) -> str:
    """Write the local operational record, including remote job identifiers."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "pyannoteai-precision2-kcsc-derived-diarization-evaluation",
        "started_at": started_at,
        "scope": (
            "Approved comparative evaluation of the managed pyannoteAI precision-2 "
            "provider on three derived KCSC conversations. Diarization only: no ASR, no "
            "emotion, no pipeline run. No product code, configuration, or credential was "
            "changed for this run."
        ),
        "provider": {
            "name": "pyannoteai",
            "model": PRECISION_MODEL_ID,
            "remote": True,
            "transmits": ["audio"],
            "request": {"transcription": False, "exclusive": True, "numSpeakers": None},
            "retention_window_hours": PYANNOTE_MEDIA_RETENTION_HOURS,
            "retention_policy_url": PYANNOTE_DATA_RETENTION_URL,
        },
        "budget": {
            "max_diarization_jobs": MAX_DIARIZATION_JOBS,
            "max_jobs_per_conversation": MAX_JOBS_PER_CONVERSATION,
            "diarization_jobs_submitted": ledger.submissions,
            "uploads": ledger.uploads,
            "retries": 0,
            "call_counts_by_kind": ledger.counts(),
            "hosts_contacted": list(ledger.hosts),
        },
        "remote_jobs": {
            "job_ids": list(ledger.job_ids),
            "terminal_statuses": sorted(
                set(ledger.job_statuses) & {"succeeded", "failed", "canceled"}
            ),
        },
        "licensed_or_private_audio_transmitted": True,
        "rights": _authorization_block(authorization),
        "caveat": EXTERNAL_PROCESSING_CAVEAT,
        "preflight": dict(preflight_summary),
        "timing": {
            "total_elapsed_seconds": round(total_elapsed_seconds, 3),
            "per_conversation_elapsed_seconds": {
                score.conversation_id: score.elapsed_seconds for score in report.scores
            },
        },
        "aggregate": {variant: dict(values) for variant, values in report.aggregate.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    path.write_text(f"{raw}\n", encoding="utf-8")
    return file_sha256(path)


#: Recorded on an artifact whose authorization block was corrected after the fact, so a
#: reader can see that the claim changed rather than finding a silently different file.
CORRECTION_NOTE = (
    "The authorization block of this artifact was corrected after it was first written. "
    "The original stated that third-party processing rights had been confirmed; that was "
    "incorrect. The transfer was authorized only by a limited operator override. No job "
    "was resubmitted and no measurement changed: the correction touches the authorization "
    "and caveat fields only."
)


def amend_authorization(path: Path, *, authorization: ExternalAuthorization) -> str:
    """Correct an already-written artifact's authorization claim, in place.

    Rewriting the file from a fresh run would mean submitting the audio again, which is
    exactly what a correction to an over-broad permission claim must not do. This edits
    the authorization and caveat fields of an existing artifact and leaves every measured
    value untouched, so the numbers stay the ones the recorded jobs actually produced.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise KcscBenchmarkError(f"artifact is not valid JSON: {path}") from None
    if not isinstance(payload, dict):
        raise KcscBenchmarkError(f"artifact has an unexpected shape: {path}")

    block = _authorization_block(authorization)
    amended = False
    processing = payload.get("processing")
    if isinstance(processing, dict):
        processing.pop("third_party_processing_confirmed", None)
        processing.update(block)
        processing["caveat"] = EXTERNAL_PROCESSING_CAVEAT
        amended = True
    if isinstance(payload.get("rights"), dict):
        payload["rights"] = dict(block)
        amended = True
    if "caveat" in payload:
        payload["caveat"] = EXTERNAL_PROCESSING_CAVEAT
        amended = True
    if not amended:
        raise KcscBenchmarkError(f"artifact carries no authorization block: {path}")
    payload["correction"] = CORRECTION_NOTE

    raw = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    path.write_text(f"{raw}\n", encoding="utf-8")
    return file_sha256(path)


def run_precision_benchmark(
    *,
    derived_root: Path,
    diarizer: Diarizer,
    conversation_ids: Sequence[str] = FROZEN_CONVERSATIONS,
) -> BenchmarkReport:
    """Score the frozen conversations through the remote provider.

    The provider raises on the first failure and this function does not catch it: a run
    that loses one conversation stops rather than submitting the rest.
    """

    if tuple(conversation_ids) != FROZEN_CONVERSATIONS:
        raise KcscBenchmarkError("conversation selection does not match the approved scope")
    return run_benchmark(
        derived_root=derived_root,
        conversation_ids=conversation_ids,
        diarizer=diarizer,
        model_name=PRECISION_MODEL_ID,
        model_tree_sha256=None,
        egress_attempts=0,
        remote=True,
    )


__all__ = [
    "AUTHORIZATION_SCOPE",
    "CORRECTION_NOTE",
    "EXTERNAL_PROCESSING_CAVEAT",
    "RIGHTS_HOLDER",
    "ExternalAuthorization",
    "FROZEN_CONVERSATIONS",
    "MAX_DIARIZATION_JOBS",
    "MAX_JOBS_PER_CONVERSATION",
    "SCHEMA_VERSION",
    "CallLedger",
    "LedgerClient",
    "PrecisionBudgetExceeded",
    "RemoteCall",
    "UnexpectedRemoteCall",
    "amend_authorization",
    "preflight",
    "run_precision_benchmark",
    "write_evaluation_record",
    "write_precision_report",
]
