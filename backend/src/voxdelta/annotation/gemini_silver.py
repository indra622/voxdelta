"""Gemini audio annotation: provisional *silver* labels that always need a human.

This adapter sends audio to Google's Gemini API and asks for a structured first pass at
diarization, transcript, and per-turn emotion. Everything it produces is provisional by
construction. There is no code path here that yields gold, and the artifact it writes
carries ``review_state = "review_required"`` as a field the promotion flow reads rather
than a convention it hopes callers honour.

Three properties are enforced rather than documented:

* **Consent is Gemini-specific.** Sending audio to Google is a different disclosure from
  sending it to pyannoteAI, so an existing consent for one is not evidence for the other.
  Without an explicit Gemini grant the adapter raises before it reads the key, let alone
  the audio.
* **The call budget is fixed and small.** One upload, one interaction, one delete. The
  ledger charges each before it is issued and refuses the second of any kind, so a retry
  loop cannot exist even if a caller writes one.
* **The uploaded file is always deleted.** The delete runs in a ``finally``, so it is
  attempted whether the interaction succeeded, failed, or raised. Its outcome — including
  failure — is recorded in the artifact, because an upload that could not be withdrawn is
  a fact the reviewer needs.

Transcript text lives in the returned annotation and in the private silver artifact. It is
never logged, never returned in an error message, and never written to a benchmark file.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from voxdelta.domain.models import ProviderProvenance
from voxdelta.providers.base import ProviderError

#: Verified against the official model list on 2026-09-02.
GEMINI_MODEL = "gemini-3.7-flash"
API_BASE = "https://generativelanguage.googleapis.com"
UPLOAD_START_URL = f"{API_BASE}/upload/v1beta/files"
INTERACTIONS_URL = f"{API_BASE}/v1beta/interactions"
API_KEY_HEADER = "x-goog-api-key"

#: One of each. Not a default a caller may raise.
MAX_UPLOADS = 1
MAX_INTERACTIONS = 1
MAX_DELETES = 1

EMOTION_LABELS: tuple[str, ...] = (
    "happiness",
    "anger",
    "disgust",
    "fear",
    "neutral",
    "sadness",
    "surprise",
    "uncertain",
)

#: The subset of JSON Schema the Gemini structured-output contract accepts.
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "speakers": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Provisional speaker identifiers, e.g. SPEAKER_00.",
        },
        "turns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number", "description": "Turn onset in seconds."},
                    "end": {"type": "number", "description": "Turn end in seconds."},
                    "speaker": {"type": "string"},
                    "transcript": {"type": "string"},
                    "emotion": {"type": "string", "enum": list(EMOTION_LABELS)},
                    "emotion_rationale": {
                        "type": "string",
                        "description": "Short reason a reviewer can check against the audio.",
                    },
                    "confidence": {"type": "number"},
                },
                "required": [
                    "start",
                    "end",
                    "speaker",
                    "transcript",
                    "emotion",
                    "emotion_rationale",
                    "confidence",
                ],
            },
        },
        "notes": {"type": "string", "description": "Anything the reviewer should check first."},
    },
    "required": ["speakers", "turns", "notes"],
}

PROMPT = (
    "You are producing a PROVISIONAL first-pass annotation of a two-party Korean phone "
    "conversation for a human reviewer to correct. Return diarized turns in time order "
    "with start/end seconds, a speaker label, a verbatim Korean transcript, one emotion "
    "label per turn, a short rationale for that emotion, and a confidence in [0,1]. Use "
    "the label 'uncertain' whenever you are not confident rather than guessing a specific "
    "emotion. Do not invent speech that is not audible."
)


class GeminiConsentError(ProviderError):
    """Raised when Gemini transmission was not explicitly and specifically consented to."""

    def __init__(self) -> None:
        super().__init__("provider_unavailable")


class GeminiBudgetExceeded(ProviderError):
    """Raised when a run would exceed the fixed one-of-each call ceiling."""

    def __init__(self) -> None:
        super().__init__("provider_unavailable")


class ContractValidationError(ProviderError):
    """A rejected annotation plus a text-free description of the first broken rule."""

    # Always present and always a plain dict here, unlike the optional base declaration.
    diagnostic: dict[str, object]

    def __init__(self, diagnostic: dict[str, object]) -> None:
        self.diagnostic = diagnostic
        super().__init__("invalid_provider_output")


@dataclass
class CallTrace:
    """Where a run got to, in facts a failed attempt can be diagnosed from.

    Deliberately status codes and stage names only. A response body from a failed auth
    call can echo credentials or account identifiers, so it is never captured.
    """

    stage: str = "not_started"
    last_status: int | None = None
    audio_bytes_sent: int = 0
    output_stage: str = "not_received"
    response_shape: dict[str, object] | None = None
    contract_diagnostic: dict[str, object] | None = None
    delete_attempted: bool = False
    delete_http_status: int | None = None
    delete_outcome: str = "not_attempted"

    def as_dict(self) -> dict[str, object]:
        return {
            "stage_reached": self.stage,
            "last_http_status": self.last_status,
            "audio_bytes_sent": self.audio_bytes_sent,
            "output_stage": self.output_stage,
            "response_shape": self.response_shape,
            "contract_diagnostic": self.contract_diagnostic,
            "delete_attempted": self.delete_attempted,
            "delete_http_status": self.delete_http_status,
            "delete_outcome": self.delete_outcome,
        }


class HttpResponse(Protocol):
    status_code: int
    headers: Mapping[str, str]
    text: str

    def json(self) -> Any: ...


class HttpClient(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = ...,
        json: Any | None = ...,
        content: bytes | None = ...,
    ) -> HttpResponse: ...


@dataclass
class CallLedger:
    """Counts outbound Gemini calls and refuses the second of any kind."""

    uploads: int = 0
    interactions: int = 0
    deletes: int = 0
    calls: list[str] = field(default_factory=list)

    def reserve(self, kind: str) -> None:
        ceilings = {
            "upload": (self.uploads, MAX_UPLOADS),
            "interaction": (self.interactions, MAX_INTERACTIONS),
            "delete": (self.deletes, MAX_DELETES),
        }
        if kind not in ceilings:
            raise GeminiBudgetExceeded()
        used, ceiling = ceilings[kind]
        if used >= ceiling:
            raise GeminiBudgetExceeded()
        if kind == "upload":
            self.uploads += 1
        elif kind == "interaction":
            self.interactions += 1
        else:
            self.deletes += 1
        self.calls.append(kind)

    def counts(self) -> dict[str, int]:
        return {
            "uploads": self.uploads,
            "interactions": self.interactions,
            "deletes": self.deletes,
        }


@dataclass(frozen=True, slots=True)
class SilverTurn:
    """One provisional turn. Carries transcript, so it never reaches a log."""

    start: float
    end: float
    speaker: str
    transcript: str
    emotion: str
    emotion_rationale: str
    confidence: float


@dataclass(frozen=True, slots=True)
class SilverAnnotation:
    """A validated provisional annotation plus what a reviewer needs to judge it."""

    turns: tuple[SilverTurn, ...]
    speakers: tuple[str, ...]
    notes: str
    dropped_turns: tuple[tuple[str, int], ...] = ()

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def dropped_turn_count(self) -> int:
        return sum(count for _rule, count in self.dropped_turns)

    @property
    def dropped_turns_by_rule(self) -> dict[str, int]:
        return dict(self.dropped_turns)

    def emotion_histogram(self) -> dict[str, int]:
        counts = dict.fromkeys(EMOTION_LABELS, 0)
        for turn in self.turns:
            counts[turn.emotion] += 1
        return counts


def provenance() -> ProviderProvenance:
    """Declares remote audio transmission explicitly; nothing infers it from the name."""

    return ProviderProvenance(
        name="gemini-annotation",
        model=GEMINI_MODEL,
        remote=True,
        transmits=("audio",),
        retention_policy_url="https://ai.google.dev/gemini-api/terms",
        revision=GEMINI_MODEL,
    )


def _value_shape(value: object, *, count: bool = False) -> dict[str, object]:
    """Report a value's type and, only for arrays, its length — never its contents."""

    if value is _MISSING:
        return {"type": "absent"}
    shape: dict[str, object] = {"type": type(value).__name__}
    if count and isinstance(value, list):
        shape["count"] = len(value)
    return shape


_MISSING = object()


def _contract_diagnostic(
    payload: object,
    *,
    rule: str,
    turn_index: int | None = None,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Bounded schema evidence for a failed model response, without any payload text."""

    if not isinstance(payload, dict):
        diagnostic: dict[str, object] = {"payload_type": type(payload).__name__}
    else:
        diagnostic = {
            "payload_type": "object",
            "top_level_keys": sorted(str(key) for key in payload)[:16],
            "turns": _value_shape(payload.get("turns", _MISSING), count=True),
            "speakers": _value_shape(payload.get("speakers", _MISSING), count=True),
            "notes": _value_shape(payload.get("notes", _MISSING)),
        }
    violation: dict[str, object] = {"rule": rule}
    if turn_index is not None:
        violation["turn_index"] = turn_index
    violation.update(metadata or {})
    diagnostic["first_violation"] = violation
    return diagnostic


def _contract_error(
    payload: object,
    rule: str,
    *,
    turn_index: int | None = None,
    metadata: Mapping[str, object] | None = None,
) -> ContractValidationError:
    return ContractValidationError(
        _contract_diagnostic(
            payload, rule=rule, turn_index=turn_index, metadata=metadata
        )
    )


def _duration_excess_bucket(excess_seconds: float) -> str:
    """Bound a timestamp overshoot without persisting an exact model-generated time."""

    if excess_seconds <= 5.0:
        return "up_to_5_seconds"
    if excess_seconds <= 15.0:
        return "5_to_15_seconds"
    if excess_seconds <= 60.0:
        return "15_to_60_seconds"
    return "over_60_seconds"


def validate(payload: object, *, duration_seconds: float | None = None) -> SilverAnnotation:
    """Turn the model's JSON into a typed annotation, refusing anything unusable.

    Validation is strict on structure and permissive on content: a turn with an unknown
    emotion or a backwards interval is a defect in the response, but a turn the reviewer
    will disagree with is the entire point of the workflow and is kept.
    """

    if not isinstance(payload, dict):
        raise _contract_error(payload, "payload.not_object")
    raw_turns = payload.get("turns")
    speakers = payload.get("speakers")
    notes = payload.get("notes", "")
    if not isinstance(raw_turns, list) or not raw_turns:
        raise _contract_error(payload, "turns.not_nonempty_list")
    if not isinstance(speakers, list) or not speakers:
        raise _contract_error(payload, "speakers.not_nonempty_list")
    if not isinstance(notes, str):
        raise _contract_error(payload, "notes.not_string")

    turns: list[SilverTurn] = []
    for index, row in enumerate(raw_turns):
        if not isinstance(row, dict):
            raise _contract_error(payload, "turn.not_object", turn_index=index)
        try:
            start = float(row["start"])
            end = float(row["end"])
            confidence = float(row["confidence"])
            speaker = row["speaker"]
            transcript = row["transcript"]
            emotion = row["emotion"]
            rationale = row["emotion_rationale"]
        except (KeyError, TypeError, ValueError):
            raise _contract_error(
                payload, "turn.missing_or_invalid_field", turn_index=index
            ) from None
        # Required as real strings rather than coerced: str(None) is "None", which would
        # turn a null transcript into a plausible-looking four-character utterance.
        if not all(isinstance(value, str) for value in (speaker, transcript, emotion, rationale)):
            raise _contract_error(payload, "turn.required_text_not_string", turn_index=index)
        if end <= start or start < 0:
            raise _contract_error(payload, "turn.invalid_interval", turn_index=index)
        if duration_seconds is not None and end > duration_seconds + 1.0:
            raise _contract_error(
                payload,
                "turn.exceeds_duration",
                turn_index=index,
                metadata={
                    "duration_excess_bucket": _duration_excess_bucket(end - duration_seconds)
                },
            )
        if emotion not in EMOTION_LABELS:
            raise _contract_error(payload, "turn.emotion_unknown", turn_index=index)
        if not 0.0 <= confidence <= 1.0:
            raise _contract_error(payload, "turn.confidence_out_of_range", turn_index=index)
        turns.append(SilverTurn(start, end, speaker, transcript, emotion, rationale, confidence))
    return SilverAnnotation(
        turns=tuple(turns),
        speakers=tuple(str(item) for item in speakers),
        notes=notes,
    )


def validate_silver(payload: object, *, duration_seconds: float | None = None) -> SilverAnnotation:
    """Keep only individually valid model turns for a human-reviewed silver draft.

    The outer contract stays strict: a malformed payload, no declared speakers, or no
    usable turns remains a provider failure.  A single malformed turn is different: it
    is omitted from provisional material and recorded by rule, without retaining its
    transcript or timestamps in diagnostic state.
    """

    if not isinstance(payload, dict):
        return validate(payload, duration_seconds=duration_seconds)
    raw_turns = payload.get("turns")
    speakers = payload.get("speakers")
    notes = payload.get("notes", "")
    if not isinstance(raw_turns, list) or not raw_turns:
        return validate(payload, duration_seconds=duration_seconds)
    if not isinstance(speakers, list) or not speakers or not isinstance(notes, str):
        return validate(payload, duration_seconds=duration_seconds)

    accepted: list[SilverTurn] = []
    dropped: dict[str, int] = {}
    for row in raw_turns:
        try:
            single = validate(
                {"turns": [row], "speakers": speakers, "notes": notes},
                duration_seconds=duration_seconds,
            )
        except ContractValidationError as error:
            violation = error.diagnostic.get("first_violation")
            rule = violation.get("rule") if isinstance(violation, dict) else None
            safe_rule = str(rule) if isinstance(rule, str) else "turn.invalid"
            dropped[safe_rule] = dropped.get(safe_rule, 0) + 1
            continue
        accepted.extend(single.turns)

    if not accepted:
        raise _contract_error(payload, "turns.no_usable_after_salvage")
    return SilverAnnotation(
        turns=tuple(accepted),
        speakers=tuple(str(item) for item in speakers),
        notes=notes,
        dropped_turns=tuple(sorted(dropped.items())),
    )


def _extract_json(body: Any) -> object:
    """Pull the structured payload out of the interaction response.

    The response envelope has changed shape across API generations, so the text is located
    by looking for it rather than by assuming one path, and a body that yields nothing
    parseable fails closed instead of being guessed at.
    """

    if isinstance(body, dict):
        for key in ("output_text", "outputText", "text"):
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                return json.loads(value)
        # Interactions REST responses place model text in a model_output step rather
        # than in output_text.  Accept only text parts; never stringify arbitrary
        # content because that could turn an API envelope into plausible annotation.
        steps = body.get("steps")
        if isinstance(steps, list):
            for step in steps:
                if not isinstance(step, dict) or step.get("type") != "model_output":
                    continue
                content = step.get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    if not isinstance(part, dict) or part.get("type") != "text":
                        continue
                    value = part.get("text")
                    if isinstance(value, str) and value.strip():
                        return json.loads(value)
        output = body.get("output") or body.get("candidates")
        if isinstance(output, list):
            for item in output:
                if isinstance(item, str) and item.strip():
                    return json.loads(item)
                if isinstance(item, dict):
                    for key in ("text", "output_text", "content"):
                        value = item.get(key)
                        if isinstance(value, str) and value.strip():
                            return json.loads(value)
    raise ProviderError("invalid_provider_output")


def _response_shape(body: object) -> dict[str, object]:
    """Summarise an Interactions response without retaining any model-generated text."""

    if not isinstance(body, dict):
        return {"body_type": type(body).__name__}

    shape: dict[str, object] = {
        "body_type": "object",
        "top_level_keys": sorted(str(key) for key in body)[:16],
    }
    steps = body.get("steps")
    if not isinstance(steps, list):
        shape["steps"] = None
        return shape

    step_shapes: list[dict[str, object]] = []
    for step in steps[:8]:
        if not isinstance(step, dict):
            step_shapes.append({"type": "non_object", "content_type": "absent", "part_types": []})
            continue
        raw_type = step.get("type")
        step_type = "model_output" if raw_type == "model_output" else "other"
        content = step.get("content")
        if not isinstance(content, list):
            step_shapes.append(
                {
                    "type": step_type,
                    "content_type": type(content).__name__,
                    "part_types": [],
                }
            )
            continue
        part_types = [
            "text" if isinstance(part, dict) and part.get("type") == "text" else "other"
            for part in content[:16]
        ]
        step_shapes.append(
            {"type": step_type, "content_type": "list", "part_types": part_types}
        )
    shape["steps"] = step_shapes
    return shape


@dataclass(frozen=True, slots=True)
class GeminiRunOutcome:
    """Everything the artifact needs about the call itself, and no transcript."""

    annotation: SilverAnnotation
    remote_file_name: str | None
    remote_file_deleted: bool
    deletion_detail: str
    call_counts: Mapping[str, int]
    model: str


class GeminiAnnotator:
    """One-shot Gemini annotation under a fixed budget and an explicit consent gate."""

    def __init__(
        self,
        *,
        api_key: str,
        client: HttpClient,
        consent_granted: bool,
        model: str = GEMINI_MODEL,
    ) -> None:
        if not consent_granted:
            # Checked before the key is touched: without consent there is nothing to
            # authenticate, because there is nothing this object is allowed to send.
            raise GeminiConsentError()
        if not api_key:
            raise ProviderError("provider_unavailable")
        self._api_key = api_key
        self._client = client
        self._model = model
        self.provenance = provenance()
        self.ledger = CallLedger()
        self.trace = CallTrace()

    def _headers(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        headers = {API_KEY_HEADER: self._api_key}
        headers.update(extra or {})
        return headers

    def _upload(self, audio_path: Path, mime_type: str) -> tuple[str, str]:
        payload = audio_path.read_bytes()
        self.ledger.reserve("upload")
        self.trace.stage = "upload_start"
        start = self._client.request(
            "POST",
            UPLOAD_START_URL,
            headers=self._headers(
                {
                    "X-Goog-Upload-Protocol": "resumable",
                    "X-Goog-Upload-Command": "start",
                    "X-Goog-Upload-Header-Content-Length": str(len(payload)),
                    "X-Goog-Upload-Header-Content-Type": mime_type,
                    "Content-Type": "application/json",
                }
            ),
            json={"file": {"display_name": audio_path.stem}},
        )
        self.trace.last_status = start.status_code
        if start.status_code >= 400:
            # No audio has been sent at this point: the start request carries metadata
            # only, and the bytes go to the session URL it would have returned.
            raise ProviderError("provider_unavailable")
        upload_url = start.headers.get("x-goog-upload-url") or start.headers.get(
            "X-Goog-Upload-URL"
        )
        if not upload_url:
            raise ProviderError("invalid_provider_output")
        self.trace.stage = "upload_bytes"
        # Charged before the request, not after it: if the transfer dies mid-flight the
        # bytes have still left this machine, and a durable "0 bytes sent" would be a
        # false assurance for exactly the failure a reviewer most needs the truth about.
        self.trace.audio_bytes_sent = len(payload)
        finalize = self._client.request(
            "POST",
            upload_url,
            headers={
                "Content-Length": str(len(payload)),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            },
            content=payload,
        )
        self.trace.last_status = finalize.status_code
        if finalize.status_code >= 400:
            raise ProviderError("provider_unavailable")
        body = finalize.json()
        info = body.get("file") if isinstance(body, dict) else None
        if not isinstance(info, dict) or not info.get("uri") or not info.get("name"):
            raise ProviderError("invalid_provider_output")
        return str(info["uri"]), str(info["name"])

    def _interact(
        self,
        file_uri: str,
        mime_type: str,
        *,
        prompt: str = PROMPT,
        response_schema: Mapping[str, Any] = RESPONSE_SCHEMA,
    ) -> object:
        self.ledger.reserve("interaction")
        self.trace.stage = "interaction"
        response = self._client.request(
            "POST",
            INTERACTIONS_URL,
            headers=self._headers({"Content-Type": "application/json"}),
            json={
                "model": self._model,
                "input": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "uri": file_uri, "mime_type": mime_type},
                ],
                "response_format": {
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": response_schema,
                },
            },
        )
        self.trace.last_status = response.status_code
        if response.status_code >= 400:
            raise ProviderError("provider_unavailable")
        body = response.json()
        self.trace.response_shape = _response_shape(body)
        self.trace.stage = "validate"
        self.trace.output_stage = "response_received"
        try:
            payload = _extract_json(body)
        except json.JSONDecodeError:
            self.trace.output_stage = "structured_text_invalid_json"
            raise ProviderError("invalid_provider_output") from None
        except ProviderError:
            self.trace.output_stage = "structured_text_unavailable"
            raise
        self.trace.output_stage = "structured_text_parsed"
        return payload

    def _delete(self, name: str) -> tuple[bool, str]:
        """Withdraw the uploaded file. Never raises: the caller is already unwinding."""

        self.trace.delete_attempted = True
        try:
            self.ledger.reserve("delete")
            response = self._client.request(
                "DELETE", f"{API_BASE}/v1beta/{name}", headers=self._headers()
            )
        except Exception as error:  # noqa: BLE001 - deletion failure must not mask the run
            self.trace.delete_outcome = "exception"
            return False, f"delete raised {type(error).__name__}"
        self.trace.delete_http_status = response.status_code
        if response.status_code >= 400:
            self.trace.delete_outcome = "http_error"
            return False, f"delete returned HTTP {response.status_code}"
        self.trace.delete_outcome = "deleted"
        return True, "deleted"

    def annotate(
        self,
        audio_path: Path,
        *,
        mime_type: str = "audio/wav",
        duration_seconds: float | None = None,
    ) -> GeminiRunOutcome:
        """Upload once, ask once, delete once. The delete happens whatever else does."""

        remote_name: str | None = None
        deleted = False
        detail = "not attempted"
        try:
            file_uri, remote_name = self._upload(audio_path, mime_type)
            payload = self._interact(file_uri, mime_type)
            try:
                annotation = validate(payload, duration_seconds=duration_seconds)
            except ContractValidationError as initial_error:
                try:
                    annotation = validate_silver(payload, duration_seconds=duration_seconds)
                except ContractValidationError as error:
                    # The initial row-level failure is more actionable than the
                    # follow-on fact that no rows survived salvage.
                    del error
                    self.trace.contract_diagnostic = initial_error.diagnostic
                    self.trace.output_stage = "contract_invalid"
                    raise
                self.trace.contract_diagnostic = {
                    **initial_error.diagnostic,
                    "salvaged": True,
                    "dropped_turn_count": annotation.dropped_turn_count,
                    "dropped_turns_by_rule": annotation.dropped_turns_by_rule,
                }
                self.trace.output_stage = "contract_salvaged"
            except ProviderError:
                self.trace.output_stage = "contract_invalid"
                raise
            self.trace.output_stage = "contract_valid"
        finally:
            if remote_name is not None:
                deleted, detail = self._delete(remote_name)
        return GeminiRunOutcome(
            annotation=annotation,
            remote_file_name=remote_name,
            remote_file_deleted=deleted,
            deletion_detail=detail,
            call_counts=self.ledger.counts(),
            model=self._model,
        )


__all__ = [
    "API_KEY_HEADER",
    "EMOTION_LABELS",
    "GEMINI_MODEL",
    "INTERACTIONS_URL",
    "MAX_DELETES",
    "MAX_INTERACTIONS",
    "MAX_UPLOADS",
    "PROMPT",
    "RESPONSE_SCHEMA",
    "UPLOAD_START_URL",
    "CallLedger",
    "CallTrace",
    "GeminiAnnotator",
    "GeminiBudgetExceeded",
    "GeminiConsentError",
    "GeminiRunOutcome",
    "SilverAnnotation",
    "SilverTurn",
    "provenance",
    "validate",
    "validate_silver",
]
