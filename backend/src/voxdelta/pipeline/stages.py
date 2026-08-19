"""Typed stage artifacts, dependencies, and deterministic cache identities."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import orjson
from pydantic import BaseModel, ConfigDict, Field, model_validator

from voxdelta.domain.models import (
    AnalysisReport,
    AudioAsset,
    EmotionResult,
    EmotionTransition,
    ProviderProvenance,
    ResponseStrategyResult,
    Role,
    SpeakerSegment,
    StageName,
    Utterance,
)

SCHEMA_VERSION: Literal["1"] = "1"
STAGE_ORDER: tuple[StageName, ...] = tuple(StageName)

UPSTREAM_STAGES: dict[StageName, tuple[StageName, ...]] = {
    StageName.NORMALIZE: (),
    StageName.DIARIZE: (StageName.NORMALIZE,),
    StageName.TRANSCRIBE: (StageName.NORMALIZE, StageName.DIARIZE),
    StageName.CONFIRM_ROLES: (StageName.TRANSCRIBE,),
    StageName.EMOTION: (StageName.NORMALIZE, StageName.CONFIRM_ROLES),
    StageName.RESPONSE_STRATEGY: (StageName.CONFIRM_ROLES,),
    StageName.TRANSITIONS: (StageName.CONFIRM_ROLES, StageName.EMOTION),
    StageName.REPORT: (
        StageName.CONFIRM_ROLES,
        StageName.EMOTION,
        StageName.RESPONSE_STRATEGY,
        StageName.TRANSITIONS,
    ),
}

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_IDENTIFIER = re.compile(r"[^a-zA-Z0-9]+")


def downstream_stages(stage: StageName) -> tuple[StageName, ...]:
    """Return the selected stage and every later stage in canonical order."""

    if not isinstance(stage, StageName):
        raise KeyError(str(stage))
    return STAGE_ORDER[STAGE_ORDER.index(stage) :]


def _canonical_cache_value(value: object) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("cache configuration numbers must be finite")
        return value
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise TypeError("cache configuration keys must be strings")
            separated = _CAMEL_BOUNDARY.sub("_", raw_key)
            normalized = _NON_IDENTIFIER.sub("_", separated).strip("_").casefold()
            if (
                normalized in {"authorization", "key", "token", "password", "secret"}
                or normalized.endswith("_key")
                or normalized.endswith("_token")
                or normalized.endswith("_password")
                or normalized.endswith("_secret")
            ):
                raise ValueError("credential-bearing cache configuration is not allowed")
            sanitized[raw_key] = _canonical_cache_value(item)
        return sanitized
    if isinstance(value, (list, tuple)):
        return [_canonical_cache_value(item) for item in value]
    raise TypeError(f"unsupported cache configuration type: {type(value).__name__}")


def cache_key_for_stage(
    stage: StageName,
    upstream_hashes: Sequence[str],
    provider: ProviderProvenance | None,
    config: Mapping[str, object],
) -> str:
    """Hash a secret-free canonical description of a stage invocation."""

    if not isinstance(stage, StageName):
        raise KeyError(str(stage))
    if any(
        len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
        for value in upstream_hashes
    ):
        raise ValueError("upstream artifact hashes must be lowercase SHA-256 values")
    payload = {
        "stage": stage.value,
        "upstream_artifact_content_hashes": list(upstream_hashes),
        "provider": (
            {"name": provider.name, "model": provider.model} if provider is not None else None
        ),
        "config": _canonical_cache_value(config),
    }
    canonical = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(canonical).hexdigest()


class StageArtifact(BaseModel):
    """Metadata common to every on-disk pipeline JSON object."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = SCHEMA_VERSION
    cache_key: str = Field(pattern=_SHA256_PATTERN)
    upstream_hashes: tuple[str, ...] = ()
    provider: ProviderProvenance | None = None

    @model_validator(mode="after")
    def hashes_are_sha256(self) -> StageArtifact:
        if any(
            len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
            for value in self.upstream_hashes
        ):
            raise ValueError("upstream hashes must be lowercase SHA-256 values")
        return self


class NormalizeArtifact(StageArtifact):
    stage: Literal[StageName.NORMALIZE] = StageName.NORMALIZE
    asset: AudioAsset


class DiarizeArtifact(StageArtifact):
    stage: Literal[StageName.DIARIZE] = StageName.DIARIZE
    segments: list[SpeakerSegment]


class TranscribeArtifact(StageArtifact):
    stage: Literal[StageName.TRANSCRIBE] = StageName.TRANSCRIBE
    utterances: list[Utterance]


class RoleArtifact(StageArtifact):
    stage: Literal[StageName.CONFIRM_ROLES] = StageName.CONFIRM_ROLES
    utterances: list[Utterance]
    suggestion: dict[str, Role] | None = None
    confirmed: bool = False
    mapping: dict[str, Role] | None = None

    @model_validator(mode="after")
    def confirmation_fields_agree(self) -> RoleArtifact:
        if self.confirmed != (self.mapping is not None):
            raise ValueError("confirmed role artifacts require a mapping")
        return self


class EmotionArtifact(StageArtifact):
    stage: Literal[StageName.EMOTION] = StageName.EMOTION
    results: list[EmotionResult]


class StrategyArtifact(StageArtifact):
    stage: Literal[StageName.RESPONSE_STRATEGY] = StageName.RESPONSE_STRATEGY
    results: list[ResponseStrategyResult]


class TransitionsArtifact(StageArtifact):
    stage: Literal[StageName.TRANSITIONS] = StageName.TRANSITIONS
    results: list[EmotionTransition]


class ReportArtifact(StageArtifact):
    stage: Literal[StageName.REPORT] = StageName.REPORT
    report: AnalysisReport


StageArtifactType = (
    NormalizeArtifact
    | DiarizeArtifact
    | TranscribeArtifact
    | RoleArtifact
    | EmotionArtifact
    | StrategyArtifact
    | TransitionsArtifact
    | ReportArtifact
)

ARTIFACT_MODELS: dict[StageName, type[StageArtifact]] = {
    StageName.NORMALIZE: NormalizeArtifact,
    StageName.DIARIZE: DiarizeArtifact,
    StageName.TRANSCRIBE: TranscribeArtifact,
    StageName.CONFIRM_ROLES: RoleArtifact,
    StageName.EMOTION: EmotionArtifact,
    StageName.RESPONSE_STRATEGY: StrategyArtifact,
    StageName.TRANSITIONS: TransitionsArtifact,
    StageName.REPORT: ReportArtifact,
}
