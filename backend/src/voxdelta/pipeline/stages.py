"""Typed stage artifacts, dependencies, and deterministic cache identities."""

from __future__ import annotations

import hashlib
import math
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
from voxdelta.security import is_credential_key

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
            if is_credential_key(raw_key):
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
    provider_identity: dict[str, object] | None = None
    if provider is not None:
        provider_identity = {"name": provider.name, "model": provider.model}
        if provider.revision is not None:
            provider_identity["revision"] = provider.revision
    payload = {
        "stage": stage.value,
        "upstream_artifact_content_hashes": list(upstream_hashes),
        "provider": provider_identity,
        "config": _canonical_cache_value(config),
    }
    canonical = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(canonical).hexdigest()


class StageArtifact(BaseModel):
    """Metadata common to every on-disk pipeline JSON object.

    Cache keys and media digests detect application-level drift; they are not cryptographic
    attestation. The SQLite database and job artifact directory are trusted local state
    protected by operating-system user permissions. Coordinated same-user modification of
    both is outside the MVP threat model.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
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


class MediaReference(BaseModel):
    """A normalized media path bound to the exact bytes accepted by the pipeline."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    sha256: str = Field(pattern=_SHA256_PATTERN)


class NormalizeArtifact(StageArtifact):
    schema_version: Literal["1"] = SCHEMA_VERSION
    stage: Literal[StageName.NORMALIZE] = StageName.NORMALIZE
    asset: AudioAsset
    normalized_media: tuple[MediaReference, ...]
    mixed_preview: MediaReference

    @model_validator(mode="after")
    def media_manifest_matches_asset(self) -> NormalizeArtifact:
        if tuple(item.path for item in self.normalized_media) != self.asset.normalized_paths:
            raise ValueError("normalized media manifest must match selected paths exactly")
        if self.asset.channel_mode == "mixed" and (
            len(self.normalized_media) != 1 or self.mixed_preview != self.normalized_media[0]
        ):
            raise ValueError("mixed media preview must match the selected normalized path")
        if self.asset.channel_mode == "separate" and self.mixed_preview.path in {
            item.path for item in self.normalized_media
        }:
            raise ValueError("separate media requires a distinct typed mixed preview")
        return self


class DiarizeArtifact(StageArtifact):
    schema_version: Literal["2"] = "2"
    stage: Literal[StageName.DIARIZE] = StageName.DIARIZE
    segments: list[SpeakerSegment]
    alignment_segments: list[SpeakerSegment]


class TranscribeArtifact(StageArtifact):
    schema_version: Literal["1"] = SCHEMA_VERSION
    stage: Literal[StageName.TRANSCRIBE] = StageName.TRANSCRIBE
    utterances: list[Utterance]


class RoleArtifact(StageArtifact):
    schema_version: Literal["1"] = SCHEMA_VERSION
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
    schema_version: Literal["1"] = SCHEMA_VERSION
    stage: Literal[StageName.EMOTION] = StageName.EMOTION
    results: list[EmotionResult]


class StrategyArtifact(StageArtifact):
    schema_version: Literal["1"] = SCHEMA_VERSION
    stage: Literal[StageName.RESPONSE_STRATEGY] = StageName.RESPONSE_STRATEGY
    results: list[ResponseStrategyResult]


class TransitionsArtifact(StageArtifact):
    schema_version: Literal["1"] = SCHEMA_VERSION
    stage: Literal[StageName.TRANSITIONS] = StageName.TRANSITIONS
    results: list[EmotionTransition]


class ReportArtifact(StageArtifact):
    schema_version: Literal["1"] = SCHEMA_VERSION
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
