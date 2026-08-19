"""Strict public request and response contracts for the local API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from voxdelta.domain.models import Role, StageName


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RoleConfirmation(StrictSchema):
    mapping: dict[str, Role]

    @model_validator(mode="after")
    def one_customer_one_agent(self) -> RoleConfirmation:
        if (
            len(self.mapping) != 2
            or any(not key for key in self.mapping)
            or sorted(self.mapping.values()) != [Role.AGENT, Role.CUSTOMER]
        ):
            raise ValueError("mapping must contain two speakers, one customer and one agent")
        return self


class RetryRequest(StrictSchema):
    stage: StageName


class JobCreated(StrictSchema):
    job_id: str
    status_url: str


class PublicError(StrictSchema):
    code: str
    message: str


class PublicErrorEnvelope(StrictSchema):
    detail: PublicError


class PublicStage(StrictSchema):
    status: str
    error: PublicError | None = None


class RoleCandidate(StrictSchema):
    speakers: list[str]
    suggested_mapping: dict[str, Role] | None = None


class PublicJob(StrictSchema):
    job_id: str
    status: str
    diagnostic_capture: bool
    created_at: str
    updated_at: str
    stages: dict[str, PublicStage]
    role_candidate: RoleCandidate | None = None


class PublicProvenance(StrictSchema):
    name: str
    model: str
    remote: bool
    schema_version: str


class ProviderDisclosure(StrictSchema):
    stage: StageName
    provenance: PublicProvenance | None
    transmits: tuple[str, ...]
    retention_policy_url: str | None


class ProviderConfiguration(StrictSchema):
    stages: list[ProviderDisclosure]


__all__ = [
    "JobCreated",
    "ProviderConfiguration",
    "ProviderDisclosure",
    "PublicError",
    "PublicErrorEnvelope",
    "PublicJob",
    "PublicProvenance",
    "PublicStage",
    "RetryRequest",
    "RoleCandidate",
    "RoleConfirmation",
]
