"""Gold-label contracts used for human evaluation annotations."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class GoldUtteranceLabel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str
    annotator: str
    operational_state: Literal["satisfied", "stable", "dissatisfied", "escalated", "uncertain"]
    negative_intensity: int = Field(ge=1, le=5)


class GoldTransitionLabel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    previous_customer_id: str
    agent_id: str
    next_customer_id: str
    annotator: str
    classification: Literal["recovery", "stable", "worsening"]
    response_strategy: Literal[
        "apology",
        "empathy",
        "clarification",
        "information",
        "solution",
        "policy_refusal",
        "greeting_closing",
        "other",
    ]
