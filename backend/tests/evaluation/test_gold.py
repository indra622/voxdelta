from __future__ import annotations

import pytest
from pydantic import ValidationError

from voxdelta.evaluation.gold import GoldTransitionLabel, GoldUtteranceLabel


def test_gold_intensity_uses_five_point_rubric() -> None:
    with pytest.raises(ValidationError):
        GoldUtteranceLabel(
            item_id="u1",
            annotator="a1",
            operational_state="dissatisfied",
            negative_intensity=6,
        )


def test_gold_utterance_label_accepts_rubric_endpoints() -> None:
    low = GoldUtteranceLabel(
        item_id="u1",
        annotator="a1",
        operational_state="stable",
        negative_intensity=1,
    )
    high = GoldUtteranceLabel(
        item_id="u2",
        annotator="a1",
        operational_state="escalated",
        negative_intensity=5,
    )

    assert (low.negative_intensity, high.negative_intensity) == (1, 5)


def test_gold_transition_label_rejects_unknown_strategy() -> None:
    with pytest.raises(ValidationError):
        GoldTransitionLabel(
            previous_customer_id="customer-1",
            agent_id="agent-1",
            next_customer_id="customer-2",
            annotator="a1",
            classification="recovery",
            response_strategy="discount",
        )


def test_gold_transition_label_accepts_exact_taxonomy() -> None:
    label = GoldTransitionLabel(
        previous_customer_id="customer-1",
        agent_id="agent-1",
        next_customer_id="customer-2",
        annotator="a1",
        classification="worsening",
        response_strategy="policy_refusal",
    )

    assert label.classification == "worsening"
    assert label.response_strategy == "policy_refusal"
