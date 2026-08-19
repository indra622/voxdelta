from __future__ import annotations

import pytest

from voxdelta.domain.models import ProviderProvenance, StageName
from voxdelta.pipeline.stages import STAGE_ORDER, cache_key_for_stage, downstream_stages


def test_stage_order_and_downstream_are_exact() -> None:
    assert STAGE_ORDER == tuple(StageName)
    assert downstream_stages(StageName.TRANSCRIBE) == (
        StageName.TRANSCRIBE,
        StageName.CONFIRM_ROLES,
        StageName.EMOTION,
        StageName.RESPONSE_STRATEGY,
        StageName.TRANSITIONS,
        StageName.REPORT,
    )


def test_cache_key_is_canonical_and_noncredential_key_names_affect_identity() -> None:
    provider = ProviderProvenance(name="fake", model="v1", remote=False)
    first = cache_key_for_stage(
        StageName.EMOTION,
        ("a" * 64, "b" * 64),
        provider,
        {
            "threshold": 0.55,
            "nested": {"z": 1, "a": 2},
            "monkey_count": 1,
            "tokenizer": "first",
        },
    )
    second = cache_key_for_stage(
        StageName.EMOTION,
        ("a" * 64, "b" * 64),
        provider,
        {
            "nested": {"a": 2, "z": 1},
            "threshold": 0.55,
            "monkey_count": 1,
            "tokenizer": "first",
        },
    )

    assert first == second
    assert len(first) == 64
    assert first != cache_key_for_stage(
        StageName.EMOTION,
        ("b" * 64, "a" * 64),
        provider,
        {"nested": {"a": 2, "z": 1}, "threshold": 0.55},
    )
    assert first != cache_key_for_stage(
        StageName.EMOTION,
        ("a" * 64, "b" * 64),
        provider.model_copy(update={"model": "v2"}),
        {"nested": {"a": 2, "z": 1}, "threshold": 0.55},
    )
    assert first != cache_key_for_stage(
        StageName.EMOTION,
        ("a" * 64, "b" * 64),
        provider,
        {
            "nested": {"a": 2, "z": 1},
            "threshold": 0.55,
            "monkey_count": 2,
            "tokenizer": "second",
        },
    )


@pytest.mark.parametrize(
    "credential_field",
    ["api_key", "gemini_api_key", "access_token", "refresh_token", "Authorization"],
)
def test_cache_key_rejects_credential_bearing_configuration(
    credential_field: str,
) -> None:
    with pytest.raises(ValueError, match="credential-bearing") as raised:
        cache_key_for_stage(
            StageName.EMOTION,
            ("a" * 64,),
            None,
            {"nested": {credential_field: "do-not-hash-this-secret"}},
        )

    assert "do-not-hash-this-secret" not in str(raised.value)


def test_cache_key_rejects_non_sha256_upstream_identifiers() -> None:
    with pytest.raises(ValueError, match="upstream artifact hashes"):
        cache_key_for_stage(StageName.TRANSCRIBE, ("not-a-content-hash",), None, {})
