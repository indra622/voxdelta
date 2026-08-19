"""Shared token-aware secret-key classification without secret values."""

from __future__ import annotations

import re

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_IDENTIFIER = re.compile(r"[^a-zA-Z0-9]+")
_CREDENTIAL_TOKENS = frozenset({"authorization", "key", "password", "secret", "token"})
_COMPACT_CREDENTIAL_ALIASES = frozenset(
    {"apikey", "accesstoken", "bearertoken", "clientsecret", "refreshtoken"}
)


def key_tokens(raw_key: str) -> tuple[str, ...]:
    separated = _CAMEL_BOUNDARY.sub("_", raw_key)
    normalized = _NON_IDENTIFIER.sub("_", separated).strip("_").casefold()
    return tuple(part for part in normalized.split("_") if part)


def is_credential_key(raw_key: str) -> bool:
    tokens = key_tokens(raw_key)
    return bool(
        _CREDENTIAL_TOKENS.intersection(tokens) or "".join(tokens) in _COMPACT_CREDENTIAL_ALIASES
    )


__all__ = ["is_credential_key", "key_tokens"]
