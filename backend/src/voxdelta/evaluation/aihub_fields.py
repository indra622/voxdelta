"""Canonical field resolution for nested AI Hub JSON metadata."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal, overload

CanonicalField = Literal["id", "call_id", "speaker_id", "transcript", "emotion"]

FIELD_ALIASES: dict[CanonicalField, tuple[str, ...]] = {
    "id": ("id", "wav_id", "audio_id"),
    "call_id": ("call_id", "conversation_id", "dialogue_id"),
    "speaker_id": ("speaker_id", "speaker", "talker_id"),
    "transcript": ("transcript", "text", "sentence", "발화문"),
    "emotion": ("emotion", "emotion_label", "label", "감정"),
}


def _walk_dictionaries(
    value: object, prefix: tuple[str, ...] = ()
) -> Iterator[tuple[dict[str, object], tuple[str, ...]]]:
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        typed_value: dict[str, object] = value
        yield typed_value, prefix
        for key, nested in typed_value.items():
            yield from _walk_dictionaries(nested, (*prefix, key))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            yield from _walk_dictionaries(nested, (*prefix, str(index)))


def _key_paths(metadata: dict[str, object]) -> list[str]:
    paths: list[str] = []

    def visit(value: object, prefix: tuple[str, ...] = ()) -> None:
        if isinstance(value, dict) and all(isinstance(key, str) for key in value):
            typed_value: dict[str, object] = value
            for key, nested in typed_value.items():
                key_path = (*prefix, key)
                paths.append(".".join(key_path))
                visit(nested, key_path)
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                visit(nested, (*prefix, str(index)))

    visit(metadata)
    return paths


@overload
def resolve_canonical_field(
    metadata: dict[str, object], field: CanonicalField, *, required: Literal[True] = True
) -> str: ...


@overload
def resolve_canonical_field(
    metadata: dict[str, object], field: CanonicalField, *, required: Literal[False]
) -> str | None: ...


def resolve_canonical_field(
    metadata: dict[str, object], field: CanonicalField, *, required: bool = True
) -> str | None:
    """Resolve an ordered alias while detecting ambiguous nested metadata."""

    matches: list[tuple[str, object]] = []
    dictionaries = list(_walk_dictionaries(metadata))
    for alias in FIELD_ALIASES[field]:
        for dictionary, prefix in dictionaries:
            if alias in dictionary:
                matches.append((".".join((*prefix, alias)), dictionary[alias]))

    if not matches:
        if not required:
            return None
        available = ", ".join(_key_paths(metadata)) or "<none>"
        raise ValueError(f"no alias matched {field}; available key paths: {available}")

    for path, value in matches:
        if not isinstance(value, str):
            raise ValueError(f"canonical field {field} must be a string at {path}")
    values = {value for _, value in matches if isinstance(value, str)}
    if len(values) != 1:
        paths = ", ".join(path for path, _ in matches)
        raise ValueError(f"multiple differing matches for {field}: {paths}")
    return next(iter(values))
