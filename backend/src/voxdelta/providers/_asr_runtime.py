"""Shared resident-candidate fencing for heavyweight local ASR models."""

from __future__ import annotations

from collections.abc import Callable

_ACTIVE_OWNER: int | None = None
_ACTIVE_UNLOAD: Callable[[], None] | None = None


def prepare_candidate_load(owner: object) -> None:
    """Evict the previous candidate before a replacement factory allocates memory."""

    global _ACTIVE_OWNER, _ACTIVE_UNLOAD
    if _ACTIVE_OWNER == id(owner):
        return
    unload = _ACTIVE_UNLOAD
    _ACTIVE_OWNER = None
    _ACTIVE_UNLOAD = None
    if unload is not None:
        try:
            unload()
        except Exception:
            pass


def activate_candidate(owner: object, unload: Callable[[], None]) -> None:
    global _ACTIVE_OWNER, _ACTIVE_UNLOAD
    if _ACTIVE_OWNER not in {None, id(owner)}:
        raise RuntimeError("ASR candidate activation was not prepared")
    _ACTIVE_OWNER = id(owner)
    _ACTIVE_UNLOAD = unload


def release_candidate(owner: object) -> None:
    global _ACTIVE_OWNER, _ACTIVE_UNLOAD
    if _ACTIVE_OWNER == id(owner):
        _ACTIVE_OWNER = None
        _ACTIVE_UNLOAD = None


__all__ = ["activate_candidate", "prepare_candidate_load", "release_candidate"]
