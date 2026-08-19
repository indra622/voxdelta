"""Non-mutating speaker-role proposals from narrow opening cues."""

from __future__ import annotations

import re

from voxdelta.domain.models import Role, Utterance

_OPENING_SECONDS = 30.0
_AGENT_CUES = ("상담원", "고객센터", "무엇을 도와", "도와드리")
_EXPLICIT_SELF_INTRODUCTION = re.compile(r"(?:저는|제\s*이름은)\s+\S(?:.*\S)?입니다[.!?…]*$")
_GREETED_NAME_INTRODUCTION = re.compile(
    r"^(?:안녕하세요|안녕하십니까)[,.!?\s]+[가-힣]{3}입니다[.!?…]*$"
)


def _is_self_introduction(transcript: str) -> bool:
    return bool(
        _EXPLICIT_SELF_INTRODUCTION.search(transcript)
        or _GREETED_NAME_INTRODUCTION.fullmatch(transcript)
    )


def suggest_roles(utterances: list[Utterance]) -> dict[str, Role] | None:
    """Return an unconfirmed two-speaker proposal when one cue score leads."""

    speaker_ids = sorted({utterance.speaker_id for utterance in utterances})
    if len(speaker_ids) != 2:
        return None

    opening = sorted(
        (utterance for utterance in utterances if utterance.start < _OPENING_SECONDS),
        key=lambda utterance: (
            utterance.start,
            utterance.end,
            utterance.id,
            utterance.speaker_id,
        ),
    )
    cues_by_speaker: dict[str, set[str]] = {speaker_id: set() for speaker_id in speaker_ids}
    first_turn_seen: set[str] = set()
    for utterance in opening:
        transcript = utterance.transcript.strip()
        cues = cues_by_speaker[utterance.speaker_id]
        for cue in _AGENT_CUES:
            if cue in transcript:
                cues.add(cue)
        if utterance.speaker_id not in first_turn_seen:
            first_turn_seen.add(utterance.speaker_id)
            if _is_self_introduction(transcript):
                cues.add("self-introduction")

    scores = {speaker_id: len(cues) for speaker_id, cues in cues_by_speaker.items()}
    lead = max(speaker_ids, key=lambda speaker_id: scores[speaker_id])
    other = next(speaker_id for speaker_id in speaker_ids if speaker_id != lead)
    if scores[lead] < 1 or scores[lead] == scores[other]:
        return None
    return {lead: Role.AGENT, other: Role.CUSTOMER}
