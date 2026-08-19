"""Non-mutating speaker-role proposals from narrow opening cues."""

from __future__ import annotations

import re

from voxdelta.domain.models import Role, Utterance

_OPENING_SECONDS = 30.0
_AGENT_CUES = ("상담원", "고객센터", "무엇을 도와", "도와드리")
_EXPLICIT_SELF_INTRODUCTION = re.compile(r"(?:저는|제\s*이름은)\s+(?P<body>.+)입니다[.!?…]*$")
_GREETED_SELF_INTRODUCTION = re.compile(
    r"^(?:안녕하세요|안녕하십니까)[,.!?\s]+(?P<body>.+)입니다[.!?…]*$"
)
_HANGUL_NAME_TOKEN = re.compile(r"[가-힣]{2,4}")
_SERVICE_OR_TITLE_PREFIXES = (
    "고객센터",
    "콜센터",
    "서비스",
    "상담원",
    "상담사",
    "담당자",
    "매니저",
    "직원",
)
_STATUS_OR_POLICY_FRAGMENTS = (
    "처리",
    "점검",
    "확인",
    "진행",
    "정책",
    "규정",
    "약관",
    "상태",
    "정상",
    "오류",
    "완료",
    "예정",
    "불가",
    "가능",
    "지연",
    "중단",
    "종료",
)
_STATUS_SUFFIXES = ("중", "완료", "예정", "불가", "가능")


def _looks_like_name_token(body: str, *, prefix_requires_service_or_title: bool) -> bool:
    parts = body.replace(",", " ").split()
    if not parts:
        return False
    final_token = parts[-1]
    if not _HANGUL_NAME_TOKEN.fullmatch(final_token):
        return False
    if any(fragment in final_token for fragment in _STATUS_OR_POLICY_FRAGMENTS):
        return False
    if final_token.endswith(_STATUS_SUFFIXES):
        return False
    if prefix_requires_service_or_title and len(parts) > 1:
        prefix = " ".join(parts[:-1])
        return any(marker in prefix for marker in _SERVICE_OR_TITLE_PREFIXES)
    return True


def _is_self_introduction(transcript: str) -> bool:
    explicit = _EXPLICIT_SELF_INTRODUCTION.search(transcript)
    if explicit is not None:
        return _looks_like_name_token(
            explicit.group("body"),
            prefix_requires_service_or_title=False,
        )
    greeted = _GREETED_SELF_INTRODUCTION.fullmatch(transcript)
    if greeted is None:
        return False
    return _looks_like_name_token(
        greeted.group("body"),
        prefix_requires_service_or_title=True,
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
