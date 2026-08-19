from __future__ import annotations

import pytest

from voxdelta.analysis.roles import suggest_roles
from voxdelta.domain.models import Role, Utterance


def _utterance(
    identifier: str,
    speaker: str,
    start: float,
    transcript: str,
    *,
    role: Role = Role.UNKNOWN,
) -> Utterance:
    return Utterance(
        id=identifier,
        start=start,
        end=start + 0.5,
        speaker_id=speaker,
        role=role,
        transcript=transcript,
        confidence=1.0,
    )


def test_suggest_roles_returns_unique_distinct_cue_leader_without_mutation() -> None:
    utterances = [
        _utterance("u1", "S1", 0.0, "상담원 상담원입니다", role=Role.CUSTOMER),
        _utterance("u2", "S2", 1.0, "문의가 있습니다", role=Role.AGENT),
    ]
    snapshot = [item.model_dump() for item in utterances]

    proposal = suggest_roles(utterances)

    assert proposal == {"S1": Role.AGENT, "S2": Role.CUSTOMER}
    assert [item.model_dump() for item in utterances] == snapshot


def test_suggest_roles_counts_each_cue_once_per_speaker_not_occurrences() -> None:
    utterances = [
        _utterance("u1", "S1", 0.0, "상담원 상담원 상담원 상담원"),
        _utterance("u2", "S2", 1.0, "고객센터에서 무엇을 도와드릴까요"),
    ]

    assert suggest_roles(utterances) == {"S1": Role.CUSTOMER, "S2": Role.AGENT}


def test_suggest_roles_ignores_cues_at_or_after_thirty_seconds() -> None:
    utterances = [
        _utterance("u1", "S1", 0.0, "일반 문의입니다"),
        _utterance("u2", "S2", 1.0, "네"),
        _utterance("u3", "S1", 30.0, "상담원 고객센터 무엇을 도와 도와드리"),
    ]

    assert suggest_roles(utterances) is None


def test_suggest_roles_requires_exactly_two_speakers_overall() -> None:
    utterances = [
        _utterance("u1", "S1", 0.0, "고객센터 상담원입니다"),
        _utterance("u2", "S2", 1.0, "문의가 있습니다"),
        _utterance("u3", "S3", 40.0, "뒤늦은 제3화자"),
    ]

    assert suggest_roles(utterances) is None


def test_suggest_roles_requires_unique_positive_lead() -> None:
    tied = [
        _utterance("u1", "S1", 0.0, "상담원입니다"),
        _utterance("u2", "S2", 1.0, "고객센터입니다"),
    ]
    no_cues = [
        _utterance("u1", "S1", 0.0, "문의가 있습니다"),
        _utterance("u2", "S2", 1.0, "네 말씀하세요"),
    ]

    assert suggest_roles(tied) is None
    assert suggest_roles(no_cues) is None


def test_suggest_roles_counts_only_explicit_opening_self_introduction() -> None:
    utterances = [
        _utterance("s2-late", "S2", 2.0, "저는 고객 김철수입니다"),
        _utterance("s1-second", "S1", 1.0, "처리 상태입니다"),
        _utterance("s1-first", "S1", 0.0, "안녕하세요. 저는 김민수입니다."),
        _utterance("s2-first", "S2", 0.5, "문의가 있습니다"),
    ]

    assert suggest_roles(utterances) == {"S1": Role.AGENT, "S2": Role.CUSTOMER}


@pytest.mark.parametrize(
    "introduction",
    [
        "안녕하세요, 김민수입니다.",
        "안녕하십니까? 홍길동입니다",
    ],
)
def test_suggest_roles_counts_common_greeted_name_introductions(introduction: str) -> None:
    utterances = [
        _utterance("s1-first", "S1", 0.0, introduction),
        _utterance("s2-first", "S2", 0.5, "문의가 있습니다"),
    ]

    assert suggest_roles(utterances) == {"S1": Role.AGENT, "S2": Role.CUSTOMER}


@pytest.mark.parametrize(
    "introduction",
    [
        "안녕하세요, 허준입니다.",
        "안녕하세요, 김민수입니다.",
        "안녕하세요, 남궁민수입니다.",
        "안녕하세요, 행복서비스 매니저 이영희입니다.",
    ],
)
def test_suggest_roles_accepts_reviewer_opening_introduction_examples(
    introduction: str,
) -> None:
    utterances = [
        _utterance("s1-first", "S1", 0.0, introduction),
        _utterance("s2-first", "S2", 0.5, "문의가 있습니다"),
    ]

    assert suggest_roles(utterances) == {"S1": Role.AGENT, "S2": Role.CUSTOMER}


@pytest.mark.parametrize("name", ["이준", "박서준", "제갈민수"])
def test_suggest_roles_accepts_two_to_four_hangul_name_tokens(name: str) -> None:
    utterances = [
        _utterance("s1-first", "S1", 0.0, f"안녕하십니까? {name}입니다"),
        _utterance("s2-first", "S2", 0.5, "문의가 있습니다"),
    ]

    assert suggest_roles(utterances) == {"S1": Role.AGENT, "S2": Role.CUSTOMER}


@pytest.mark.parametrize(
    "introduction",
    [
        "저는 허준입니다.",
        "제 이름은 김민수입니다.",
        "안녕하세요. 저는 남궁민수입니다.",
        "저는 행복서비스 담당자 이영희입니다.",
    ],
)
def test_suggest_roles_preserves_explicit_self_introduction_forms(
    introduction: str,
) -> None:
    utterances = [
        _utterance("s1-first", "S1", 0.0, introduction),
        _utterance("s2-first", "S2", 0.5, "문의가 있습니다"),
    ]

    assert suggest_roles(utterances) == {"S1": Role.AGENT, "S2": Role.CUSTOMER}


@pytest.mark.parametrize(
    "non_introduction",
    [
        "현재 처리 상태입니다.",
        "약관상 환불 불가입니다.",
        "안녕하세요, 정책입니다.",
        "안녕하십니까? 이용약관입니다.",
    ],
)
def test_suggest_roles_does_not_score_policy_or_status_sentences(
    non_introduction: str,
) -> None:
    utterances = [
        _utterance("s1-first", "S1", 0.0, non_introduction),
        _utterance("s2-first", "S2", 0.5, "문의가 있습니다"),
    ]

    assert suggest_roles(utterances) is None


@pytest.mark.parametrize(
    "status_token",
    [
        "정책",
        "규정",
        "정상",
        "오류",
        "완료",
        "예정",
        "불가",
        "가능",
        "처리중",
        "점검중",
        "확인중",
        "진행중",
        "처리완료",
        "점검예정",
        "이용약관",
        "처리불가",
    ],
)
def test_suggest_roles_rejects_two_to_four_hangul_status_or_policy_tokens(
    status_token: str,
) -> None:
    utterances = [
        _utterance("s1-first", "S1", 0.0, f"안녕하세요, {status_token}입니다."),
        _utterance("s2-first", "S2", 0.5, "문의가 있습니다"),
    ]

    assert suggest_roles(utterances) is None


def test_suggest_roles_does_not_treat_non_opening_or_generic_imnida_as_introduction() -> None:
    utterances = [
        _utterance("s1-first", "S1", 0.0, "처리 상태입니다"),
        _utterance("s2-first", "S2", 0.5, "문의가 있습니다"),
        _utterance("s1-second", "S1", 1.0, "저는 김민수입니다"),
    ]

    assert suggest_roles(utterances) is None
