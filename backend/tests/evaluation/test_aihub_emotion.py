from __future__ import annotations

import csv
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from voxdelta.evaluation.aihub_emotion import plan_emotion_import

HEADERS = (
    "wav_id",
    "발화문",
    "상황",
    "1번 감정",
    "1번 감정세기",
    "2번 감정",
    "2번 감정세기",
    "3번 감정",
    "3번 감정세기",
    "4번 감정",
    "4번감정세기",
    "5번 감정",
    "5번 감정세기",
    "나이",
    "성별",
)


def _row(item_id: str, votes: list[str]) -> dict[str, str]:
    assert len(votes) == 5
    row = {
        "wav_id": item_id,
        "발화문": "PRIVATE_TRANSCRIPT_SENTINEL",
        "상황": "fixture situation",
        "나이": "30대",
        "성별": "female",
    }
    for index, vote in enumerate(votes, start=1):
        row[f"{index}번 감정"] = vote
        intensity = f"{index}번감정세기" if index == 4 else f"{index}번 감정세기"
        row[intensity] = "1"
    return row


def _write_official_fixture(
    root: Path,
    *,
    rows: list[dict[str, str]],
    audio_ids: list[str] | None = None,
    extra_members: dict[str, bytes] | None = None,
) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "4차년도.csv"
    with csv_path.open("w", encoding="cp949", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    zip_path = root / "4차년도.zip"
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        for item_id in audio_ids if audio_ids is not None else [row["wav_id"] for row in rows]:
            archive.writestr(f"4차년도_wav/{item_id}.wav", b"fixture-wave")
        for member, payload in (extra_members or {}).items():
            archive.writestr(member, payload)
    return csv_path, zip_path


def test_plan_accepts_three_of_five_votes_and_omits_transcript(tmp_path: Path) -> None:
    csv_path, zip_path = _write_official_fixture(
        tmp_path,
        rows=[_row("item-a", ["angry", "angry", "angry", "neutral", "sadness"])],
    )

    plan = plan_emotion_import(
        csv_path,
        zip_path,
        max_missing_audio=0,
        max_orphan_audio=0,
    )

    assert [(row.item_id, row.emotion) for row in plan.accepted] == [("item-a", "anger")]
    assert plan.accepted[0].transcript == ""


def test_plan_quarantines_votes_without_three_person_consensus(tmp_path: Path) -> None:
    csv_path, zip_path = _write_official_fixture(
        tmp_path,
        rows=[_row("item-a", ["angry", "angry", "neutral", "neutral", "sadness"])],
    )

    plan = plan_emotion_import(
        csv_path,
        zip_path,
        max_missing_audio=0,
        max_orphan_audio=0,
    )

    assert plan.accepted == ()
    assert plan.ambiguous_ids == ("item-a",)


def test_plan_rejects_duplicate_csv_ids(tmp_path: Path) -> None:
    duplicate = _row("item-a", ["neutral"] * 5)
    csv_path, zip_path = _write_official_fixture(
        tmp_path,
        rows=[duplicate, duplicate],
        audio_ids=["item-a"],
    )

    with pytest.raises(ValueError, match="duplicate metadata ID"):
        plan_emotion_import(csv_path, zip_path, max_missing_audio=0, max_orphan_audio=0)


def test_plan_rejects_duplicate_zip_stems(tmp_path: Path) -> None:
    csv_path, zip_path = _write_official_fixture(
        tmp_path,
        rows=[_row("item-a", ["neutral"] * 5)],
        audio_ids=[],
        extra_members={
            "first/item-a.wav": b"first",
            "second/item-a.wav": b"second",
        },
    )

    with pytest.raises(ValueError, match="duplicate archive audio ID"):
        plan_emotion_import(csv_path, zip_path, max_missing_audio=0, max_orphan_audio=0)


def test_plan_rejects_non_wav_archive_members(tmp_path: Path) -> None:
    csv_path, zip_path = _write_official_fixture(
        tmp_path,
        rows=[_row("item-a", ["neutral"] * 5)],
        extra_members={"PRIVATE_TRANSCRIPT_SENTINEL.txt": b"private"},
    )

    with pytest.raises(ValueError, match="unexpected archive member") as raised:
        plan_emotion_import(csv_path, zip_path, max_missing_audio=0, max_orphan_audio=0)

    assert "PRIVATE_TRANSCRIPT_SENTINEL" not in str(raised.value)


def test_plan_rejects_invalid_cp949_without_echoing_bytes(tmp_path: Path) -> None:
    csv_path, zip_path = _write_official_fixture(
        tmp_path,
        rows=[_row("item-a", ["neutral"] * 5)],
    )
    csv_path.write_bytes(b"PRIVATE_TRANSCRIPT_SENTINEL\xff")

    with pytest.raises(ValueError, match="invalid CP949 emotion metadata") as raised:
        plan_emotion_import(csv_path, zip_path, max_missing_audio=0, max_orphan_audio=0)

    assert "PRIVATE_TRANSCRIPT_SENTINEL" not in str(raised.value)


def test_plan_rejects_wrong_official_headers(tmp_path: Path) -> None:
    csv_path, zip_path = _write_official_fixture(
        tmp_path,
        rows=[_row("item-a", ["neutral"] * 5)],
    )
    csv_path.write_text("wav_id,wrong\nitem-a,value\n", encoding="cp949")

    with pytest.raises(ValueError, match="invalid AI Hub 263 metadata headers"):
        plan_emotion_import(csv_path, zip_path, max_missing_audio=0, max_orphan_audio=0)


def test_plan_reports_explicitly_allowed_unmatched_ids(tmp_path: Path) -> None:
    csv_path, zip_path = _write_official_fixture(
        tmp_path,
        rows=[
            _row("matched", ["neutral"] * 5),
            _row("missing", ["sadness"] * 5),
        ],
        audio_ids=["matched", "orphan"],
    )

    plan = plan_emotion_import(
        csv_path,
        zip_path,
        max_missing_audio=1,
        max_orphan_audio=1,
    )

    assert plan.missing_audio_ids == ("missing",)
    assert plan.orphan_audio_ids == ("orphan",)
    assert [row.item_id for row in plan.accepted] == ["matched"]


def test_plan_rejects_unmatched_ids_above_allowance(tmp_path: Path) -> None:
    csv_path, zip_path = _write_official_fixture(
        tmp_path,
        rows=[_row("missing", ["neutral"] * 5)],
        audio_ids=["orphan"],
    )

    with pytest.raises(ValueError, match="mismatch exceeds allowance"):
        plan_emotion_import(csv_path, zip_path, max_missing_audio=0, max_orphan_audio=0)


def test_plan_rejects_symlinked_zip(tmp_path: Path) -> None:
    csv_path, zip_path = _write_official_fixture(
        tmp_path,
        rows=[_row("item-a", ["neutral"] * 5)],
    )
    linked = tmp_path / "linked.zip"
    try:
        linked.symlink_to(zip_path)
    except OSError as error:  # pragma: no cover - platform capability guard
        pytest.skip(f"symlinks unavailable: {error}")

    with pytest.raises(ValueError, match="symlink"):
        plan_emotion_import(csv_path, linked, max_missing_audio=0, max_orphan_audio=0)
