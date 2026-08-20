from __future__ import annotations

import csv
import json
import os
import struct
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

import pytest

import voxdelta.evaluation.aihub_emotion as aihub_emotion
from voxdelta.evaluation.aihub_emotion import import_emotion_dataset, plan_emotion_import
from voxdelta.evaluation.manifest import load_manifest

BACKEND = Path(__file__).resolve().parents[2]

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
    release_name: str = "4차년도",
    force_zip64: bool = False,
) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / f"{release_name}.csv"
    with csv_path.open("w", encoding="cp949", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    zip_path = root / f"{release_name}.zip"
    compression = ZIP_STORED if force_zip64 else ZIP_DEFLATED
    with ZipFile(zip_path, "w", compression=compression) as archive:
        for item_id in audio_ids if audio_ids is not None else [row["wav_id"] for row in rows]:
            member_name = f"{release_name}_wav/{item_id}.wav"
            if force_zip64:
                info = ZipInfo(member_name)
                info.compress_type = ZIP_STORED
                with archive.open(info, "w", force_zip64=True) as member:
                    member.write(b"fixture-wave")
            else:
                archive.writestr(member_name, b"fixture-wave")
        for member, payload in (extra_members or {}).items():
            archive.writestr(member, payload)
    return csv_path, zip_path


def _write_three_release_fixture(root: Path) -> Path:
    for index, release_name in enumerate(("4차년도", "5차년도", "5차년도_2차")):
        _write_official_fixture(
            root,
            rows=[_row(f"item-{index}", ["angry"] * 5)],
            release_name=release_name,
            force_zip64=release_name == "5차년도",
        )
    return root


def _run_preparer(source: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/prepare_aihub_emotion.py",
            "--source-root",
            str(source),
            "--output-root",
            str(output),
            "--max-missing-audio",
            "0",
            "--max-orphan-audio",
            "0",
        ],
        cwd=BACKEND,
        text=True,
        capture_output=True,
        check=False,
    )


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


def test_plan_rejects_malformed_row_without_echoing_values(tmp_path: Path) -> None:
    csv_path, zip_path = _write_official_fixture(
        tmp_path,
        rows=[_row("item-a", ["neutral"] * 5)],
    )
    header = ",".join(HEADERS)
    csv_path.write_text(
        f"{header}\nitem-a,PRIVATE_TRANSCRIPT_SENTINEL\n",
        encoding="cp949",
    )

    with pytest.raises(ValueError, match="malformed AI Hub 263 metadata row") as raised:
        plan_emotion_import(csv_path, zip_path, max_missing_audio=0, max_orphan_audio=0)

    assert "PRIVATE_TRANSCRIPT_SENTINEL" not in str(raised.value)


def test_plan_rejects_audio_member_above_size_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path, zip_path = _write_official_fixture(
        tmp_path,
        rows=[_row("item-a", ["neutral"] * 5)],
    )
    monkeypatch.setattr(aihub_emotion, "_MAX_AUDIO_BYTES", 4)

    with pytest.raises(ValueError, match="audio member exceeds size limit"):
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


def test_import_writes_sharded_pairs_and_transcript_free_report(tmp_path: Path) -> None:
    source = _write_three_release_fixture(tmp_path / "source")
    output = tmp_path / "normalized"

    report = import_emotion_dataset(
        source,
        output,
        max_missing_audio=0,
        max_orphan_audio=0,
    )

    metadata_path = output / "pairs" / "it" / "item-0.json"
    audio_path = output / "pairs" / "it" / "item-0.wav"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    report_text = (output / "import-report.json").read_text(encoding="utf-8")
    assert audio_path.read_bytes() == b"fixture-wave"
    assert metadata["transcript"] == ""
    assert metadata["emotion"] == "anger"
    assert metadata["call_id"] == "aihub-263-item:item-0"
    assert metadata["speaker_id"] == "aihub-263-item:item-0"
    assert report.accepted == 3
    assert "PRIVATE_TRANSCRIPT_SENTINEL" not in report_text
    assert os.stat(audio_path).st_mode & 0o777 == 0o600
    assert os.stat(metadata_path).st_mode & 0o777 == 0o600


def test_import_preserves_source_bytes(tmp_path: Path) -> None:
    source = _write_three_release_fixture(tmp_path / "source")
    before = {path.name: path.read_bytes() for path in source.iterdir()}

    import_emotion_dataset(
        source,
        tmp_path / "normalized",
        max_missing_audio=0,
        max_orphan_audio=0,
    )

    assert {path.name: path.read_bytes() for path in source.iterdir()} == before


def test_import_rejects_preexisting_output(tmp_path: Path) -> None:
    source = _write_three_release_fixture(tmp_path / "source")
    output = tmp_path / "normalized"
    output.mkdir()

    with pytest.raises(ValueError, match="output already exists"):
        import_emotion_dataset(
            source,
            output,
            max_missing_audio=0,
            max_orphan_audio=0,
        )


def test_import_rejects_symlinked_output_ancestor(tmp_path: Path) -> None:
    source = _write_three_release_fixture(tmp_path / "source")
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(real, target_is_directory=True)
    except OSError as error:  # pragma: no cover - platform capability guard
        pytest.skip(f"symlinks unavailable: {error}")

    with pytest.raises(ValueError, match="symlink"):
        import_emotion_dataset(
            source,
            linked / "normalized",
            max_missing_audio=0,
            max_orphan_audio=0,
        )


def test_import_rejects_incomplete_release_set(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_official_fixture(
        source,
        rows=[_row("item-a", ["neutral"] * 5)],
        release_name="4차년도",
    )

    with pytest.raises(ValueError, match="official release set"):
        import_emotion_dataset(
            source,
            tmp_path / "normalized",
            max_missing_audio=0,
            max_orphan_audio=0,
        )


def test_import_rejects_total_audio_above_size_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_three_release_fixture(tmp_path / "source")
    monkeypatch.setattr(aihub_emotion, "_MAX_TOTAL_AUDIO_BYTES", 20)

    with pytest.raises(ValueError, match="dataset audio exceeds size limit"):
        import_emotion_dataset(
            source,
            tmp_path / "normalized",
            max_missing_audio=0,
            max_orphan_audio=0,
        )


def test_import_failure_leaves_no_published_or_staging_output(tmp_path: Path) -> None:
    source = _write_three_release_fixture(tmp_path / "source")
    corrupt = source / "5차년도.zip"
    payload = bytearray(corrupt.read_bytes())
    marker = payload.index(b"fixture-wave")
    payload[marker] ^= 0x01
    corrupt.write_bytes(payload)
    output = tmp_path / "normalized"

    with pytest.raises(ValueError, match="archive read failed"):
        import_emotion_dataset(
            source,
            output,
            max_missing_audio=0,
            max_orphan_audio=0,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".normalized.*"))


def test_cli_prepares_pairs_consumed_by_manifest_builder(tmp_path: Path) -> None:
    source = _write_three_release_fixture(tmp_path / "source")
    normalized = tmp_path / "normalized"

    prepared = _run_preparer(source, normalized)

    assert prepared.returncode == 0
    assert prepared.stdout == ""
    assert prepared.stderr == ""
    consultation = tmp_path / "empty-consultation"
    consultation.mkdir()
    manifests = tmp_path / "manifests"
    built = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/build_aihub_manifests.py",
            "--consultation-root",
            str(consultation),
            "--emotion-root",
            str(normalized / "pairs"),
            "--output-root",
            str(manifests),
        ],
        cwd=BACKEND,
        text=True,
        capture_output=True,
        check=False,
    )
    assert built.returncode == 0, built.stderr
    items = load_manifest(manifests / "emotion.jsonl")
    assert len(items) == 3
    assert {item.source for item in items} == {"emotion"}
    assert {item.transcript for item in items} == {""}


def test_cli_sanitizes_import_failures(tmp_path: Path) -> None:
    source = tmp_path / "PRIVATE_SOURCE_SENTINEL"
    source.mkdir()
    output = tmp_path / "PRIVATE_OUTPUT_SENTINEL"

    result = _run_preparer(source, output)

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "emotion import failed\n"
    assert "PRIVATE_SOURCE_SENTINEL" not in result.stderr
    assert "PRIVATE_OUTPUT_SENTINEL" not in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_sanitizes_unsupported_zip_compression(tmp_path: Path) -> None:
    source = _write_three_release_fixture(tmp_path / "source")
    archive_path = source / "4차년도.zip"
    payload = bytearray(archive_path.read_bytes())
    local_header = payload.index(b"PK\x03\x04")
    central_header = payload.index(b"PK\x01\x02")
    struct.pack_into("<H", payload, local_header + 8, 99)
    struct.pack_into("<H", payload, central_header + 10, 99)
    archive_path.write_bytes(payload)

    result = _run_preparer(source, tmp_path / "normalized")

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "emotion import failed\n"
    assert "Traceback" not in result.stderr
