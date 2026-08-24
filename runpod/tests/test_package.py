from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
import wave
from collections import Counter
from pathlib import Path

import pytest
from voxdelta.evaluation.manifest import DatasetItem

from voxdelta_runpod.config import CANONICAL_LABELS, load_experiment_config
from voxdelta_runpod.package import (
    AuditDecision,
    PackageError,
    PackageSidecar,
    _publish_package,
    archive_members_are_safe,
    derive_holdout_identity,
    extract_verified_package,
    select_audit_queue,
    summarize_audit,
)

RUNPOD = Path(__file__).parents[1]
CONFIG = RUNPOD / "config" / "experiment.toml"


def _sha(index: int) -> str:
    return hashlib.sha256(f"audio-{index}".encode()).hexdigest()


def _item(index: int, split: str, label: str, audio: Path | None = None) -> DatasetItem:
    return DatasetItem.model_validate(
        {
            "id": f"private-id-{index}",
            "call_id": f"private-call-{index}",
            "speaker_id": f"private-speaker-{index}",
            "audio_path": str(audio or Path(f"/nonexistent/private-{index}.wav")),
            "transcript": f"private transcript {index}",
            "split": split,
            "source": "emotion",
            "emotion": label,
            "start": None,
            "end": None,
            "sha256": _sha(index),
        }
    )


def _manifest(path: Path, items: list[DatasetItem]) -> Path:
    path.write_text("".join(item.model_dump_json() + "\n" for item in items))
    path.chmod(0o600)
    return path.resolve()


def _holdout_manifests(tmp_path: Path) -> tuple[Path, Path]:
    holdout_counts = {
        "anger": 695,
        "disgust": 212,
        "fear": 216,
        "happiness": 330,
        "neutral": 578,
        "sadness": 1_485,
        "surprise": 69,
    }
    full: list[DatasetItem] = []
    exposed: list[DatasetItem] = []
    index = 0
    for label in CANONICAL_LABELS:
        for _ in range(holdout_counts[label] + 5):
            item = _item(index, "test", label)
            full.append(item)
            if sum(1 for candidate in exposed if candidate.emotion == label) < 5:
                exposed.append(item.model_copy(update={"transcript": ""}))
            index += 1
    return _manifest(tmp_path / "full.jsonl", full), _manifest(tmp_path / "smoke.jsonl", exposed)


def _wav(path: Path, sample: int) -> Path:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(sample.to_bytes(2, "little", signed=True) * 32)
    path.chmod(0o600)
    return path.resolve()


def _audio_item(index: int, split: str, label: str, path: Path) -> DatasetItem:
    item = _item(index, split, label, path)
    return item.model_copy(update={"sha256": hashlib.sha256(path.read_bytes()).hexdigest()})


def test_holdout_identity_excludes_exact_exposed_fingerprints_without_audio_io(
    tmp_path: Path,
) -> None:
    source, exposed = _holdout_manifests(tmp_path)
    config = load_experiment_config(CONFIG.resolve())

    identity = derive_holdout_identity(source, exposed, config)

    assert identity.nominal_test_count == 3_620
    assert identity.exposed_count == 35
    assert identity.final_holdout_count == 3_585
    assert identity.exposed_label_counts == {label: 5 for label in CANONICAL_LABELS}
    assert identity.holdout_label_counts == {
        "anger": 695,
        "disgust": 212,
        "fear": 216,
        "happiness": 330,
        "neutral": 578,
        "sadness": 1_485,
        "surprise": 69,
    }
    assert len(identity.identity_sha256) == 64


def test_holdout_identity_rejects_wrong_exposed_label_distribution(tmp_path: Path) -> None:
    source, exposed = _holdout_manifests(tmp_path)
    rows = exposed.read_text().splitlines()
    records = [json.loads(row) for row in rows]
    source_records = [json.loads(row) for row in source.read_text().splitlines()]
    sixth_anger = next(
        record
        for record in source_records
        if record["emotion"] == "anger"
        and record["sha256"] not in {candidate["sha256"] for candidate in records}
    )
    records[-1] = sixth_anger
    _manifest(exposed, [DatasetItem.model_validate(record) for record in records])

    with pytest.raises(PackageError, match="^invalid_exposed_holdout$"):
        derive_holdout_identity(source, exposed, load_experiment_config(CONFIG.resolve()))


def test_audit_queue_is_deterministic_balanced_and_summary_drops_notes(tmp_path: Path) -> None:
    items: list[DatasetItem] = []
    losses: dict[str, float] = {}
    index = 0
    for label in CANONICAL_LABELS:
        for _ in range(20):
            items.append(_item(index, "train", label))
            index += 1
        for rank in range(20):
            item = _item(index, "validation", label)
            items.append(item)
            losses[item.sha256] = float(rank)
            index += 1
    source = _manifest(tmp_path / "audit.jsonl", items)

    queue = select_audit_queue(source, losses)
    repeated = select_audit_queue(source, losses)

    assert queue == repeated
    assert len(queue) == 210
    assert Counter((item.emotion, item.split) for item in queue) == Counter(
        {(label, split): 15 for label in CANONICAL_LABELS for split in ("train", "validation")}
    )
    decisions = [
        AuditDecision(item_key=item.item_key, outcome="agree", note="private reviewer note")
        for item in queue
    ]
    summary = summarize_audit(queue, decisions)
    assert summary.total_count == 210
    assert all(outcomes["agree"] == 30 for outcomes in summary.label_outcome_counts.values())
    assert "private reviewer note" not in summary.model_dump_json()


def test_small_package_is_deterministic_private_and_contains_allowlisted_fields(
    tmp_path: Path,
) -> None:
    items: list[DatasetItem] = []
    for index, label in enumerate(CANONICAL_LABELS):
        path = _wav(tmp_path / f"source-{index}.wav", index + 1)
        items.append(_audio_item(index, "train", label, path))
    for offset, label in enumerate(CANONICAL_LABELS, start=len(CANONICAL_LABELS)):
        path = _wav(tmp_path / f"source-{offset}.wav", offset + 1)
        items.append(_audio_item(offset, "validation", label, path))

    first = _publish_package(
        items,
        (tmp_path / "packet-a").resolve(),
        package_kind="train-validation",
        archive_name="train-validation.tar.zst",
        sidecar_name="train-validation.sidecar.json",
    )
    second = _publish_package(
        items,
        (tmp_path / "packet-b").resolve(),
        package_kind="train-validation",
        archive_name="train-validation.tar.zst",
        sidecar_name="train-validation.sidecar.json",
    )

    archive_a = first / "train-validation.tar.zst"
    archive_b = second / "train-validation.tar.zst"
    assert (
        hashlib.sha256(archive_a.read_bytes()).digest()
        == hashlib.sha256(archive_b.read_bytes()).digest()
    )
    assert first.stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o077 == 0 for path in first.iterdir())
    sidecar = PackageSidecar.model_validate_json(
        (first / "train-validation.sidecar.json").read_text()
    )
    assert sidecar.audio_file_count == 14
    assert sidecar.archive_member_count == 15
    assert sidecar.split_counts == {"train": 7, "validation": 7}
    assert sidecar.label_counts == {label: 2 for label in CANONICAL_LABELS}

    tar_path = tmp_path / "unpacked.tar"
    with tar_path.open("wb") as output:
        completed = subprocess.run(
            ["zstd", "--quiet", "--decompress", "--stdout", str(archive_a)],
            stdout=output,
            check=False,
        )
    assert completed.returncode == 0
    with tarfile.open(tar_path, "r:") as stream:
        names = stream.getnames()
        assert archive_members_are_safe(names)
        manifest_stream = stream.extractfile("manifest.jsonl")
        assert manifest_stream is not None
        records = [json.loads(line) for line in manifest_stream.read().splitlines()]
    assert set(records[0]) == {"item_key", "audio_path", "split", "emotion", "audio_sha256"}
    serialized = json.dumps(records)
    for private_value in (
        "private-id",
        "private-call",
        "private-speaker",
        "transcript",
        str(tmp_path),
    ):
        assert private_value not in serialized

    extracted = extract_verified_package(
        archive_a.resolve(),
        (first / "train-validation.sidecar.json").resolve(),
        (tmp_path / "extracted").resolve(),
    )
    assert len(tuple((extracted / "audio").glob("*.wav"))) == 14
    assert (extracted / "manifest.jsonl").stat().st_mode & 0o077 == 0

    with pytest.raises(PackageError, match="^package_exists$"):
        _publish_package(
            items,
            first,
            package_kind="train-validation",
            archive_name="train-validation.tar.zst",
            sidecar_name="train-validation.sidecar.json",
        )


def test_package_rejects_symlinked_or_non_pcm16_audio(tmp_path: Path) -> None:
    real = _wav(tmp_path / "real.wav", 1)
    linked = tmp_path / "linked.wav"
    linked.symlink_to(real)
    symlinked = _audio_item(1, "train", "anger", real).model_copy(
        update={"audio_path": str(linked)}
    )
    with pytest.raises(PackageError, match="^invalid_package_audio$"):
        _publish_package(
            [symlinked],
            (tmp_path / "symlink-packet").resolve(),
            package_kind="train-validation",
            archive_name="train-validation.tar.zst",
            sidecar_name="train-validation.sidecar.json",
        )

    stereo = tmp_path / "stereo.wav"
    with wave.open(str(stereo), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\0\0" * 4)
    invalid = _audio_item(2, "train", "anger", stereo.resolve())
    with pytest.raises(PackageError, match="^invalid_package_audio$"):
        _publish_package(
            [invalid],
            (tmp_path / "stereo-packet").resolve(),
            package_kind="train-validation",
            archive_name="train-validation.tar.zst",
            sidecar_name="train-validation.sidecar.json",
        )


def test_package_rejects_symlinked_output_parent(tmp_path: Path) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(PackageError, match="^invalid_package_target$"):
        _publish_package(
            [],
            linked_parent / "packet",
            package_kind="train-validation",
            archive_name="train-validation.tar.zst",
            sidecar_name="train-validation.sidecar.json",
        )


@pytest.mark.parametrize("names", [["../escape"], ["/absolute"], ["audio/not-a-key.wav"]])
def test_archive_member_allowlist_rejects_traversal_and_unexpected_names(names: list[str]) -> None:
    assert not archive_members_are_safe(names)


def test_package_cli_error_is_stable_and_path_free(tmp_path: Path) -> None:
    private = tmp_path / "private-person-name.jsonl"
    completed = subprocess.run(
        [
            "python",
            str(RUNPOD / "scripts" / "package_data.py"),
            "training",
            "--config",
            str(CONFIG),
            "--manifest",
            str(private),
            "--output",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert completed.stdout == "package_error\n"
    assert completed.stderr == ""
    assert "private-person" not in completed.stdout + completed.stderr
