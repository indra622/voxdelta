from __future__ import annotations

import errno
import json
import os
from pathlib import Path

import pytest

import voxdelta.jobs.artifacts as artifacts_module
from voxdelta.domain.models import AnalysisReport, StageName
from voxdelta.jobs.artifacts import ArtifactStore


def analysis_report(*, warning: str | None = None) -> AnalysisReport:
    return AnalysisReport(
        job_id="j1",
        summary={
            "start_state": "stable",
            "end_state": "stable",
            "peak_customer_utterance_id": "u1",
            "overall_delta": 0.0,
            "valid_coverage": 1.0,
            "recovery_count": 0,
            "worsening_count": 0,
        },
        utterances=[],
        emotions=[],
        strategies=[],
        transitions=[],
        warnings=[] if warning is None else [warning],
    )


def test_artifact_round_trip_retains_model_schema_version(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "jobs")
    report = analysis_report()

    path = store.write_model("j1", StageName.REPORT, report)

    assert path.name == "report.v1.json"
    assert json.loads(path.read_bytes()) == report.model_dump(mode="json")
    assert json.loads(path.read_bytes())["schema_version"] == "1"
    assert store.read_model("j1", StageName.REPORT, AnalysisReport) == report


def test_stage_artifact_hash_and_exact_deletion_do_not_touch_audio(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "jobs")
    report_path = store.write_model("j1", StageName.REPORT, analysis_report())
    audio_generation = store.job_dir("j1") / "audio-generation" / "mixed.wav"
    audio_generation.parent.mkdir()
    audio_generation.write_bytes(b"normalized audio")

    digest = store.content_hash("j1", StageName.REPORT)
    store.delete_stage("j1", StageName.REPORT)

    assert len(digest) == 64
    assert not report_path.exists()
    assert audio_generation.read_bytes() == b"normalized audio"


def test_artifact_write_replaces_from_a_temporary_in_the_target_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ArtifactStore(tmp_path / "jobs")
    replacements: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def observed_replace(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        source_path = Path(source)
        target_path = Path(target)
        assert source_path.parent == target_path.parent
        assert source_path != target_path
        assert source_path.exists()
        replacements.append((source_path, target_path))
        real_replace(source_path, target_path)

    monkeypatch.setattr(artifacts_module.os, "replace", observed_replace)

    target = store.write_model("j1", StageName.REPORT, analysis_report())

    assert len(replacements) == 1
    temporary, replacement_target = replacements[0]
    assert replacement_target == target
    assert not temporary.exists()


def test_artifact_write_cleans_temporary_and_preserves_target_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ArtifactStore(tmp_path / "jobs")
    original = analysis_report()
    target = store.write_model("j1", StageName.REPORT, original)

    def failed_replace(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(artifacts_module.os, "replace", failed_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        store.write_model("j1", StageName.REPORT, analysis_report(warning="new"))

    assert store.read_model("j1", StageName.REPORT, AnalysisReport) == original
    assert list(target.parent.iterdir()) == [target]


def test_artifact_write_fsyncs_containing_directory_after_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ArtifactStore(tmp_path / "jobs")
    expected_directory = tmp_path / "jobs" / "j1"
    events: list[tuple[str, Path]] = []
    real_replace = os.replace

    def observed_replace(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        real_replace(source, target)
        events.append(("replace", Path(target)))

    def observed_directory_fsync(directory: Path) -> None:
        assert (directory / "report.v1.json").exists()
        events.append(("directory_fsync", directory))

    monkeypatch.setattr(artifacts_module.os, "replace", observed_replace)
    monkeypatch.setattr(
        artifacts_module, "_fsync_directory", observed_directory_fsync, raising=False
    )

    target = store.write_model("j1", StageName.REPORT, analysis_report())

    assert events == [("replace", target), ("directory_fsync", expected_directory)]


@pytest.mark.skipif(os.name != "posix", reason="directory descriptors are POSIX-specific")
def test_directory_fsync_opens_syncs_and_closes_the_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[tuple[str, object]] = []
    expected_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)

    def observed_open(path: str | os.PathLike[str], flags: int) -> int:
        events.append(("open", (Path(path), flags)))
        return 41

    def observed_fsync(descriptor: int) -> None:
        events.append(("fsync", descriptor))

    def observed_close(descriptor: int) -> None:
        events.append(("close", descriptor))

    monkeypatch.setattr(artifacts_module.os, "open", observed_open)
    monkeypatch.setattr(artifacts_module.os, "fsync", observed_fsync)
    monkeypatch.setattr(artifacts_module.os, "close", observed_close)

    artifacts_module._fsync_directory(tmp_path)

    assert events == [
        ("open", (tmp_path, expected_flags)),
        ("fsync", 41),
        ("close", 41),
    ]


@pytest.mark.skipif(os.name != "posix", reason="directory descriptors are POSIX-specific")
def test_directory_fsync_falls_back_when_the_filesystem_does_not_support_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[int] = []
    monkeypatch.setattr(artifacts_module.os, "open", lambda path, flags: 42)

    def unsupported_fsync(descriptor: int) -> None:
        raise OSError(errno.EINVAL, "directory fsync unsupported")

    monkeypatch.setattr(artifacts_module.os, "fsync", unsupported_fsync)
    monkeypatch.setattr(artifacts_module.os, "close", closed.append)

    artifacts_module._fsync_directory(tmp_path)

    assert closed == [42]


@pytest.mark.parametrize(
    "job_id",
    [
        "",
        ".",
        "..",
        "../escape",
        "nested/job",
        r"nested\job",
        "/absolute",
        r"C:\absolute",
        "C:",
        "C:..",
        "C:foo",
    ],
)
@pytest.mark.parametrize("operation", ["job_dir", "write_model", "read_model", "delete_job"])
def test_every_artifact_path_operation_rejects_unsafe_job_ids(
    job_id: str, operation: str, tmp_path: Path
) -> None:
    store = ArtifactStore(tmp_path / "jobs")

    with pytest.raises(ValueError, match="job ID"):
        if operation == "job_dir":
            store.job_dir(job_id)
        elif operation == "write_model":
            store.write_model(job_id, StageName.REPORT, analysis_report())
        elif operation == "read_model":
            store.read_model(job_id, StageName.REPORT, AnalysisReport)
        else:
            store.delete_job(job_id)


@pytest.mark.parametrize("target_location", ["outside", "sibling"])
@pytest.mark.parametrize("operation", ["job_dir", "write_model", "read_model"])
def test_artifact_operations_reject_existing_job_directory_symlinks(
    target_location: str, operation: str, tmp_path: Path
) -> None:
    root = tmp_path / "jobs"
    root.mkdir()
    target = tmp_path / "outside" if target_location == "outside" else root / "j10"
    target.mkdir()
    artifact = target / "report.v1.json"
    original = analysis_report(warning="sentinel").model_dump_json()
    artifact.write_text(original, encoding="utf-8")
    (root / "j1").symlink_to(target, target_is_directory=True)
    store = ArtifactStore(root)

    with pytest.raises(ValueError, match="job directory"):
        if operation == "job_dir":
            store.job_dir("j1")
        elif operation == "write_model":
            store.write_model("j1", StageName.REPORT, analysis_report(warning="overwrite"))
        else:
            store.read_model("j1", StageName.REPORT, AnalysisReport)

    assert artifact.read_text(encoding="utf-8") == original


def test_delete_job_removes_only_the_exact_job_directory(tmp_path: Path) -> None:
    root = tmp_path / "jobs"
    store = ArtifactStore(root)
    (store.job_dir("j1") / "artifact.json").write_text("j1", encoding="utf-8")
    sibling = store.job_dir("j10") / "artifact.json"
    sibling.write_text("j10", encoding="utf-8")

    store.delete_job("j1")

    assert root.is_dir()
    assert not (root / "j1").exists()
    assert sibling.read_text(encoding="utf-8") == "j10"


def test_delete_job_rejects_a_symlink_that_resolves_outside_the_jobs_root(tmp_path: Path) -> None:
    root = tmp_path / "jobs"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    (root / "j1").symlink_to(outside, target_is_directory=True)
    store = ArtifactStore(root)

    with pytest.raises(ValueError, match="jobs root"):
        store.delete_job("j1")

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert (root / "j1").is_symlink()


def test_delete_job_rejects_a_symlink_to_a_sibling_job(tmp_path: Path) -> None:
    root = tmp_path / "jobs"
    sibling = root / "j10"
    sibling.mkdir(parents=True)
    sentinel = sibling / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    (root / "j1").symlink_to(sibling, target_is_directory=True)
    store = ArtifactStore(root)

    with pytest.raises(ValueError, match="jobs root"):
        store.delete_job("j1")

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert (root / "j1").is_symlink()
