from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
import time
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


@pytest.mark.skipif(os.name != "posix", reason="cross-process flock is POSIX-specific")
def test_job_operation_lock_serializes_separate_processes(tmp_path: Path) -> None:
    jobs_root = tmp_path / "jobs"
    store = ArtifactStore(jobs_root)
    store.job_dir("j1")
    marker = tmp_path / "child-acquired"
    script = "\n".join(
        (
            "from pathlib import Path",
            "from voxdelta.jobs.artifacts import ArtifactStore",
            f"store = ArtifactStore(Path({str(jobs_root)!r}))",
            "with store.operation_lock('j1'):",
            f"    Path({str(marker)!r}).write_text('acquired')",
        )
    )

    with store.operation_lock("j1"):
        child = subprocess.Popen([sys.executable, "-c", script])
        time.sleep(0.2)
        assert not marker.exists()

    assert child.wait(timeout=5) == 0
    assert marker.read_text(encoding="utf-8") == "acquired"


def test_job_operation_lock_rejects_a_linked_lock_file(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "jobs")
    lock_directory = store.root / ".locks"
    lock_directory.mkdir(parents=True)
    outside = tmp_path / "outside.lock"
    outside.write_text("outside", encoding="utf-8")
    (lock_directory / "j1.lock").symlink_to(outside)

    with pytest.raises(ValueError, match="operation lock"):
        with store.operation_lock("j1"):
            pytest.fail("linked operation lock was accepted")

    assert outside.read_text(encoding="utf-8") == "outside"


def test_external_job_operation_lock_rejects_linked_control_directory(tmp_path: Path) -> None:
    root = tmp_path / "jobs"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / ".locks").symlink_to(outside, target_is_directory=True)
    store = ArtifactStore(root)

    with pytest.raises(ValueError, match="operation lock"):
        with store.operation_lock("j1"):
            pytest.fail("linked external operation lock directory was accepted")

    assert not list(outside.iterdir())


@pytest.mark.parametrize("cycle", ["self", "two-link"])
def test_job_operation_lock_rejects_cyclic_links_without_disclosing_paths(
    cycle: str,
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "jobs")
    lock_directory = store.root / ".locks"
    lock_directory.mkdir(parents=True)
    lock_path = lock_directory / "j1.lock"
    if cycle == "self":
        lock_path.symlink_to(lock_path.name)
    else:
        second = lock_directory / ".operation-lock-cycle"
        lock_path.symlink_to(second.name)
        second.symlink_to(lock_path.name)

    with pytest.raises(ValueError) as raised:
        with store.operation_lock("j1"):
            pytest.fail("cyclic operation lock was accepted")

    assert str(raised.value) == "job operation lock could not be validated safely"
    assert str(tmp_path) not in str(raised.value)


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


def test_operation_lock_keeps_durable_file_but_releases_process_registry_entry(
    tmp_path: Path,
) -> None:
    root = tmp_path / "jobs"
    store = ArtifactStore(root)
    lock_path = root / ".locks" / "j1.lock"

    with store.operation_lock("j1"):
        assert lock_path.absolute() in artifacts_module._OPERATION_LOCKS  # noqa: SLF001

    assert lock_path.is_file()
    assert lock_path.absolute() not in artifacts_module._OPERATION_LOCKS  # noqa: SLF001


def test_incoming_upload_is_private_then_atomically_adopted_into_exact_new_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "jobs"
    store = ArtifactStore(root)
    synced: list[Path] = []
    monkeypatch.setattr(artifacts_module, "_fsync_directory", synced.append)
    descriptor, incoming = store.open_incoming_upload("j1", ".wav")
    os.write(descriptor, b"admitted")
    os.fsync(descriptor)
    os.close(descriptor)

    adopted = store.adopt_incoming_upload("j1", incoming, ".wav")

    assert incoming.parent == root / ".incoming"
    assert not incoming.exists()
    assert adopted == root / "j1" / "source-upload.wav"
    assert adopted.read_bytes() == b"admitted"
    assert adopted.stat().st_mode & 0o777 == 0o600
    assert (root / ".incoming").stat().st_mode & 0o777 == 0o700
    assert not (root / ".deleted").exists()
    assert not (root / ".locks").exists()
    assert root / ".incoming" in synced
    assert adopted.parent in synced
    assert root in synced


def test_active_old_incoming_upload_is_owned_across_writer_close_until_adoption(
    tmp_path: Path,
) -> None:
    root = tmp_path / "jobs"
    owner = ArtifactStore(root)
    reconciler = ArtifactStore(root)
    descriptor, incoming = owner.open_incoming_upload("a" * 32, ".wav")
    os.write(descriptor, b"admitted")
    os.close(descriptor)
    os.utime(incoming, (0, 0))
    lease_descriptor = artifacts_module._INCOMING_OWNERS[  # noqa: SLF001
        incoming.absolute()
    ].lease_descriptor

    assert reconciler.remove_stale_incoming_uploads(lease_seconds=10, now=1000) == 0
    assert incoming.is_file()

    adopted = owner.adopt_incoming_upload("a" * 32, incoming, ".wav")

    assert adopted.read_bytes() == b"admitted"
    assert incoming.absolute() not in artifacts_module._INCOMING_OWNERS  # noqa: SLF001
    if lease_descriptor is not None:
        with pytest.raises(OSError) as closed:
            os.fstat(lease_descriptor)
        assert closed.value.errno == errno.EBADF


def test_non_posix_incoming_owner_registry_preserves_active_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(artifacts_module, "fcntl", None)
    root = tmp_path / "jobs"
    owner = ArtifactStore(root)
    descriptor, incoming = owner.open_incoming_upload("c" * 32, ".wav")
    os.close(descriptor)
    os.utime(incoming, (0, 0))

    assert (
        ArtifactStore(root).remove_stale_incoming_uploads(
            lease_seconds=10,
            now=1000,
        )
        == 0
    )
    owner.discard_incoming_upload(incoming)

    assert not incoming.exists()
    assert incoming.absolute() not in artifacts_module._INCOMING_OWNERS  # noqa: SLF001


@pytest.mark.skipif(os.name != "posix", reason="cross-process flock is POSIX-specific")
def test_stale_incoming_becomes_reclaimable_after_owner_process_exits(tmp_path: Path) -> None:
    root = tmp_path / "jobs"
    script = "\n".join(
        (
            "import os, sys",
            "from pathlib import Path",
            "from voxdelta.jobs.artifacts import ArtifactStore",
            f"store = ArtifactStore(Path({str(root)!r}))",
            "descriptor, incoming = store.open_incoming_upload('b' * 32, '.wav')",
            "os.write(descriptor, b'crash-leftover')",
            "os.close(descriptor)",
            "os.utime(incoming, (0, 0))",
            "print(incoming, flush=True)",
            "sys.stdin.read()",
        )
    )
    child = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        incoming = Path(child.stdout.readline().strip())
        reconciler = ArtifactStore(root)

        assert reconciler.remove_stale_incoming_uploads(lease_seconds=10, now=1000) == 0
        assert incoming.is_file()

        assert child.stdin is not None
        child.stdin.close()
        assert child.wait(timeout=5) == 0

        assert reconciler.remove_stale_incoming_uploads(lease_seconds=10, now=1000) == 1
        assert not incoming.exists()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def test_discard_incoming_leaves_no_tombstone_lock_or_public_job_discard(tmp_path: Path) -> None:
    root = tmp_path / "jobs"
    store = ArtifactStore(root)
    descriptor, incoming = store.open_incoming_upload("j1", ".wav")
    os.close(descriptor)
    store.discard_incoming_upload(incoming)

    assert not incoming.exists()
    assert not (root / ".deleted").exists()
    assert not (root / ".locks").exists()
    assert not hasattr(store, "discard_unregistered_job")
    assert incoming.absolute() not in artifacts_module._INCOMING_OWNERS  # noqa: SLF001


def test_unregistered_job_discard_rejects_direct_call_without_absence_proof(
    tmp_path: Path,
) -> None:
    root = tmp_path / "jobs"
    store = ArtifactStore(root)
    job_id = "a" * 32
    sentinel = store.job_dir(job_id) / "sentinel"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(ValueError, match="absence proof"):
        store._discard_unregistered_job_under_absence_proof(  # noqa: SLF001
            job_id,
            None,
        )

    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_existing_tombstone_fsync_failure_is_retried_before_job_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "jobs"
    store = ArtifactStore(root)
    store.job_dir("j1")
    store.mark_deletion_tombstone("j1")
    real_fsync = artifacts_module.os.fsync
    fsync_attempts = 0

    def fail_once(descriptor: int) -> None:
        nonlocal fsync_attempts
        fsync_attempts += 1
        if fsync_attempts == 1:
            raise OSError("tombstone fsync interrupted")
        real_fsync(descriptor)

    monkeypatch.setattr(artifacts_module.os, "fsync", fail_once)

    with pytest.raises(OSError, match="interrupted"):
        store.delete_job("j1")
    assert (root / "j1").is_dir()
    store.delete_job("j1")

    assert fsync_attempts >= 2
    assert not (root / "j1").exists()


def test_partial_atomic_tombstone_repair_failure_retries_and_preserves_expected_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "jobs"
    store = ArtifactStore(root)
    store.job_dir("j1")
    deleted = root / ".deleted"
    deleted.mkdir(mode=0o700)
    tombstone = deleted / "j1.tombstone"
    tombstone.write_bytes(b"j")
    tombstone.chmod(0o600)
    real_write = artifacts_module.os.write
    write_calls = 0

    def fail_after_partial_write(descriptor: int, payload: object) -> int:
        nonlocal write_calls
        write_calls += 1
        if write_calls == 1:
            return real_write(descriptor, bytes(payload)[:1])
        if write_calls == 2:
            raise OSError("partial tombstone repair")
        return real_write(descriptor, payload)  # type: ignore[arg-type]

    monkeypatch.setattr(artifacts_module.os, "write", fail_after_partial_write)

    with pytest.raises(OSError, match="partial tombstone repair"):
        store.delete_job("j1")
    assert tombstone.read_bytes() == b"j"
    assert (root / "j1").is_dir()
    assert not list(deleted.glob("*.tmp"))

    store.delete_job("j1")

    assert tombstone.read_bytes() == b"j1"
    assert not (root / "j1").exists()
    assert not list(deleted.glob("*.tmp"))


def test_delete_job_persists_tombstone_that_blocks_stale_process_logger(tmp_path: Path) -> None:
    root = tmp_path / "jobs"
    store = ArtifactStore(root)
    store.job_dir("j1")
    ready = tmp_path / "ready"
    release = tmp_path / "release"
    blocked = tmp_path / "blocked"
    recreated = tmp_path / "recreated"
    script = "\n".join(
        (
            "import time",
            "from pathlib import Path",
            "from voxdelta.jobs.artifacts import ArtifactStore",
            "from voxdelta.jobs.logging import PipelineLogger",
            f"root = Path({str(root)!r})",
            "store = ArtifactStore(root)",
            "store.job_dir('j1')",
            f"Path({str(ready)!r}).write_text('ready')",
            f"release = Path({str(release)!r})",
            "while not release.exists(): time.sleep(0.01)",
            "try:",
            "    PipelineLogger(store).event(",
            "        job_id='j1', stage='report', event='late', duration_ms=0,",
            "        provider=None, model=None, error_code=None,",
            "    )",
            "except FileNotFoundError:",
            f"    Path({str(blocked)!r}).write_text('blocked')",
            "else:",
            f"    Path({str(recreated)!r}).write_text('recreated')",
        )
    )
    child = subprocess.Popen([sys.executable, "-c", script])
    deadline = time.monotonic() + 5
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ready.exists()

    store.delete_job("j1")
    release.write_text("release", encoding="utf-8")

    assert child.wait(timeout=5) == 0
    assert blocked.read_text(encoding="utf-8") == "blocked"
    assert not recreated.exists()
    assert not (root / "j1").exists()
    assert (root / ".deleted" / "j1.tombstone").is_file()


def test_delete_job_rejects_linked_tombstone_control_directory(tmp_path: Path) -> None:
    root = tmp_path / "jobs"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / ".deleted").symlink_to(outside, target_is_directory=True)
    store = ArtifactStore(root)

    with pytest.raises(ValueError, match="tombstone"):
        store.delete_job("j1")

    assert not list(outside.iterdir())


@pytest.mark.parametrize("control_name", [".deleted", ".locks", ".incoming"])
def test_reserved_control_components_cannot_be_used_as_job_ids_or_deleted(
    control_name: str, tmp_path: Path
) -> None:
    root = tmp_path / "jobs"
    control = root / control_name
    control.mkdir(parents=True)
    sentinel = control / "sentinel"
    sentinel.write_text("preserve", encoding="utf-8")
    store = ArtifactStore(root)

    with pytest.raises(ValueError, match="reserved"):
        store.job_dir(control_name)
    with pytest.raises(ValueError, match="reserved"):
        store.delete_job(control_name)

    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_stale_reconciliation_candidates_exclude_incoming_and_job_links(tmp_path: Path) -> None:
    root = tmp_path / "jobs"
    incoming = root / ".incoming"
    incoming.mkdir(parents=True)
    outside_file = tmp_path / "outside.wav"
    outside_file.write_bytes(b"preserve")
    incoming_link = incoming / f".upload-{'a' * 32}-linked.wav"
    incoming_link.symlink_to(outside_file)
    outside_directory = tmp_path / "outside-job"
    outside_directory.mkdir()
    job_link = root / ("b" * 32)
    job_link.symlink_to(outside_directory, target_is_directory=True)
    store = ArtifactStore(root)

    removed = store.remove_stale_incoming_uploads(lease_seconds=1, now=10**10)
    candidates = store.stale_unregistered_job_candidates(lease_seconds=1, now=10**10)

    assert removed == 0
    assert candidates == ()
    assert incoming_link.is_symlink()
    assert job_link.is_symlink()
    assert outside_file.read_bytes() == b"preserve"


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
