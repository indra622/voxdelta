"""The local encoder bundle must be verified before anything is loaded from it."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from voxdelta.providers.encoder_bundle import (
    CHECKSUMS_NAME,
    EMOTION2VEC_ENCODER_ID,
    EMOTION2VEC_ENCODER_REVISION,
    MANIFEST_NAME,
    REQUIRED_FILES,
    EncoderBundleError,
    encoder_bundle_manifest,
    verify_encoder_bundle,
)
from voxdelta.providers.encoder_publication import (
    EncoderPublicationError,
    publish_encoder_bundle,
)

CONTENTS: dict[str, bytes] = {
    "config.yaml": b"model: emotion2vec\n",
    "configuration.json": b'{"framework":"pytorch"}',
    "model.pt": b"synthetic-encoder-weights",
    "tokens.txt": b"happy\nsad\n",
}


@pytest.fixture
def synthetic_pin(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    """Re-pin the modules to tiny synthetic content for structural tests.

    This is a test-only rebinding of the module constants. It is deliberately *not* a
    parameter on any public function: production callers get the real pin and have no
    argument with which to relax it.
    """

    from voxdelta.providers import _attested_tree, encoder_bundle, encoder_publication

    payloads = {
        name: (hashlib.sha256(data).hexdigest(), len(data)) for name, data in CONTENTS.items()
    }
    records = [
        {"path": name, "sha256": payloads[name][0], "bytes": payloads[name][1]}
        for name in sorted(payloads)
    ]
    tree = _attested_tree.canonical_sha256(records)
    required = tuple(sorted(payloads))
    for module in (encoder_bundle, encoder_publication):
        monkeypatch.setattr(module, "PINNED_PAYLOADS", payloads, raising=False)
        monkeypatch.setattr(module, "PINNED_BUNDLE_TREE_SHA256", tree, raising=False)
        monkeypatch.setattr(module, "REQUIRED_FILES", required, raising=False)
    return dict(CONTENTS)


def _canonical(document: object) -> bytes:
    return (
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        + b"\n"
    )


def _build(root: Path, *, contents: dict[str, bytes] | None = None, **updates: object) -> Path:
    files = dict(CONTENTS if contents is None else contents)
    root.mkdir(parents=True)
    for name, data in files.items():
        (root / name).write_bytes(data)
    payloads = [
        {
            "path": name,
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
        for name, data in sorted(files.items())
    ]
    manifest = encoder_bundle_manifest(payloads)
    manifest.update(updates)
    raw = _canonical(manifest)
    (root / MANIFEST_NAME).write_bytes(raw)
    lines = [f"{hashlib.sha256(raw).hexdigest()}  {MANIFEST_NAME}"]
    lines.extend(f"{record['sha256']}  {record['path']}" for record in payloads)
    (root / CHECKSUMS_NAME).write_text(
        "".join(f"{line}\n" for line in sorted(lines, key=lambda line: line.split("  ", 1)[1])),
        encoding="utf-8",
    )
    return root


def test_a_well_formed_bundle_verifies_and_reports_its_identity(
    synthetic_pin: dict[str, bytes], tmp_path: Path
) -> None:
    bundle = _build(tmp_path / "encoder")

    verified = verify_encoder_bundle(bundle)

    assert verified.encoder_id == EMOTION2VEC_ENCODER_ID
    assert verified.encoder_revision == EMOTION2VEC_ENCODER_REVISION
    assert verified.init_param_sha256 == hashlib.sha256(CONTENTS["model.pt"]).hexdigest()
    assert len(verified.bundle_tree_sha256) == 64
    assert verified.path == bundle.resolve()


def test_a_relative_path_is_refused(synthetic_pin: dict[str, bytes], tmp_path: Path) -> None:
    _build(tmp_path / "encoder")

    with pytest.raises(EncoderBundleError):
        verify_encoder_bundle(Path("encoder"))


def test_a_tampered_payload_is_refused(synthetic_pin: dict[str, bytes], tmp_path: Path) -> None:
    bundle = _build(tmp_path / "encoder")
    (bundle / "model.pt").write_bytes(b"swapped-weights")

    with pytest.raises(EncoderBundleError):
        verify_encoder_bundle(bundle)


def test_a_truncated_payload_is_refused(synthetic_pin: dict[str, bytes], tmp_path: Path) -> None:
    bundle = _build(tmp_path / "encoder")
    (bundle / "model.pt").write_bytes(CONTENTS["model.pt"][:5])

    with pytest.raises(EncoderBundleError):
        verify_encoder_bundle(bundle)


def test_a_missing_required_file_is_refused(
    synthetic_pin: dict[str, bytes], tmp_path: Path
) -> None:
    reduced = {name: data for name, data in CONTENTS.items() if name != "tokens.txt"}
    bundle = _build(tmp_path / "encoder", contents=reduced)

    with pytest.raises(EncoderBundleError):
        verify_encoder_bundle(bundle)


@pytest.mark.parametrize("stray", [".DS_Store", "model.pt.incomplete", "README.md", "logo.png"])
def test_an_unexpected_file_makes_the_bundle_malformed(
    synthetic_pin: dict[str, bytes], tmp_path: Path, stray: str
) -> None:
    """Cache droppings and partial downloads are a defect, not untidiness."""

    bundle = _build(tmp_path / "encoder")
    (bundle / stray).write_bytes(b"stray")

    with pytest.raises(EncoderBundleError):
        verify_encoder_bundle(bundle)


def test_an_unattested_subdirectory_is_refused(
    synthetic_pin: dict[str, bytes], tmp_path: Path
) -> None:
    bundle = _build(tmp_path / "encoder")
    (bundle / "example").mkdir()
    (bundle / "example" / "test.wav").write_bytes(b"RIFF")

    with pytest.raises(EncoderBundleError):
        verify_encoder_bundle(bundle)


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlinks are required")
def test_a_symlinked_payload_is_refused(synthetic_pin: dict[str, bytes], tmp_path: Path) -> None:
    outside = tmp_path / "outside.pt"
    outside.write_bytes(CONTENTS["model.pt"])
    bundle = _build(tmp_path / "encoder")
    (bundle / "model.pt").unlink()
    (bundle / "model.pt").symlink_to(outside)

    with pytest.raises(EncoderBundleError):
        verify_encoder_bundle(bundle)


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlinks are required")
def test_a_symlinked_bundle_root_is_refused(
    synthetic_pin: dict[str, bytes], tmp_path: Path
) -> None:
    real = _build(tmp_path / "encoder")
    link = tmp_path / "link"
    link.symlink_to(real)

    with pytest.raises(EncoderBundleError):
        verify_encoder_bundle(link)


def test_the_verifier_exposes_no_argument_that_could_relax_the_pin() -> None:
    """The pin must not be a parameter, or a caller could accept a different encoder."""

    import inspect

    parameters = inspect.signature(verify_encoder_bundle).parameters
    assert list(parameters) == ["path"]

    with pytest.raises(TypeError):
        verify_encoder_bundle(Path("/tmp"), encoder_revision="v9.9.9")  # type: ignore[call-arg]


def test_a_manifest_claiming_a_different_revision_is_refused(
    synthetic_pin: dict[str, bytes], tmp_path: Path
) -> None:
    bundle = _build(tmp_path / "encoder", encoder_revision="v9.9.9")

    with pytest.raises(EncoderBundleError):
        verify_encoder_bundle(bundle)


def test_a_manifest_that_disagrees_with_its_own_tree_digest_is_refused(
    synthetic_pin: dict[str, bytes], tmp_path: Path
) -> None:
    bundle = _build(tmp_path / "encoder", bundle_tree_sha256="a" * 64)

    with pytest.raises(EncoderBundleError):
        verify_encoder_bundle(bundle)


def test_a_checksum_file_that_does_not_attest_the_manifest_is_refused(
    synthetic_pin: dict[str, bytes], tmp_path: Path
) -> None:
    bundle = _build(tmp_path / "encoder")
    lines = [
        line
        for line in (bundle / CHECKSUMS_NAME).read_text(encoding="utf-8").splitlines()
        if not line.endswith(MANIFEST_NAME)
    ]
    (bundle / CHECKSUMS_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(EncoderBundleError):
        verify_encoder_bundle(bundle)


def test_a_malformed_manifest_is_refused(synthetic_pin: dict[str, bytes], tmp_path: Path) -> None:
    bundle = _build(tmp_path / "encoder")
    (bundle / MANIFEST_NAME).write_bytes(b"{not json")

    with pytest.raises(EncoderBundleError):
        verify_encoder_bundle(bundle)


# --- publication ---


def _snapshot(root: Path) -> Path:
    root.mkdir(parents=True)
    for name, data in CONTENTS.items():
        (root / name).write_bytes(data)
    # The real cache carries all of this beside the four files that matter.
    (root / ".DS_Store").write_bytes(b"junk")
    (root / ".gitattributes").write_bytes(b"* filter=lfs")
    (root / "README.md").write_bytes(b"# encoder")
    (root / "model.pt.incomplete").write_bytes(b"partial-download")
    (root / "logo.png").write_bytes(b"\x89PNG")
    (root / "example").mkdir()
    (root / "example" / "test.wav").write_bytes(b"RIFF")
    return root


def test_publication_copies_only_the_loaders_file_set(
    synthetic_pin: dict[str, bytes], tmp_path: Path
) -> None:
    source = _snapshot(tmp_path / "snapshot")

    bundle = publish_encoder_bundle(source, tmp_path / "encoder")

    published = {entry.name for entry in bundle.path.iterdir()}
    assert published == set(REQUIRED_FILES) | {MANIFEST_NAME, CHECKSUMS_NAME}
    for stray in (".DS_Store", "model.pt.incomplete", "README.md", "logo.png", "example"):
        assert not (bundle.path / stray).exists()


def test_publication_leaves_the_cache_untouched(
    synthetic_pin: dict[str, bytes], tmp_path: Path
) -> None:
    source = _snapshot(tmp_path / "snapshot")
    before = {entry.name: entry.stat().st_mtime_ns for entry in source.iterdir()}

    publish_encoder_bundle(source, tmp_path / "encoder")

    after = {entry.name: entry.stat().st_mtime_ns for entry in source.iterdir()}
    assert before == after


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlinks are required")
def test_publication_dereferences_a_symlinked_weight_into_a_real_file(
    synthetic_pin: dict[str, bytes], tmp_path: Path
) -> None:
    """A bundle whose weights follow a link into another cache is not immutable."""

    blob = tmp_path / "blob.pt"
    blob.write_bytes(CONTENTS["model.pt"])
    source = _snapshot(tmp_path / "snapshot")
    (source / "model.pt").unlink()
    (source / "model.pt").symlink_to(blob)

    bundle = publish_encoder_bundle(source, tmp_path / "encoder")

    weights = bundle.path / "model.pt"
    assert not weights.is_symlink()
    assert weights.read_bytes() == CONTENTS["model.pt"]
    blob.unlink()
    assert verify_encoder_bundle(bundle.path).bundle_tree_sha256 == bundle.bundle_tree_sha256


def test_published_files_are_private(synthetic_pin: dict[str, bytes], tmp_path: Path) -> None:
    source = _snapshot(tmp_path / "snapshot")

    bundle = publish_encoder_bundle(source, tmp_path / "encoder")

    for entry in bundle.path.iterdir():
        assert entry.stat().st_mode & 0o077 == 0


def test_publication_refuses_to_overwrite_an_existing_bundle(
    synthetic_pin: dict[str, bytes], tmp_path: Path
) -> None:
    source = _snapshot(tmp_path / "snapshot")
    publish_encoder_bundle(source, tmp_path / "encoder")

    with pytest.raises(EncoderPublicationError) as error:
        publish_encoder_bundle(source, tmp_path / "encoder")

    assert error.value.code == "encoder_bundle_already_exists"


def test_publication_refuses_a_snapshot_missing_a_required_file(
    synthetic_pin: dict[str, bytes], tmp_path: Path
) -> None:
    source = _snapshot(tmp_path / "snapshot")
    (source / "tokens.txt").unlink()

    with pytest.raises(EncoderPublicationError) as error:
        publish_encoder_bundle(source, tmp_path / "encoder")

    assert error.value.code == "encoder_snapshot_incomplete"
    assert not (tmp_path / "encoder").exists()


def test_a_failed_publication_leaves_no_partial_bundle(
    synthetic_pin: dict[str, bytes], tmp_path: Path
) -> None:
    source = _snapshot(tmp_path / "snapshot")
    (source / "model.pt").unlink()
    (source / "model.pt").mkdir()

    with pytest.raises(EncoderPublicationError):
        publish_encoder_bundle(source, tmp_path / "encoder")

    assert not (tmp_path / "encoder").exists()
    assert not (tmp_path / ".encoder.staging").exists()


def test_publication_refuses_relative_paths(
    synthetic_pin: dict[str, bytes], tmp_path: Path
) -> None:
    with pytest.raises(EncoderPublicationError) as error:
        publish_encoder_bundle(Path("snapshot"), tmp_path / "encoder")

    assert error.value.code == "encoder_bundle_path_not_absolute"


def test_a_published_bundle_verifies_through_the_production_verifier(
    synthetic_pin: dict[str, bytes], tmp_path: Path
) -> None:
    source = _snapshot(tmp_path / "snapshot")

    bundle = publish_encoder_bundle(source, tmp_path / "encoder")

    assert verify_encoder_bundle(bundle.path).bundle_tree_sha256 == bundle.bundle_tree_sha256


# --- the pin is the anchor: self-consistency is not identity ---

BACKEND = Path(__file__).resolve().parents[2]


def _resign(root: Path, contents: dict[str, bytes]) -> Path:
    """Write a bundle that is entirely self-consistent but carries different content."""

    root.mkdir(parents=True)
    for name, data in contents.items():
        (root / name).write_bytes(data)
    payloads = [
        {"path": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        for name, data in sorted(contents.items())
    ]
    from voxdelta.providers._attested_tree import canonical_sha256

    manifest = {
        "schema_version": "1",
        "artifact_kind": "local-encoder-bundle",
        "encoder_id": EMOTION2VEC_ENCODER_ID,
        "encoder_revision": EMOTION2VEC_ENCODER_REVISION,
        "init_param": "model.pt",
        "payloads": payloads,
        "bundle_tree_sha256": canonical_sha256(payloads),
    }
    raw = _canonical(manifest)
    (root / MANIFEST_NAME).write_bytes(raw)
    lines = [f"{hashlib.sha256(raw).hexdigest()}  {MANIFEST_NAME}"]
    lines.extend(f"{record['sha256']}  {record['path']}" for record in payloads)
    (root / CHECKSUMS_NAME).write_text(
        "".join(f"{line}\n" for line in sorted(lines, key=lambda line: line.split("  ", 1)[1])),
        encoding="utf-8",
    )
    return root


def test_a_fully_resigned_alternate_bundle_is_rejected_by_the_pin(tmp_path: Path) -> None:
    """Rewriting all four payloads plus the manifest and checksums must not help."""

    hostile = _resign(
        tmp_path / "hostile",
        {
            "config.yaml": b"model: attacker\n",
            "configuration.json": b'{"framework":"pytorch"}',
            "model.pt": b"attacker-weights",
            "tokens.txt": b"a\nb\n",
        },
    )

    # Internally consistent by construction...
    payload = json.loads((hostile / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert payload["encoder_id"] == EMOTION2VEC_ENCODER_ID
    for record in payload["payloads"]:
        actual = hashlib.sha256((hostile / record["path"]).read_bytes()).hexdigest()
        assert actual == record["sha256"]

    # ...and still refused, because it is not the pinned encoder.
    with pytest.raises(EncoderBundleError):
        verify_encoder_bundle(hostile)


def test_a_resigned_bundle_is_rejected_before_any_provider_is_constructed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Identity must be settled while it is still a path, not after a model load."""

    import voxdelta.providers.emotion2vec_emotion as emotion2vec
    from voxdelta.evaluation.shadow_replay import ShadowReplayError, _build_emotion2vec_primary

    constructed: list[object] = []

    def explode(*args: object, **kwargs: object) -> object:
        constructed.append(args)
        raise AssertionError("the provider must never be constructed for a rejected bundle")

    monkeypatch.setattr(emotion2vec, "Emotion2VecEmotionProvider", explode)

    hostile = _resign(
        tmp_path / "hostile",
        {
            "config.yaml": b"x\n",
            "configuration.json": b"{}",
            "model.pt": b"attacker-weights",
            "tokens.txt": b"y\n",
        },
    )
    checkpoint = (tmp_path / "checkpoint").resolve()
    checkpoint.mkdir()

    with pytest.raises(ShadowReplayError) as error:
        _build_emotion2vec_primary(checkpoint, hostile, "cpu", object())

    assert error.value.code == "shadow_encoder_bundle_rejected"
    assert constructed == []


def test_publication_refuses_a_snapshot_whose_content_is_not_the_pinned_encoder(
    tmp_path: Path,
) -> None:
    """Wrong bytes are destroyed in staging and never reach the target."""

    source = tmp_path / "snapshot"
    source.mkdir()
    for name in ("config.yaml", "configuration.json", "model.pt", "tokens.txt"):
        (source / name).write_bytes(b"not-the-real-encoder")

    with pytest.raises(EncoderPublicationError) as error:
        publish_encoder_bundle(source, tmp_path / "encoder")

    assert error.value.code == "encoder_source_does_not_match_pin"
    assert not (tmp_path / "encoder").exists()
    assert not (tmp_path / ".encoder.staging").exists()


def test_an_empty_target_reserved_by_another_writer_is_never_overwritten(
    synthetic_pin: dict[str, bytes], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real race: renaming a directory onto an existing *empty* one succeeds on POSIX.

    Injected at the last possible moment — inside the reservation itself — so the target
    appears after every earlier existence check has already passed.
    """

    import voxdelta.providers.encoder_publication as publication

    source = _snapshot(tmp_path / "snapshot")
    target = tmp_path / "encoder"
    real_reserve = publication._reserve
    observed: dict[str, int] = {}

    def racing_reserve(path: Path) -> None:
        if not path.exists():
            path.mkdir(mode=0o700)
            observed["inode"] = os.stat(path).st_ino
        real_reserve(path)

    monkeypatch.setattr(publication, "_reserve", racing_reserve)

    with pytest.raises(EncoderPublicationError) as error:
        publish_encoder_bundle(source, target)

    assert error.value.code == "encoder_bundle_already_exists"
    assert target.is_dir()
    assert list(target.iterdir()) == [], "the other writer's empty target was overwritten"
    assert os.stat(target).st_ino == observed["inode"], "the target directory was replaced"
    assert not (tmp_path / ".encoder.staging").exists()


def test_a_non_empty_target_from_another_writer_is_never_overwritten(
    synthetic_pin: dict[str, bytes], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import voxdelta.providers.encoder_publication as publication

    source = _snapshot(tmp_path / "snapshot")
    target = tmp_path / "encoder"
    real_reserve = publication._reserve

    def racing_reserve(path: Path) -> None:
        if not path.exists():
            path.mkdir(mode=0o700)
            (path / "other-writer").write_text("keep", encoding="utf-8")
        real_reserve(path)

    monkeypatch.setattr(publication, "_reserve", racing_reserve)

    with pytest.raises(EncoderPublicationError) as error:
        publish_encoder_bundle(source, target)

    assert error.value.code == "encoder_bundle_already_exists"
    assert (target / "other-writer").read_text(encoding="utf-8") == "keep"


def test_the_manifest_is_the_last_file_written(
    synthetic_pin: dict[str, bytes], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Commit-marker ordering: payloads, then checksums, then ENCODER.json."""

    import voxdelta.providers.encoder_publication as publication

    order: list[str] = []
    real_write = publication._write_private

    def recording_write(path: Path, payload: bytes | str) -> None:
        order.append(path.name)
        real_write(path, payload)

    monkeypatch.setattr(publication, "_write_private", recording_write)
    source = _snapshot(tmp_path / "snapshot")

    publish_encoder_bundle(source, tmp_path / "encoder")

    assert order == [CHECKSUMS_NAME, MANIFEST_NAME]


def test_a_reserved_but_incomplete_target_is_rejected_by_the_verifier(
    synthetic_pin: dict[str, bytes], tmp_path: Path
) -> None:
    """Until the commit marker exists, a reader must refuse the directory."""

    complete = publish_encoder_bundle(_snapshot(tmp_path / "snapshot"), tmp_path / "encoder")

    partial = tmp_path / "partial"
    partial.mkdir(mode=0o700)
    for name in REQUIRED_FILES:
        (partial / name).write_bytes(CONTENTS[name])
    # Payloads present, checksums present, commit marker absent.
    (partial / CHECKSUMS_NAME).write_bytes((complete.path / CHECKSUMS_NAME).read_bytes())

    with pytest.raises(EncoderBundleError):
        verify_encoder_bundle(partial)

    # Bare reservation, nothing in it at all.
    bare = tmp_path / "bare"
    bare.mkdir(mode=0o700)
    with pytest.raises(EncoderBundleError):
        verify_encoder_bundle(bare)


def test_a_failure_after_reservation_removes_only_our_own_target(
    synthetic_pin: dict[str, bytes], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import voxdelta.providers.encoder_publication as publication

    def fail_verification(path: Path) -> object:
        raise EncoderBundleError()

    monkeypatch.setattr(publication, "verify_encoder_bundle", fail_verification)
    source = _snapshot(tmp_path / "snapshot")

    with pytest.raises(EncoderPublicationError):
        publish_encoder_bundle(source, tmp_path / "encoder")

    assert not (tmp_path / "encoder").exists()
    assert not (tmp_path / ".encoder.staging").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlinks are required")
@pytest.mark.parametrize("name", ["config.yaml", "configuration.json", "tokens.txt"])
def test_publication_refuses_a_symlink_for_any_file_but_the_weights(
    synthetic_pin: dict[str, bytes], tmp_path: Path, name: str
) -> None:
    """Only model.pt legitimately arrives through a link; the rest must be real files."""

    outside = tmp_path / "outside"
    outside.write_bytes(CONTENTS[name])
    source = _snapshot(tmp_path / "snapshot")
    (source / name).unlink()
    (source / name).symlink_to(outside)

    with pytest.raises(EncoderPublicationError) as error:
        publish_encoder_bundle(source, tmp_path / "encoder")

    assert error.value.code == "encoder_source_is_a_symlink"
    assert not (tmp_path / "encoder").exists()


# --- CLI path contracts ---


def _run_cli(module: str, argv: list[str], monkeypatch: pytest.MonkeyPatch) -> tuple[int, str]:
    import io
    from contextlib import redirect_stdout

    monkeypatch.syspath_prepend(str(BACKEND))
    imported = __import__(f"scripts.{module}", fromlist=["main"])
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = imported.main(argv)
    return code, buffer.getvalue().strip()


def test_the_builder_cli_refuses_a_relative_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, output = _run_cli(
        "build_encoder_bundle",
        ["--snapshot", "relative-snapshot", "--output", str(tmp_path / "encoder")],
        monkeypatch,
    )

    assert code == 2
    assert output == "encoder_bundle_path_not_absolute"


def test_the_builder_cli_refuses_a_relative_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _snapshot(tmp_path / "snapshot")

    code, output = _run_cli(
        "build_encoder_bundle",
        ["--snapshot", str(source), "--output", "relative-output"],
        monkeypatch,
    )

    assert code == 2
    assert output == "encoder_bundle_path_not_absolute"
    assert not Path("relative-output").exists()


def test_the_replay_cli_refuses_a_relative_encoder_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, output = _run_cli(
        "run_shadow_replay",
        [
            "--release",
            str(tmp_path / "release"),
            "--calibration",
            str(tmp_path / "calibration"),
            "--primary-encoder-bundle",
            "relative-bundle",
            "--primary-checkpoint",
            str(tmp_path / "checkpoint"),
            "--output",
            str(tmp_path / "out"),
        ],
        monkeypatch,
    )

    assert code == 2
    assert output in {"invalid_shadow_input", "shadow_artifacts_rejected"}
