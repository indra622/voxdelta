from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

import voxdelta.evaluation.wav2vec_base as wav2vec_base
from voxdelta.evaluation.wav2vec_base import (
    WAV2VEC_ARTIFACTS,
    WAV2VEC_MODEL_ID,
    WAV2VEC_MODEL_REVISION,
    WAV2VEC_WEIGHTS_SHA256,
    Wav2VecArtifact,
    prepare_wav2vec_base,
    validate_wav2vec_base,
)


def _payloads() -> dict[str, bytes]:
    return {
        "config.json": json.dumps({"model_type": "wav2vec2"}).encode(),
        "preprocessor_config.json": json.dumps({"sampling_rate": 16_000}).encode(),
        "pytorch_model.bin": b"test-only-model-weights",
    }


def _patch_small_artifacts(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    payloads = _payloads()
    artifacts = tuple(
        Wav2VecArtifact(
            name=name,
            size=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        for name, payload in payloads.items()
    )
    monkeypatch.setattr(wav2vec_base, "WAV2VEC_ARTIFACTS", artifacts)
    return payloads


def _downloader(payloads: dict[str, bytes]):
    def download(url: str, output: Path) -> None:
        output.write_bytes(payloads[url.rsplit("/", 1)[-1]])

    return download


def test_pinned_artifacts_have_exact_official_identity() -> None:
    assert WAV2VEC_MODEL_ID == "facebook/wav2vec2-xls-r-300m"
    assert WAV2VEC_MODEL_REVISION == "1a640f32ac3e39899438a2931f9924c02f080a54"
    assert WAV2VEC_WEIGHTS_SHA256 == (
        "d5e490574712ad0a6736923b9ed11d4cd51c78609c36205f704fc4e87b11d2e0"
    )
    assert {item.name: (item.size, item.sha256) for item in WAV2VEC_ARTIFACTS} == {
        "config.json": (
            1_568,
            "0bffa0d0e98153e883b828d86491f3c6062cb563dc9d7a9cfd1790da30c286ac",
        ),
        "preprocessor_config.json": (
            212,
            "a2254a5b58f72cd4de3632f8eee64f3f098b7c1402128d2f419e7d00ae13e335",
        ),
        "pytorch_model.bin": (1_269_737_156, WAV2VEC_WEIGHTS_SHA256),
    }


def test_prepare_is_atomic_private_and_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = _patch_small_artifacts(monkeypatch)
    target = tmp_path / "base"

    prepared = prepare_wav2vec_base(target, downloader=_downloader(payloads))

    assert prepared.path == target.resolve()
    assert prepared.model_id == WAV2VEC_MODEL_ID
    assert prepared.revision == WAV2VEC_MODEL_REVISION
    assert prepared.weights_sha256 == WAV2VEC_WEIGHTS_SHA256
    assert stat.S_IMODE(target.stat().st_mode) == 0o700
    assert {path.name for path in target.iterdir()} == set(payloads)
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in target.iterdir())
    assert validate_wav2vec_base(target) == prepared


def test_prepare_rejects_existing_target_without_downloading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_small_artifacts(monkeypatch)
    target = tmp_path / "base"
    target.mkdir(mode=0o700)
    called = False

    def download(_url: str, _output: Path) -> None:
        nonlocal called
        called = True

    with pytest.raises(ValueError, match="wav2vec_base_preparation_failed"):
        prepare_wav2vec_base(target, downloader=download)

    assert called is False
    assert list(target.iterdir()) == []


def test_failed_download_leaves_no_final_or_staging_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_small_artifacts(monkeypatch)
    target = tmp_path / "base"

    def fail(_url: str, output: Path) -> None:
        output.write_bytes(b"partial")
        raise OSError("private detail")

    with pytest.raises(ValueError, match="wav2vec_base_preparation_failed"):
        prepare_wav2vec_base(target, downloader=fail)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("candidate", [Path("relative"), Path("../escape")])
def test_prepare_rejects_non_absolute_or_traversing_output(candidate: Path) -> None:
    with pytest.raises(ValueError, match="wav2vec_base_preparation_failed"):
        prepare_wav2vec_base(candidate, downloader=lambda _url, _output: None)


def test_validation_rejects_unexpected_tampered_and_public_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = _patch_small_artifacts(monkeypatch)
    target = tmp_path / "base"
    prepare_wav2vec_base(target, downloader=_downloader(payloads))

    (target / "unexpected.txt").write_text("no", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_wav2vec_base"):
        validate_wav2vec_base(target)
    (target / "unexpected.txt").unlink()

    weights = target / "pytorch_model.bin"
    weights.write_bytes(b"tampered")
    os.chmod(weights, 0o600)
    with pytest.raises(ValueError, match="invalid_wav2vec_base"):
        validate_wav2vec_base(target)

    weights.write_bytes(payloads["pytorch_model.bin"])
    os.chmod(weights, 0o644)
    with pytest.raises(ValueError, match="invalid_wav2vec_base"):
        validate_wav2vec_base(target)


def test_validation_rejects_symlinked_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = _patch_small_artifacts(monkeypatch)
    target = tmp_path / "base"
    prepare_wav2vec_base(target, downloader=_downloader(payloads))
    config = target / "config.json"
    real = tmp_path / "real-config.json"
    config.replace(real)
    config.symlink_to(real)

    with pytest.raises(ValueError, match="invalid_wav2vec_base"):
        validate_wav2vec_base(target)


def test_cli_sanitizes_preparation_failure(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2]))
    from scripts import prepare_wav2vec_base as prepare_cli

    assert prepare_cli.main(["--output", "relative"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "wav2vec_base_error: wav2vec_base_preparation_failed\n"
    assert "relative" not in captured.err
