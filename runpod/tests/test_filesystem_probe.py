from __future__ import annotations

import importlib.util
import os
import stat
from pathlib import Path
from typing import Any

import pytest

PROBE = Path(__file__).parents[1] / "scripts" / "probe_filesystem.py"


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("probe_filesystem", PROBE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_accepts_a_filesystem_that_honors_private_modes(tmp_path: Path) -> None:
    module = _module()
    target = (tmp_path / "workspace" / "run").resolve()

    assert module.main(["--path", str(target)]) == 0
    assert target.is_dir()
    assert stat.S_IMODE(target.stat().st_mode) == 0o700
    # the probe leaves nothing behind on the sensitive filesystem
    assert list(target.iterdir()) == []


def test_probe_rejects_a_mount_that_ignores_requested_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EU-SE-1 mounts /workspace over MooseFS, which reports 0777/0666 regardless."""
    module = _module()
    target = (tmp_path / "moosefs" / "run").resolve()
    monkeypatch.setattr(module, "_mode", lambda path: 0o777)

    assert module.main(["--path", str(target)]) == 2


def test_probe_reports_every_requested_path_and_cleans_up(tmp_path: Path) -> None:
    module = _module()
    first = (tmp_path / "run").resolve()
    second = (tmp_path / "models").resolve()

    assert module.main(["--path", str(first), "--path", str(second)]) == 0
    for target in (first, second):
        assert stat.S_IMODE(target.stat().st_mode) == 0o700
        assert list(target.iterdir()) == []


def test_probe_rejects_relative_and_symlinked_targets(tmp_path: Path) -> None:
    module = _module()
    real = (tmp_path / "real").resolve()
    real.mkdir(mode=0o700)
    link = (tmp_path / "link").resolve()
    os.symlink(real, link)

    assert module.main(["--path", "relative/path"]) == 2
    with pytest.raises(module.FilesystemProbeError, match="^invalid_probe_target$"):
        module.probe_path(Path("relative/path"))
    with pytest.raises(module.FilesystemProbeError, match="^invalid_probe_target$"):
        module.probe_path(link)
