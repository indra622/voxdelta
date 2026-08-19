from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
KEYS = ("HUGGINGFACE_TOKEN", "GEMINI_API_KEY", "PYANNOTEAI_API_KEY")


def isolated_backend(tmp_path: Path, env_content: str, mode: int = 0o600) -> Path:
    backend = tmp_path / "backend"
    shutil.copytree(BACKEND / "src", backend / "src")
    (backend / "scripts").mkdir(parents=True)
    shutil.copy2(BACKEND / "scripts" / "check_credentials.py", backend / "scripts")
    env_file = backend / ".env"
    env_file.write_text(env_content, encoding="utf-8")
    if os.name == "posix":
        env_file.chmod(mode)
    return backend


def run_check(backend: Path, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for key in KEYS:
        env.pop(key, None)
    env["PYTHONPATH"] = str(backend / "src")
    return subprocess.run(
        [sys.executable, str(backend / "scripts" / "check_credentials.py")],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_prints_only_safe_statuses_and_succeeds_from_another_cwd(tmp_path: Path) -> None:
    secrets = ("hf-cli-secret", "gemini-cli-secret")
    backend = isolated_backend(
        tmp_path,
        f"HUGGINGFACE_TOKEN={secrets[0]}\nGEMINI_API_KEY={secrets[1]}\n",
    )
    cwd = tmp_path / "caller"
    cwd.mkdir()

    result = run_check(backend, cwd)

    assert result.returncode == 0
    assert result.stdout.splitlines() == [
        "HUGGINGFACE_TOKEN configured",
        "GEMINI_API_KEY configured",
        "PYANNOTEAI_API_KEY missing",
    ]
    assert result.stderr == ""
    assert all(secret not in result.stdout + result.stderr for secret in secrets)


def test_cli_returns_one_when_a_required_value_is_empty(tmp_path: Path) -> None:
    backend = isolated_backend(
        tmp_path,
        "HUGGINGFACE_TOKEN=hf-present\nGEMINI_API_KEY=\n",
    )
    cwd = tmp_path / "caller"
    cwd.mkdir()

    result = run_check(backend, cwd)

    assert result.returncode == 1
    assert result.stdout.splitlines() == [
        "HUGGINGFACE_TOKEN configured",
        "GEMINI_API_KEY missing",
        "PYANNOTEAI_API_KEY missing",
    ]
    assert result.stderr == ""
    assert "hf-present" not in result.stdout + result.stderr


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission modes are required")
def test_cli_returns_two_with_safe_remediation_for_unsafe_permissions(tmp_path: Path) -> None:
    secret = "unsafe-file-secret"
    backend = isolated_backend(
        tmp_path,
        f"HUGGINGFACE_TOKEN={secret}\nGEMINI_API_KEY=gemini-secret\n",
        mode=0o644,
    )
    cwd = tmp_path / "caller"
    cwd.mkdir()

    result = run_check(backend, cwd)

    assert result.returncode == 2
    assert result.stdout == ""
    assert str(backend / ".env") in result.stderr
    assert "chmod 600 backend/.env" in result.stderr
    assert secret not in result.stdout + result.stderr
