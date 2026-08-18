from __future__ import annotations

import subprocess
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
BACKEND = REPOSITORY / "backend"


def check_ignore(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "check-ignore", "--no-index", "--quiet", str(path)],
        cwd=REPOSITORY,
        text=True,
        capture_output=True,
        check=False,
    )


def test_local_env_is_ignored_but_tracked_example_is_explicitly_unignored() -> None:
    assert check_ignore(BACKEND / ".env").returncode == 0
    assert check_ignore(BACKEND / ".env.example").returncode == 1


def test_env_example_contains_exactly_the_supported_empty_placeholders() -> None:
    assert (BACKEND / ".env.example").read_text(encoding="utf-8").splitlines() == [
        "HUGGINGFACE_TOKEN=",
        "GEMINI_API_KEY=",
        "PYANNOTEAI_API_KEY=",
    ]
