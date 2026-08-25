"""Every external binary the code shells out to must actually exist where it runs.

Three separate Pod runs were lost to a tool the image did not contain -- first
``openssh-server``, then ``rsync``, then ``zstd``. Presence of a package is not
something to assert by eye: this derives the set from the source itself, so a new
``subprocess`` call site fails here rather than on a paid GPU.
"""

from __future__ import annotations

import ast
from pathlib import Path

RUNPOD = Path(__file__).parents[1]
DOCKERFILE = RUNPOD / "docker" / "Dockerfile"

#: Shipped by the pinned CUDA base image rather than an apt line of ours.
BASE_IMAGE_PROVIDED = frozenset({"nvidia-smi"})

#: Only ever invoked on the operator's machine, never inside the container.
OPERATOR_ONLY = frozenset({"ssh", "shasum", "docker", "git", "skopeo", "mktemp"})

_SUBPROCESS_CALLS = frozenset({"run", "Popen", "call", "check_call", "check_output"})


def _binaries_in(path: Path) -> set[str]:
    """Literal argv[0] of every subprocess invocation in one module."""
    found: set[str] = set()
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in _SUBPROCESS_CALLS:
            continue
        value = node.func.value
        if not (isinstance(value, ast.Name) and value.id == "subprocess"):
            continue
        if not node.args or not isinstance(node.args[0], ast.List | ast.Tuple):
            continue
        first = node.args[0].elts[0] if node.args[0].elts else None
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            found.add(Path(first.value).name)
    return found


def _discovered_binaries() -> set[str]:
    found: set[str] = set()
    for directory in ("src", "scripts"):
        for path in sorted((RUNPOD / directory).rglob("*.py")):
            found |= _binaries_in(path)
    return found


def _apt_packages() -> set[str]:
    """Package names on the Dockerfile's apt-get install continuation lines."""
    text = DOCKERFILE.read_text()
    start = text.index("apt-get install")
    end = text.index("rm -rf /var/lib/apt/lists", start)
    packages: set[str] = set()
    for line in text[start:end].splitlines()[1:]:
        candidate = line.strip().removesuffix("\\").strip()
        if candidate and not candidate.startswith("&&"):
            packages.add(candidate)
    return packages


def test_image_installs_every_binary_the_container_shells_out_to() -> None:
    discovered = _discovered_binaries()
    # the derivation itself must keep working; these are known call sites
    assert {"zstd", "nvidia-smi"} <= discovered

    packages = _apt_packages()
    required = discovered - BASE_IMAGE_PROVIDED - OPERATOR_ONLY
    missing = {binary for binary in required if binary not in packages}

    assert not missing, (
        f"binaries invoked in-container but not installed by the Dockerfile: {sorted(missing)}"
    )


def test_decompression_paths_are_covered_by_the_image() -> None:
    """The two remote extraction paths that stopped stage 01 on the container-disk Pod."""
    for module in ("model_bundle.py", "package.py"):
        assert "zstd" in _binaries_in(RUNPOD / "src" / "voxdelta_runpod" / module)
    assert "zstd" in _apt_packages()


def test_image_still_carries_every_previously_missing_tool() -> None:
    """One case per Pod run lost to a tool the image did not ship."""
    packages = _apt_packages()
    assert "openssh-server" in packages  # no SSH at all: operator locked out
    assert "rsync" in packages  # transfers had no receiver
    assert "zstd" in packages  # extraction could not decompress
