"""Stable, path-free identities for trusted local checkpoint trees."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path


def _fail() -> ValueError:
    return ValueError("invalid_local_checkpoint")


def checkpoint_tree_digest(checkpoint: str | Path) -> str:
    """Hash regular checkpoint files by normalized relative path and streamed bytes.

    The absolute checkpoint location is deliberately excluded. Every directory entry is
    inspected without following links; special files and identity changes fail closed.
    """

    try:
        candidate = Path(checkpoint)
        if ".." in candidate.parts or candidate.is_symlink():
            raise _fail()
        root = candidate.resolve(strict=True)
        if not root.is_dir() or root.is_symlink():
            raise _fail()
        root_metadata = root.stat(follow_symlinks=False)
        if not stat.S_ISDIR(root_metadata.st_mode):
            raise _fail()

        entries: list[tuple[str, Path]] = []
        for directory, directory_names, filenames in os.walk(root, followlinks=False):
            directory_path = Path(directory)
            for name in sorted(directory_names):
                child = directory_path / name
                metadata = child.stat(follow_symlinks=False)
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                    raise _fail()
                if root not in child.resolve(strict=True).parents:
                    raise _fail()
            for name in sorted(filenames):
                child = directory_path / name
                metadata = child.stat(follow_symlinks=False)
                if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                    raise _fail()
                resolved = child.resolve(strict=True)
                if root not in resolved.parents:
                    raise _fail()
                relative = child.relative_to(root).as_posix()
                if not relative or relative.startswith("/") or ".." in Path(relative).parts:
                    raise _fail()
                entries.append((relative, child))
        if not entries:
            raise _fail()

        digest = hashlib.sha256()
        for relative, path in sorted(entries):
            encoded = relative.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
            try:
                before = os.fstat(descriptor)
                if not stat.S_ISREG(before.st_mode):
                    raise _fail()
                digest.update(before.st_size.to_bytes(8, "big"))
                while chunk := os.read(descriptor, 1024 * 1024):
                    digest.update(chunk)
                after = os.fstat(descriptor)
                if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                ):
                    raise _fail()
            finally:
                os.close(descriptor)
        return digest.hexdigest()
    except ValueError:
        raise
    except (OSError, RuntimeError, UnicodeError):
        raise _fail() from None


__all__ = ["checkpoint_tree_digest"]
