"""Atomic, job-scoped persistence for versioned Pydantic artifacts."""

from __future__ import annotations

import errno
import hashlib
import os
import shutil
import stat
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any, Protocol, TypeVar
from uuid import uuid4

import orjson
from pydantic import BaseModel

from voxdelta.domain.models import StageName
from voxdelta.jobs._ids import (
    JobAbsenceProof,
    validate_canonical_job_id,
    validate_job_absence_proof,
    validate_job_id,
)

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised only on non-POSIX platforms
    fcntl = None  # type: ignore[assignment]

T = TypeVar("T", bound=BaseModel)


_UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS = {
    errno.EBADF,
    errno.EINVAL,
    getattr(errno, "ENOTSUP", errno.EINVAL),
    getattr(errno, "EOPNOTSUPP", errno.EINVAL),
}
_LINKED_JOB_DIRECTORY_ERROR = (
    "job directory beneath jobs root must not be a symlink or reparse point"
)
_OPERATION_LOCK_VALIDATION_ERROR = "job operation lock could not be validated safely"
_TOMBSTONE_VALIDATION_ERROR = "job deletion tombstone could not be validated safely"
_INCOMING_VALIDATION_ERROR = "incoming upload could not be validated safely"
_LEGACY_ARTIFACT_LEASE_GRACE_SECONDS = 3600.0
_PROCESS_OWNER_TOKEN = uuid4().hex
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_SYNCHRONIZE = 0x00100000
_WAIT_OBJECT_0 = 0x00000000
_WAIT_TIMEOUT = 0x00000102
_ERROR_ACCESS_DENIED = 5
_ERROR_INVALID_PARAMETER = 87


@dataclass(slots=True)
class _ProcessLockEntry:
    lock: threading.Lock
    users: int = 0


_OPERATION_LOCKS: dict[Path, _ProcessLockEntry] = {}
_OPERATION_LOCKS_GUARD = threading.Lock()


@dataclass(slots=True)
class _IncomingOwner:
    lease_descriptor: int | None


_INCOMING_OWNERS: dict[Path, _IncomingOwner] = {}
_INCOMING_OWNERS_GUARD = threading.Lock()
_ACTIVE_ARTIFACT_LEASES: dict[Path, str] = {}
_ACTIVE_ARTIFACT_LEASES_GUARD = threading.Lock()


class _WindowsProcessApi(Protocol):
    def open_process(self, pid: int) -> tuple[int | None, int]: ...

    def wait_for_single_object(self, handle: int) -> int: ...

    def close_handle(self, handle: int) -> None: ...


class _CtypesWindowsProcessApi:
    """Minimal non-destructive wrapper around documented kernel32 process APIs."""

    def __init__(self) -> None:
        import ctypes

        loader = getattr(ctypes, "WinDLL", None)
        if loader is None:
            raise OSError("Win32 process APIs are unavailable")
        kernel32: Any = loader("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.WaitForSingleObject.restype = ctypes.c_uint32
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int
        self._get_last_error: Any = vars(ctypes)["get_last_error"]
        self._kernel32 = kernel32

    def open_process(self, pid: int) -> tuple[int | None, int]:
        access = _PROCESS_QUERY_LIMITED_INFORMATION | _SYNCHRONIZE
        handle = self._kernel32.OpenProcess(access, False, pid)
        if handle is None:
            return None, int(self._get_last_error())
        return int(handle), 0

    def wait_for_single_object(self, handle: int) -> int:
        return int(self._kernel32.WaitForSingleObject(handle, 0))

    def close_handle(self, handle: int) -> None:
        self._kernel32.CloseHandle(handle)


@contextmanager
def _operation_process_lock(path: Path) -> Iterator[None]:
    """Reference-count the process fallback while leaving the cross-process file durable."""

    lexical = path if path.is_absolute() else path.absolute()
    with _OPERATION_LOCKS_GUARD:
        entry = _OPERATION_LOCKS.get(lexical)
        if entry is None:
            entry = _ProcessLockEntry(threading.Lock())
            _OPERATION_LOCKS[lexical] = entry
        entry.users += 1
    try:
        with entry.lock:
            yield
    finally:
        with _OPERATION_LOCKS_GUARD:
            entry.users -= 1
            if entry.users == 0 and _OPERATION_LOCKS.get(lexical) is entry:
                del _OPERATION_LOCKS[lexical]


def _incoming_owner_key(path: Path) -> Path:
    return path if path.is_absolute() else path.absolute()


def _artifact_lease_key(directory: Path) -> Path:
    lexical = directory if directory.is_absolute() else directory.absolute()
    return lexical / ".lease"


def _acquire_artifact_lease(directory: Path) -> int:
    """Create a durable owner record plus the strongest available process lock."""

    lease = _artifact_lease_key(directory)
    with _ACTIVE_ARTIFACT_LEASES_GUARD:
        if lease in _ACTIVE_ARTIFACT_LEASES:
            raise ValueError("artifact workspace already has an active owner")
        _ACTIVE_ARTIFACT_LEASES[lease] = _PROCESS_OWNER_TOKEN
    descriptor = -1
    lease_created = False
    try:
        flags = os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(lease, flags, 0o600)
        lease_created = True
        if os.name == "posix":
            os.fchmod(descriptor, 0o600)
        payload = orjson.dumps(
            {
                "pid": os.getpid(),
                "owner_token": _PROCESS_OWNER_TOKEN,
            }
        )
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("artifact lease write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        return descriptor
    except BaseException:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if lease_created:
            try:
                lease.unlink(missing_ok=True)
            except OSError:
                pass
        with _ACTIVE_ARTIFACT_LEASES_GUARD:
            _ACTIVE_ARTIFACT_LEASES.pop(lease, None)
        raise


def _release_artifact_lease(directory: Path, descriptor: int) -> None:
    """Release one lease without letting cleanup errors mask pipeline state."""

    lease = _artifact_lease_key(directory)
    try:
        lease.unlink(missing_ok=True)
    except OSError:
        pass
    try:
        os.close(descriptor)
    except OSError:
        pass
    with _ACTIVE_ARTIFACT_LEASES_GUARD:
        _ACTIVE_ARTIFACT_LEASES.pop(lease, None)


@contextmanager
def _relocate_artifact_lease(source: Path, destination: Path) -> Iterator[None]:
    """Move the process owner registration atomically with its directory rename."""

    source_lease = _artifact_lease_key(source)
    destination_lease = _artifact_lease_key(destination)
    with _ACTIVE_ARTIFACT_LEASES_GUARD:
        owner_token = _ACTIVE_ARTIFACT_LEASES.get(source_lease)
        if owner_token is None or destination_lease in _ACTIVE_ARTIFACT_LEASES:
            raise ValueError("artifact workspace lease cannot be relocated safely")
        yield
        del _ACTIVE_ARTIFACT_LEASES[source_lease]
        _ACTIVE_ARTIFACT_LEASES[destination_lease] = owner_token


def _windows_pid_is_alive(pid: int, api: _WindowsProcessApi | None = None) -> bool:
    selected_api = _CtypesWindowsProcessApi() if api is None else api
    try:
        handle, error = selected_api.open_process(pid)
    except OSError:
        return True
    if handle is None:
        if error == _ERROR_ACCESS_DENIED:
            return True
        if error == _ERROR_INVALID_PARAMETER:
            return False
        # Unknown query errors fail closed to avoid deleting an active owner's media.
        return True
    try:
        try:
            wait_result = selected_api.wait_for_single_object(handle)
        except OSError:
            return True
    finally:
        selected_api.close_handle(handle)
    if wait_result == _WAIT_OBJECT_0:
        return False
    if wait_result == _WAIT_TIMEOUT:
        return True
    return True


def _pid_is_alive(
    pid: int,
    *,
    platform_name: str | None = None,
    windows_api: _WindowsProcessApi | None = None,
) -> bool:
    if pid <= 0:
        return False
    selected_platform = os.name if platform_name is None else platform_name
    if selected_platform == "nt":
        return _windows_pid_is_alive(pid, windows_api)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, OverflowError):
        return False
    return True


def _portable_lease_owner_is_active(path: Path) -> bool | None:
    """Return active/dead for a valid record, or unknown for a legacy record."""

    key = path if path.is_absolute() else path.absolute()
    with _ACTIVE_ARTIFACT_LEASES_GUARD:
        if key in _ACTIVE_ARTIFACT_LEASES:
            return True
    descriptor = -1
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        current = path.lstat()
        if (
            _is_link_like(path)
            or not stat.S_ISREG(opened.st_mode)
            or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            return None
        payload = os.read(descriptor, 4097)
        if len(payload) > 4096:
            return None
        parsed = orjson.loads(payload)
    except (FileNotFoundError, OSError, orjson.JSONDecodeError):
        return None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(parsed, dict):
        return None
    pid = parsed.get("pid")
    owner_token = parsed.get("owner_token")
    if not isinstance(pid, int) or isinstance(pid, bool) or not isinstance(owner_token, str):
        return None
    if pid == os.getpid():
        # A live lease in this process is always registered. An unregistered record belongs to
        # an abandoned operation or to a prior process that reused this PID.
        return False
    return _pid_is_alive(pid)


def _register_incoming_owner(path: Path, descriptor: int) -> None:
    lease_descriptor: int | None = None
    if fcntl is not None:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        lease_descriptor = os.dup(descriptor)
    key = _incoming_owner_key(path)
    with _INCOMING_OWNERS_GUARD:
        if key in _INCOMING_OWNERS:
            if lease_descriptor is not None:
                os.close(lease_descriptor)
            raise ValueError(_INCOMING_VALIDATION_ERROR)
        _INCOMING_OWNERS[key] = _IncomingOwner(lease_descriptor)


def _release_incoming_owner(path: Path) -> None:
    key = _incoming_owner_key(path)
    with _INCOMING_OWNERS_GUARD:
        owner = _INCOMING_OWNERS.pop(key, None)
    if owner is not None and owner.lease_descriptor is not None:
        os.close(owner.lease_descriptor)


@contextmanager
def _stale_incoming_ownership(path: Path) -> Iterator[bool]:
    """Try to own a stale candidate without disrupting an active writer/adopter.

    On platforms without ``fcntl``, the registry protects overlapping stores in this process;
    cross-process ownership detection is unavailable there.
    """

    key = _incoming_owner_key(path)
    fallback_owner: _IncomingOwner | None = None
    with _INCOMING_OWNERS_GUARD:
        already_owned = key in _INCOMING_OWNERS
        if not already_owned and fcntl is None:
            fallback_owner = _IncomingOwner(None)
            _INCOMING_OWNERS[key] = fallback_owner
    if already_owned:
        yield False
        return

    descriptor = -1
    try:
        flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
            opened = os.fstat(descriptor)
            current = path.lstat()
        except (FileNotFoundError, OSError):
            yield False
            return
        if (
            _is_link_like(path)
            or not stat.S_ISREG(opened.st_mode)
            or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            yield False
            return
        if fcntl is not None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                if error.errno in {errno.EACCES, errno.EAGAIN}:
                    yield False
                    return
                raise
        yield True
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if fallback_owner is not None:
            with _INCOMING_OWNERS_GUARD:
                if _INCOMING_OWNERS.get(key) is fallback_owner:
                    del _INCOMING_OWNERS[key]


@contextmanager
def _exclusive_posix_lease(path: Path) -> Iterator[bool]:
    """Own one regular no-follow lease file, or fail closed off POSIX."""

    if fcntl is None:
        yield False
        return
    descriptor = -1
    try:
        flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
            opened = os.fstat(descriptor)
            current = path.lstat()
        except (FileNotFoundError, OSError):
            yield False
            return
        if (
            _is_link_like(path)
            or not stat.S_ISREG(opened.st_mode)
            or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            yield False
            return
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in {errno.EACCES, errno.EAGAIN}:
                yield False
                return
            raise
        yield True
    finally:
        if descriptor >= 0:
            os.close(descriptor)


@contextmanager
def _collectable_artifact_lease(path: Path, *, legacy_expired: bool) -> Iterator[bool]:
    """Fence active owners on POSIX and use durable owner records everywhere else."""

    if _is_link_like(path):
        yield False
        return
    if not path.exists():
        yield legacy_expired
        return
    if fcntl is not None:
        with _exclusive_posix_lease(path) as acquired:
            yield acquired
        return
    owner_active = _portable_lease_owner_is_active(path)
    if owner_active is None:
        yield legacy_expired
        return
    yield not owner_active


def _is_link_like(path: Path) -> bool:
    """Detect symlinks and Windows reparse points without following them."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(file_attributes & reparse_attribute)


def _fsync_directory(directory: Path) -> None:
    """Persist a rename on POSIX, falling back when directory fsync is unsupported.

    ``os.replace`` remains atomic on non-POSIX platforms and filesystems that do not expose
    syncable directory descriptors, but those platforms cannot provide this extra durability
    barrier through Python's portable APIs.
    """

    if os.name != "posix":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(directory, flags)
    except OSError as error:
        if error.errno in _UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS:
            return
        raise
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            if error.errno not in _UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS:
                raise
    finally:
        os.close(descriptor)


class PreparedArtifact:
    """A durable temporary artifact that is harmless until its fenced publish."""

    def __init__(
        self,
        store: ArtifactStore,
        job_id: str,
        target: Path,
        temporary: Path,
        content_hash: str,
    ) -> None:
        self._store = store
        self._job_id = job_id
        self.target = target
        self.temporary = temporary
        self.content_hash = content_hash
        self._published = False

    def publish(self) -> None:
        """Atomically replace the target after revalidating its job directory."""

        directory = self._store._job_path(  # noqa: SLF001 - paired capability type
            self._job_id, create=False, require_directory=True
        )
        if directory != self.target.parent or self.temporary.parent != directory:
            raise ValueError("prepared artifact directory changed before publication")
        os.replace(self.temporary, self.target)
        _fsync_directory(directory)
        self._published = True

    def discard(self) -> None:
        """Remove an unpublished temporary artifact."""

        if not self._published:
            self.temporary.unlink(missing_ok=True)


class ArtifactStore:
    """Store one versioned JSON artifact per pipeline stage and job."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve(strict=False)

    def _control_directory(self, name: str, *, create: bool, error_message: str) -> Path:
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
        resolved_root = self.root.resolve()
        candidate = self.root / name
        directory_created = False
        if _is_link_like(candidate):
            raise ValueError(error_message)
        if create:
            try:
                candidate.mkdir(mode=0o700)
                directory_created = True
            except FileExistsError:
                pass
        if candidate.exists():
            try:
                resolved = candidate.resolve()
            except (OSError, RuntimeError):
                raise ValueError(error_message) from None
            if (
                _is_link_like(candidate)
                or not candidate.is_dir()
                or resolved != resolved_root / name
                or resolved.parent != resolved_root
            ):
                raise ValueError(error_message)
            if create:
                os.chmod(candidate, 0o700)
        if directory_created:
            _fsync_directory(self.root)
        return candidate

    def _tombstone_path(self, job_id: str, *, create_directory: bool) -> Path:
        validate_job_id(job_id)
        directory = self._control_directory(
            ".deleted",
            create=create_directory,
            error_message=_TOMBSTONE_VALIDATION_ERROR,
        )
        return directory / f"{job_id}.tombstone"

    def deletion_tombstone_exists(self, job_id: str) -> bool:
        """Return whether a durable, regular tombstone fences this job ID."""

        target = self._tombstone_path(job_id, create_directory=False)
        if not target.exists() and not _is_link_like(target):
            return False
        try:
            metadata = target.lstat()
            parent = target.parent.resolve()
        except (OSError, RuntimeError):
            raise ValueError(_TOMBSTONE_VALIDATION_ERROR) from None
        if (
            _is_link_like(target)
            or not stat.S_ISREG(metadata.st_mode)
            or parent != self.root.resolve() / ".deleted"
        ):
            raise ValueError(_TOMBSTONE_VALIDATION_ERROR)
        return True

    def _tombstone_payload(self, target: Path, *, sync: bool) -> bytes | None:
        if not target.exists() and not _is_link_like(target):
            return None
        if _is_link_like(target):
            raise ValueError(_TOMBSTONE_VALIDATION_ERROR)
        flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(target, flags)
        except FileNotFoundError:
            return None
        except OSError:
            raise ValueError(_TOMBSTONE_VALIDATION_ERROR) from None
        try:
            opened = os.fstat(descriptor)
            current = target.lstat()
            try:
                target_is_local = target.resolve().parent == self.root.resolve() / ".deleted"
            except (OSError, RuntimeError):
                raise ValueError(_TOMBSTONE_VALIDATION_ERROR) from None
            if (
                _is_link_like(target)
                or not target_is_local
                or not stat.S_ISREG(opened.st_mode)
                or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)
            ):
                raise ValueError(_TOMBSTONE_VALIDATION_ERROR)
            os.fchmod(descriptor, 0o600)
            os.lseek(descriptor, 0, os.SEEK_SET)
            payload = os.read(descriptor, 4096)
            if sync:
                os.fsync(descriptor)
            return payload
        finally:
            os.close(descriptor)

    def _replace_tombstone(self, target: Path, payload: bytes) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            remaining = memoryview(payload)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("tombstone write made no progress")
                remaining = remaining[written:]
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            # Never replace a linked or non-regular entry, even inside the private directory.
            self._tombstone_payload(target, sync=False)
            os.replace(temporary, target)
            persisted = self._tombstone_payload(target, sync=True)
            if persisted != payload:
                raise ValueError(_TOMBSTONE_VALIDATION_ERROR)
            _fsync_directory(target.parent)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)

    def mark_deletion_tombstone(self, job_id: str) -> Path:
        """Atomically and durably fence every future write for one random job ID."""

        target = self._tombstone_path(job_id, create_directory=True)
        expected = job_id.encode("ascii")
        if self._tombstone_payload(target, sync=True) != expected:
            self._replace_tombstone(target, expected)
        else:
            _fsync_directory(target.parent)
        return target

    def _job_path(
        self,
        job_id: str,
        *,
        create: bool,
        require_directory: bool,
        allow_deleted: bool = False,
    ) -> Path:
        validate_job_id(job_id)
        if not allow_deleted and self.deletion_tombstone_exists(job_id):
            raise FileNotFoundError(job_id)
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
        resolved_root = self.root.resolve()
        unresolved = self.root / job_id
        if _is_link_like(unresolved):
            raise ValueError(_LINKED_JOB_DIRECTORY_ERROR)
        if create:
            try:
                unresolved.mkdir()
            except FileExistsError:
                pass
        if _is_link_like(unresolved):
            raise ValueError(_LINKED_JOB_DIRECTORY_ERROR)
        try:
            resolved = unresolved.resolve()
        except RuntimeError as error:
            raise ValueError("job directory could not be resolved safely") from error
        expected = resolved_root / job_id
        if resolved != expected or resolved.parent != resolved_root:
            raise ValueError("job directory must resolve directly beneath the configured jobs root")
        if require_directory and not resolved.is_dir():
            raise FileNotFoundError(resolved)
        if _is_link_like(unresolved) or unresolved.resolve() != resolved:
            raise ValueError("job directory changed while it was being validated")
        if not allow_deleted and self.deletion_tombstone_exists(job_id):
            if create:
                try:
                    unresolved.rmdir()
                except OSError:
                    pass
            raise FileNotFoundError(job_id)
        return resolved

    def job_dir(self, job_id: str) -> Path:
        """Return the validated job directory, creating it when necessary."""

        return self._job_path(job_id, create=True, require_directory=True)

    def _validate_upload_suffix(self, suffix: str) -> None:
        if suffix not in {".wav", ".mp3", ".m4a", ".mp4"}:
            raise ValueError(_INCOMING_VALIDATION_ERROR)

    def _validated_incoming_path(self, path: Path, *, require_file: bool) -> Path:
        directory = self._control_directory(
            ".incoming",
            create=False,
            error_message=_INCOMING_VALIDATION_ERROR,
        )
        candidate = Path(path)
        try:
            is_local = candidate.parent.resolve() == directory.resolve()
        except (OSError, RuntimeError):
            raise ValueError(_INCOMING_VALIDATION_ERROR) from None
        if not is_local or _is_link_like(candidate):
            raise ValueError(_INCOMING_VALIDATION_ERROR)
        if not candidate.exists():
            if require_file:
                raise FileNotFoundError(candidate)
            return candidate
        try:
            metadata = candidate.lstat()
        except OSError:
            raise ValueError(_INCOMING_VALIDATION_ERROR) from None
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(_INCOMING_VALIDATION_ERROR)
        return candidate

    def open_incoming_upload(self, job_id: str, suffix: str) -> tuple[int, Path]:
        """Create one private upload file outside every public job directory."""

        validate_job_id(job_id)
        self._validate_upload_suffix(suffix)
        directory = self._control_directory(
            ".incoming",
            create=True,
            error_message=_INCOMING_VALIDATION_ERROR,
        )
        descriptor, name = tempfile.mkstemp(
            dir=directory,
            prefix=f".upload-{job_id}-",
            suffix=suffix,
        )
        path = Path(name)
        try:
            os.fchmod(descriptor, 0o600)
            _register_incoming_owner(path, descriptor)
        except BaseException:
            os.close(descriptor)
            path.unlink(missing_ok=True)
            _fsync_directory(directory)
            raise
        return descriptor, path

    def discard_incoming_upload(self, path: Path) -> None:
        """Remove one exact staged upload and persist its directory entry removal."""

        candidate = Path(path)
        try:
            candidate = self._validated_incoming_path(candidate, require_file=True)
            candidate.unlink(missing_ok=True)
            _fsync_directory(candidate.parent)
        finally:
            _release_incoming_owner(candidate)

    def adopt_incoming_upload(self, job_id: str, path: Path, suffix: str) -> Path:
        """Move one admitted upload into a new exact job directory and persist the move."""

        incoming = Path(path)
        try:
            validate_job_id(job_id)
            self._validate_upload_suffix(suffix)
            incoming = self._validated_incoming_path(incoming, require_file=True)
            if self.deletion_tombstone_exists(job_id):
                raise FileExistsError(job_id)
            self.root.mkdir(parents=True, exist_ok=True)
            directory = self.root / job_id
            if _is_link_like(directory):
                raise ValueError(_LINKED_JOB_DIRECTORY_ERROR)
            try:
                directory.mkdir(mode=0o700)
            except FileExistsError:
                raise FileExistsError(job_id) from None
            destination = directory / f"source-upload{suffix}"
            try:
                os.replace(incoming, destination)
                _fsync_directory(incoming.parent)
                flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                flags |= getattr(os, "O_NOFOLLOW", 0)
                descriptor = os.open(destination, flags)
                try:
                    opened = os.fstat(descriptor)
                    current = destination.lstat()
                    if (
                        _is_link_like(destination)
                        or not stat.S_ISREG(opened.st_mode)
                        or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)
                    ):
                        raise ValueError(_INCOMING_VALIDATION_ERROR)
                    os.fchmod(descriptor, 0o600)
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                _fsync_directory(directory)
                _fsync_directory(self.root)
                return destination
            except BaseException:
                try:
                    if directory.exists() and not _is_link_like(directory):
                        shutil.rmtree(directory)
                        _fsync_directory(self.root)
                except Exception:
                    pass
                try:
                    self.discard_incoming_upload(incoming)
                except Exception:
                    pass
                raise
        finally:
            _release_incoming_owner(incoming)

    def _discard_unregistered_job_under_absence_proof(
        self,
        job_id: str,
        proof: JobAbsenceProof,
    ) -> None:
        """Repository callback: remove an adopted directory during a locked absence proof."""

        validate_job_absence_proof(proof, job_id)
        directory = self._job_path(job_id, create=False, require_directory=False)
        if directory.exists():
            shutil.rmtree(directory)
        _fsync_directory(self.root)

    def remove_stale_incoming_uploads(self, *, lease_seconds: float, now: float) -> int:
        """Remove only old canonical regular files from the private incoming directory."""

        if lease_seconds <= 0:
            raise ValueError("reconciliation lease must be positive")
        directory = self._control_directory(
            ".incoming",
            create=False,
            error_message=_INCOMING_VALIDATION_ERROR,
        )
        if not directory.is_dir():
            return 0
        removed = 0
        for candidate in directory.iterdir():
            name = candidate.name
            if not name.startswith(".upload-") or len(name) < 42:
                continue
            job_id = name[8:40]
            if name[40] != "-" or candidate.suffix not in {".wav", ".mp3", ".m4a", ".mp4"}:
                continue
            try:
                validate_canonical_job_id(job_id)
                metadata = candidate.lstat()
            except (OSError, ValueError):
                continue
            if (
                _is_link_like(candidate)
                or not stat.S_ISREG(metadata.st_mode)
                or now - metadata.st_mtime < lease_seconds
            ):
                continue
            with _stale_incoming_ownership(candidate) as acquired:
                if not acquired:
                    continue
                candidate.unlink()
                removed += 1
        if removed:
            _fsync_directory(directory)
        return removed

    def stale_unregistered_job_candidates(
        self, *, lease_seconds: float, now: float
    ) -> tuple[str, ...]:
        """Return old canonical direct job directories, excluding every control component/link."""

        if lease_seconds <= 0:
            raise ValueError("reconciliation lease must be positive")
        if not self.root.is_dir() or _is_link_like(self.root):
            return ()
        candidates: list[str] = []
        for directory in self.root.iterdir():
            try:
                validate_canonical_job_id(directory.name)
                metadata = directory.lstat()
            except (OSError, ValueError):
                continue
            if (
                _is_link_like(directory)
                or not stat.S_ISDIR(metadata.st_mode)
                or now - metadata.st_mtime < lease_seconds
            ):
                continue
            candidates.append(directory.name)
        return tuple(sorted(candidates))

    def remove_stale_ingest_workspaces(self, *, lease_seconds: float, now: float) -> int:
        """Remove expired abandoned decodes while preserving every locked active workspace.

        Cross-process lease inspection uses ``flock`` when available and durable PID ownership
        records elsewhere. Pre-lease upgrade residue receives a one-hour grace before collection.
        """

        if lease_seconds <= 0:
            raise ValueError("reconciliation lease must be positive")
        if not self.root.is_dir() or _is_link_like(self.root):
            return 0
        parents: list[Path] = []
        incoming = self._control_directory(
            ".incoming",
            create=False,
            error_message=_INCOMING_VALIDATION_ERROR,
        )
        if incoming.is_dir():
            parents.append(incoming)
        for candidate in self.root.iterdir():
            try:
                validate_canonical_job_id(candidate.name)
                metadata = candidate.lstat()
            except (OSError, ValueError):
                continue
            if not _is_link_like(candidate) and stat.S_ISDIR(metadata.st_mode):
                parents.append(candidate)

        removed = 0
        for parent in parents:
            for workspace in parent.iterdir():
                if not workspace.name.startswith(".ingest-"):
                    continue
                try:
                    metadata = workspace.lstat()
                except OSError:
                    continue
                if (
                    _is_link_like(workspace)
                    or not stat.S_ISDIR(metadata.st_mode)
                    or now - metadata.st_mtime < lease_seconds
                ):
                    continue
                legacy_expired = now - metadata.st_mtime >= _LEGACY_ARTIFACT_LEASE_GRACE_SECONDS
                with _collectable_artifact_lease(
                    workspace / ".lease",
                    legacy_expired=legacy_expired,
                ) as acquired:
                    if not acquired:
                        continue
                    try:
                        current = workspace.lstat()
                    except OSError:
                        continue
                    if (
                        _is_link_like(workspace)
                        or not stat.S_ISDIR(current.st_mode)
                        or now - current.st_mtime < lease_seconds
                    ):
                        continue
                    shutil.rmtree(workspace)
                    removed += 1
            if removed:
                _fsync_directory(parent)
        return removed

    def canonical_source_exists(self, job_id: str, source_name: str) -> bool:
        """Validate one adopted source path without following links."""

        try:
            validate_canonical_job_id(job_id)
            directory = self._job_path(job_id, create=False, require_directory=True)
            source = Path(source_name)
            if source.parent != directory or source.name not in {
                "source-upload.wav",
                "source-upload.mp3",
                "source-upload.m4a",
            }:
                return False
            metadata = source.lstat()
            return not _is_link_like(source) and stat.S_ISREG(metadata.st_mode)
        except (OSError, ValueError):
            return False

    def remove_unreferenced_audio_generations(
        self,
        job_id: str,
        keep_paths: tuple[str, ...],
    ) -> int:
        """Best-effort remove obsolete direct audio generations after fenced publication."""

        directory = self._job_path(job_id, create=False, require_directory=True)
        keep: set[Path] = set()
        for raw_path in keep_paths:
            path = Path(raw_path)
            generation = path.parent
            if (
                not path.is_absolute()
                or generation.parent != directory
                or not generation.name.startswith("audio-")
                or _is_link_like(generation)
                or not generation.is_dir()
            ):
                raise ValueError("retained audio path must belong to one direct generation")
            keep.add(generation)
        removed = 0
        for candidate in directory.iterdir():
            if candidate in keep or not candidate.name.startswith("audio-"):
                continue
            try:
                metadata = candidate.lstat()
            except OSError:
                continue
            if _is_link_like(candidate) or not stat.S_ISDIR(metadata.st_mode):
                continue
            lease = candidate / ".lease"
            if lease.exists() or _is_link_like(lease):
                legacy_expired = time() - metadata.st_mtime >= _LEGACY_ARTIFACT_LEASE_GRACE_SECONDS
                with _collectable_artifact_lease(
                    lease,
                    legacy_expired=legacy_expired,
                ) as acquired:
                    if not acquired:
                        continue
                    try:
                        shutil.rmtree(candidate)
                        removed += 1
                    except OSError:
                        continue
            else:
                try:
                    shutil.rmtree(candidate)
                    removed += 1
                except OSError:
                    # Open descriptors remain valid on POSIX. Windows may defer cleanup until a
                    # later retry once active preview handles close.
                    continue
        if removed:
            _fsync_directory(directory)
        return removed

    @contextmanager
    def operation_lock(self, job_id: str) -> Iterator[None]:
        """Serialize one job's claim/reset transitions across runner processes.

        POSIX uses ``flock`` on a validated root-control file. Platforms without ``fcntl``
        retain process-wide per-path serialization but cannot promise cross-process locking.
        Lock files intentionally remain durable because unlinking a potentially shared inode can
        split later callers onto a different lock; only the reference-counted memory entry expires.
        """

        validate_job_id(job_id)
        directory = self._control_directory(
            ".locks",
            create=True,
            error_message=_OPERATION_LOCK_VALIDATION_ERROR,
        )
        target = directory / f"{job_id}.lock"
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        with _operation_process_lock(target):
            if _is_link_like(target):
                raise ValueError(_OPERATION_LOCK_VALIDATION_ERROR)
            try:
                descriptor = os.open(target, flags, 0o600)
            except OSError:
                if _is_link_like(target):
                    raise ValueError(_OPERATION_LOCK_VALIDATION_ERROR) from None
                raise
            try:
                opened = os.fstat(descriptor)
                current = target.lstat()
                try:
                    target_is_local = target.resolve().parent == directory.resolve()
                except (OSError, RuntimeError):
                    raise ValueError(_OPERATION_LOCK_VALIDATION_ERROR) from None
                if (
                    _is_link_like(target)
                    or not target_is_local
                    or not stat.S_ISREG(opened.st_mode)
                    or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)
                ):
                    raise ValueError(_OPERATION_LOCK_VALIDATION_ERROR)
                os.fchmod(descriptor, 0o600)
                if fcntl is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def write_model(self, job_id: str, stage: StageName, value: BaseModel) -> Path:
        """Atomically persist a model without dropping its version fields."""

        prepared = self.prepare_model(job_id, stage, value)
        try:
            prepared.publish()
        finally:
            prepared.discard()
        return prepared.target

    def prepare_model(self, job_id: str, stage: StageName, value: BaseModel) -> PreparedArtifact:
        """Validate and durably stage a model without replacing the fixed artifact."""

        directory = self.job_dir(job_id)
        target = directory / f"{stage.value}.v1.json"
        validated = type(value).model_validate(value.model_dump(mode="python"))
        payload = orjson.dumps(validated.model_dump(mode="json"), option=orjson.OPT_INDENT_2)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as file:
                file.write(payload)
                file.flush()
                os.fsync(file.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return PreparedArtifact(
            self,
            job_id,
            target,
            temporary,
            hashlib.sha256(payload).hexdigest(),
        )

    def artifact_path(self, job_id: str, stage: StageName) -> Path:
        """Return the exact path reserved for one validated stage artifact."""

        if not isinstance(stage, StageName):
            raise KeyError(str(stage))
        directory = self._job_path(job_id, create=False, require_directory=True)
        return directory / f"{stage.value}.v1.json"

    def content_hash(self, job_id: str, stage: StageName) -> str:
        """Hash the exact persisted artifact bytes with SHA-256."""

        digest = hashlib.sha256()
        with self.artifact_path(job_id, stage).open("rb") as artifact:
            while chunk := artifact.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def delete_stage(self, job_id: str, stage: StageName) -> None:
        """Delete only one stage JSON, preserving audio and all other job files."""

        path = self.artifact_path(job_id, stage)
        path.unlink(missing_ok=True)
        _fsync_directory(path.parent)

    def read_model(self, job_id: str, stage: StageName, model_type: type[T]) -> T:
        """Read and validate a versioned stage artifact."""

        directory = self._job_path(job_id, create=False, require_directory=True)
        path = directory / f"{stage.value}.v1.json"
        return model_type.model_validate_json(path.read_bytes())

    def delete_job(self, job_id: str) -> None:
        """Durably fence and remove exactly one job directory; safe to retry."""

        self.mark_deletion_tombstone(job_id)
        resolved_target = self._job_path(
            job_id,
            create=False,
            require_directory=False,
            allow_deleted=True,
        )
        if resolved_target.exists():
            shutil.rmtree(resolved_target)
        _fsync_directory(self.root)
