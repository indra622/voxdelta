"""Make "offline" an enforced property rather than a requested one.

Environment variables such as ``HF_HUB_OFFLINE`` are honoured only by the libraries that
choose to read them. A model loader that consults a different hub, or that treats a cache
miss as a reason to fetch, will happily reach the network anyway — and if it catches its
own connection error and falls back, the caller sees a generic failure and never learns
that a fetch was attempted at all.

This guard closes both gaps. While it is active every outbound connection attempt fails
immediately, so nothing can leave the host, and each attempt is counted even when the
caller swallows the resulting exception. "It loaded under the guard" is therefore positive
proof that the assets were local; "it attempted egress" is proof that they were not.

The guard patches the shared ``socket`` module, so it affects the whole process for its
duration and is intended for single-purpose verification tools, not for serving traffic.
It deliberately does not block loopback-free local IPC that never touches ``socket``.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


class NetworkEgressAttempted(OSError):
    """Raised in place of a real connection while the guard is active."""

    def __init__(self) -> None:
        super().__init__("network_egress_blocked")


@dataclass
class EgressLog:
    """How many outbound attempts were blocked, and where they were aimed.

    Targets are recorded as coarse host strings for diagnosis only; nothing here is
    published into an artifact.
    """

    attempts: int = 0
    hosts: list[str] = field(default_factory=list)

    @property
    def attempted(self) -> bool:
        return self.attempts > 0

    def record(self, target: object) -> None:
        self.attempts += 1
        host = ""
        if isinstance(target, tuple) and target:
            host = str(target[0])
        elif isinstance(target, str):
            host = target
        if host and host not in self.hosts:
            self.hosts.append(host)


@contextmanager
def block_network_egress() -> Iterator[EgressLog]:
    """Fail every outbound connection for the duration, counting each attempt."""

    log = EgressLog()
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection
    original_getaddrinfo = socket.getaddrinfo

    def deny_connect(self: socket.socket, address: object) -> None:
        log.record(address)
        raise NetworkEgressAttempted()

    def deny_connect_ex(self: socket.socket, address: object) -> int:
        log.record(address)
        raise NetworkEgressAttempted()

    def deny_create_connection(address: object, *args: object, **kwargs: object) -> object:
        log.record(address)
        raise NetworkEgressAttempted()

    def deny_getaddrinfo(host: object, port: object, *args: object, **kwargs: object) -> object:
        log.record(host)
        raise NetworkEgressAttempted()

    socket.socket.connect = deny_connect  # type: ignore[method-assign,assignment]
    socket.socket.connect_ex = deny_connect_ex  # type: ignore[method-assign,assignment]
    socket.create_connection = deny_create_connection  # type: ignore[assignment]
    socket.getaddrinfo = deny_getaddrinfo  # type: ignore[assignment]
    try:
        yield log
    finally:
        socket.socket.connect = original_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = original_connect_ex  # type: ignore[method-assign]
        socket.create_connection = original_create_connection
        socket.getaddrinfo = original_getaddrinfo


__all__ = ["EgressLog", "NetworkEgressAttempted", "block_network_egress"]
