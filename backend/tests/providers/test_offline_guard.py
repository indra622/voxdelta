"""The egress guard must block, count, and then fully restore the socket module."""

from __future__ import annotations

import socket

import pytest

from voxdelta.providers.offline_guard import (
    NetworkEgressAttempted,
    block_network_egress,
)


def test_a_connection_attempt_is_blocked_and_counted() -> None:
    with block_network_egress() as log:
        with pytest.raises(NetworkEgressAttempted):
            socket.create_connection(("example.invalid", 443))

        assert log.attempted is True
        assert log.attempts == 1
        assert "example.invalid" in log.hosts


def test_name_resolution_is_blocked_too_so_nothing_leaves_the_host() -> None:
    with block_network_egress() as log:
        with pytest.raises(NetworkEgressAttempted):
            socket.getaddrinfo("example.invalid", 443)

        assert log.attempts == 1


def test_a_direct_socket_connect_is_blocked() -> None:
    with block_network_egress() as log:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(NetworkEgressAttempted):
                probe.connect(("example.invalid", 443))
        finally:
            probe.close()

        assert log.attempts == 1


def test_attempts_are_counted_even_when_the_caller_swallows_the_error() -> None:
    """A loader that catches its own connection error must still be detected."""

    with block_network_egress() as log:
        for host in ("first.invalid", "second.invalid", "first.invalid"):
            try:
                socket.create_connection((host, 443))
            except OSError:
                pass

        assert log.attempts == 3
        assert log.hosts == ["first.invalid", "second.invalid"]


def test_the_socket_module_is_restored_exactly_afterwards() -> None:
    before = (
        socket.socket.connect,
        socket.socket.connect_ex,
        socket.create_connection,
        socket.getaddrinfo,
    )

    with block_network_egress():
        assert socket.create_connection is not before[2]

    after = (
        socket.socket.connect,
        socket.socket.connect_ex,
        socket.create_connection,
        socket.getaddrinfo,
    )
    assert before == after


def test_the_socket_module_is_restored_even_when_the_body_raises() -> None:
    before = socket.getaddrinfo

    with pytest.raises(RuntimeError, match="inner"):
        with block_network_egress():
            raise RuntimeError("inner")

    assert socket.getaddrinfo is before


def test_a_clean_run_reports_no_attempt() -> None:
    with block_network_egress() as log:
        pass

    assert log.attempted is False
    assert log.attempts == 0
    assert log.hosts == []
