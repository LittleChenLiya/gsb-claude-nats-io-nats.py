"""Tests for request-waiter closure on permanent connection close.

Uses in-memory fake connections (no real sockets, no Docker) to test:
normal response, successful reconnect, permanent close, and response-vs-close race.
"""

from __future__ import annotations

import asyncio

import pytest
from nats.client import Client, ClientStatus, ServerInfo
from nats.client.connection import Connection


class FakeConnection:
    """A Connection backed by an in-memory asyncio.StreamReader."""

    def __init__(self) -> None:
        self.reader: asyncio.StreamReader = asyncio.StreamReader()
        self._closed = False
        self.written: list[bytes] = []

    def feed(self, data: bytes) -> None:
        self.reader.feed_data(data)

    def feed_eof(self) -> None:
        self.reader.feed_eof()

    async def read(self, n: int) -> bytes:
        return await self.reader.read(n)

    async def readline(self) -> bytes:
        return await self.reader.readline()

    async def readexactly(self, n: int) -> bytes:
        return await self.reader.readexactly(n)

    async def write(self, data: bytes) -> None:
        self.written.append(data)

    def is_connected(self) -> bool:
        return not self._closed

    async def close(self) -> None:
        self._closed = True


FAKE_INFO: dict = {
    "server_id": "fake-server",
    "server_name": "fake",
    "version": "1.0.0",
    "go": "go1.21",
    "host": "localhost",
    "port": 4222,
    "headers": True,
    "auth_required": False,
    "tls_required": False,
    "tls_available": False,
    "max_payload": 1048576,
    "proto": 1,
}


def _make_server_info() -> ServerInfo:
    return ServerInfo(
        server_id="fake-server",
        server_name="fake",
        version="1.0.0",
        go_version="go1.21",
        host="localhost",
        port=4222,
        headers=True,
        auth_required=False,
        tls_required=False,
        tls_available=False,
        tls_verify=False,
        max_payload=1048576,
        proto=1,
    )


def _make_client(conn: Connection, **kwargs) -> Client:
    defaults = dict(
        servers=["localhost:4222"],
        allow_reconnect=True,
        reconnect_time_wait=0.01,
        reconnect_max_attempts=3,
    )
    defaults.update(kwargs)
    client = Client(conn, _make_server_info(), **defaults)
    client._status = ClientStatus.CONNECTED
    client._request_prefix = "_INBOX.test."
    return client


async def _request(client: Client) -> asyncio.Task:
    task = asyncio.create_task(client.request("test.subject", b"hello", timeout=5.0))
    for _ in range(100):
        if client._request_futures:
            break
        await asyncio.sleep(0.001)
    assert client._request_futures, "request future was not registered"
    return task


def _msg_frame(token: int, payload: bytes) -> bytes:
    subject = f"_INBOX.test.{token}"
    return f"MSG {subject} 0 {len(payload)}\r\n".encode() + payload + b"\r\n"


async def test_request_normal_response():
    conn = FakeConnection()
    client = _make_client(conn)
    task = await _request(client)
    conn.feed(_msg_frame(0, b"world"))
    response = await task
    assert response.data == b"world"
    assert 0 not in client._request_futures


async def test_request_survives_successful_reconnect(monkeypatch):
    import nats.client as client_module

    new_conn = FakeConnection()

    async def fake_establish(url, *, timeout, wants_tls, tls=None, tls_hostname=None, tls_handshake_first=False):
        new_conn.feed(b"PONG\r\n")
        return new_conn, dict(FAKE_INFO), False

    monkeypatch.setattr(client_module, "establish_connection", fake_establish)

    conn = FakeConnection()
    client = _make_client(conn, allow_reconnect=True, reconnect_time_wait=0.01)
    task = await _request(client)

    conn.feed_eof()

    for _ in range(200):
        if client._status == ClientStatus.CONNECTED and client._connection is new_conn:
            break
        await asyncio.sleep(0.01)
    assert client._status == ClientStatus.CONNECTED
    assert client._connection is new_conn
    assert 0 in client._request_futures
    assert not client._request_futures[0].done()

    new_conn.feed(_msg_frame(0, b"world"))
    response = await task
    assert response.data == b"world"


async def test_request_fails_on_permanent_close():
    conn = FakeConnection()
    client = _make_client(conn, allow_reconnect=False)
    task = await _request(client)
    conn.feed_eof()
    with pytest.raises(ConnectionError, match="Connection closed"):
        await task
    assert client._status == ClientStatus.CLOSED


async def test_request_fails_on_close():
    conn = FakeConnection()
    client = _make_client(conn)
    task = await _request(client)
    await client.close()
    with pytest.raises(ConnectionError, match="Connection closed"):
        await task


async def test_request_fails_when_reconnect_exhausted(monkeypatch):
    import nats.client as client_module

    async def fake_establish(url, *, timeout, wants_tls, tls=None, tls_hostname=None, tls_handshake_first=False):
        raise ConnectionError("no server available")

    monkeypatch.setattr(client_module, "establish_connection", fake_establish)

    conn = FakeConnection()
    client = _make_client(conn, allow_reconnect=True, reconnect_max_attempts=1, reconnect_time_wait=0.01)
    task = await _request(client)
    conn.feed_eof()

    for _ in range(200):
        if client._status == ClientStatus.CLOSED:
            break
        await asyncio.sleep(0.01)
    assert client._status == ClientStatus.CLOSED
    with pytest.raises(ConnectionError, match="Connection closed"):
        await task


async def test_request_close_race_response_wins():
    conn = FakeConnection()
    client = _make_client(conn, allow_reconnect=False)
    task = await _request(client)
    future = client._request_futures[0]

    conn.feed(_msg_frame(0, b"world"))
    conn.feed_eof()

    response = await task
    assert response.data == b"world"
    assert future.done()
    assert future.exception() is None
    assert future.result().data == b"world"


async def test_request_close_race_close_wins():
    conn = FakeConnection()
    client = _make_client(conn, allow_reconnect=False)
    task = await _request(client)
    future = client._request_futures[0]

    conn.feed_eof()

    with pytest.raises(ConnectionError, match="Connection closed"):
        await task
    assert future.done()
    with pytest.raises(ConnectionError, match="Connection closed"):
        future.result()


async def test_request_future_completed_exactly_once():
    conn = FakeConnection()
    client = _make_client(conn, allow_reconnect=False)
    task = await _request(client)
    future = client._request_futures[0]

    conn.feed(_msg_frame(0, b"world"))
    await asyncio.sleep(0.02)
    assert future.done()

    client._fail_pending_requests(ConnectionError("Connection closed"))
    response = await task
    assert response.data == b"world"


async def test_request_future_fail_then_result_is_noop():
    conn = FakeConnection()
    client = _make_client(conn, allow_reconnect=False)
    task = await _request(client)
    future = client._request_futures[0]

    client._fail_pending_requests(ConnectionError("Connection closed"))
    assert future.done()

    conn.feed(_msg_frame(0, b"world"))
    with pytest.raises(ConnectionError, match="Connection closed"):
        await task
    with pytest.raises(ConnectionError, match="Connection closed"):
        future.result()
