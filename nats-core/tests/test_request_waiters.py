"""Request-waiter close-out tests.

These tests drive a real ``Client`` (read/write loops, reconnect state
machine, request multiplexer) against an in-process asyncio fake transport.
No NATS server binary or Docker is required.

Scenarios covered:

1. Normal response is delivered to the waiter.
2. A transient disconnect followed by a successful reconnect must not fail
   the in-flight request; the response arrives on the new connection.
3. A permanent close (explicit ``close()``, reconnection disabled, or the
   reconnect budget exhausted) fails the waiter immediately with an
   explicit ``ConnectionClosedError`` instead of a bare
   ``asyncio.CancelledError`` or waiting out the request timeout.
4. A response racing a close completes the waiter exactly once.
"""

import asyncio

import nats.client as client_module
import pytest
from nats.client import Client, ClientStatus, ConnectionClosedError, ServerInfo

PONG_FRAME = b"PONG\r\n"

RECONNECTED_INFO: dict = {
    "server_id": "reconnected",
    "server_name": "reconnected",
    "version": "0.0.0",
    "go": "test",
    "host": "localhost",
    "port": 4222,
    "headers": True,
    "auth_required": False,
    "tls_required": False,
    "tls_available": False,
    "tls_verify": False,
    "max_payload": 1048576,
    "proto": 1,
}


def msg_frame(subject: str, payload: bytes, *, sid: str = "0", reply: str | None = None) -> bytes:
    """Build a raw NATS MSG frame for sid 0 (the request multiplexer)."""
    if reply is not None:
        header = f"MSG {subject} {sid} {reply} {len(payload)}\r\n".encode()
    else:
        header = f"MSG {subject} {sid} {len(payload)}\r\n".encode()
    return header + payload + b"\r\n"


class FakeConnection:
    """Scripted byte-stream transport implementing the Connection protocol."""

    def __init__(self, scripted: bytes = b"") -> None:
        self._buffer = bytearray(scripted)
        self._wake = asyncio.Event()
        self._connected = True
        self.written = bytearray()
        self.closed_by_client = False

    async def readline(self) -> bytes:
        while b"\n" not in self._buffer:
            if not self._connected:
                return b""
            self._wake.clear()
            await self._wake.wait()
        index = self._buffer.index(b"\n") + 1
        line = bytes(self._buffer[:index])
        del self._buffer[:index]
        return line

    async def readexactly(self, n: int) -> bytes:
        while len(self._buffer) < n:
            if not self._connected:
                raise asyncio.IncompleteReadError(bytes(self._buffer), n)
            self._wake.clear()
            await self._wake.wait()
        data = bytes(self._buffer[:n])
        del self._buffer[:n]
        return data

    async def read(self, n: int) -> bytes:
        if not self._buffer:
            if not self._connected:
                return b""
            self._wake.clear()
            await self._wake.wait()
        data = bytes(self._buffer[:n])
        del self._buffer[:n]
        return data

    async def write(self, data: bytes) -> None:
        if not self._connected:
            raise ConnectionError("Not connected")
        self.written += data

    def is_connected(self) -> bool:
        return self._connected

    async def close(self) -> None:
        self.closed_by_client = True
        self.drop()

    # Test-only helpers --------------------------------------------------

    def feed(self, data: bytes) -> None:
        """Deliver server-to-client bytes."""
        self._buffer.extend(data)
        self._wake.set()

    def drop(self) -> None:
        """Simulate the server vanishing mid-stream: reads EOF, writes fail."""
        self._connected = False
        self._wake.set()


def make_server_info() -> ServerInfo:
    return ServerInfo(
        server_id="test",
        server_name="test",
        version="0.0.0",
        go_version="test",
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


def make_client(connection: FakeConnection, **kwargs: object) -> Client:
    """Build a CONNECTED client on top of a fake transport.

    ``_status`` is flipped to CONNECTED synchronously, before the event loop
    gets a chance to schedule the background tasks -- mirroring ``connect()``.
    """
    options: dict = {
        "allow_reconnect": True,
        "reconnect_max_attempts": 3,
        "reconnect_time_wait": 0.02,
        "reconnect_time_wait_max": 0.02,
        "reconnect_jitter": 0.0,
        "reconnect_timeout": 1.0,
        "ping_interval": 1000.0,
    }
    options.update(kwargs)
    client = Client(connection, make_server_info(), servers=["localhost:4222"], **options)
    client._status = ClientStatus.CONNECTED
    return client


async def wait_until(predicate, timeout: float = 1.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() >= deadline:
            raise AssertionError("condition not met before timeout")
        await asyncio.sleep(0)


async def wait_registered(client: Client, token: int = 0) -> str:
    """Wait for request() to register its waiter and flush the PUB frame."""
    await wait_until(lambda: token in client._request_futures)
    await client._force_flush()
    return f"{client._request_prefix}{token}"


# ---------------------------------------------------------------------------
# 1. Normal response
# ---------------------------------------------------------------------------


async def test_request_waiter_receives_normal_response():
    conn = FakeConnection()
    client = make_client(conn, allow_reconnect=False)
    try:
        task = asyncio.create_task(client.request("svc", b"hi", timeout=2.0))
        inbox = await wait_registered(client)
        assert b"PUB svc" in conn.written

        conn.feed(msg_frame(inbox, b"hello"))

        message = await asyncio.wait_for(task, timeout=1.0)
        assert message.data == b"hello"
        assert client._request_futures == {}
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# 2. Transient disconnect + successful reconnect must not pre-fail the waiter
# ---------------------------------------------------------------------------


async def test_request_waiter_survives_transient_reconnect(monkeypatch):
    conn = FakeConnection()
    client = make_client(conn)
    new_conn = FakeConnection(PONG_FRAME)
    gate = asyncio.Event()

    async def fake_establish(*args: object, **kwargs: object):
        await gate.wait()
        return new_conn, RECONNECTED_INFO, False

    monkeypatch.setattr(client_module, "establish_connection", fake_establish)

    try:
        task = asyncio.create_task(client.request("svc", b"hi", timeout=5.0))
        inbox = await wait_registered(client)

        # Server disappears; the read loop enters the reconnect cycle.
        conn.drop()
        await wait_until(lambda: client.status == ClientStatus.RECONNECTING)

        # The waiter must survive the brief outage: no early failure.
        assert not task.done(), "waiter failed during a recoverable reconnect"

        # Let the reconnection land; PONG is already scripted on new_conn.
        gate.set()
        await wait_until(lambda: client.status == ClientStatus.CONNECTED)
        assert not task.done(), "waiter failed even though reconnect succeeded"

        # Response arrives on the new connection and resolves the same future.
        new_conn.feed(msg_frame(inbox, b"back"))
        message = await asyncio.wait_for(task, timeout=1.0)
        assert message.data == b"back"
        assert client._request_futures == {}
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# 3. Permanent close fails the waiter immediately with ConnectionClosedError
# ---------------------------------------------------------------------------


async def test_request_waiter_fails_on_explicit_close():
    conn = FakeConnection()
    client = make_client(conn, allow_reconnect=False)
    task = asyncio.create_task(client.request("svc", b"hi", timeout=5.0))
    await wait_registered(client)

    await client.close()

    with pytest.raises(ConnectionClosedError):
        await asyncio.wait_for(task, timeout=1.0)
    assert client.status == ClientStatus.CLOSED
    assert client._request_futures == {}


async def test_request_after_close_raises_connection_closed_error():
    conn = FakeConnection()
    client = make_client(conn, allow_reconnect=False)
    await client.close()
    with pytest.raises(ConnectionClosedError):
        await client.request("svc", b"hi")


async def test_request_waiter_fails_when_reconnect_disabled():
    conn = FakeConnection()
    client = make_client(conn, allow_reconnect=False)
    task = asyncio.create_task(client.request("svc", b"hi", timeout=5.0))
    await wait_registered(client)

    loop = asyncio.get_running_loop()
    start = loop.time()
    conn.drop()  # read loop -> _force_disconnect -> CLOSED (no reconnect)

    with pytest.raises(ConnectionClosedError):
        await asyncio.wait_for(task, timeout=1.0)
    # Fails immediately, well before the 5s request timeout.
    assert loop.time() - start < 1.0
    assert client.status == ClientStatus.CLOSED


async def test_request_waiter_fails_when_reconnect_budget_exhausted(monkeypatch):
    conn = FakeConnection()
    client = make_client(
        conn,
        allow_reconnect=True,
        reconnect_max_attempts=1,
        reconnect_time_wait=0.01,
    )

    async def failing_establish(*args: object, **kwargs: object):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(client_module, "establish_connection", failing_establish)

    task = asyncio.create_task(client.request("svc", b"hi", timeout=5.0))
    await wait_registered(client)

    loop = asyncio.get_running_loop()
    start = loop.time()
    conn.drop()

    with pytest.raises(ConnectionClosedError):
        await asyncio.wait_for(task, timeout=2.0)
    assert loop.time() - start < 1.0
    assert client.status == ClientStatus.CLOSED
    assert client._request_futures == {}


# ---------------------------------------------------------------------------
# 4. Response vs. close race: the waiter completes exactly once
# ---------------------------------------------------------------------------


async def test_race_response_wins():
    conn = FakeConnection()
    client = make_client(conn, allow_reconnect=False)
    try:
        task = asyncio.create_task(client.request("svc", b"hi", timeout=2.0))
        inbox = await wait_registered(client)

        # Response is dispatched first; the future is popped and resolved.
        await client._handle_msg(inbox, "0", None, b"ok")
        # Close then races the already-resolved waiter: must be a no-op.
        await client.close()

        message = await asyncio.wait_for(task, timeout=1.0)
        assert message.data == b"ok"
        assert client._request_futures == {}
    finally:
        await client.close()


async def test_race_close_wins_late_response_ignored():
    conn = FakeConnection()
    client = make_client(conn, allow_reconnect=False)
    task = asyncio.create_task(client.request("svc", b"hi", timeout=2.0))
    inbox = await wait_registered(client)

    await client.close()
    # A late response for the just-closed waiter must be ignored, not raise
    # InvalidStateError or resurrect the waiter.
    await client._handle_msg(inbox, "0", None, b"too late")

    with pytest.raises(ConnectionClosedError):
        await asyncio.wait_for(task, timeout=1.0)
    assert client._request_futures == {}


async def test_race_concurrent_completes_exactly_once():
    """Stress both interleavings; the waiter always ends in exactly one state."""
    outcomes: set[str] = set()
    for _ in range(10):
        # --- Mode A: response dispatch and close() are scheduled together,
        #     the response handler runs first (its body is await-free once
        #     scheduled) -> response wins.
        conn = FakeConnection()
        client = make_client(conn, allow_reconnect=False)
        task = asyncio.create_task(client.request("svc", b"hi", timeout=5.0))
        inbox = await wait_registered(client)

        handler = asyncio.create_task(client._handle_msg(inbox, "0", None, b"ok"))
        closer = asyncio.create_task(client.close())
        await asyncio.gather(handler, closer, return_exceptions=True)

        results = await asyncio.gather(task, return_exceptions=True)
        outcome = results[0]
        if isinstance(outcome, BaseException):
            raise AssertionError(f"expected response, got {outcome!r}")
        assert outcome.data == b"ok"
        outcomes.add("response")
        assert task.done()
        assert client._request_futures == {}

        # --- Mode B: close() runs on its own concurrently and reaches the
        #     fail-futures step; the response handler runs after that ->
        #     close wins, late response is ignored.
        conn = FakeConnection()
        client = make_client(conn, allow_reconnect=False)
        task = asyncio.create_task(client.request("svc", b"hi", timeout=5.0))
        inbox = await wait_registered(client)

        closer = asyncio.create_task(client.close())
        # Let the close task drive itself through its real awaits until the
        # waiter has been failed and popped.
        await wait_until(lambda: not client._request_futures or closer.done())
        handler = asyncio.create_task(client._handle_msg(inbox, "0", None, b"ok"))
        await asyncio.gather(handler, closer, return_exceptions=True)

        results = await asyncio.gather(task, return_exceptions=True)
        outcome = results[0]
        assert isinstance(outcome, ConnectionClosedError)
        outcomes.add("closed")
        assert task.done()
        assert client._request_futures == {}

    # Both orderings must have been observed.
    assert outcomes == {"closed", "response"}
