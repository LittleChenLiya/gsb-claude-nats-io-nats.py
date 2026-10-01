"""Tests for batch pull partial-result semantics.

These tests use a FAKE JetStream client (no nats-server / Docker required)
that scripts the server's behavior: it delivers a fixed number of messages
and then times out, so the old and new modes can be compared directly.
"""

import asyncio
from datetime import UTC, datetime

import pytest
from nats.client.message import Message as ClientMessage
from nats.client.message import Status
from nats.jetstream.consumer import ConsumerConfig, ConsumerInfo
from nats.jetstream.consumer.pull import PullConsumer


def _consumer_info() -> ConsumerInfo:
    return ConsumerInfo(
        stream_name="S",
        name="C",
        config=ConsumerConfig(),
        created=datetime.now(UTC),
        delivered={},
        ack_floor={},
        num_ack_pending=0,
        num_redelivered=0,
        num_waiting=0,
        num_pending=0,
    )


def _real_message(i: int) -> ClientMessage:
    return ClientMessage(
        subject=f"S.{i}",
        data=f"msg-{i}".encode(),
        reply=None,
        headers=None,
        status=None,
    )


def _heartbeat() -> ClientMessage:
    return ClientMessage(subject="S", data=b"", status=Status("100", "Idle Heartbeat"))


class FakeSubscription:
    """A subscription that yields scripted messages, then times out.

    Mirrors the real client's ``subscription.next(timeout=...)``: if the next
    message does not arrive within ``timeout``, ``asyncio.TimeoutError`` is
    raised.
    """

    def __init__(self, script: list[ClientMessage], *, block_forever: bool = False):
        self._script = list(script)
        self._block_forever = block_forever
        self.unsubscribed = False
        self.subject = "inbox.fake"

    async def next(self, timeout: float | None = None) -> ClientMessage:
        if self._block_forever:
            await asyncio.sleep(3600)
        if self._script:
            return self._script.pop(0)
        if timeout is not None:
            await asyncio.sleep(timeout)
        raise asyncio.TimeoutError()

    async def unsubscribe(self) -> None:
        self.unsubscribed = True


class FakeClient:
    def __init__(self, script: list[ClientMessage], *, block_forever: bool = False):
        self._script = script
        self._block_forever = block_forever
        self.subscriptions: dict[int, FakeSubscription] = {}
        self._sid = 0
        self.published: list[tuple] = []

    async def subscribe(self, subject: str, **kwargs) -> FakeSubscription:
        self._sid += 1
        sub = FakeSubscription(self._script, block_forever=self._block_forever)
        self.subscriptions[self._sid] = sub
        return sub

    async def publish(self, subject: str, payload: bytes | None = None, reply: str | None = None, **kwargs) -> None:
        self.published.append((subject, payload, reply))

    def add_disconnected_callback(self, cb) -> None:
        pass

    def add_reconnected_callback(self, cb) -> None:
        pass


class FakeJetStream:
    def __init__(self, client: FakeClient):
        self._client = client
        self.api_prefix = "$JS.API"

    @property
    def client(self) -> FakeClient:
        return self._client


class FakeStream:
    def __init__(self, js: FakeJetStream):
        self._jetstream = js


def _make_consumer(client: FakeClient) -> PullConsumer:
    return PullConsumer(FakeStream(FakeJetStream(client)), _consumer_info())


async def _collect(batch) -> list:
    msgs = []
    async for msg in batch:
        msgs.append(msg)
    return msgs


@pytest.mark.asyncio
async def test_default_mode_timeout_silently_ends_batch():
    """Old mode: two messages then timeout -> batch ends, no error, no partial signal."""
    client = FakeClient([_real_message(1), _real_message(2)])
    consumer = _make_consumer(client)

    batch = await consumer.fetch(max_messages=5, max_wait=0.3)
    msgs = await _collect(batch)

    assert [m.data for m in msgs] == [b"msg-1", b"msg-2"]
    assert batch.error is None
    # Legacy mode does not report partial completion.
    assert batch.complete is True


@pytest.mark.asyncio
async def test_partial_mode_timeout_returns_messages_and_marks_incomplete():
    """New mode: two messages then timeout -> messages returned, batch flagged incomplete."""
    client = FakeClient([_real_message(1), _real_message(2)])
    consumer = _make_consumer(client)

    batch = await consumer.fetch(max_messages=5, max_wait=0.3, partial=True)
    msgs = await _collect(batch)

    assert [m.data for m in msgs] == [b"msg-1", b"msg-2"]
    assert batch.error is None
    assert batch.complete is False


@pytest.mark.asyncio
async def test_partial_mode_full_batch_is_complete():
    """Partial mode with the full batch delivered -> complete=True."""
    client = FakeClient([_real_message(1), _real_message(2), _real_message(3)])
    consumer = _make_consumer(client)

    batch = await consumer.fetch(max_messages=3, max_wait=0.3, partial=True)
    msgs = await _collect(batch)

    assert [m.data for m in msgs] == [b"msg-1", b"msg-2", b"msg-3"]
    assert batch.error is None
    assert batch.complete is True


@pytest.mark.asyncio
async def test_status_frames_do_not_count_toward_message_count():
    """Heartbeat status frames are skipped and never decrement the message count."""
    client = FakeClient([_heartbeat(), _real_message(1), _real_message(2)])
    consumer = _make_consumer(client)

    # Batch of 2: the heartbeat must not count, so both real messages are returned.
    batch = await consumer.fetch(max_messages=2, max_wait=0.3, partial=True)
    msgs = await _collect(batch)

    assert [m.data for m in msgs] == [b"msg-1", b"msg-2"]
    assert batch.complete is True


@pytest.mark.asyncio
async def test_cancellation_cleans_up_subscription_in_both_modes():
    """Cancelling a fetch mid-wait tears down the inbox subscription in both modes."""
    for partial in (False, True):
        client = FakeClient([], block_forever=True)
        consumer = _make_consumer(client)

        async def consume():
            batch = await consumer.fetch(max_messages=5, max_wait=10.0, partial=partial)
            return await _collect(batch)

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(consume(), 0.1)

        assert len(client.subscriptions) == 1
        assert all(sub.unsubscribed for sub in client.subscriptions.values())
