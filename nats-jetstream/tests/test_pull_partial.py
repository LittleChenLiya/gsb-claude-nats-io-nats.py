"""Acceptance tests for partial-result batch pull semantics.

These tests run against a *fake* JetStream (no server, no Docker): a scripted
subscription feeds the pull consumer raw client messages -- two real messages
interleaved with status frames, followed by a silence that triggers the
deadline. They compare the default mode (unchanged behavior) with the new
partial-result mode and verify resources are released, including after
cancellation.
"""

import asyncio
from types import SimpleNamespace

import pytest
from nats.client.message import Message as ClientMessage
from nats.client.message import Status
from nats.jetstream.consumer.pull import PartialFetchResult, PullConsumer, PullMessageBatch


class FakeSubscription:
    """Scripted stand-in for nats.client.Subscription."""

    def __init__(self, items: list[ClientMessage]):
        self._queue: asyncio.Queue[ClientMessage] = asyncio.Queue()
        for item in items:
            self._queue.put_nowait(item)
        self.unsubscribed = False

    @property
    def subject(self) -> str:
        return "_INBOX.fake"

    async def next(self, timeout: float | None = None) -> ClientMessage:
        try:
            if timeout is None:
                return await self._queue.get()
            return await asyncio.wait_for(self._queue.get(), timeout)
        except asyncio.TimeoutError:
            raise TimeoutError("nats: timeout") from None

    async def unsubscribe(self) -> None:
        self.unsubscribed = True


class FakeClient:
    """Minimal client: hands out the scripted subscription, records publishes."""

    def __init__(self, subscription: FakeSubscription):
        self._subscription = subscription
        self.published: list[tuple[str, bytes, str | None]] = []

    async def subscribe(self, subject: str) -> FakeSubscription:
        return self._subscription

    async def publish(self, subject: str, payload: bytes = b"", reply: str | None = None) -> None:
        self.published.append((subject, payload, reply))

    def add_disconnected_callback(self, cb) -> None:
        pass

    def add_reconnected_callback(self, cb) -> None:
        pass


def data_msg(i: int) -> ClientMessage:
    return ClientMessage(subject=f"ORDERS.{i}", data=f"msg-{i}".encode())


def heartbeat() -> ClientMessage:
    return ClientMessage(subject="_INBOX.fake", data=b"", status=Status(code="100", description="Idle Heartbeat"))


def status(code: str, description: str = "") -> ClientMessage:
    return ClientMessage(subject="_INBOX.fake", data=b"", status=Status(code=code, description=description))


def make_consumer(items: list[ClientMessage]) -> tuple[PullConsumer, FakeSubscription, FakeClient]:
    subscription = FakeSubscription(items)
    client = FakeClient(subscription)
    jetstream = SimpleNamespace(_client=client, client=client, api_prefix="$JS.API")
    stream = SimpleNamespace(_jetstream=jetstream)
    info = SimpleNamespace(name="pull_consumer", stream_name="ORDERS")
    return PullConsumer(stream, info), subscription, client


async def test_partial_mode_returns_received_messages_when_deadline_expires():
    """New mode: two messages then silence -> partial result, complete=False."""
    consumer, subscription, _client = make_consumer([data_msg(1), data_msg(2)])

    result = await consumer.fetch_partial(max_messages=5, max_wait=0.2)

    assert isinstance(result, PartialFetchResult)
    assert [m.data for m in result.messages] == [b"msg-1", b"msg-2"]
    assert result.received == 2
    assert result.complete is False
    assert result.error is None
    # Resources released after the batch ended
    assert subscription.unsubscribed is True


async def test_default_fetch_behavior_unchanged_on_timeout():
    """Old mode: same fake stream -> iteration yields the two messages and
    simply ends; no exception is raised and no result object is involved."""
    consumer, subscription, _client = make_consumer([data_msg(1), data_msg(2)])

    batch = await consumer.fetch(max_messages=5, max_wait=0.2)
    received = [msg.data async for msg in batch]

    assert received == [b"msg-1", b"msg-2"]
    assert batch.received == 2
    assert batch.complete is False
    assert batch.error is None
    assert subscription.unsubscribed is True


async def test_default_next_still_raises_timeout_error():
    """Default timeout exception behavior is preserved for next()."""
    consumer, subscription, _client = make_consumer([])

    with pytest.raises(asyncio.TimeoutError):
        await consumer.next(max_wait=0.1)

    assert subscription.unsubscribed is True


async def test_partial_mode_marks_complete_when_batch_fills():
    """New mode: all requested messages arrive -> complete=True."""
    consumer, subscription, _client = make_consumer([data_msg(1), data_msg(2), data_msg(3)])

    result = await consumer.fetch_partial(max_messages=3, max_wait=1.0)

    assert result.complete is True
    assert result.received == 3
    assert result.error is None
    assert subscription.unsubscribed is True


async def test_status_frames_do_not_count_as_messages():
    """Heartbeat (100) and timeout (408) status frames must not inflate the
    received count in either mode."""
    items = [heartbeat(), data_msg(1), heartbeat(), data_msg(2), status("408", "Request Timeout")]
    consumer, subscription, _client = make_consumer(items)

    result = await consumer.fetch_partial(max_messages=5, max_wait=1.0)

    assert result.received == 2
    assert [m.data for m in result] == [b"msg-1", b"msg-2"]
    assert result.complete is False
    assert subscription.unsubscribed is True

    # Same script through the default batch: counters only track real messages.
    consumer2, subscription2, _ = make_consumer(items)
    batch = await consumer2.fetch(max_messages=5, max_wait=1.0)
    received = [msg.data async for msg in batch]
    assert received == [b"msg-1", b"msg-2"]
    assert batch.received == 2
    assert batch.complete is False
    assert subscription2.unsubscribed is True


async def test_408_status_frame_ends_batch_without_counting():
    consumer, subscription, _client = make_consumer([data_msg(1), status("408", "Request Timeout")])

    result = await consumer.fetch_partial(max_messages=5, max_wait=1.0)

    assert result.received == 1
    assert result.complete is False
    assert result.error is None
    assert subscription.unsubscribed is True


async def test_cancel_releases_subscription():
    """Cancelling the task that iterates a batch must release the subscription."""
    consumer, subscription, _client = make_consumer([data_msg(1)])

    batch = await consumer.fetch(max_messages=5, max_wait=30.0)

    async def drain():
        async for _ in batch:
            pass

    task = asyncio.create_task(drain())
    await asyncio.sleep(0.1)  # let it consume the message and block waiting
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert subscription.unsubscribed is True
    assert batch._terminated is True


async def test_cancelled_fetch_partial_releases_subscription():
    """Cancelling fetch_partial itself must also release the subscription."""
    consumer, subscription, _client = make_consumer([data_msg(1)])

    task = asyncio.create_task(consumer.fetch_partial(max_messages=5, max_wait=30.0))
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert subscription.unsubscribed is True


async def test_batch_close_is_idempotent():
    consumer, subscription, _client = make_consumer([])
    batch = await consumer.fetch(max_messages=2, max_wait=0.1)
    assert isinstance(batch, PullMessageBatch)
    await batch.close()
    await batch.close()
    assert subscription.unsubscribed is True
