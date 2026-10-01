"""Tests for subscription iterator cancellation reclamation.

These tests use a locally injected message queue (a :class:`FakeClient` plus
direct :meth:`Subscription._enqueue` calls) so they are fully deterministic and
do not require a running NATS server. They cover:

- cancelling the consumer before any message arrives (cancel while waiting),
- cancelling the consumer after a message was delivered (cancel while processing),
- repeated / concurrent cancellation and idempotent teardown,
- resource counting (client registry, pending counters, delivered count),
- late messages being rejected after cancellation,
- compatibility of the manual :meth:`Subscription.unsubscribe` path.
"""

import asyncio

import pytest
from nats.client.message import Message
from nats.client.subscription import Subscription


class FakeClient:
    """Minimal stand-in for :class:`~nats.client.Client`.

    Records UNSUB calls and mirrors the client's subscription registry so that
    resource release can be asserted without a network connection.
    """

    def __init__(self):
        self._subscriptions: dict[str, Subscription] = {}
        self.unsub_calls: list[tuple[str, int | None, bool]] = []

    async def _unsubscribe(self, sid: str, *, max_messages: int | None = None, keep: bool = False) -> None:
        self.unsub_calls.append((sid, max_messages, keep))
        if sid in self._subscriptions:
            del self._subscriptions[sid]


def make_subscription(
    subject: str = "test.subject",
    sid: str = "1",
    queue: str = "",
    client: FakeClient | None = None,
    max_pending_messages: int | None = None,
    max_pending_bytes: int | None = None,
) -> tuple[Subscription, FakeClient]:
    """Create a subscription registered with a fake client."""
    client = client or FakeClient()
    subscription = Subscription(
        subject,
        sid,
        queue,
        client,
        max_pending_messages=max_pending_messages,
        max_pending_bytes=max_pending_bytes,
    )
    client._subscriptions[sid] = subscription
    return subscription, client


def inject(subscription: Subscription, data: bytes = b"msg", subject: str | None = None) -> None:
    """Inject a message directly into the subscription's queue (no network)."""
    subscription._enqueue(Message(subject=subject or subscription.subject, data=data))


@pytest.mark.asyncio
async def test_cancel_while_waiting_releases_subscription():
    """Consumer cancelled while waiting for a message: subscription is released and late messages are rejected."""
    subscription, client = make_subscription()
    received: list[bytes] = []

    async def consumer() -> None:
        async for message in subscription.messages:
            received.append(message.data)

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.05)  # let the consumer start waiting

    # Cancel while waiting for a message.
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The subscription is torn down and released from the client registry.
    assert subscription.closed
    assert subscription._sid not in client._subscriptions
    assert len(client._subscriptions) == 0
    assert subscription.pending == (0, 0)
    assert client.unsub_calls == [(subscription._sid, None, False)]

    # The internal wait is unblocked: next() raises RuntimeError immediately,
    # not TimeoutError.
    with pytest.raises(RuntimeError, match="Subscription is closed"):
        await subscription.next(timeout=0.1)

    # A late message must not wake the cancelled iterator.
    with pytest.raises(asyncio.QueueShutDown):
        inject(subscription, b"late")
    assert received == []
    assert subscription.delivered == 0


@pytest.mark.asyncio
async def test_cancel_before_message_then_inject():
    """Cancel before any message arrives; a late message is rejected and not delivered."""
    subscription, client = make_subscription()
    received: list[bytes] = []

    async def consumer() -> None:
        async for message in subscription.messages:
            received.append(message.data)

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.05)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Inject a late message after cancellation.
    with pytest.raises(asyncio.QueueShutDown):
        inject(subscription, b"late")

    assert received == []
    assert subscription.delivered == 0
    assert subscription.closed
    assert subscription._sid not in client._subscriptions
    assert client.unsub_calls == [(subscription._sid, None, False)]


@pytest.mark.asyncio
async def test_cancel_after_message_delivered_releases_subscription():
    """Consumer cancelled after receiving a message (while processing): subscription is released."""
    subscription, client = make_subscription()
    received: list[bytes] = []
    got_message = asyncio.Event()
    generator = subscription.messages

    async def consumer() -> None:
        async for message in generator:
            received.append(message.data)
            got_message.set()
            await asyncio.sleep(10)  # simulate processing -> cancelled here

    task = asyncio.create_task(consumer())
    inject(subscription, b"first")
    await asyncio.wait_for(got_message.wait(), timeout=1.0)

    # Cancel while the consumer is processing the message.
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The generator is suspended at yield; close it as GC finalization would.
    await generator.aclose()

    assert subscription.closed
    assert subscription._sid not in client._subscriptions
    assert subscription.pending == (0, 0)
    assert received == [b"first"]
    assert subscription.delivered == 1
    assert client.unsub_calls == [(subscription._sid, None, False)]


@pytest.mark.asyncio
async def test_cancel_while_processing_releases_via_gc_finalization():
    """Consumer cancelled while processing; subscription is released via GC finalization."""
    subscription, client = make_subscription()
    got_message = asyncio.Event()
    holder: dict[str, object] = {}

    async def consumer() -> None:
        generator = subscription.messages
        holder["generator"] = generator
        async for _ in generator:
            got_message.set()
            await asyncio.sleep(10)

    task = asyncio.create_task(consumer())
    inject(subscription, b"first")
    await asyncio.wait_for(got_message.wait(), timeout=1.0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Drop the generator so the async-gen finalization hook closes it.
    holder["generator"] = None
    for _ in range(20):
        if subscription.closed:
            break
        await asyncio.sleep(0.05)

    assert subscription.closed
    assert subscription._sid not in client._subscriptions
    assert subscription.pending == (0, 0)
    assert client.unsub_calls == [(subscription._sid, None, False)]


@pytest.mark.asyncio
async def test_repeated_cancel_is_idempotent():
    """Repeated cancellation / cleanup calls are idempotent."""
    subscription, client = make_subscription()

    async def consumer() -> None:
        async for _ in subscription.messages:
            pass

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.05)

    task.cancel()
    task.cancel()  # repeated cancel
    with pytest.raises(asyncio.CancelledError):
        await task

    # Cleanup is idempotent: only one UNSUB call.
    assert client.unsub_calls == [(subscription._sid, None, False)]
    assert subscription.closed

    # Calling unsubscribe again is a no-op.
    await subscription.unsubscribe()
    assert client.unsub_calls == [(subscription._sid, None, False)]
    assert subscription.closed


@pytest.mark.asyncio
async def test_multiple_consumers_cancel_releases_once():
    """Multiple consumers on the same subscription: cancelling both releases once."""
    subscription, client = make_subscription()

    async def consumer() -> None:
        async for _ in subscription.messages:
            pass

    task1 = asyncio.create_task(consumer())
    task2 = asyncio.create_task(consumer())
    await asyncio.sleep(0.05)

    task1.cancel()
    task2.cancel()
    await asyncio.gather(task1, task2, return_exceptions=True)

    assert subscription.closed
    assert subscription._sid not in client._subscriptions
    assert len(client._subscriptions) == 0
    # Both generators finalized; cleanup is idempotent.
    assert client.unsub_calls == [(subscription._sid, None, False)]


@pytest.mark.asyncio
async def test_resource_counting_after_cancel():
    """Resource counters: registry count drops to zero and pending counters reset."""
    client = FakeClient()
    subscriptions: list[Subscription] = []
    for i in range(3):
        subscription, _ = make_subscription(sid=str(i), client=client)
        subscriptions.append(subscription)

    assert len(client._subscriptions) == 3

    tasks: list[asyncio.Task[None]] = []
    for subscription in subscriptions:

        async def consumer(s: Subscription = subscription) -> None:
            async for _ in s.messages:
                pass

        tasks.append(asyncio.create_task(consumer()))
    await asyncio.sleep(0.05)

    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    # All subscriptions released.
    assert len(client._subscriptions) == 0
    for subscription in subscriptions:
        assert subscription.closed
        assert subscription.pending == (0, 0)
        with pytest.raises(RuntimeError, match="Subscription is closed"):
            await subscription.next(timeout=0.1)


@pytest.mark.asyncio
async def test_manual_unsubscribe_still_works():
    """Manual unsubscribe behavior is unchanged and rejects late messages."""
    subscription, client = make_subscription()
    inject(subscription, b"msg")
    assert subscription.pending == (1, 3)

    await subscription.unsubscribe()

    assert subscription.closed
    assert subscription._sid not in client._subscriptions
    assert subscription.pending == (0, 0)
    assert client.unsub_calls == [(subscription._sid, None, False)]

    # Late message is rejected.
    with pytest.raises(asyncio.QueueShutDown):
        inject(subscription, b"late")


@pytest.mark.asyncio
async def test_cancel_then_queued_messages_are_discarded():
    """Messages already queued at cancellation time are discarded by the immediate shutdown."""
    subscription, client = make_subscription()
    for i in range(3):
        inject(subscription, f"msg-{i}".encode())
    assert subscription.pending == (3, 15)

    async def consumer() -> None:
        async for _ in subscription.messages:
            pass

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.05)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The immediate shutdown discarded the queued messages.
    assert subscription.closed
    assert subscription.pending == (0, 0)
    assert subscription.delivered == 3
    assert client.unsub_calls == [(subscription._sid, None, False)]
