"""Local tests for subscription iterator cancellation reclamation.

These tests drive :class:`~nats.client.subscription.Subscription` with a
locally injected message queue -- no NATS server, sockets or Docker involved.
They verify that:

* cancelling a consumer parked inside ``async for`` detaches the internal
  queue wait immediately and releases the iterator's subscription reference;
* messages arriving *after* cancellation never wake the cancelled iterator;
* repeated cancellation / ``aclose()`` is idempotent;
* cancellation never unsubscribes -- the sid stays registered and other
  consumers keep working -- while manual ``unsubscribe()`` semantics are
  unchanged.
"""

import asyncio
import gc
import weakref
from typing import Any, cast

import pytest
from nats.client.message import Message
from nats.client.subscription import Subscription, _MessageIterator


class FakeClient:
    """Minimal stand-in for Client exposing only what Subscription needs.

    Records ``UNSUB`` traffic and owns the sid registry exactly like the real
    client, so local teardown paths (``_close_local`` / ``unsubscribe``) can
    be exercised without a server.
    """

    def __init__(self) -> None:
        self._subscriptions: dict[str, Subscription] = {}
        self.unsub_calls: list[tuple[str, int | None, bool]] = []

    async def _unsubscribe(self, sid: str, *, max_messages: int | None = None, keep: bool = False) -> None:
        self.unsub_calls.append((sid, max_messages, keep))
        self._subscriptions.pop(sid, None)


def make_subscription(client: FakeClient | None = None) -> tuple[FakeClient, Subscription]:
    """Build a subscription backed by an injected in-process queue."""
    client = client or FakeClient()
    # The fake only implements the surface Subscription collaborates with.
    subscription = Subscription("test.subject", "sid-1", "", cast(Any, client))
    client._subscriptions["sid-1"] = subscription
    return client, subscription


def inject(subscription: Subscription, data: bytes) -> None:
    """Push a message straight into the subscription's pending queue."""
    subscription._enqueue(Message(subject="test.subject", data=data))


def parked_waiters(subscription: Subscription) -> int:
    """Count consumers currently parked on the internal queue ``get()``.

    The asyncio.Queue stores pending getters in its private ``_getters`` deque;
    expose it through one cast so the tests can assert waiter teardown without
    repeating private-attribute access.
    """
    return len(cast(Any, subscription._pending_queue)._getters)


# ---------------------------------------------------------------------------
# Behaviour before cancellation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_iterator_delivers_injected_messages_before_cancel():
    """Sanity check: the iterator consumes messages from the injected queue."""
    _, subscription = make_subscription()

    received: list[bytes] = []

    async def consume() -> None:
        async for message in subscription.messages:
            received.append(message.data)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)  # let the consumer park on the queue

    inject(subscription, b"one")
    inject(subscription, b"two")

    for _ in range(50):
        if len(received) == 2:
            break
        await asyncio.sleep(0.01)
    assert received == [b"one", b"two"]

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_each_messages_access_returns_independent_iterator():
    """Every async-for / .messages access gets its own iterator instance."""
    _, subscription = make_subscription()

    iterator_a = subscription.messages
    iterator_b = subscription.messages
    assert isinstance(iterator_a, _MessageIterator)
    assert iterator_a is not iterator_b
    assert iterator_a is not subscription.messages


# ---------------------------------------------------------------------------
# Cancellation while waiting for a message
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_parked_iterator_detaches_wait_and_releases_reference():
    """Cancelling a parked consumer tears the internal wait down at once."""
    _, subscription = make_subscription()

    async def consume() -> None:
        async for message in subscription.messages:
            raise AssertionError(f"unexpected message: {message!r}")

    task = asyncio.create_task(consume())
    for _ in range(50):
        if parked_waiters(subscription):
            break
        await asyncio.sleep(0.01)
    assert parked_waiters(subscription) == 1  # one waiter parked on the injected queue

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The internal wait must be gone without needing garbage collection.
    assert parked_waiters(subscription) == 0
    assert task.cancelled()


def test_cancelled_iterator_releases_subscription_reference():
    """After cancellation the iterator no longer keeps the subscription alive.

    A cancelled task's ``CancelledError`` traceback pins the awaiting frames
    (and their locals) for the lifetime of the frame that awaited it -- that is
    standard CPython exception-context retention, independent of this library.
    The meaningful reclaim test therefore runs the whole cancellation lifecycle
    inside a *separate, short-lived* event loop whose frames unwind completely,
    leaving the (immediately detached) iterator as the only possible leaker.

    This is intentionally a synchronous test so ``asyncio.run`` owns the only
    loop and tears it (and every task/traceback frame) down before the
    assertion.
    """
    box: dict[str, object] = {}

    async def lifecycle() -> None:
        client, subscription = make_subscription()
        iterator = subscription.messages

        async def consume() -> None:
            async for message in iterator:
                raise AssertionError(f"unexpected message: {message!r}")

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # The iterator detaches immediately; cancellation sends no UNSUB.
        assert iterator._subscription is None
        assert iterator._closed is True
        assert client.unsub_calls == []
        assert "sid-1" in client._subscriptions

        # Teardown removes the registry's strong reference, so the dead
        # iterator is the only thing that could keep the subscription alive.
        client._subscriptions.pop("sid-1")
        box["ref"] = weakref.ref(subscription)

    asyncio.run(lifecycle())
    gc.collect()

    subscription_ref = box["ref"]
    assert isinstance(subscription_ref, weakref.ReferenceType)
    assert subscription_ref() is None, "cancelled iterator leaked the subscription"


@pytest.mark.asyncio
async def test_iterator_back_reference_nulled_immediately_on_cancel():
    """The iterator drops its subscription reference synchronously on cancel.

    The old generator-based implementation released its closure-held
    subscription only later, when garbage collection ran the async-generator
    finalizer. This iterator detaches as soon as the CancelledError reaches it,
    while the same loop is still running and even though this test frame keeps
    the CancelledError traceback pinned.
    """
    _, subscription = make_subscription()
    iterator = subscription.messages

    async def consume() -> None:
        async for message in iterator:
            raise AssertionError(f"unexpected message: {message!r}")

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # No gc.collect(); the surrounding frame still pins the traceback, yet the
    # iterator's own back-reference is gone immediately.
    assert iterator._closed is True
    assert iterator._subscription is None


@pytest.mark.asyncio
async def test_late_messages_do_not_wake_cancelled_iterator():
    """A message enqueued after cancellation never reaches the dead iterator."""
    client, subscription = make_subscription()
    received: list[bytes] = []

    async def consume() -> None:
        async for message in subscription.messages:
            received.append(message.data)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Inject messages *after* cancellation. The dead iterator must stay dead.
    inject(subscription, b"late-1")
    inject(subscription, b"late-2")
    await asyncio.sleep(0.05)
    assert received == []

    # The late messages are not lost: they remain queued for other consumers.
    assert subscription.pending == (2, len(b"late-1") + len(b"late-2"))
    assert (await subscription.next(timeout=0.5)).data == b"late-1"

    # A fresh iterator picks up where the cancelled one left off.
    async with asyncio.timeout(0.5):
        message = await subscription.messages.__anext__()
    assert message.data == b"late-2"

    assert "sid-1" in client._subscriptions  # cancellation did not unsubscribe
    await subscription.unsubscribe()


@pytest.mark.asyncio
async def test_cancelled_iterator_raises_stop_on_reuse():
    """Re-entering iteration on a cancelled iterator ends it immediately."""
    _, subscription = make_subscription()
    iterator = subscription.messages

    task = asyncio.create_task(iterator.__anext__())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert iterator._closed is True
    with pytest.raises(StopAsyncIteration):
        await iterator.__anext__()


@pytest.mark.asyncio
async def test_cancel_while_processing_message_does_not_lose_later_messages():
    """Cancellation landing while the body runs clears the wait; later msgs queue."""
    client, subscription = make_subscription()
    gate = asyncio.Event()
    received: list[bytes] = []

    async def consume() -> None:
        async for message in subscription.messages:
            received.append(message.data)
            await gate.wait()  # parked in the consumer body, not in the iterator

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)

    inject(subscription, b"in-flight")
    for _ in range(50):
        if received:
            break
        await asyncio.sleep(0.01)
    assert received == [b"in-flight"]

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert parked_waiters(subscription) == 0

    # The iterator is gone; a message arriving afterwards is retained on the
    # subscription and reachable through next()/a fresh iterator.
    inject(subscription, b"after-body-cancel")
    await asyncio.sleep(0.05)
    assert received == [b"in-flight"]
    assert subscription.pending == (1, len(b"after-body-cancel"))
    assert (await subscription.next(timeout=0.5)).data == b"after-body-cancel"
    assert "sid-1" in client._subscriptions
    await subscription.unsubscribe()


# ---------------------------------------------------------------------------
# Repeated cancellation / close
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repeated_task_cancellation_is_safe():
    """Cancelling the consumer again after it already died raises no new errors."""
    _, subscription = make_subscription()

    async def consume() -> None:
        async for message in subscription.messages:
            raise AssertionError(f"unexpected message: {message!r}")

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The task is already finished; further cancels must be no-ops.
    task.cancel()
    task.cancel()
    assert task.cancelled()
    assert parked_waiters(subscription) == 0


@pytest.mark.asyncio
async def test_aclose_is_idempotent_and_ends_iteration():
    """aclose() can be called any number of times, including after cancellation."""
    _, subscription = make_subscription()
    iterator = subscription.messages

    task = asyncio.create_task(iterator.__anext__())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await iterator.aclose()
    await iterator.aclose()
    await iterator.aclose()  # repeated close after cancellation: no-op, no raise

    with pytest.raises(StopAsyncIteration):
        await iterator.__anext__()


@pytest.mark.asyncio
async def test_aclose_without_cancellation_releases_reference():
    """Explicit aclose() detaches an idle iterator even if it was never cancelled."""
    _, subscription = make_subscription()
    iterator = subscription.messages

    await iterator.aclose()
    assert iterator._closed is True
    assert iterator._subscription is None

    with pytest.raises(StopAsyncIteration):
        await iterator.__anext__()
    with pytest.raises(StopAsyncIteration):
        await iterator.__anext__()


# ---------------------------------------------------------------------------
# Cancellation must not interfere with subscriptions / other consumers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_does_not_unsubscribe():
    """Cancelling a consumer sends no UNSUB and leaves the sid registered."""
    client, subscription = make_subscription()

    async def consume() -> None:
        async for message in subscription.messages:
            raise AssertionError(f"unexpected message: {message!r}")

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert client.unsub_calls == []
    assert "sid-1" in client._subscriptions
    assert subscription.closed is False

    # Subscription still accepts and delivers messages via next().
    inject(subscription, b"still-alive")
    assert (await subscription.next(timeout=0.5)).data == b"still-alive"


@pytest.mark.asyncio
async def test_cancelling_one_consumer_keeps_other_consumer_running():
    """A cancelled iterator must not disturb independent iterators on the same sub."""
    _, subscription = make_subscription()
    received_b: list[bytes] = []

    async def consume_a() -> None:
        async for message in subscription.messages:
            raise AssertionError(f"consumer A should be cancelled, got {message.data!r}")

    stop_b = asyncio.Event()

    async def consume_b() -> None:
        async for message in subscription.messages:
            received_b.append(message.data)
            if stop_b.is_set():
                break

    task_a = asyncio.create_task(consume_a())
    task_b = asyncio.create_task(consume_b())
    await asyncio.sleep(0.05)
    assert parked_waiters(subscription) == 2

    task_a.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task_a
    assert parked_waiters(subscription) == 1  # consumer B still parked

    inject(subscription, b"for-b")
    for _ in range(50):
        if received_b:
            break
        await asyncio.sleep(0.01)
    assert received_b == [b"for-b"]

    stop_b.set()
    await subscription.unsubscribe()
    await asyncio.wait_for(task_b, timeout=1.0)


# ---------------------------------------------------------------------------
# Manual unsubscribe / drain compatibility
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_manual_unsubscribe_ends_iterator_and_is_idempotent():
    """Existing unsubscribe behaviour is unchanged: iteration ends, repeated call OK."""
    client, subscription = make_subscription()
    received: list[bytes] = []

    async def consume() -> None:
        async for message in subscription.messages:
            received.append(message.data)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)

    await subscription.unsubscribe()
    await subscription.unsubscribe()  # idempotent today; stays idempotent

    await asyncio.wait_for(task, timeout=1.0)
    assert received == []
    assert subscription.closed is True
    assert "sid-1" not in client._subscriptions
    # UNSUB is sent once; the second unsubscribe() is a no-op once closed.
    assert client.unsub_calls == [("sid-1", None, False)]

    with pytest.raises(RuntimeError, match="Subscription is closed"):
        await subscription.next(timeout=0.5)


@pytest.mark.asyncio
async def test_drain_still_delivers_pending_messages_then_ends_iterator():
    """drain() keeps its graceful semantics: pending messages flow, iteration then ends."""
    client, subscription = make_subscription()
    inject(subscription, b"pending-0")
    inject(subscription, b"pending-1")
    assert subscription.pending == (2, 18)

    await subscription.drain()
    assert subscription.closed is True
    assert "sid-1" not in client._subscriptions

    received: list[bytes] = []
    async for message in subscription.messages:
        received.append(message.data)
    assert received == [b"pending-0", b"pending-1"]

    with pytest.raises(RuntimeError, match="Subscription is closed"):
        await subscription.next(timeout=0.5)


# ---------------------------------------------------------------------------
# Resource accounting across the lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resource_counts_through_cancel_and_new_consumer_lifecycle():
    """Queue waiter and pending-message accounting stay consistent end to end."""
    client, subscription = make_subscription()

    async def consume() -> list[bytes]:
        got: list[bytes] = []
        async for message in subscription.messages:
            got.append(message.data)
        return got

    task = asyncio.create_task(consume())
    for _ in range(50):
        if parked_waiters(subscription):
            break
        await asyncio.sleep(0.01)
    assert parked_waiters(subscription) == 1
    assert subscription.pending == (0, 0)
    assert "sid-1" in client._subscriptions

    # One message delivered before cancel keeps its counters straight.
    inject(subscription, b"x")
    for _ in range(50):
        if subscription.pending == (0, 0):
            break
        await asyncio.sleep(0.01)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert parked_waiters(subscription) == 0
    assert subscription._pending_queue.qsize() == 0

    # A late message is retained, not dispatched to the dead consumer.
    inject(subscription, b"late")
    await asyncio.sleep(0.05)
    assert parked_waiters(subscription) == 0
    assert subscription.pending == (1, 4)

    # A new consumer lifecycle drains the late message and ends on unsubscribe.
    task2 = asyncio.create_task(consume())
    for _ in range(50):
        if subscription.pending == (0, 0):
            break
        await asyncio.sleep(0.01)
    await subscription.unsubscribe()
    got = await asyncio.wait_for(task2, timeout=1.0)
    assert got == [b"late"]
    assert parked_waiters(subscription) == 0
    assert subscription.pending == (0, 0)
    assert "sid-1" not in client._subscriptions
