"""Deterministic tests for the ordered consumer's generation fence.

These tests do not need a NATS server: they drive ``_OrderedMessageStream``
with controllable fakes to reproduce, in a repeatable way:

1. a disconnect/reconnect style rebuild of the underlying consumer;
2. a stale message arriving *late* on the old subscription after the rebuild;
3. a fresh message arriving on the new subscription;
4. a new generation that wrongly skips the expected stream sequence;
5. an old subscription being released (stopped) exactly once.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from nats.jetstream.consumer import OrderedConsumerConfig
from nats.jetstream.consumer.ordered import OrderedConsumer, _OrderedMessageStream
from nats.jetstream.message import Message, Metadata, SequencePair


def _make_msg(*, stream_seq: int, consumer_seq: int, consumer_name: str) -> Message:
    """Build a JetStream Message carrying explicit metadata."""
    return Message(
        data=f"s{stream_seq}".encode(),
        subject="TEST",
        reply=None,
        headers=None,
        metadata=Metadata(
            sequence=SequencePair(consumer=consumer_seq, stream=stream_seq),
            num_delivered=1,
            num_pending=0,
            timestamp=datetime.now(UTC),
            stream="TEST_STREAM",
            consumer=consumer_name,
        ),
    )


class FakeInner:
    """A scripted stand-in for PullMessageStream.

    ``script`` is a list of items consumed in order:
    - a Message is returned from __anext__;
    - a StopAsyncIteration instance ends the inner stream (heartbeat gap /
      consumer deleted -> disconnect-like rebuild trigger).
    """

    def __init__(self, name: str, script: list):
        self._fake_name = name
        self._script = list(script)
        self.stop_calls = 0
        self._terminated = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._script:
            raise StopAsyncIteration
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    async def stop(self) -> None:
        self.stop_calls += 1
        self._terminated = True


class FakePullConsumer:
    """Minimal stand-in for PullConsumer: hands out scripted inner streams."""

    def __init__(self, name: str, inner_factory):
        self.name = name
        self._inner_factory = inner_factory

    async def messages(self, **kwargs):
        return self._inner_factory()


class FakeStream:
    name = "TEST_STREAM"

    def __init__(self):
        self.deleted_consumers: list[str] = []

    async def delete_consumer(self, name: str) -> None:
        self.deleted_consumers.append(name)


def _make_ordered_consumer() -> OrderedConsumer:
    return OrderedConsumer(FakeStream(), OrderedConsumerConfig(name_prefix="fence"))  # type: ignore[arg-type]


def _install_generation(oc: OrderedConsumer, serial: int, inner_factory) -> FakePullConsumer:
    """Simulate a successful (re)creation of a server-side consumer.

    Mirrors a real reset for generations past the first: delivery sequence
    restarts at 1 while the stream cursor is preserved.
    """
    consumer = FakePullConsumer(f"fence_{serial}", inner_factory)
    oc._current_consumer = consumer  # type: ignore[assignment]
    oc._serial = serial
    oc._generation = serial
    if serial > 1:
        oc._cursor.deliver_seq = 0
    return consumer


async def test_stale_message_after_rebuild_is_dropped_and_new_message_delivered():
    """Late message from the old generation is fenced out; new one is accepted."""
    oc = _make_ordered_consumer()

    # Generation 1 delivers stream seq 1 and still holds a queued late
    # message on its subscription when the rebuild happens.
    late_from_old = _make_msg(stream_seq=2, consumer_seq=2, consumer_name="fence_1")
    old_inner = FakeInner(
        "old",
        [_make_msg(stream_seq=1, consumer_seq=1, consumer_name="fence_1"), late_from_old],
    )
    _install_generation(oc, 1, lambda: old_inner)

    messages = _OrderedMessageStream(oc, max_wait=5.0)

    # First message comes through normally.
    first = await messages.__anext__()
    assert first.metadata.sequence.stream == 1
    assert oc._cursor.stream_seq == 1
    assert oc._cursor.deliver_seq == 1

    # Simulate disconnect + rebuild while the old subscription still has the
    # late message queued. A fresh consumer generation now exists.
    fresh = _make_msg(stream_seq=2, consumer_seq=1, consumer_name="fence_2")
    new_inner = FakeInner("new", [fresh])
    _install_generation(oc, 2, lambda: new_inner)

    # The stale late message is dropped; the iterator returns the fresh
    # message from the new generation instead.
    second = await messages.__anext__()
    assert second.metadata.sequence.stream == 2
    assert second.metadata.consumer == "fence_2"
    assert oc._cursor.stream_seq == 2
    assert oc._cursor.deliver_seq == 1

    # The old subscription was released exactly once; the current one is open.
    assert old_inner.stop_calls == 1
    assert old_inner in messages._released_inners
    assert new_inner.stop_calls == 0
    assert new_inner not in messages._released_inners


async def test_old_subscription_released_only_once_even_if_fence_repeats():
    """Repeated stale arrivals / stop() calls release the old inner once."""
    oc = _make_ordered_consumer()

    # Two messages are queued on the old subscription before the rebuild.
    old_inner = FakeInner(
        "old",
        [
            _make_msg(stream_seq=1, consumer_seq=1, consumer_name="fence_1"),
            _make_msg(stream_seq=2, consumer_seq=2, consumer_name="fence_1"),
        ],
    )
    _install_generation(oc, 1, lambda: old_inner)

    messages = _OrderedMessageStream(oc, max_wait=5.0)
    first = await messages.__anext__()
    assert first.metadata.sequence.stream == 1

    # Rebuild to generation 2 BEFORE the old inner yields its queued late msg.
    fresh = _make_msg(stream_seq=2, consumer_seq=1, consumer_name="fence_2")
    new_inner = FakeInner("new", [fresh])
    _install_generation(oc, 2, lambda: new_inner)

    second = await messages.__anext__()
    assert second.metadata.sequence.stream == 2
    assert second.metadata.consumer == "fence_2"

    assert old_inner.stop_calls == 1

    # stop() must not release the old inner a second time; the current inner
    # is stopped exactly once, and stop() itself is idempotent.
    await messages.stop()
    assert old_inner.stop_calls == 1
    assert new_inner.stop_calls == 1
    await messages.stop()
    assert new_inner.stop_calls == 1


async def test_inner_stream_ending_triggers_rebuild_and_resumes_in_sequence():
    """A dead inner stream (heartbeat gap/delete) triggers exactly one rebuild."""
    oc = _make_ordered_consumer()

    # gen 1 delivers seq 1, then dies (like a disconnect / consumer deletion).
    old_inner = FakeInner(
        "old",
        [_make_msg(stream_seq=1, consumer_seq=1, consumer_name="fence_1"), StopAsyncIteration()],
    )
    _install_generation(oc, 1, lambda: old_inner)

    resets: list[int] = []
    new_inner = FakeInner("new", [_make_msg(stream_seq=2, consumer_seq=1, consumer_name="fence_2")])

    async def fake_reset():
        # Mirror the real reset: recreate from cursor+1 under a new generation.
        resets.append(oc._serial + 1)
        _install_generation(oc, 2, lambda: new_inner)

    oc._reset = fake_reset  # type: ignore[assignment]

    messages = _OrderedMessageStream(oc, max_wait=5.0)

    first = await messages.__anext__()
    assert first.metadata.sequence.stream == 1

    second = await messages.__anext__()
    assert second.metadata.sequence.stream == 2
    assert second.metadata.consumer == "fence_2"
    assert resets == [2]

    # The dead old subscription was released exactly once; the new one is open.
    assert old_inner.stop_calls == 1
    assert new_inner.stop_calls == 0


async def test_new_generation_must_not_skip_expected_stream_sequence():
    """A recreated consumer starting past cursor+1 is rejected and re-rebuilt."""
    oc = _make_ordered_consumer()

    # Simulate one message already consumed before a rebuild.
    _install_generation(oc, 1, lambda: None)
    oc._cursor.deliver_seq = 1
    oc._cursor.stream_seq = 1

    # The post-rebuild generation 2 wrongly skips seq 2 and starts at seq 3.
    bad_inner = FakeInner("bad", [_make_msg(stream_seq=3, consumer_seq=1, consumer_name="fence_2")])
    _install_generation(oc, 2, lambda: bad_inner)

    messages = _OrderedMessageStream(oc, max_wait=5.0)
    # Bind the stream directly to the skipping generation, like a rebuild that
    # recreated the consumer at the wrong start sequence.
    messages._inner = bad_inner
    messages._generation = 2

    # Good generation 3: resumes exactly at seq 2.
    good_inner = FakeInner("good", [_make_msg(stream_seq=2, consumer_seq=1, consumer_name="fence_3")])

    rebuild_count = 0

    async def fake_reset():
        nonlocal rebuild_count
        rebuild_count += 1
        _install_generation(oc, 3, lambda: good_inner)

    oc._reset = fake_reset  # type: ignore[assignment]

    second = await messages.__anext__()
    assert second.metadata.sequence.stream == 2
    assert second.metadata.consumer == "fence_3"
    assert rebuild_count == 1

    # The skipping generation's subscription is released; seq 3 never delivered.
    assert bad_inner.stop_calls == 1
    assert [1, second.metadata.sequence.stream] == [1, 2]


async def test_consumer_name_fence_drops_late_message_within_observed_generation():
    """Even when observed generations match, a foreign consumer name is stale."""
    oc = _make_ordered_consumer()

    old_inner = FakeInner("old", [_make_msg(stream_seq=1, consumer_seq=1, consumer_name="fence_1")])
    _install_generation(oc, 1, lambda: old_inner)
    messages = _OrderedMessageStream(oc, max_wait=5.0)
    first = await messages.__anext__()
    assert first.metadata.sequence.stream == 1

    # A gen-2 consumer is active, but the stream still holds an inner bound to
    # the old generation that emits a late fence_1 message. Force the stream's
    # observed generation to match so only the consumer-name fence can catch it.
    fresh = _make_msg(stream_seq=2, consumer_seq=1, consumer_name="fence_2")
    new_inner = FakeInner("new", [fresh])
    _install_generation(oc, 2, lambda: new_inner)

    stale_inner = FakeInner("stale", [_make_msg(stream_seq=2, consumer_seq=2, consumer_name="fence_1")])
    messages._inner = stale_inner
    messages._generation = oc._generation  # intentionally equal

    second = await messages.__anext__()
    assert second.metadata.sequence.stream == 2
    assert second.metadata.consumer == "fence_2"
    assert stale_inner.stop_calls == 1
    assert new_inner.stop_calls == 0


async def test_ordered_consumer_releases_each_server_consumer_once():
    """The consumer-level release dedupes server-side deletes by name."""
    oc = _make_ordered_consumer()
    c1 = _install_generation(oc, 1, lambda: None)
    c2 = _install_generation(oc, 2, lambda: None)

    # Scheduling the same old name twice must only delete it once.
    oc._schedule_consumer_delete(c1.name)
    oc._schedule_consumer_delete(c1.name)
    # Let the fire-and-forget delete tasks run.
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert oc._stream.deleted_consumers.count("fence_1") == 1  # type: ignore[attr-defined]

    # close() releases the current consumer exactly once and is idempotent.
    assert oc._current_consumer is c2
    await oc.close()
    assert oc._current_consumer is None
    assert oc._stream.deleted_consumers.count("fence_2") == 1  # type: ignore[attr-defined]
    await oc.close()
    assert oc._stream.deleted_consumers.count("fence_2") == 1  # type: ignore[attr-defined]
