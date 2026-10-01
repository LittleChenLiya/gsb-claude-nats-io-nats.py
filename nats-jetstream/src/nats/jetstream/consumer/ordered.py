"""Ordered consumer implementation for JetStream."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, AsyncIterator, overload

from nats.jetstream.consumer import (
    Consumer,
    ConsumerConfig,
    ConsumerInfo,
    ConsumerReset,
    MessageBatch,
    MessageStream,
    OrderedConsumerConfig,
)
from nats.jetstream.consumer.pull import PullConsumer, PullMessageStream
from nats.jetstream.errors import OrderedConsumerClosedError, OrderedConsumerResetError
from nats.jetstream.message import Message

if TYPE_CHECKING:
    from nats.jetstream.stream import Stream

logger = logging.getLogger(__name__)

# How long a pull request rests on the server before expiring, and how often the
# server reports it is still alive while it waits.
_DEFAULT_MAX_WAIT = 30.0
_DEFAULT_HEARTBEAT = 15.0


@dataclass
class _Cursor:
    stream_seq: int = 0
    deliver_seq: int = 0


class OrderedConsumer(Consumer):
    """An ordered consumer that guarantees in-order message delivery.

    Ordered consumers are ephemeral, client-managed pull consumers. When a
    sequence gap, heartbeat miss, or consumer deletion is detected, the library
    automatically recreates the underlying server-side consumer from the last
    successfully consumed stream sequence.
    """

    _stream: Stream
    _cfg: OrderedConsumerConfig
    _current_consumer: PullConsumer | None
    _cursor: _Cursor
    _name_prefix: str
    _serial: int
    # Monotonically increasing generation of the active server-side consumer.
    # Every successful (re)creation bumps it; streams fence messages that carry
    # an older generation so a stale subscription can never deliver late.
    _generation: int
    _closed: bool
    _last_batch: _OrderedMessageBatch | None
    _delete_tasks: set[asyncio.Task[None]]
    # Names of server-side consumers whose release has already been scheduled,
    # so each old subscription/consumer is released at most once.
    _released_consumers: set[str]

    def __init__(
        self,
        stream: Stream,
        cfg: OrderedConsumerConfig,
    ) -> None:
        self._stream = stream
        self._cfg = cfg
        self._current_consumer = None
        self._cursor = _Cursor()
        self._name_prefix = cfg.name_prefix or uuid.uuid4().hex
        self._serial = 0
        self._generation = 0
        self._closed = False
        self._last_batch = None
        self._delete_tasks = set()
        self._released_consumers = set()

    @classmethod
    async def create(cls, stream: Stream, cfg: OrderedConsumerConfig) -> OrderedConsumer:
        """Create an ordered consumer with its initial server-side consumer."""
        oc = cls(stream, cfg)

        # Build initial consumer config and create the server-side consumer
        consumer_config = oc._get_consumer_config()
        consumer = await stream.create_or_update_consumer(consumer_config)
        oc._current_consumer = consumer  # type: ignore[assignment]
        oc._generation = oc._serial

        return oc

    @property
    def name(self) -> str:
        if self._current_consumer is None:
            return f"{self._name_prefix}_0"
        return self._current_consumer.name

    @property
    def stream_name(self) -> str:
        return self._stream.name

    @property
    def info(self) -> ConsumerInfo:
        if self._current_consumer is None:
            raise OrderedConsumerClosedError("Ordered consumer has no active consumer")
        return self._current_consumer.info

    async def get_info(self) -> ConsumerInfo:
        if self._current_consumer is None:
            raise OrderedConsumerClosedError("Ordered consumer has no active consumer")
        return await self._current_consumer.get_info()

    async def reset(self, seq: int | None = None) -> ConsumerReset:
        """Not supported on ordered consumers.

        Ordered consumers manage their own delivery sequence and recover
        from gaps by recreating the underlying consumer. Per ADR-60, an
        external ``$JS.API.CONSUMER.RESET`` should not be issued against
        them — instead, let the existing gap-detection path recreate the
        consumer.
        """
        raise NotImplementedError(
            "reset() is not supported on ordered consumers; recovery happens via consumer recreation"
        )

    async def close(self) -> None:
        """Close the ordered consumer and clean up resources.

        Deletes the current server-side consumer and cancels any pending
        delete tasks. After closing, all operations will raise
        OrderedConsumerClosedError.
        """
        if self._closed:
            return
        self._closed = True

        # Delete current consumer
        if self._current_consumer is not None:
            await self._delete_consumer_once(self._current_consumer.name)
            self._current_consumer = None

        # Cancel and await pending delete tasks
        for task in self._delete_tasks:
            if not task.done():
                task.cancel()
        if self._delete_tasks:
            await asyncio.gather(*self._delete_tasks, return_exceptions=True)
            self._delete_tasks.clear()

    async def __aenter__(self) -> OrderedConsumer:
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.close()

    async def _prepare_fetch(self) -> None:
        """Validate state and reset consumer if needed before a fetch operation."""
        if self._closed:
            raise OrderedConsumerClosedError("Ordered consumer is closed")

        # Update cursor from previous fetch batch, then reset consumer.
        # On the very first fetch, skip reset and use the initially-created consumer.
        if self._last_batch is not None:
            last_sseq = self._last_batch.last_stream_seq
            if last_sseq > 0:
                self._cursor.stream_seq = last_sseq
            await self._reset()

    @overload
    async def fetch(
        self,
        *,
        max_messages: int,
        max_wait: float | None = 30.0,
        heartbeat: float | None = None,
    ) -> MessageBatch: ...

    @overload
    async def fetch(
        self,
        *,
        max_bytes: int,
        max_wait: float | None = 30.0,
        heartbeat: float | None = None,
    ) -> MessageBatch: ...

    async def fetch(
        self,
        *,
        max_messages: int | None = None,
        max_bytes: int | None = None,
        max_wait: float | None = 30.0,
        heartbeat: float | None = None,
    ) -> MessageBatch:
        """Fetch a batch of messages.

        Note: Each fetch call resets the ordered consumer by creating a
        new server-side consumer, even if the previous fetch returned no
        messages. Use messages() for continuous consumption.
        """
        await self._prepare_fetch()
        if self._current_consumer is None:
            raise OrderedConsumerClosedError("Ordered consumer has no active consumer")

        if max_bytes is not None:
            batch = await self._current_consumer.fetch(
                max_bytes=max_bytes,
                max_wait=max_wait,
                heartbeat=heartbeat,
            )
        elif max_messages is not None:
            batch = await self._current_consumer.fetch(
                max_messages=max_messages,
                max_wait=max_wait,
                heartbeat=heartbeat,
            )
        else:
            raise ValueError("Either max_messages or max_bytes must be provided")

        ordered_batch = _OrderedMessageBatch(self, batch)
        self._last_batch = ordered_batch
        return ordered_batch

    @overload
    async def fetch_nowait(
        self,
        *,
        max_messages: int,
    ) -> MessageBatch: ...

    @overload
    async def fetch_nowait(
        self,
        *,
        max_bytes: int,
    ) -> MessageBatch: ...

    async def fetch_nowait(
        self,
        *,
        max_messages: int | None = None,
        max_bytes: int | None = None,
    ) -> MessageBatch:
        """Fetch available messages without waiting.

        Note: Each fetch call resets the ordered consumer, which is less
        efficient than using messages() for continuous consumption.
        """
        await self._prepare_fetch()
        if self._current_consumer is None:
            raise OrderedConsumerClosedError("Ordered consumer has no active consumer")

        if max_bytes is not None:
            batch = await self._current_consumer.fetch_nowait(
                max_bytes=max_bytes,
            )
        elif max_messages is not None:
            batch = await self._current_consumer.fetch_nowait(
                max_messages=max_messages,
            )
        else:
            raise ValueError("Either max_messages or max_bytes must be provided")

        ordered_batch = _OrderedMessageBatch(self, batch)
        self._last_batch = ordered_batch
        return ordered_batch

    async def next(self, max_wait: float = 5.0) -> Message:
        """Fetch the next single message.

        Note: Each call resets the ordered consumer. Use messages() for
        efficient continuous consumption.
        """
        batch = await self.fetch(max_messages=1, max_wait=max_wait)
        async for msg in batch:
            return msg

        if batch.error is not None:
            raise batch.error

        raise asyncio.TimeoutError("No message received within timeout")

    async def messages(
        self,
        *,
        heartbeat: float | None = None,
        max_wait: float | None = None,
        max_messages: int | None = None,
        max_bytes: int | None = None,
    ) -> MessageStream:
        """Get an async iterator for continuous ordered message consumption.

        This is the most efficient way to consume messages from an ordered
        consumer. On sequence gaps or errors, the consumer is automatically
        recreated and consumption resumes from the correct position.

        The max_messages and max_bytes parameters control the internal pull
        batch size, not the total number of messages delivered. The iterator
        continues indefinitely until stopped or closed.
        """
        if self._closed:
            raise OrderedConsumerClosedError("Ordered consumer is closed")

        return _OrderedMessageStream(
            consumer=self,
            heartbeat=heartbeat,
            max_wait=max_wait,
            max_messages=max_messages,
            max_bytes=max_bytes,
        )

    def _get_consumer_config(self) -> ConsumerConfig:
        """Build the server-side ConsumerConfig for the current serial."""
        self._serial += 1
        name = f"{self._name_prefix}_{self._serial}"

        # Determine start sequence
        if self._cursor.stream_seq == 0:
            # No messages consumed yet - use user's deliver policy
            opt_start_seq = self._cfg.opt_start_seq
            deliver_policy = self._cfg.deliver_policy
            opt_start_time = self._cfg.opt_start_time

            # For certain policies, opt_start_seq should not be set
            if deliver_policy in ("last_per_subject", "last", "new", "all"):
                opt_start_seq = None
            elif deliver_policy == "by_start_time":
                opt_start_seq = None
            # For by_start_sequence, keep the user's opt_start_seq
        else:
            # Resume from next sequence after last consumed
            deliver_policy = "by_start_sequence"
            opt_start_seq = self._cursor.stream_seq + 1
            opt_start_time = None

        inactive_threshold = self._cfg.inactive_threshold or timedelta(minutes=5)

        filter_subject = None
        filter_subjects = None
        if self._cfg.filter_subjects:
            if len(self._cfg.filter_subjects) == 1:
                filter_subject = self._cfg.filter_subjects[0]
            else:
                filter_subjects = self._cfg.filter_subjects

        # For last_per_subject with no filter, use ">" to match all
        if deliver_policy == "last_per_subject" and not self._cfg.filter_subjects:
            filter_subjects = [">"]

        return ConsumerConfig(
            name=name,
            deliver_policy=deliver_policy,
            opt_start_seq=opt_start_seq,
            opt_start_time=opt_start_time,
            ack_policy="none",
            inactive_threshold=inactive_threshold,
            num_replicas=1,
            headers_only=self._cfg.headers_only,
            mem_storage=True,
            metadata=self._cfg.metadata,
            replay_policy=self._cfg.replay_policy,
            filter_subject=filter_subject,
            filter_subjects=filter_subjects,
        )

    async def _reset(self) -> None:
        """Reset the ordered consumer by deleting old and creating new.

        A successful reset bumps the generation fence: inner streams created
        before the reset belong to the old generation, and any messages they
        still deliver are discarded.
        """
        if self._closed:
            raise OrderedConsumerClosedError("Ordered consumer is closed")

        # Delete old consumer (fire-and-forget, at most once per name)
        if self._current_consumer is not None:
            old_name = self._current_consumer.name
            self._schedule_consumer_delete(old_name)

        # Reset delivery sequence tracking
        self._cursor.deliver_seq = 0

        # Build new config
        config = self._get_consumer_config()

        # Create new consumer with exponential backoff retry
        interval = 1.0
        max_interval = 10.0
        factor = 2.0
        attempts = 0

        while True:
            if self._closed:
                raise OrderedConsumerClosedError("Ordered consumer is closed")

            try:
                consumer = await self._stream.create_or_update_consumer(config)
                self._current_consumer = consumer  # type: ignore[assignment]
                # Raise the generation fence only after the new consumer exists.
                self._generation = self._serial
                return
            except Exception as e:
                attempts += 1
                max_attempts = self._cfg.max_reset_attempts
                if max_attempts is not None and attempts >= max_attempts:
                    raise OrderedConsumerResetError(
                        f"Failed to reset ordered consumer after {attempts} attempts: {e}"
                    ) from e

                logger.warning(
                    "Ordered consumer reset attempt %d failed: %s. Retrying in %.1fs",
                    attempts,
                    e,
                    interval,
                )

                await asyncio.sleep(interval)
                interval = min(interval * factor, max_interval)

    def _schedule_consumer_delete(self, name: str) -> None:
        """Fire-and-forget release of an old server-side consumer, once per name."""
        if name in self._released_consumers:
            return
        self._released_consumers.add(name)
        task = asyncio.create_task(self._delete_consumer(name))
        self._delete_tasks.add(task)
        task.add_done_callback(self._delete_tasks.discard)

    async def _delete_consumer_once(self, name: str) -> None:
        """Delete a server-side consumer synchronously, at most once per name."""
        if name in self._released_consumers:
            # A fire-and-forget delete was scheduled earlier; await it so the
            # release is complete by the time close() returns.
            pending = [t for t in self._delete_tasks if not t.done()]
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            return
        self._released_consumers.add(name)
        await self._delete_consumer(name)

    async def _delete_consumer(self, name: str) -> None:
        """Delete a server-side consumer, ignoring errors."""
        try:
            await self._stream.delete_consumer(name)
        except Exception:
            pass


class _OrderedMessageBatch(MessageBatch):
    """Wrapper around a PullMessageBatch that tracks stream sequence for ordered consumers."""

    def __init__(self, consumer: OrderedConsumer, inner: MessageBatch) -> None:
        self._consumer = consumer
        self._inner = inner
        self.last_stream_seq: int = 0

    @property
    def error(self) -> Exception | None:
        return self._inner.error

    def __aiter__(self) -> AsyncIterator[Message]:
        return self

    async def __anext__(self) -> Message:
        msg = await self._inner.__anext__()
        # Track the last consumed stream sequence
        self.last_stream_seq = msg.metadata.sequence.stream
        return msg


class _OrderedMessageStream(MessageStream):
    """Ordered message stream that validates sequence and auto-resets on gaps."""

    def __init__(
        self,
        consumer: OrderedConsumer,
        heartbeat: float | None = None,
        max_wait: float | None = None,
        max_messages: int | None = None,
        max_bytes: int | None = None,
    ) -> None:
        self._consumer = consumer
        self._heartbeat = heartbeat
        self._max_wait = max_wait
        self._max_messages = max_messages
        self._max_bytes = max_bytes
        self._inner: PullMessageStream | None = None
        self._closed = False
        # Generation the current inner stream belongs to. Messages delivered
        # by an inner stream from an older generation are stale late arrivals
        # and must be dropped.
        self._generation = 0
        # Set of inner streams whose release has already run, so an old
        # subscription is torn down at most once even if several resets race.
        self._released_inners: set[PullMessageStream] = set()

    async def _create_inner_stream(self) -> PullMessageStream:
        """Create a new inner PullMessageStream from the current consumer.

        The returned stream is fenced to the consumer's current generation.
        """
        if self._consumer._current_consumer is None:
            raise OrderedConsumerClosedError("No active consumer")
        max_wait = self._max_wait if self._max_wait is not None else _DEFAULT_MAX_WAIT
        stream = await self._consumer._current_consumer.messages(
            # An ordered consumer resets when its inner stream ends, and without
            # idle heartbeats nothing ends it: a consumer that stops answering
            # leaves the iterator waiting forever. Always ask for heartbeats,
            # keeping them under the request expiry the server enforces.
            heartbeat=self._heartbeat if self._heartbeat is not None else min(_DEFAULT_HEARTBEAT, max_wait / 2),
            max_wait=max_wait,
            max_messages=self._max_messages,
            max_bytes=self._max_bytes,
        )
        self._generation = self._consumer._generation
        return stream  # type: ignore[return-value]

    async def _release_inner(self, inner: PullMessageStream) -> None:
        """Release (stop) an old inner subscription exactly once."""
        if inner in self._released_inners:
            return
        self._released_inners.add(inner)
        with contextlib.suppress(Exception):
            await inner.stop()

    async def _rebuild(self, inner: PullMessageStream | None) -> PullMessageStream | None:
        """Release the given inner stream and rebuild the consumer generation.

        Returns the new inner stream, or None if the stream/consumer is closed
        or the reset failed terminally.
        """
        if self._closed or self._consumer._closed:
            return None

        if inner is not None:
            await self._release_inner(inner)

        try:
            await self._consumer._reset()
            return await self._create_inner_stream()
        except (OrderedConsumerClosedError, OrderedConsumerResetError):
            self._closed = True
            return None

    async def _rebind(self, inner: PullMessageStream | None) -> bool:
        """Release a stale inner stream and bind one to the *current* consumer.

        Used when the generation already advanced without this stream's own
        rebuild path (e.g. an external reset): no second reset is needed.
        Returns False if the stream is terminally closed.
        """
        if inner is not None:
            await self._release_inner(inner)

        if self._closed or self._consumer._closed or self._consumer._current_consumer is None:
            self._closed = True
            return False

        self._inner = await self._create_inner_stream()
        return True

    def __aiter__(self) -> AsyncIterator[Message]:
        return self

    async def __anext__(self) -> Message:
        if self._closed:
            raise StopAsyncIteration

        # Create inner stream on first access
        if self._inner is None:
            self._inner = await self._create_inner_stream()

        while True:
            inner = self._inner
            generation = self._generation
            expected_consumer = self._consumer._current_consumer

            try:
                msg = await inner.__anext__()
            except StopAsyncIteration:
                # Inner stream ended - this could be heartbeat miss or other error.
                # Rebuild the consumer generation and retry.
                new_inner = await self._rebuild(inner)
                if new_inner is None:
                    raise StopAsyncIteration
                self._inner = new_inner
                continue

            # --- Generation fence -------------------------------------------------
            # A rebuild (heartbeat gap, consumer deletion, sequence mismatch) may
            # have created a newer consumer while this old subscription was still
            # draining. Anything delivered by the old inner stream is a stale,
            # late arrival: release the old subscription once and drop it.
            if generation != self._consumer._generation:
                logger.debug(
                    "Ordered consumer dropping stale message: inner generation %d, current %d",
                    generation,
                    self._consumer._generation,
                )
                if not await self._rebind(inner):
                    raise StopAsyncIteration
                continue

            # Belt-and-braces: the message must name the active consumer. With a
            # real server a late message from the deleted subscription carries
            # the previous consumer name even if generations were observed equal.
            if (
                expected_consumer is not None
                and msg.metadata.consumer
                and msg.metadata.consumer != expected_consumer.name
            ):
                logger.debug(
                    "Ordered consumer dropping stale message from consumer %r (active %r)",
                    msg.metadata.consumer,
                    expected_consumer.name,
                )
                if not await self._rebind(inner):
                    raise StopAsyncIteration
                continue

            # --- Sequence validation ---------------------------------------------
            dseq = msg.metadata.sequence.consumer
            sseq = msg.metadata.sequence.stream

            # New generation boundary (we consumed at least one message before):
            # the recreated consumer must resume exactly at the next stream
            # sequence. Starting past it means messages were silently skipped.
            if generation > 1 and self._consumer._cursor.deliver_seq == 0 and self._consumer._cursor.stream_seq > 0:
                expected_stream_seq = self._consumer._cursor.stream_seq + 1
                if sseq != expected_stream_seq:
                    logger.error(
                        "Ordered consumer recreated at wrong stream sequence: expected %d, got %d. Resetting.",
                        expected_stream_seq,
                        sseq,
                    )
                    new_inner = await self._rebuild(inner)
                    if new_inner is None:
                        raise StopAsyncIteration
                    self._inner = new_inner
                    continue

            # Within a generation consumer_seq must be consecutive.
            expected_dseq = self._consumer._cursor.deliver_seq + 1
            if dseq != expected_dseq:
                logger.debug(
                    "Ordered consumer sequence mismatch: expected %d, got %d. Resetting.",
                    expected_dseq,
                    dseq,
                )
                new_inner = await self._rebuild(inner)
                if new_inner is None:
                    raise StopAsyncIteration
                self._inner = new_inner
                continue

            # Update cursor
            self._consumer._cursor.deliver_seq = dseq
            self._consumer._cursor.stream_seq = sseq
            return msg

    async def stop(self) -> None:
        """Stop the ordered message stream."""
        self._closed = True
        if self._inner is not None:
            await self._release_inner(self._inner)
