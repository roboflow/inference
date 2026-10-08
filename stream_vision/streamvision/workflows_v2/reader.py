"""Bounded reader core shared by the video sources.

One ``MemberReader`` per camera; one ``FrameCollector`` per source::

    member thread i:  VideoSource(B frames) ──read_frame──▶ handoff deque (S frames)
                                                                │  shared Condition
    engine reader:    FrameCollector.collect() ◀────────────────┘
                        wait for >= 1 ready member, then up to collection_timeout
                        for batch_size ready members; take <= 1 frame per member,
                        starting after the member served last (rotation)

Every frame waits in at most two bounded places: the VideoSource buffer (B)
and the member's handoff (S). A frame replaced in a full handoff is counted
``evicted`` and released after the shared lock is left; hardware frames keep
the producers' own CUDA synchronisation and lifetime handling.

Stopping a member never blocks the engine: a terminator thread calls
``VideoSource.terminate`` (which joins the capture thread without a deadline
and is refused while VideoSource is INITIALISING) while the member thread keeps
draining the VideoSource buffer until its end marker. ``FrameCollector.close``
waits for all of it up to ``stop_timeout`` and raises ``VideoStopError`` when
any thread is still alive or a producer release failed; the member statistics
then say ``closed=False``.

VideoSource owns producer release and exposes failures through its
``release_error`` observation. After cleanup, the member checks this for every
decoder, including ``auto`` and its OpenCV fallback. Forced and injected
producers use ``_ProducerProxy`` to retain native counters before release and
to observe when release has finished.

VideoSource reports both a finite end and a capture-thread error as the same
end marker. The member tells them apart by the VideoSource state (ERROR versus
ENDED). VideoSource also reports a live stream whose ``grab`` fails as an
ordinary end, so a live end is treated as the failure ``live_stream_ended``.
"""

import threading
import time
from collections import deque
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    get_args,
)

from roboflow_workflows.execution_engine.v2.sources import engine_observation
from streamvision.camera.buffer_strategies import (
    BufferConsumptionStrategy,
    BufferFillingStrategy,
)
from streamvision.camera.entities import StatusUpdate, VideoFrame
from streamvision.camera.exceptions import (
    EndOfStreamError,
    StreamOperationNotAllowedError,
)
from streamvision.camera.source_reference_sanitizer import (
    redact_credentials_in_text,
    sanitize_source_reference,
)
from streamvision.camera.video_source import (
    FRAME_CAPTURED_EVENT,
    FRAME_DROPPED_EVENT,
    SOURCE_ERROR_EVENT,
    StreamState,
    VideoSource,
)
from streamvision.workflows_v2.statistics import MemberStatistics

__all__ = [
    "DECODERS",
    "FrameCollector",
    "MemberReader",
    "VideoSourceFailure",
    "VideoStopError",
    "check_positive_int",
]

Decoder = Literal["auto", "opencv", "gstreamer_cuda", "jetson"]
"""``auto`` lets VideoSource choose (hardware only in tensor mode, OpenCV
fallback); the others force one decoder without fallback."""

EndPolicy = Literal["all", "any"]
FailurePolicy = Literal["fail", "end_member"]

DECODERS = get_args(Decoder)
END_POLICIES = get_args(EndPolicy)
FAILURE_POLICIES = get_args(FailurePolicy)

FACTORY_DECODER = "factory"
"""Statistics ``decoder`` of a member whose producer factory was injected."""

# The one place of the defaults; the source Params use them too.
DEFAULT_DECODER = "auto"
DEFAULT_BUFFER_SIZE = 2
DEFAULT_HANDOFF_SIZE = 2
DEFAULT_COLLECTION_TIMEOUT = 0.003
DEFAULT_END = "all"
DEFAULT_FAILURE_POLICY = "fail"
DEFAULT_STOP_TIMEOUT = 5.0

END_OF_STREAM = "end_of_stream"
STOPPED = "stopped"
FAILURE_REASONS = (
    "open_failed",
    "source_error",
    "live_stream_ended",
    "stalled",
    "frame_error",
)

POLL_SECONDS = 0.02
"""Bound of every internal wait, so stop requests are noticed promptly."""

TERMINATE_RETRY_SECONDS = 0.01

VideoReference = Union[str, int, Callable[[], Any]]


class VideoSourceFailure(RuntimeError):
    """A member failed and the failure policy is ``fail``.

    Args:
        member_id: Member that failed.
        reason: One of ``FAILURE_REASONS``.
        detail: Human-readable detail.
    """

    def __init__(self, member_id: str, *, reason: str, detail: str) -> None:
        super().__init__(f"video member {member_id!r} failed ({reason}): {detail}")
        self.member_id = member_id
        self.reason = reason
        self.detail = detail


class VideoStopError(RuntimeError):
    """Stopping did not finish within the stop timeout, or reported errors."""


def check_positive_int(name: str, value: Any) -> int:
    """Return ``value`` if it is an ``int`` above zero (``bool`` rejected).

    Args:
        name: Parameter name for the error message.
        value: Value to check.

    Returns:
        The value.

    Raises:
        ValueError: When the value is not a strictly positive ``int``.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value!r}")

    return value


def check_non_negative_seconds(name: str, value: Any, *, optional: bool) -> Any:
    """Return ``value`` if it is a finite non-negative number of seconds.

    Args:
        name: Parameter name for the error message.
        value: Value to check.
        optional: Whether ``None`` is accepted.

    Returns:
        The value.

    Raises:
        ValueError: When the value is not a finite number ``>= 0``.
    """
    if value is None and optional:
        return value

    valid = (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and 0 <= value < float("inf")
    )
    if not valid:
        raise ValueError(f"{name} must be a finite number >= 0, got {value!r}")

    return value


def _join_until(
    label: str,
    thread: Optional[threading.Thread],
    deadline: float,
    errors: List[str],
) -> None:
    # Appends to errors when the thread outlives the deadline.
    if thread is None:
        return

    thread.join(timeout=max(0.0, deadline - time.monotonic()))
    if thread.is_alive():
        errors.append(f"{label} still alive after the stop timeout")


class _ProducerProxy:
    """Preserve native counters before release and signal its completion.

    The native pipeline cannot be queried after release. VideoSource records
    any release exception; this proxy leaves that behavior intact.
    """

    def __init__(self, producer: Any) -> None:
        self._producer = producer
        self.native: Optional[Dict[str, int]] = None
        self.released = threading.Event()

    def release(self) -> None:
        try:
            stats = self._producer.tensor_bridge_stats
            self.native = {key: int(value) for key, value in stats.items()}
        except Exception:  # noqa: BLE001 - counters are optional facts
            self.native = None
        try:
            self._producer.release()
        finally:
            self.released.set()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._producer, name)


def forced_producer_factory(
    reference: Union[str, int], *, decoder: str
) -> Optional[Callable[[], Any]]:
    """Return the producer factory of a forced decoder.

    Args:
        reference: File path, URL or device index.
        decoder: One of ``DECODERS``.

    Returns:
        A factory building the decoder's producer; ``None`` for ``auto``,
        where VideoSource chooses the producer itself.

    Raises:
        ValueError: On an unknown decoder.
    """
    if decoder not in DECODERS:
        raise ValueError(f"decoder must be one of {DECODERS}, got {decoder!r}")

    if decoder == "auto":
        return None

    def build() -> Any:
        if decoder == "opencv":
            from streamvision.camera.video_source import CV2VideoFrameProducer

            return CV2VideoFrameProducer(reference)

        if decoder == "gstreamer_cuda":
            from streamvision.camera.gstreamer_cuda_producer import (
                GstreamerCudaVideoFrameProducer as producer_class,
            )
        else:
            from streamvision.camera.jetson_producer import (
                JetsonVideoFrameProducer as producer_class,
            )

        return producer_class(reference, output_tensor=True)

    return build


class MemberReader:
    """One camera: a VideoSource, its member thread and its handoff deque.

    Args:
        member_id: ``"<source>"`` or ``"<source>/<index>"``.
        reference: Video reference, or a producer factory (a callable that
            VideoSource calls to build its ``VideoFrameProducer``).
        condition: Condition shared with the collector; guards the handoff.
        convert: Builds the payload of one frame on the member thread;
            receives the frame and the engine-clock acquisition time.
        decoder: One of ``DECODERS``; ignored with a producer factory.
        buffer_size: VideoSource buffer capacity (B).
        handoff_size: Handoff capacity (S).
        producer_factory: Builds the producer instead of ``decoder``; the
            statistics then report ``reference`` (sanitized) and the decoder
            ``FACTORY_DECODER``.

    Raises:
        ValueError: On an invalid capacity or decoder.
    """

    def __init__(
        self,
        member_id: str,
        *,
        reference: VideoReference,
        condition: threading.Condition,
        convert: Callable[[VideoFrame, Any], Any],
        decoder: str = DEFAULT_DECODER,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        handoff_size: int = DEFAULT_HANDOFF_SIZE,
        producer_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.member_id = member_id
        self.buffer_size = check_positive_int("buffer_size", buffer_size)
        self.handoff_size = check_positive_int("handoff_size", handoff_size)
        if callable(reference):
            name = getattr(reference, "__name__", "?")
            reference, producer_factory = f"<producer factory {name}>", reference
        if producer_factory is None:
            producer_factory = forced_producer_factory(reference, decoder=decoder)
            self.decoder = decoder
        else:
            self.decoder = FACTORY_DECODER
        self.reference = sanitize_source_reference(str(reference))
        # Every producer this member builds; VideoSource builds the auto one.
        self._producers: List[_ProducerProxy] = []
        self._video_reference: VideoReference = (
            reference if producer_factory is None else self._recorded(producer_factory)
        )
        self._condition = condition
        self._convert = convert

        # Guarded by the shared condition.
        self.handoff: Deque[Any] = deque()
        self.last_arrival = time.monotonic()
        self.ended = False
        self.end_reason: Optional[str] = None
        self.end_detail: Optional[str] = None
        self.read = 0
        self.evicted = 0
        self.consumed = 0
        self.cleared = 0
        self.discarded = 0

        # Written by the VideoSource capture thread only.
        self.grabbed = 0
        self.video_source_dropped = 0
        self._error_detail: Optional[str] = None

        # Guarded by _lifecycle.
        self._lifecycle = threading.Lock()
        self._stopping = False
        self._started = False
        self._terminator: Optional[threading.Thread] = None
        self.is_file: Optional[bool] = None
        self.closed = False
        self.close_errors: List[str] = []

        self._video_source: Optional[VideoSource] = None
        self._thread: Optional[threading.Thread] = None

    # Lifecycle ------------------------------------------------------------

    def start(self) -> None:
        """Start the member thread; it connects and reads in the background."""
        self.last_arrival = time.monotonic()
        self._thread = threading.Thread(
            target=self._run, name=f"video-member-{self.member_id}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Ask the member to stop; returns at once. Repeated calls are no-ops."""
        with self._lifecycle:
            if self._stopping:
                return
            self._stopping = True
            start_terminator = self._started
        if start_terminator:
            self._start_terminator()

    def join(self, *, deadline: float) -> None:
        """Wait for the member's threads until ``deadline`` (monotonic seconds).

        Records ``closed`` and ``close_errors``; never raises.

        Args:
            deadline: ``time.monotonic()`` value to give up at.
        """
        errors = []
        _join_until("member thread", self._thread, deadline, errors)
        # Read only now: until the member thread ends, it may start the
        # terminator (when a stop arrives while VideoSource is opening).
        with self._lifecycle:
            terminator = self._terminator
        _join_until("terminator thread", terminator, deadline, errors)
        # VideoSource may release the producer after its end marker was read.
        for producer in list(self._producers):
            if not producer.released.wait(max(0.0, deadline - time.monotonic())):
                errors.append("producer not released after the stop timeout")
        source = self._video_source
        if source is not None and source.release_error is not None:
            errors.append(f"producer release failed: {source.release_error}")
        with self._lifecycle:
            self.close_errors.extend(errors)
            self.closed = not self.close_errors

    def native(self) -> Optional[Dict[str, int]]:
        """Bridge counters frozen at the last producer release, if it had any."""
        if not self._producers:
            return None

        native = self._producers[-1].native

        return native

    def statistics(self) -> MemberStatistics:
        """Return a snapshot of the member's counters and end.

        Returns:
            The snapshot.
        """
        with self._condition:
            handoff = dict(
                read=self.read,
                discarded=self.discarded,
                evicted=self.evicted,
                consumed=self.consumed,
                cleared=self.cleared,
                end_reason=self.end_reason,
                end_detail=self.end_detail,
            )
        with self._lifecycle:
            closed = self.closed
            close_errors = tuple(self.close_errors)
            is_file = self.is_file
        snapshot = MemberStatistics(
            member_id=self.member_id,
            reference=self.reference,
            decoder=self.decoder,
            is_file=is_file,
            grabbed=self.grabbed,
            video_source_dropped=self.video_source_dropped,
            native=self.native(),
            closed=closed,
            close_errors=close_errors,
            **handoff,
        )

        return snapshot

    # Member thread --------------------------------------------------------

    def _recorded(self, factory: Callable[[], Any]) -> Callable[[], _ProducerProxy]:
        def build() -> _ProducerProxy:
            producer = _ProducerProxy(factory())
            self._producers.append(producer)

            return producer

        return build

    def _run(self) -> None:
        try:
            self._video_source = VideoSource.init(
                video_reference=self._video_reference,
                buffer_size=self.buffer_size,
                status_update_handlers=[self._on_status_update],
                buffer_filling_strategy=BufferFillingStrategy.DROP_OLDEST,
                buffer_consumption_strategy=BufferConsumptionStrategy.LAZY,
                allow_tensor_frames=True,
            )
            with self._lifecycle:
                stopping = self._stopping
            if stopping:
                self._end(STOPPED, "stopped before the source was opened")
                return
            self._video_source.start()
        except Exception as error:  # noqa: BLE001 - reported as the end reason
            self._end("open_failed", f"{type(error).__name__}: {error}")
            return

        properties = self._video_source.describe_source().source_properties
        with self._lifecycle:
            self.is_file = None if properties is None else properties.is_file
            self._started = True
            start_terminator = self._stopping
        if start_terminator:
            self._start_terminator()
        self._read_until_end()

    def _read_until_end(self) -> None:
        source = self._video_source
        while True:
            try:
                frame = source.read_frame(timeout=POLL_SECONDS)
            except EndOfStreamError:
                break
            if frame is None:
                continue
            self._accept(frame)
            del frame

        state = source.get_state()
        with self._lifecycle:
            stopping = self._stopping
        if state is StreamState.ERROR:
            self._end("source_error", self._error_detail or "VideoSource error")
        elif stopping:
            self._end(STOPPED, "stop requested")
        elif self.is_file:
            self._end(END_OF_STREAM, "the video file ended")
        else:
            self._end(
                "live_stream_ended",
                "a live stream ended; VideoSource cannot tell a disconnect "
                "from a regular end",
            )

    def _accept(self, frame: VideoFrame) -> None:
        acquired = engine_observation()
        payload = None
        failure = None
        with self._condition:
            self.read += 1
            accepting = not self.ended
            if not accepting:
                self.discarded += 1
        if accepting:
            try:
                payload = self._convert(frame, acquired)
            except Exception as error:  # noqa: BLE001 - reported as end reason
                failure = f"{type(error).__name__}: {error}"
        if failure is not None:
            with self._condition:
                self.discarded += 1
            self._end("frame_error", failure)
            return

        if payload is None:
            return

        evicted = None
        with self._condition:
            if self.ended:
                self.discarded += 1
            else:
                if len(self.handoff) == self.handoff_size:
                    evicted = self.handoff.popleft()
                    self.evicted += 1
                self.handoff.append(payload)
                self.last_arrival = time.monotonic()
                self._condition.notify_all()
        # An evicted or discarded frame dies here, after the lock is left.
        del evicted, payload

    def _on_status_update(self, update: StatusUpdate) -> None:
        # VideoSource capture thread.
        if update.event_type == FRAME_CAPTURED_EVENT:
            self.grabbed += 1
        elif update.event_type == FRAME_DROPPED_EVENT:
            self.video_source_dropped += 1
        elif update.event_type == SOURCE_ERROR_EVENT:
            payload = update.payload or {}
            self._error_detail = (
                f"{payload.get('error_type')}: {payload.get('error_message')}"
            )

    # Ending ---------------------------------------------------------------

    def _end(self, reason: str, detail: str) -> None:
        """Record the first end reason, clear the handoff, stop the source."""
        cleared: List[Any] = []
        with self._condition:
            if not self.ended:
                self.ended = True
                self.end_reason = reason
                self.end_detail = detail
            if reason != END_OF_STREAM:
                cleared = list(self.handoff)
                self.handoff.clear()
                self.cleared += len(cleared)
            self._condition.notify_all()
        # Cleared frames die here, after the lock is left.
        del cleared
        self.stop()

    def fail(self, reason: str, detail: str) -> None:
        """End the member with a failure decided outside it (a stall).

        Args:
            reason: One of ``FAILURE_REASONS``.
            detail: Human-readable detail.
        """
        self._end(reason, detail)

    def _start_terminator(self) -> None:
        with self._lifecycle:
            if self._terminator is not None:
                return
            self._terminator = threading.Thread(
                target=self._terminate,
                name=f"video-terminate-{self.member_id}",
                daemon=True,
            )
            terminator = self._terminator
        terminator.start()

    def _terminate(self) -> None:
        source = self._video_source
        while True:
            try:
                # The member thread keeps draining, so the capture thread can
                # put its end marker even into a full buffer.
                source.terminate(
                    wait_on_frames_consumption=False, purge_frames_buffer=False
                )
                return
            except StreamOperationNotAllowedError:
                # INITIALISING right after start(); RUNNING follows shortly.
                if not self._thread.is_alive():
                    return
                time.sleep(TERMINATE_RETRY_SECONDS)
            except Exception as error:  # noqa: BLE001 - reported at close
                with self._lifecycle:
                    self.close_errors.append(
                        f"terminate failed: {type(error).__name__}: {error}"
                    )
                return


class FrameCollector:
    """Collects frames of several members into bounded batches.

    Args:
        readers: Members in index order; the collector owns their lifecycle.
        condition: The condition shared with every reader.
        batch_size: Most members per collection; at most ``len(readers)``.
        collection_timeout: Seconds to wait for more ready members after the
            first one, before a partial batch is returned.
        end: ``"all"`` ends the source once every member ended and its frames
            were taken; ``"any"`` ends it as soon as one member ended, even
            while other members have frames ready (``close`` clears those).
        on_member_failure: ``"fail"`` raises ``VideoSourceFailure`` on the
            first member failure; ``"end_member"`` ends that member only.
        stall_timeout: Seconds without a new frame after which a member fails
            with ``stalled``; ``None`` disables the check.
        stop_timeout: Seconds ``close`` waits for all threads.
        stop_event: Engine stop request; ``collect`` returns ``None`` once set.

    Raises:
        ValueError: On an invalid setting.
    """

    def __init__(
        self,
        readers: Sequence[MemberReader],
        *,
        condition: threading.Condition,
        batch_size: int,
        collection_timeout: float = DEFAULT_COLLECTION_TIMEOUT,
        end: str = DEFAULT_END,
        on_member_failure: str = DEFAULT_FAILURE_POLICY,
        stall_timeout: Optional[float] = None,
        stop_timeout: float = DEFAULT_STOP_TIMEOUT,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        if not readers:
            raise ValueError("a video source needs at least one member")
        check_positive_int("batch_size", batch_size)
        if batch_size > len(readers):
            raise ValueError(
                f"batch_size {batch_size} exceeds the number of streams "
                f"{len(readers)}"
            )
        if end not in END_POLICIES:
            raise ValueError(f"end must be one of {END_POLICIES}, got {end!r}")
        if on_member_failure not in FAILURE_POLICIES:
            raise ValueError(
                f"on_member_failure must be one of {FAILURE_POLICIES}, got "
                f"{on_member_failure!r}"
            )
        check_non_negative_seconds(
            "collection_timeout", collection_timeout, optional=False
        )
        check_non_negative_seconds("stall_timeout", stall_timeout, optional=True)
        check_non_negative_seconds("stop_timeout", stop_timeout, optional=False)
        self.readers = list(readers)
        self.batch_size = batch_size
        self.collection_timeout = collection_timeout
        self.end = end
        self.on_member_failure = on_member_failure
        self.stall_timeout = stall_timeout
        self.stop_timeout = stop_timeout
        self.stop_event = stop_event or threading.Event()
        self._condition = condition
        self._next = 0
        self._closed = False

    def start(self) -> None:
        """Start every member thread."""
        for reader in self.readers:
            reader.start()

    def collect(self) -> Optional[List[Tuple[int, Any]]]:
        """Wait for the next batch.

        Returns:
            ``[(member position, payload), ...]`` in rotation order, at most
            one per member and at most ``batch_size`` items; ``None`` when the
            source ended or the engine asked it to stop.

        Raises:
            VideoSourceFailure: When a member failed under policy ``fail``.
        """
        with self._condition:
            ready = self._wait_for_ready()
            if not ready:
                return None

            batch = []
            for position in ready[: self.batch_size]:
                reader = self.readers[position]
                batch.append((position, reader.handoff.popleft()))
                reader.consumed += 1
            self._next = (batch[-1][0] + 1) % len(self.readers)

        return batch

    def close(self) -> None:
        """Stop every member and wait for its threads up to ``stop_timeout``.

        Raises:
            VideoStopError: When a thread is still alive at the deadline or a
                member reported a stop error.
        """
        if self._closed:
            return

        self._closed = True
        for reader in self.readers:
            reader.stop()
        deadline = time.monotonic() + self.stop_timeout
        for reader in self.readers:
            reader.join(deadline=deadline)

        cleared: List[Any] = []
        with self._condition:
            for reader in self.readers:
                cleared.extend(reader.handoff)
                reader.cleared += len(reader.handoff)
                reader.handoff.clear()
        # Frames left at close die here, after the lock is left.
        del cleared

        problems = [
            f"{reader.member_id}: {error}"
            for reader in self.readers
            for error in reader.statistics().close_errors
        ]
        if problems:
            raise VideoStopError(
                "video source did not stop cleanly: " + "; ".join(problems)
            )

    # Under the condition --------------------------------------------------

    def _wait_for_ready(self) -> List[int]:
        # Returns ready member positions in rotation order; [] at the end or
        # on stop. _finished runs before the first look and after every wait,
        # so neither ready members nor a long collection_timeout delay it.
        while True:
            if self._finished():
                return []
            ready = self._ready()
            if not ready:
                self._condition.wait(POLL_SECONDS)
                continue

            deadline = time.monotonic() + self.collection_timeout
            while 0 < len(ready) < self.batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                # Bounded: a stop request does not notify the condition.
                self._condition.wait(min(remaining, POLL_SECONDS))
                if self._finished():
                    return []
                # A member that failed meanwhile under end_member has no frames.
                ready = self._ready()
            if ready:
                return ready

    def _finished(self) -> bool:
        # Stop first, then a failure (raised), then the end policy.
        if self.stop_event.is_set():
            return True
        self._expire_stalled_members()
        self._raise_on_failure()
        finished = self._source_ended()

        return finished

    def _ready(self) -> List[int]:
        count = len(self.readers)
        order = [(self._next + offset) % count for offset in range(count)]
        ready = [position for position in order if self.readers[position].handoff]

        return ready

    def _failure(self, reader: MemberReader) -> bool:
        return reader.ended and reader.end_reason in FAILURE_REASONS

    def _raise_on_failure(self) -> None:
        if self.on_member_failure != "fail":
            return

        for reader in self.readers:
            if self._failure(reader):
                raise VideoSourceFailure(
                    reader.member_id,
                    reason=reader.end_reason,
                    detail=reader.end_detail or "",
                )

    def _source_ended(self) -> bool:
        finished = [reader.ended and not reader.handoff for reader in self.readers]
        if self.end == "any":
            ended = any(reader.ended for reader in self.readers)
        else:
            ended = all(finished)

        return ended

    def _expire_stalled_members(self) -> None:
        if self.stall_timeout is None:
            return

        now = time.monotonic()
        stalled = [
            reader
            for reader in self.readers
            if not reader.ended
            and not reader.handoff
            and now - reader.last_arrival > self.stall_timeout
        ]
        for reader in stalled:
            # _end takes the (re-entrant) condition and starts the terminator.
            reader.fail("stalled", f"no frame for more than {self.stall_timeout} s")
