"""Rolling encoded-video ring: protocol, packet store and the shared base class.

A ring buffers one `video_identifier`'s processed frames as H.264 packets with
an IDR at least every output second, so any window can be cut GOP-aligned and
remuxed (never re-encoded) into an MP4.

Threads (all owned by the ring, never a shared executor):

* the caller (pipeline) thread runs `submit()`: grid check, the backend's
  `_prepare_frame` (conversion / copy, so no reference to the caller's frame is
  kept) and a non-blocking enqueue;
* the encoder thread runs the backend's `_encode_frame`, appends packets,
  extends and completes captures and expires stalled ones;
* the finalizer thread remuxes completed captures and resolves their futures.

Captures share packets by reference count; the store evicts whole GOPs by age
(pre-roll + one output second + margin) and by the byte cap
``max(8 MiB, (pre_roll + one output second) * measured bytes/s * 1.5)``.
Byte-cap evictions are counted as ``ring_drops``; a capture whose window start
was already evicted is marked ``partial="ring_evicted"``.

Backends subclass `BaseEncodedRing` and implement the encoder hooks only.
"""

import abc
import logging
import math
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from fractions import Fraction
from functools import partial
from pathlib import Path
from queue import Empty, Queue
from typing import (
    Any,
    ClassVar,
    Deque,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    runtime_checkable,
)

from roboflow_workflows.core_steps.common.video_clips.admission import (
    Budget,
    default_budget,
)
from roboflow_workflows.core_steps.common.video_clips.artifact import (
    ClipArtifact,
    allocate_clip_path,
)
from roboflow_workflows.core_steps.common.video_clips.capture import (
    PARTIAL_RESOLUTION_CHANGE,
    PARTIAL_RING_EVICTED,
    PARTIAL_SHUTDOWN,
    PARTIAL_STALLED,
    STALL_GRACE_SECONDS,
    CaptureHandle,
    ClipEncodeError,
    ClipShutdownError,
    ClipUnavailableError,
    PendingCapture,
)
from roboflow_workflows.core_steps.common.video_clips.grid import (
    PTS_CLOCK_RATE,
    SLOT_EPSILON,
    GridSelector,
    TimelinePlan,
    plan_timeline,
    to_seconds,
)
from roboflow_workflows.core_steps.common.video_clips.report import (
    ClipReport,
    LossCounts,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BaseEncodedRing",
    "CaptureHandle",
    "EncodedOutput",
    "EncodedPacket",
    "PacketStore",
    "PreparedFrame",
    "Ring",
    "RingStats",
    "VideoExtraMissingError",
    "import_av",
    "mux_h264_packets",
]

DEFAULT_QUEUE_DEPTH = 30
DEFAULT_CLOSE_DEADLINE_SECONDS = 30.0
MIN_RING_BYTE_CAP = 8 * 1024 * 1024
RING_BYTE_CAP_FACTOR = 1.5
RETENTION_MARGIN_SECONDS = 0.5
MIN_RATE_SPAN_SECONDS = 1.0
COST_SAMPLES = 600
LOSS_EVENT_HISTORY = 65536
MAX_CONSECUTIVE_ENCODE_ERRORS = 3
WORKER_POLL_SECONDS = 0.1
TIME_EPSILON = 1e-6
LOGGED_ERRORS_BEFORE_THROTTLE = 5
LOG_EVERY_NTH_ERROR = 100
UNAVAILABLE_CLOSING = "unavailable:closing"


class VideoExtraMissingError(RuntimeError):
    """PyAV (`av`) is not importable, so clips cannot be encoded or muxed."""


def import_av() -> Any:
    """Import PyAV on demand.

    Returns:
        The `av` module.

    Raises:
        VideoExtraMissingError: If PyAV is not installed.
    """
    try:
        import av
    except ImportError as error:
        raise VideoExtraMissingError(
            "Video clips need PyAV (`av`), a dependency of roboflow-workflows; "
            "reinstall the package to restore it."
        ) from error

    return av


@dataclass(frozen=True)
class PreparedFrame:
    """A frame converted on the caller's thread, waiting for the encoder.

    Attributes:
        payload: Backend-specific data (an I420 array, a GPU pool slot...).
        width: Encoded width (even).
        height: Encoded height (even).
        nbytes: Bytes counted against the budget while queued.
    """

    payload: Any
    width: int
    height: int
    nbytes: int


@dataclass(frozen=True)
class EncodedOutput:
    """One packet returned by a backend encoder (H.264 Annex B).

    Attributes:
        payload: The packet bytes.
        is_keyframe: True for an IDR access unit.
    """

    payload: bytes
    is_keyframe: bool


class EncodedPacket:
    """One encoded frame in a ring, shared by captures through a refcount.

    Args:
        frame_timestamp: Source timestamp of the frame.
        t: The same timestamp in POSIX seconds.
        frame_number: Source frame number.
        payload: H.264 Annex B access unit.
        is_keyframe: True for an IDR.
        width: Frame width.
        height: Frame height.
    """

    __slots__ = (
        "frame_timestamp",
        "t",
        "frame_number",
        "payload",
        "is_keyframe",
        "width",
        "height",
        "refs",
        "in_store",
    )

    def __init__(
        self,
        *,
        frame_timestamp: datetime,
        t: float,
        frame_number: int,
        payload: bytes,
        is_keyframe: bool,
        width: int,
        height: int,
    ):
        self.frame_timestamp = frame_timestamp
        self.t = t
        self.frame_number = frame_number
        self.payload = payload
        self.is_keyframe = is_keyframe
        self.width = width
        self.height = height
        self.refs = 0
        self.in_store = False

    @property
    def nbytes(self) -> int:
        """Size of the packet payload in bytes."""
        return len(self.payload)


class PacketStore:
    """Packets of one ring in decode order, evicted GOP by GOP.

    Not thread-safe: the owning ring serialises access with its lock. Packets
    leaving the store while captures still reference them stay counted in
    ``held_bytes`` until their last reference is released.
    """

    def __init__(self):
        self._packets: Deque[EncodedPacket] = deque()
        self._keyframes: Deque[EncodedPacket] = deque()
        self.bytes = 0
        self.held_bytes = 0
        self.evicted_until_t: Optional[float] = None

    def __len__(self) -> int:
        return len(self._packets)

    @property
    def oldest(self) -> Optional[EncodedPacket]:
        """The oldest packet still in the store."""
        return self._packets[0] if self._packets else None

    @property
    def newest(self) -> Optional[EncodedPacket]:
        """The newest packet in the store."""
        return self._packets[-1] if self._packets else None

    def append(self, packet: EncodedPacket) -> None:
        """Add a packet at the end of the store.

        Args:
            packet: The packet; it must follow the previous one in decode order.
        """
        packet.in_store = True
        self._packets.append(packet)
        self.bytes += packet.nbytes
        if packet.is_keyframe:
            self._keyframes.append(packet)

    def acquire(self, packets: Iterable[EncodedPacket]) -> None:
        """Take one reference on each packet.

        Args:
            packets: Packets a capture now holds.
        """
        for packet in packets:
            packet.refs += 1

    def release(self, packets: Iterable[EncodedPacket]) -> None:
        """Drop one reference on each packet, freeing evicted unreferenced ones.

        Args:
            packets: Packets a capture no longer holds.
        """
        for packet in packets:
            packet.refs -= 1
            if packet.refs == 0 and not packet.in_store:
                self.held_bytes -= packet.nbytes

    def evict_older_than(self, cutoff_t: float) -> List[EncodedPacket]:
        """Evict leading GOPs that end before ``cutoff_t`` (normal ageing).

        A GOP goes once the next keyframe is at or before ``cutoff_t``, so the
        store always keeps the GOP covering ``cutoff_t``.

        Args:
            cutoff_t: Oldest source time that must stay covered.

        Returns:
            The evicted packets.
        """
        evicted: List[EncodedPacket] = []
        while len(self._keyframes) >= 2 and self._keyframes[1].t <= cutoff_t:
            evicted.extend(self._pop_gop())

        return evicted

    def enforce_cap(self, cap_bytes: int) -> List[EncodedPacket]:
        """Evict leading GOPs until the store fits ``cap_bytes``.

        The newest GOP is always kept.

        Args:
            cap_bytes: The byte cap.

        Returns:
            The evicted packets.
        """
        evicted: List[EncodedPacket] = []
        while self.bytes > cap_bytes and len(self._keyframes) >= 2:
            evicted.extend(self._pop_gop())

        return evicted

    def clear(self) -> List[EncodedPacket]:
        """Remove every packet (referenced ones stay held).

        Returns:
            The removed packets.
        """
        removed: List[EncodedPacket] = []
        while self._packets:
            removed.append(self._pop_head(record_eviction=False))

        return removed

    def byte_rate(self) -> Optional[float]:
        """Measured bytes per source second over the store contents.

        Returns:
            The rate, or ``None`` while the store spans less than a second.
        """
        if len(self._packets) < 2:
            return None

        span = self._packets[-1].t - self._packets[0].t
        if span < MIN_RATE_SPAN_SECONDS:
            return None

        rate = self.bytes / span

        return rate

    def snapshot(
        self, start_t: float, end_t: float, *, max_lead_seconds: float
    ) -> List[EncodedPacket]:
        """Return the GOP-aligned packets that already cover a window.

        The slice starts at the latest keyframe at or before ``start_t``; when
        that keyframe leads the window by more than ``max_lead_seconds`` the
        first keyframe within the lead is used instead. When no keyframe is at
        or before ``start_t`` the first keyframe in the window is used. The
        slice ends before the first packet after ``end_t`` (inclusive end).

        Args:
            start_t: Window start, source seconds.
            end_t: Inclusive window end, source seconds.
            max_lead_seconds: Largest allowed lead (one output second).

        Returns:
            Packets in decode order, starting with a keyframe, or an empty list.
        """
        packets = list(self._packets)
        start_index = None
        for index, packet in enumerate(packets):
            if packet.is_keyframe and packet.t <= start_t + TIME_EPSILON:
                start_index = index
        if (
            start_index is not None
            and start_t - packets[start_index].t > max_lead_seconds + TIME_EPSILON
        ):
            start_index = next(
                (
                    index
                    for index in range(start_index + 1, len(packets))
                    if packets[index].is_keyframe
                    and packets[index].t >= start_t - max_lead_seconds - TIME_EPSILON
                ),
                None,
            )
            if start_index is None:
                return []
        if start_index is None:
            start_index = next(
                (
                    index
                    for index, packet in enumerate(packets)
                    if packet.is_keyframe and packet.t <= end_t + TIME_EPSILON
                ),
                None,
            )
        if start_index is None:
            return []

        selected: List[EncodedPacket] = []
        for packet in packets[start_index:]:
            if packet.t > end_t + TIME_EPSILON:
                break
            selected.append(packet)

        return selected

    def _pop_gop(self) -> List[EncodedPacket]:
        popped = [self._pop_head(record_eviction=True)]
        while self._packets and not self._packets[0].is_keyframe:
            popped.append(self._pop_head(record_eviction=True))

        return popped

    def _pop_head(self, *, record_eviction: bool) -> EncodedPacket:
        packet = self._packets.popleft()
        self.bytes -= packet.nbytes
        packet.in_store = False
        if packet.refs:
            self.held_bytes += packet.nbytes
        if self._keyframes and self._keyframes[0] is packet:
            self._keyframes.popleft()
        if record_eviction and (
            self.evicted_until_t is None or packet.t > self.evicted_until_t
        ):
            self.evicted_until_t = packet.t

        return packet


@dataclass(frozen=True)
class RingStats:
    """Point-in-time statistics of one ring.

    Attributes:
        backend: Encoder backend name.
        bytes: Bytes of packets in the store.
        packets: Packets in the store.
        held_bytes: Bytes of evicted packets still referenced by captures.
        queue_bytes: Bytes of frames waiting for the encoder.
        queue_depth: Frames waiting for the encoder.
        pending_captures: Captures waiting for their post-roll.
        frames_submitted: Frames handed to `submit()`.
        frames_encoded: Frames the encoder accepted.
        decimated: Frames dropped by the output grid.
        queue_drops: Frames dropped because the queue was full or dropped at
            shutdown.
        ring_drops: Packets evicted by the byte cap.
        stale_dropped: Frames the recorder reported as stale.
        encode_errors: Frames the encoder failed on.
        byte_cap_bytes: Current byte cap.
        ring_cost_ms_p50: Median `submit()` time on the caller thread.
        ring_cost_ms_p95: 95th percentile of the same.
        encode_ms_p50: Median encode time per frame on the encoder thread.
        encode_ms_p95: 95th percentile of the same.
        extra: Backend-specific statistics.
    """

    backend: str
    bytes: int
    packets: int
    held_bytes: int
    queue_bytes: int
    queue_depth: int
    pending_captures: int
    frames_submitted: int
    frames_encoded: int
    decimated: int
    queue_drops: int
    ring_drops: int
    stale_dropped: int
    encode_errors: int
    byte_cap_bytes: int
    ring_cost_ms_p50: Optional[float]
    ring_cost_ms_p95: Optional[float]
    encode_ms_p50: Optional[float]
    encode_ms_p95: Optional[float]
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return the statistics as a plain dictionary.

        Returns:
            JSON-compatible dictionary.
        """
        serialized = {
            "backend": self.backend,
            "bytes": self.bytes,
            "packets": self.packets,
            "held_bytes": self.held_bytes,
            "queue_bytes": self.queue_bytes,
            "queue_depth": self.queue_depth,
            "pending_captures": self.pending_captures,
            "frames_submitted": self.frames_submitted,
            "frames_encoded": self.frames_encoded,
            "decimated": self.decimated,
            "queue_drops": self.queue_drops,
            "ring_drops": self.ring_drops,
            "stale_dropped": self.stale_dropped,
            "encode_errors": self.encode_errors,
            "byte_cap_bytes": self.byte_cap_bytes,
            "ring_cost_ms_p50": self.ring_cost_ms_p50,
            "ring_cost_ms_p95": self.ring_cost_ms_p95,
            "encode_ms_p50": self.encode_ms_p50,
            "encode_ms_p95": self.encode_ms_p95,
            "extra": dict(self.extra),
        }

        return serialized


@runtime_checkable
class Ring(Protocol):
    """What the recorder needs from an encoder backend (one per identifier)."""

    def submit(self, frame_timestamp: datetime, frame_number: int, image: Any) -> None:
        """Hand a processed frame to the ring; never blocks on the encoder."""

    def capture(self, start: datetime, end_inclusive: datetime) -> CaptureHandle:
        """Snapshot a GOP-aligned window and register it as pending."""

    def prepare_close(self, deadline: Optional[float]) -> None:
        """Stop ingest, drop unencoded frames and finalise pending captures."""

    def close(self) -> None:
        """Join the ring's threads within the deadline and release everything."""

    def stats(self) -> RingStats:
        """Return the ring's statistics."""


def mux_h264_packets(
    packets: Sequence[EncodedPacket],
    *,
    pts: Sequence[int],
    final_duration_ticks: int,
    width: int,
    height: int,
    path: Path,
    frame_rate: Optional[float] = None,
) -> None:
    """Write H.264 Annex B packets to an MP4 file without re-encoding.

    Args:
        packets: Packets in decode order, starting with a keyframe.
        pts: Output PTS per packet (90 kHz, strictly increasing, first 0).
        final_duration_ticks: Display duration of the last packet.
        width: Frame width.
        height: Frame height.
        path: Output file (written with ``movflags=+faststart``).
        frame_rate: Nominal frame rate stored as stream metadata.

    Raises:
        VideoExtraMissingError: If PyAV is not installed.
        ValueError: If ``packets`` is empty or ``pts`` does not match it.
    """
    if not packets:
        raise ValueError("Cannot mux an empty clip")
    if len(pts) != len(packets):
        raise ValueError("Every packet needs exactly one PTS")

    av = import_av()
    time_base = Fraction(1, PTS_CLOCK_RATE)
    nominal_rate = Fraction(frame_rate or 30.0).limit_denominator(1001)
    with av.open(
        str(path), mode="w", format="mp4", options={"movflags": "+faststart"}
    ) as container:
        stream = container.add_stream("h264", rate=nominal_rate)
        stream.width = width
        stream.height = height
        stream.time_base = time_base
        for index, item in enumerate(packets):
            packet = av.Packet(item.payload)
            packet.pts = pts[index]
            packet.dts = pts[index]
            packet.time_base = time_base
            packet.duration = (
                pts[index + 1] - pts[index]
                if index + 1 < len(packets)
                else final_duration_ticks
            )
            packet.is_keyframe = item.is_keyframe
            packet.stream = stream
            container.mux(packet)


class _CaptureState:
    """Ring-side bookkeeping of one pending capture (ring lock held)."""

    __slots__ = (
        "pending",
        "start_t",
        "end_t",
        "packets",
        "started",
        "partial",
        "stall_deadline",
    )

    def __init__(
        self,
        *,
        start_t: float,
        end_t: float,
        packets: List[EncodedPacket],
        partial: Optional[str],
        stall_deadline: float,
    ):
        self.pending: Optional[PendingCapture] = None
        self.start_t = start_t
        self.end_t = end_t
        self.packets = packets
        self.started = bool(packets)
        self.partial = partial
        self.stall_deadline = stall_deadline


_FRAME = "frame"
_TICK = "tick"
_STOP = "stop"


@dataclass
class _Job:
    kind: str
    t: float = 0.0
    frame_timestamp: Optional[datetime] = None
    frame_number: int = 0
    frame: Optional[PreparedFrame] = None
    reason: Optional[str] = None


@dataclass
class _InFlight:
    t: float
    frame_timestamp: datetime
    frame_number: int
    width: int
    height: int


class BaseEncodedRing(abc.ABC):
    """Backend-neutral ring: queue, grid, packet store, captures and remux.

    Subclasses set `backend` and implement `_prepare_frame`, `_encode_frame`,
    `_flush_encoder` and `_close_encoder`; `_discard_frame` and
    `_backend_stats` are optional.

    Args:
        video_identifier: Source this ring buffers.
        pre_roll_seconds: The recorder's pre-roll; sets time retention and the
            byte cap.
        playback_speed: Playback speed ``s``; one output second (the GOP) is
            ``s`` source seconds.
        clip_fps: Output frame-rate cap; ``None`` keeps source timing.
        max_width: Downscale frames wider than this (backend-specific).
        declared_fps: Fps declared by the source; non-positive means unknown.
        budget: Budget the ring's bytes count against; `default_budget()`
            when ``None``.
        queue_depth: Maximum frames waiting for the encoder.
        byte_cap_bytes: Fixed byte cap overriding the measured cap.
        stall_grace_seconds: Seconds without frames (and after the expected
            post-roll end) before a capture is finalised as
            ``partial="stalled"``; every submitted frame extends the deadline.

    Raises:
        ValueError: If a setting is out of range.
    """

    backend: ClassVar[str] = "base"

    def __init__(
        self,
        video_identifier: str,
        *,
        pre_roll_seconds: float,
        playback_speed: float = 1.0,
        clip_fps: Optional[float] = None,
        max_width: Optional[int] = None,
        declared_fps: Optional[float] = None,
        budget: Optional[Budget] = None,
        queue_depth: Optional[int] = None,
        byte_cap_bytes: Optional[int] = None,
        stall_grace_seconds: float = STALL_GRACE_SECONDS,
    ):
        if pre_roll_seconds < 0:
            raise ValueError(f"pre_roll_seconds must be >= 0, got {pre_roll_seconds}")
        if queue_depth is not None and queue_depth <= 0:
            raise ValueError(f"queue_depth must be > 0, got {queue_depth}")

        self.video_identifier = video_identifier
        self.pre_roll_seconds = pre_roll_seconds
        self.playback_speed = playback_speed
        self.clip_fps = clip_fps
        self.max_width = max_width
        self.declared_fps = declared_fps if declared_fps and declared_fps > 0 else None
        self._grid = GridSelector(playback_speed=playback_speed, clip_fps=clip_fps)
        self._queue_depth = queue_depth or DEFAULT_QUEUE_DEPTH
        self._byte_cap_override = byte_cap_bytes
        self._stall_grace_seconds = stall_grace_seconds
        self._budget = budget or default_budget()

        self._lock = threading.RLock()
        self._close_lock = threading.Lock()
        self._queue: "Queue[_Job]" = Queue()
        self._finalize_queue: "Queue[Optional[_CaptureState]]" = Queue()
        self._store = PacketStore()
        self._pending: List[_CaptureState] = []
        self._outstanding: Set[_CaptureState] = set()
        self._encoder_thread: Optional[threading.Thread] = None
        self._finalizer_thread: Optional[threading.Thread] = None

        self._closing = False
        self._prepared = False
        self._closed = False
        self._close_deadline: Optional[float] = None
        self._stop_reason: Optional[str] = None

        self._queue_items = 0
        self._queue_bytes = 0
        self._frames_submitted = 0
        self._frames_encoded = 0
        self._decimated = 0
        self._queue_drops = 0
        self._ring_drops = 0
        self._stale_dropped = 0
        self._encode_errors = 0
        self._error_counts: Dict[str, int] = {}
        self._decimated_times: Deque[float] = deque(maxlen=LOSS_EVENT_HISTORY)
        self._queue_drop_times: Deque[float] = deque(maxlen=LOSS_EVENT_HISTORY)
        self._ring_drop_times: Deque[float] = deque(maxlen=LOSS_EVENT_HISTORY)
        self._stale_times: Deque[float] = deque(maxlen=LOSS_EVENT_HISTORY)
        self._ring_cost_ms: Deque[float] = deque(maxlen=COST_SAMPLES)
        self._encode_ms: Deque[float] = deque(maxlen=COST_SAMPLES)
        self._newest_submitted_t: Optional[float] = None
        self._processed_until_t: Optional[float] = None

        # Encoder-thread state (touched only by the encoder thread).
        self._t0: Optional[float] = None
        self._last_pts = -1
        self._last_keyframe_slot: Optional[int] = None
        self._frame_size: Optional[Tuple[int, int]] = None
        self._in_flight: Deque[_InFlight] = deque()
        self._consecutive_encode_errors = 0

        self._budget.register_source(self)

    # ---- backend hooks ----------------------------------------------------------

    @abc.abstractmethod
    def _prepare_frame(self, image: Any) -> Optional[PreparedFrame]:
        """Convert or copy a frame on the caller's thread.

        Returns ``None`` when the frame cannot be queued (counted as a queue
        drop). Must not keep a reference to the caller's image.
        """

    @abc.abstractmethod
    def _encode_frame(
        self, frame: PreparedFrame, *, pts: int, force_keyframe: bool
    ) -> List[EncodedOutput]:
        """Encode one frame on the encoder thread; outputs map FIFO to inputs."""

    @abc.abstractmethod
    def _flush_encoder(self) -> List[EncodedOutput]:
        """Drain delayed packets (stop, retire, resolution change)."""

    @abc.abstractmethod
    def _close_encoder(self) -> None:
        """Release the encoder on the encoder thread; the next encode reopens."""

    def _discard_frame(self, frame: PreparedFrame) -> None:
        """Release a prepared frame that will never be encoded."""
        return None

    def _backend_stats(self) -> Dict[str, Any]:
        """Backend-specific statistics merged into `RingStats.extra`."""
        return {}

    # ---- caller (pipeline) thread ----------------------------------------------

    def submit(self, frame_timestamp: datetime, frame_number: int, image: Any) -> None:
        """Hand a processed frame to the ring.

        Never blocks on the encoder and never raises: failures are logged and
        counted. Frames are ignored once the ring is closing.

        Args:
            frame_timestamp: Source timestamp of the frame.
            frame_number: Source frame number.
            image: The workflow image (`WorkflowImageData`).
        """
        started = time.perf_counter()
        try:
            self._submit(
                frame_timestamp=frame_timestamp,
                frame_number=frame_number,
                image=image,
            )
        except Exception as error:
            self._log_error("submit", error)
        finally:
            with self._lock:
                self._ring_cost_ms.append((time.perf_counter() - started) * 1000)

    def record_stale(self, frame_timestamp: datetime) -> None:
        """Count a frame the recorder skipped for excessive processing lag.

        Args:
            frame_timestamp: Source timestamp of the skipped frame.
        """
        t = to_seconds(frame_timestamp)
        with self._lock:
            self._stale_dropped += 1
            self._stale_times.append(t)

    def capture(self, start: datetime, end_inclusive: datetime) -> PendingCapture:
        """Snapshot ``[start, end_inclusive]`` and register it as pending.

        Synchronous and cheap: takes references on the GOP-aligned packets that
        already exist; later packets extend the capture until a frame after
        ``end_inclusive`` arrives, then the finalizer thread remuxes it.

        Args:
            start: Window start.
            end_inclusive: Window end (inclusive).

        Returns:
            The pending capture.

        Raises:
            ClipUnavailableError: If the ring is closing.
            ValueError: If the window ends before it starts.
        """
        start_t = to_seconds(start)
        end_t = to_seconds(end_inclusive)
        if end_t < start_t:
            raise ValueError("The capture window ends before it starts")

        with self._lock:
            if self._closing:
                raise ClipUnavailableError(UNAVAILABLE_CLOSING)

            snapshot = self._store.snapshot(
                start_t, end_t, max_lead_seconds=self.playback_speed
            )
            self._store.acquire(snapshot)
            evicted_until_t = self._store.evicted_until_t
            partial_reason = (
                PARTIAL_RING_EVICTED
                if evicted_until_t is not None
                and evicted_until_t >= start_t - TIME_EPSILON
                else None
            )
            reference_t = (
                self._newest_submitted_t
                if self._newest_submitted_t is not None
                else start_t
            )
            remaining_seconds = max(0.0, end_t - reference_t)
            state = _CaptureState(
                start_t=start_t,
                end_t=end_t,
                packets=snapshot,
                partial=partial_reason,
                stall_deadline=time.monotonic()
                + remaining_seconds
                + self._stall_grace_seconds,
            )
            oldest = self._store.oldest
            pending = PendingCapture(
                capture_id=uuid.uuid4().hex,
                video_identifier=self.video_identifier,
                backend=self.backend,
                requested_window=(start, end_inclusive),
                available_from=oldest.frame_timestamp if oldest is not None else None,
                estimated_size_bytes=self._estimate_size(state),
                on_cancel=partial(self._cancel_capture, state),
            )
            state.pending = pending
            self._pending.append(state)
            if (
                self._processed_until_t is not None
                and self._processed_until_t > end_t + TIME_EPSILON
            ):
                self._complete(state, partial=None)
        self._ensure_encoder_thread()

        return pending

    def retire(self, reason: str) -> None:
        """End this ring after a source reset (resolution change, frame reset).

        Non-blocking: frames already queued are still encoded, pending captures
        are then finalised with ``partial=reason`` and the ring's threads exit
        on their own. New frames and captures are refused. Call `close()` later
        to release the remaining packets.

        Args:
            reason: Partial reason for the pending captures.
        """
        with self._lock:
            if self._closing:
                return None
            self._closing = True
            self._stop_reason = reason
        self._stop_encoder(reason)

    def prepare_close(self, deadline: Optional[float] = None) -> None:
        """Start closing: drop unencoded frames, finalise pending as shutdown.

        Idempotent and thread-safe. The dropped frames count as queue drops.

        Args:
            deadline: Seconds from now `close()` may spend joining the ring's
                threads; ``None`` means `DEFAULT_CLOSE_DEADLINE_SECONDS`.
        """
        with self._lock:
            if self._prepared:
                return None
            self._prepared = True
            self._closing = True
            self._close_deadline = time.monotonic() + (
                DEFAULT_CLOSE_DEADLINE_SECONDS
                if deadline is None
                else max(0.0, deadline)
            )
            reason = self._stop_reason or PARTIAL_SHUTDOWN
            dropped = self._drain_queue_locked()
        for frame in dropped:
            self._discard_safely(frame)
        self._stop_encoder(reason)

    def close(self) -> None:
        """Join the ring's threads within the deadline and release its packets.

        Calls `prepare_close(None)` first when needed. Idempotent and safe under
        concurrent calls: a second caller waits for the first. Futures that
        could not be finalised in time fail with `ClipShutdownError`.
        """
        self.prepare_close(None)
        with self._close_lock:
            if self._closed:
                return None

            self._join(self._encoder_thread, name="encoder")
            with self._lock:
                for state in list(self._pending):
                    self._complete(state, partial=PARTIAL_SHUTDOWN)
                if self._outstanding:
                    self._ensure_finalizer_thread()
            self._finalize_queue.put(None)
            self._join(self._finalizer_thread, name="finalizer")
            with self._lock:
                for state in list(self._outstanding):
                    _set_exception_safely(
                        state.pending.future,
                        ClipShutdownError(
                            "The video clip could not be finalised before the "
                            "close deadline"
                        ),
                    )
                self._store.clear()
                self._closed = True
            self._budget.unregister_source(self)

    def stats(self) -> RingStats:
        """Return the ring's statistics.

        Returns:
            A `RingStats` snapshot.
        """
        with self._lock:
            ring_costs = list(self._ring_cost_ms)
            encode_costs = list(self._encode_ms)
            stats = RingStats(
                backend=self.backend,
                bytes=self._store.bytes,
                packets=len(self._store),
                held_bytes=self._store.held_bytes,
                queue_bytes=self._queue_bytes,
                queue_depth=self._queue_items,
                pending_captures=len(self._pending),
                frames_submitted=self._frames_submitted,
                frames_encoded=self._frames_encoded,
                decimated=self._decimated,
                queue_drops=self._queue_drops,
                ring_drops=self._ring_drops,
                stale_dropped=self._stale_dropped,
                encode_errors=self._encode_errors,
                byte_cap_bytes=self._byte_cap(),
                ring_cost_ms_p50=_percentile(ring_costs, 0.5),
                ring_cost_ms_p95=_percentile(ring_costs, 0.95),
                encode_ms_p50=_percentile(encode_costs, 0.5),
                encode_ms_p95=_percentile(encode_costs, 0.95),
                extra=self._backend_stats(),
            )

        return stats

    def budget_bytes(self) -> int:
        """Bytes this ring holds for the budget (store, held packets, queue).

        Returns:
            The byte count (read without locking).
        """
        held = self._store.bytes + self._store.held_bytes + self._queue_bytes

        return held

    def _submit(
        self, *, frame_timestamp: datetime, frame_number: int, image: Any
    ) -> None:
        t = to_seconds(frame_timestamp)
        with self._lock:
            if self._closing:
                return None
            self._frames_submitted += 1
            if self._newest_submitted_t is None or t > self._newest_submitted_t:
                self._newest_submitted_t = t
            self._extend_stall_deadlines_locked()
            if not self._grid.accept(t):
                self._decimated += 1
                self._decimated_times.append(t)
                if self._pending:
                    self._queue.put(_Job(kind=_TICK, t=t))
                return None
            if self._queue_items >= self._queue_depth:
                self._record_queue_drop(t)
                return None
            self._queue_items += 1
        self._ensure_encoder_thread()

        try:
            frame = self._prepare_frame(image)
        except Exception:
            with self._lock:
                self._queue_items -= 1
                self._record_queue_drop(t)
            raise

        with self._lock:
            if frame is None or self._closing:
                self._queue_items -= 1
                self._record_queue_drop(t)
                discard = frame
            else:
                self._queue_bytes += frame.nbytes
                self._queue.put(
                    _Job(
                        kind=_FRAME,
                        t=t,
                        frame_timestamp=frame_timestamp,
                        frame_number=frame_number,
                        frame=frame,
                    )
                )
                discard = None
        if discard is not None:
            self._discard_safely(discard)

    # ---- encoder thread ----------------------------------------------------------

    def _encoder_loop(self) -> None:
        while True:
            try:
                job: Optional[_Job] = self._queue.get(timeout=WORKER_POLL_SECONDS)
            except Empty:
                job = None
            if job is not None and job.kind == _STOP:
                self._handle_stop(job.reason or PARTIAL_SHUTDOWN)
                return None
            try:
                if job is not None and job.kind == _FRAME:
                    self._encode_job(job)
                elif job is not None and job.kind == _TICK:
                    with self._lock:
                        self._observe_processed(job.t)
                with self._lock:
                    self._expire_stalled()
            except Exception as error:
                self._log_error("encoder", error)

    def _encode_job(self, job: _Job) -> None:
        frame = job.frame
        with self._lock:
            self._queue_items -= 1
            self._queue_bytes -= frame.nbytes

        size = (frame.width, frame.height)
        if self._frame_size is not None and size != self._frame_size:
            self._restart_stream(PARTIAL_RESOLUTION_CHANGE)
        self._frame_size = size
        if self._t0 is None:
            self._t0 = job.t
        elapsed_output_seconds = (job.t - self._t0) / self.playback_speed
        pts = round(elapsed_output_seconds * PTS_CLOCK_RATE)
        if pts <= self._last_pts:
            pts = self._last_pts + 1
        self._last_pts = pts
        keyframe_slot = math.floor(elapsed_output_seconds + SLOT_EPSILON)
        force_keyframe = keyframe_slot != self._last_keyframe_slot
        if force_keyframe:
            self._last_keyframe_slot = keyframe_slot

        self._in_flight.append(
            _InFlight(
                t=job.t,
                frame_timestamp=job.frame_timestamp,
                frame_number=job.frame_number,
                width=frame.width,
                height=frame.height,
            )
        )
        started = time.perf_counter()
        try:
            outputs = self._encode_frame(frame, pts=pts, force_keyframe=force_keyframe)
        except Exception as error:
            self._in_flight.pop()
            self._on_encode_error(frame, error)
            return None

        encode_ms = (time.perf_counter() - started) * 1000
        self._consecutive_encode_errors = 0
        with self._lock:
            self._frames_encoded += 1
            self._encode_ms.append(encode_ms)
        self._append_outputs(outputs, processed_t=job.t)

    def _on_encode_error(self, frame: PreparedFrame, error: Exception) -> None:
        with self._lock:
            self._encode_errors += 1
        self._log_error("encode", error)
        self._discard_safely(frame)
        self._consecutive_encode_errors += 1
        if self._consecutive_encode_errors >= MAX_CONSECUTIVE_ENCODE_ERRORS:
            self._consecutive_encode_errors = 0
            self._close_encoder_safely()
            self._in_flight.clear()
            self._last_keyframe_slot = None

    def _append_outputs(
        self, outputs: List[EncodedOutput], *, processed_t: Optional[float]
    ) -> None:
        with self._lock:
            for output in outputs:
                if not self._in_flight:
                    self._log_error(
                        "encode",
                        RuntimeError("The encoder returned more packets than frames"),
                    )
                    break
                meta = self._in_flight.popleft()
                self._add_packet(
                    EncodedPacket(
                        frame_timestamp=meta.frame_timestamp,
                        t=meta.t,
                        frame_number=meta.frame_number,
                        payload=output.payload,
                        is_keyframe=output.is_keyframe,
                        width=meta.width,
                        height=meta.height,
                    )
                )
            if processed_t is not None:
                self._observe_processed(processed_t)

    def _add_packet(self, packet: EncodedPacket) -> None:
        self._store.append(packet)
        for state in list(self._pending):
            if packet.t > state.end_t + TIME_EPSILON:
                self._complete(state, partial=None)
            elif state.started or packet.is_keyframe:
                state.started = True
                state.packets.append(packet)
                self._store.acquire([packet])
        self._store.evict_older_than(packet.t - self._retention_seconds())
        for evicted in self._store.enforce_cap(self._byte_cap()):
            self._ring_drops += 1
            self._ring_drop_times.append(evicted.t)

    def _observe_processed(self, t: float) -> None:
        if self._in_flight:
            return None
        if self._processed_until_t is None or t > self._processed_until_t:
            self._processed_until_t = t
        for state in list(self._pending):
            if t > state.end_t + TIME_EPSILON:
                self._complete(state, partial=None)

    def _extend_stall_deadlines_locked(self) -> None:
        # A stall is `stall_grace_seconds` without frames, not a fixed deadline
        # from capture time: a source processed slower than real time (a video
        # file on CPU inference) keeps its pending captures alive while frames
        # still arrive.
        if not self._pending:
            return None

        extended_deadline = time.monotonic() + self._stall_grace_seconds
        for state in self._pending:
            if extended_deadline > state.stall_deadline:
                state.stall_deadline = extended_deadline

    def _expire_stalled(self) -> None:
        now = time.monotonic()
        for state in list(self._pending):
            if now >= state.stall_deadline:
                self._complete(state, partial=PARTIAL_STALLED)

    def _restart_stream(self, reason: str) -> None:
        self._append_outputs(self._flush_safely(), processed_t=None)
        self._in_flight.clear()
        self._close_encoder_safely()
        with self._lock:
            for state in list(self._pending):
                self._complete(state, partial=reason)
            self._store.clear()
        self._t0 = None
        self._last_pts = -1
        self._last_keyframe_slot = None

    def _handle_stop(self, reason: str) -> None:
        try:
            self._append_outputs(self._flush_safely(), processed_t=None)
            self._in_flight.clear()
            with self._lock:
                for state in list(self._pending):
                    self._complete(state, partial=reason)
                dropped = self._drain_queue_locked()
            for frame in dropped:
                self._discard_safely(frame)
        finally:
            self._close_encoder_safely()
            self._finalize_queue.put(None)

    # ---- finalizer thread ----------------------------------------------------------

    def _finalizer_loop(self) -> None:
        while True:
            state = self._finalize_queue.get()
            if state is None:
                if self._finalize_queue.empty():
                    return None
                continue
            try:
                self._finalize(state)
            except Exception as error:
                self._log_error("finalize", error)

    def _finalize(self, state: _CaptureState) -> None:
        future = state.pending.future
        try:
            if future.done() or not future.set_running_or_notify_cancel():
                self._release_packets(state.packets)
                return None
            try:
                artifact = self._build_artifact(state)
            except Exception as error:
                self._release_packets(state.packets)
                encode_error = (
                    error
                    if isinstance(error, ClipEncodeError)
                    else ClipEncodeError(f"Muxing the video clip failed: {error!r}")
                )
                _set_exception_safely(future, encode_error)
                return None
            if not _set_result_safely(future, artifact):
                artifact.release()
        finally:
            with self._lock:
                self._outstanding.discard(state)

    def _build_artifact(self, state: _CaptureState) -> ClipArtifact:
        started = time.perf_counter()
        packets = state.packets
        if not packets:
            raise ClipEncodeError(
                "No frames were buffered in the requested window",
                report=self._build_report(
                    state, packets=packets, plan=None, finalize_ms=None
                ),
            )

        plan = plan_timeline(
            [packet.t for packet in packets],
            playback_speed=self.playback_speed,
            clip_fps=self.clip_fps,
            declared_fps=self.declared_fps,
        )
        path = allocate_clip_path()
        try:
            mux_h264_packets(
                packets,
                pts=plan.pts,
                final_duration_ticks=plan.final_duration_ticks,
                width=packets[0].width,
                height=packets[0].height,
                path=path,
                frame_rate=self._nominal_output_fps(),
            )
        except Exception:
            path.unlink(missing_ok=True)
            raise
        finalize_ms = (time.perf_counter() - started) * 1000
        report = self._build_report(
            state, packets=packets, plan=plan, finalize_ms=finalize_ms
        )
        artifact = ClipArtifact.from_muxed_file(
            path,
            duration_seconds=plan.duration_ticks / PTS_CLOCK_RATE,
            report=report,
        )
        # The MP4 holds its own copy of the clip, so the ring's packets are
        # released now rather than when the upload finishes.
        self._release_packets(packets)

        return artifact

    def _build_report(
        self,
        state: _CaptureState,
        *,
        packets: List[EncodedPacket],
        plan: Optional[TimelinePlan],
        finalize_ms: Optional[float],
    ) -> ClipReport:
        start, end = state.pending.requested_window
        with self._lock:
            loss = LossCounts(
                decimated=_count_between(
                    self._decimated_times, state.start_t, state.end_t
                ),
                ring_drops=_count_between(
                    self._ring_drop_times, state.start_t, state.end_t
                ),
                queue_drops=_count_between(
                    self._queue_drop_times, state.start_t, state.end_t
                ),
                stale_dropped=_count_between(
                    self._stale_times, state.start_t, state.end_t
                ),
                inference_gaps=plan.inference_gaps if plan else 0,
                pts_clamped=plan.pts_clamped if plan else 0,
            )
            ring_costs = list(self._ring_cost_ms)
            encode_costs = list(self._encode_ms)
        actual_start = packets[0].frame_timestamp if packets else None
        actual_end = (
            actual_start + timedelta(seconds=plan.actual_end_offset_seconds)
            if actual_start is not None and plan is not None
            else None
        )
        span_seconds = plan.actual_end_offset_seconds if plan else 0.0

        report = ClipReport(
            requested_window=(start, end),
            actual_start=actual_start,
            actual_end=actual_end,
            lead_seconds=max(0.0, state.start_t - packets[0].t) if packets else 0.0,
            frames=len(packets),
            frame_density_fps=len(packets) / span_seconds if span_seconds > 0 else 0.0,
            source_fps_declared=self.declared_fps,
            playback_speed=self.playback_speed,
            clip_fps=self.clip_fps,
            width=packets[0].width if packets else None,
            height=packets[0].height if packets else None,
            backend=self.backend,
            partial=state.partial,
            loss=loss,
            wall_clock_jump_flagged=plan.wall_clock_jump_flagged if plan else False,
            encode_ms_p50=_percentile(encode_costs, 0.5),
            encode_ms_p95=_percentile(encode_costs, 0.95),
            ring_cost_ms_p50=_percentile(ring_costs, 0.5),
            ring_cost_ms_p95=_percentile(ring_costs, 0.95),
            finalize_ms=finalize_ms,
        )

        return report

    # ---- shared helpers (ring lock held where noted) -----------------------------

    def _complete(self, state: _CaptureState, *, partial: Optional[str]) -> None:
        # Ring lock held.
        if state in self._pending:
            self._pending.remove(state)
        if partial is not None and state.partial is None:
            state.partial = partial
        if self._closed:
            self._release_packets_locked(state.packets)
            _set_exception_safely(
                state.pending.future,
                ClipShutdownError("The ring closed before the clip was finalised"),
            )
            return None
        self._outstanding.add(state)
        self._ensure_finalizer_thread()
        self._finalize_queue.put(state)

    def _cancel_capture(self, state: _CaptureState) -> None:
        with self._lock:
            if state not in self._pending:
                return None
            self._pending.remove(state)
            self._release_packets_locked(state.packets)

    def _release_packets(self, packets: List[EncodedPacket]) -> None:
        with self._lock:
            self._release_packets_locked(packets)

    def _release_packets_locked(self, packets: List[EncodedPacket]) -> None:
        self._store.release(packets)
        packets.clear()

    def _record_queue_drop(self, t: float) -> None:
        # Ring lock held.
        self._queue_drops += 1
        self._queue_drop_times.append(t)

    def _drain_queue_locked(self) -> List[PreparedFrame]:
        dropped: List[PreparedFrame] = []
        while True:
            try:
                job = self._queue.get_nowait()
            except Empty:
                break
            if job.kind != _FRAME:
                continue
            self._queue_items -= 1
            self._queue_bytes -= job.frame.nbytes
            self._record_queue_drop(job.t)
            dropped.append(job.frame)

        return dropped

    def _stop_encoder(self, reason: str) -> None:
        with self._lock:
            encoder_alive = (
                self._encoder_thread is not None and self._encoder_thread.is_alive()
            )
            if encoder_alive:
                self._queue.put(_Job(kind=_STOP, reason=reason))
                return None
            for state in list(self._pending):
                self._complete(state, partial=reason)
            if self._finalizer_thread is not None:
                self._finalize_queue.put(None)

    def _ensure_encoder_thread(self) -> None:
        with self._lock:
            if self._closing or (
                self._encoder_thread is not None and self._encoder_thread.is_alive()
            ):
                return None
            self._encoder_thread = threading.Thread(
                target=self._encoder_loop,
                name=f"video-clip-{self.backend}-encoder-{self.video_identifier}",
                daemon=True,
            )
            self._encoder_thread.start()

    def _ensure_finalizer_thread(self) -> None:
        # Ring lock held.
        if self._finalizer_thread is not None and self._finalizer_thread.is_alive():
            return None
        self._finalizer_thread = threading.Thread(
            target=self._finalizer_loop,
            name=f"video-clip-{self.backend}-finalizer-{self.video_identifier}",
            daemon=True,
        )
        self._finalizer_thread.start()

    def _join(self, thread: Optional[threading.Thread], *, name: str) -> None:
        if thread is None or thread is threading.current_thread():
            return None
        remaining = max(
            0.0, (self._close_deadline or time.monotonic()) - time.monotonic()
        )
        thread.join(timeout=remaining)
        if thread.is_alive():
            logger.warning(
                "Video clip %s thread for %r did not stop before the close deadline.",
                name,
                self.video_identifier,
            )

    def _retention_seconds(self) -> float:
        retention = (
            self.pre_roll_seconds + self.playback_speed + RETENTION_MARGIN_SECONDS
        )

        return retention

    def _byte_cap(self) -> int:
        # Ring lock held.
        if self._byte_cap_override is not None:
            return self._byte_cap_override

        rate = self._store.byte_rate()
        if rate is None:
            return MIN_RING_BYTE_CAP

        measured_cap = int(
            (self.pre_roll_seconds + self.playback_speed) * rate * RING_BYTE_CAP_FACTOR
        )
        cap = max(MIN_RING_BYTE_CAP, measured_cap)

        return cap

    def _estimate_size(self, state: _CaptureState) -> int:
        # Ring lock held.
        rate = self._store.byte_rate() or 0.0
        lead = max(0.0, state.start_t - state.packets[0].t) if state.packets else 0.0
        estimate = int(rate * (state.end_t - state.start_t + lead))

        return estimate

    def _nominal_output_fps(self) -> float:
        if self.clip_fps:
            return self.clip_fps
        if self.declared_fps:
            return self.declared_fps / self.playback_speed

        return 30.0

    def _flush_safely(self) -> List[EncodedOutput]:
        try:
            outputs = self._flush_encoder()
        except Exception as error:
            self._log_error("flush", error)
            return []

        return outputs

    def _close_encoder_safely(self) -> None:
        try:
            self._close_encoder()
        except Exception as error:
            self._log_error("close_encoder", error)

    def _discard_safely(self, frame: PreparedFrame) -> None:
        try:
            self._discard_frame(frame)
        except Exception as error:
            self._log_error("discard", error)

    def _log_error(self, operation: str, error: Exception) -> None:
        with self._lock:
            count = self._error_counts.get(operation, 0) + 1
            self._error_counts[operation] = count
        if count <= LOGGED_ERRORS_BEFORE_THROTTLE or count % LOG_EVERY_NTH_ERROR == 0:
            logger.warning(
                "Video clip ring (%s, %r) %s error #%s: %r",
                self.backend,
                self.video_identifier,
                operation,
                count,
                error,
            )


def _percentile(samples: Sequence[float], quantile: float) -> Optional[float]:
    if not samples:
        return None

    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))

    return ordered[index]


def _count_between(times: Iterable[float], start_t: float, end_t: float) -> int:
    count = sum(1 for t in times if start_t - TIME_EPSILON <= t <= end_t + TIME_EPSILON)

    return count


def _set_result_safely(future: Future, result: Any) -> bool:
    try:
        future.set_result(result)
    except InvalidStateError:
        return False

    return True


def _set_exception_safely(future: Future, error: BaseException) -> bool:
    try:
        future.set_exception(error)
    except InvalidStateError:
        return False

    return True
