"""`ClipAnchor` and `PendingCapture`: how a sink asks a recorder for a clip.

The recorder emits one immutable `ClipAnchor` per frame. The sink calls
`anchor.capture(pre, post)`, which synchronously snapshots the ring packets
that already cover ``[anchor - pre, anchor + post]`` (end inclusive) and
returns a `PendingCapture`. Its future resolves to a `ClipArtifact` on the
ring's own finalizer thread once the post-roll has arrived, or earlier as a
partial clip (stall, resolution change, shutdown).

Future outcomes:

* result: a `ClipArtifact` (``report.partial`` says whether it is partial);
* exception `ClipEncodeError`: no clip could be produced (no frames, mux error);
* exception `ClipShutdownError`: the ring was closed before the clip finished;
* cancelled: the capture was released before it finished.
"""

import threading
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

if TYPE_CHECKING:
    from roboflow_workflows.core_steps.common.video_clips.artifact import ClipArtifact
    from roboflow_workflows.core_steps.common.video_clips.report import ClipReport

AVAILABILITY_OK = "ok"

PARTIAL_STALLED = "stalled"
PARTIAL_RESOLUTION_CHANGE = "resolution_change"
PARTIAL_SHUTDOWN = "shutdown"
PARTIAL_RING_EVICTED = "ring_evicted"
PARTIAL_REASONS = (
    PARTIAL_STALLED,
    PARTIAL_RESOLUTION_CHANGE,
    PARTIAL_SHUTDOWN,
    PARTIAL_RING_EVICTED,
)

STALL_GRACE_SECONDS = 10.0


class ClipUnavailableError(RuntimeError):
    """A clip cannot be captured (anchor unavailable or recorder closing).

    Args:
        reason: Availability string such as ``"unavailable:closing"``.
    """

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class ClipEncodeError(RuntimeError):
    """No clip could be produced for a capture.

    Args:
        message: What went wrong.
        report: The (partial) report, when one could be built.
    """

    def __init__(self, message: str, *, report: Optional["ClipReport"] = None):
        super().__init__(message)
        self.report = report


class ClipShutdownError(RuntimeError):
    """The ring closed before the capture could be finalised in time."""


class PendingCapture:
    """A capture registered with a ring, resolved later by the ring's worker.

    Created by `Ring.capture()`; callers never construct it.

    Args:
        capture_id: Unique id of the capture.
        video_identifier: Source the capture belongs to.
        backend: Encoder backend of the ring.
        requested_window: Requested ``(start, end)``; the end is inclusive.
        available_from: Earliest buffered timestamp at capture time, if any.
        estimated_size_bytes: Expected clip size from the ring's measured
            byte rate (0 when unknown).
        on_cancel: Called once when the capture is released before its future
            resolved (the ring drops the registration and its packet refs).
    """

    def __init__(
        self,
        *,
        capture_id: str,
        video_identifier: str,
        backend: str,
        requested_window: Tuple[datetime, datetime],
        available_from: Optional[datetime],
        estimated_size_bytes: int,
        on_cancel: Optional[Callable[[], None]] = None,
    ):
        self.capture_id = capture_id
        self.video_identifier = video_identifier
        self.backend = backend
        self.requested_window = requested_window
        self.available_from = available_from
        self.estimated_size_bytes = estimated_size_bytes
        self.captured_at = datetime.now(timezone.utc)
        self.future: "Future[ClipArtifact]" = Future()
        self._on_cancel = on_cancel
        self._lock = threading.Lock()
        self._released = False

    @property
    def released(self) -> bool:
        """True once `release()` was called."""
        return self._released

    def requested_window_dict(self) -> Dict[str, str]:
        """Return the requested window with ISO-8601 timestamps.

        Returns:
            ``{"start": ..., "end": ...}``.
        """
        start, end = self.requested_window
        window = {"start": start.isoformat(), "end": end.isoformat()}

        return window

    def release(self) -> None:
        """Give the capture up and free everything it holds.

        Before resolution the capture is cancelled and its packet refs are
        dropped; after resolution the artifact is released. Idempotent and
        thread-safe.
        """
        with self._lock:
            if self._released:
                return None
            self._released = True
            on_cancel, self._on_cancel = self._on_cancel, None

        if self.future.cancel():
            if on_cancel is not None:
                on_cancel()
            return None

        self.future.add_done_callback(_release_artifact_of)


@dataclass(frozen=True)
class ClipAnchor:
    """Reference to a recorder's buffered video for one frame.

    Attributes:
        recorder_step: Name of the recorder step that emitted the anchor.
        video_identifier: Source of the frame.
        frame_timestamp: Timestamp of the frame (naive UTC or aware).
        frame_number: Frame number of the frame.
        availability: ``"ok"`` or ``"unavailable:<reason>"``.
        label: Label of the recorder (the video's caption).
        pre_roll_seconds: The recorder's pre-roll, for the sink's capture.
        post_roll_seconds: The recorder's post-roll, for the sink's capture.
        _capture: Bound capture function; ``None`` when unavailable.
    """

    recorder_step: str
    video_identifier: str
    frame_timestamp: Optional[datetime]
    frame_number: int
    availability: str
    label: str
    pre_roll_seconds: float
    post_roll_seconds: float
    _capture: Optional[Callable[[float, float], PendingCapture]]

    @property
    def available(self) -> bool:
        """True when a clip can be captured from this anchor."""
        return self.availability == AVAILABILITY_OK and self._capture is not None

    def capture(
        self, pre_roll_seconds: float, post_roll_seconds: float
    ) -> PendingCapture:
        """Capture ``[anchor - pre, anchor + post]`` (end inclusive).

        Synchronous and cheap: snapshots packet references and registers the
        capture with the ring.

        Args:
            pre_roll_seconds: Seconds before the anchor frame.
            post_roll_seconds: Seconds after the anchor frame.

        Returns:
            The pending capture.

        Raises:
            ClipUnavailableError: If the anchor is unavailable or the recorder
                is closing.
        """
        if not self.available:
            raise ClipUnavailableError(
                self.availability
                if self.availability != AVAILABILITY_OK
                else "unavailable:no_encoder"
            )

        pending_capture = self._capture(pre_roll_seconds, post_roll_seconds)

        return pending_capture

    def to_metadata(self) -> Dict[str, Any]:
        """Return the serialisable metadata of the anchor (no frames, no packets).

        Returns:
            The `video_clip_anchor` kind's serialised form.
        """
        metadata = {
            "kind": "video_clip_anchor",
            "recorder": self.recorder_step,
            "video_identifier": self.video_identifier,
            "frame_timestamp": (
                self.frame_timestamp.isoformat()
                if self.frame_timestamp is not None
                else None
            ),
            "frame_number": self.frame_number,
            "availability": self.availability,
        }

        return metadata


CaptureHandle = PendingCapture


def bind_capture(
    ring: Any, frame_timestamp: datetime
) -> Callable[[float, float], PendingCapture]:
    """Bind a ring and an anchor timestamp into a `ClipAnchor._capture` function.

    Args:
        ring: Any `Ring` implementation.
        frame_timestamp: The anchor frame's timestamp.

    Returns:
        ``f(pre, post)`` returning ``ring.capture(ts - pre, ts + post)``.
    """

    def _capture(pre_roll_seconds: float, post_roll_seconds: float) -> PendingCapture:
        start = frame_timestamp - timedelta(seconds=pre_roll_seconds)
        end_inclusive = frame_timestamp + timedelta(seconds=post_roll_seconds)

        return ring.capture(start, end_inclusive)

    return _capture


def _release_artifact_of(future: "Future[ClipArtifact]") -> None:
    if future.cancelled() or future.exception() is not None:
        return None

    future.result().release()
