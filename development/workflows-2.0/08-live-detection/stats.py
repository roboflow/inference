"""Rolling speed statistics of the live runner; every container is bounded.

Timing boundaries (``time.perf_counter``, seconds):

    captured_at    right after VideoCapture.read() returned the frame
                   (sensor and driver latency are not included)
    completed_at   the graph worker holds the run result
    presented_at   cv2.waitKey returned after the frame's cv2.imshow

    output FPS          presented frames per second (window only)
    processed FPS       completed runs per second
    capture->present    presented_at - captured_at
    capture->result     completed_at - captured_at
    pre/model/post/boxes/labels   per-call milliseconds the blocks return

Nothing is derived from model time: output FPS counts presented frames only.
Stage times are wall times observed inside each call; in pipeline mode they
include contention between overlapping stages. Drops are counted where they
happen:

    capture_overwritten   a newer camera frame replaced one the graph had not
                          taken yet (serial run busy, or max_in_flight pending)
    result_replaced       a newer result replaced one the display never showed

The pipeline itself never drops a frame: a submit that stalls is an error.
"""

import math
import statistics
import threading
from collections import deque
from typing import Deque, Dict, List, Mapping, Optional

STAGES = ("pre_ms", "model_ms", "post_ms", "boxes_ms", "labels_ms")
"""Workflow timing outputs, in milliseconds, in graph order."""

DROPS = ("capture_overwritten", "result_replaced")

RATE_WINDOW_SECONDS = 2.0
SAMPLE_WINDOW = 120
MAX_RATE_EVENTS = 1024


class RollingRate:
    """Events per second over the last ``window_seconds``.

    Args:
        window_seconds: Length of the window.
        max_events: Bound of remembered timestamps.
    """

    def __init__(
        self,
        *,
        window_seconds: float = RATE_WINDOW_SECONDS,
        max_events: int = MAX_RATE_EVENTS,
    ):
        self.window_seconds = window_seconds
        self.total = 0
        self._times: Deque[float] = deque(maxlen=max_events)

    def add(self, at: float) -> None:
        """Record one event.

        Args:
            at: ``time.perf_counter()`` of the event.
        """
        self.total += 1
        self._times.append(at)

    def per_second(self, now: float) -> Optional[float]:
        """Return the event rate in the window, or None below two events.

        Args:
            now: Current ``time.perf_counter()``.

        Returns:
            Events per second between the first and last event in the window.
        """
        recent = [at for at in self._times if now - at <= self.window_seconds]
        if len(recent) < 2 or recent[-1] == recent[0]:
            return None

        rate = (len(recent) - 1) / (recent[-1] - recent[0])

        return rate


class RollingSamples:
    """The last ``size`` values of one measurement.

    Args:
        size: Bound of remembered values.
    """

    def __init__(self, *, size: int = SAMPLE_WINDOW):
        self._values: Deque[float] = deque(maxlen=size)

    def add(self, value: float) -> None:
        """Remember one value.

        Args:
            value: The measurement.
        """
        self._values.append(value)

    def median(self) -> Optional[float]:
        """Return the median, or None without values."""
        if not self._values:
            return None

        median = statistics.median(self._values)

        return median

    def p95(self) -> Optional[float]:
        """Return the 95th percentile (nearest rank), or None without values."""
        if not self._values:
            return None

        # Nearest rank: the ceil(0.95 * n)-th smallest value. 95 * n / 100 is
        # exact when integral; 0.95 * n can round above an integer.
        ordered = sorted(self._values)
        p95 = ordered[math.ceil(95 * len(ordered) / 100) - 1]

        return p95


class LiveStats:
    """Thread-safe rolling statistics shared by reader, worker and display."""

    def __init__(self):
        self._lock = threading.Lock()
        self._read = RollingRate()
        self._completed = RollingRate()
        self._presented = RollingRate()
        self._result_age_ms = RollingSamples()
        self._present_age_ms = RollingSamples()
        self._stages = {stage: RollingSamples() for stage in STAGES}
        self._drops = dict.fromkeys(DROPS, 0)
        self._detections = 0
        self._size_hw: Optional[tuple] = None

    def frame_read(self, *, at: float, size_hw: tuple) -> None:
        """Record a frame the source delivered.

        Args:
            at: Time right after the read returned.
            size_hw: Frame ``(height, width)``.
        """
        with self._lock:
            self._read.add(at)
            self._size_hw = size_hw

    def result_completed(
        self,
        *,
        at: float,
        captured_at: float,
        timings: Mapping[str, float],
        detections: int,
    ) -> None:
        """Record a finished run.

        Args:
            at: Time the worker received the result.
            captured_at: Time its frame was read.
            timings: Milliseconds of every name in ``STAGES``.
            detections: Number of detections of the run.
        """
        with self._lock:
            self._completed.add(at)
            self._result_age_ms.add((at - captured_at) * 1000)
            for stage in STAGES:
                self._stages[stage].add(timings[stage])
            self._detections = detections

    def result_presented(self, *, at: float, captured_at: float) -> None:
        """Record a result shown on screen for the first time.

        Args:
            at: Time imshow and waitKey returned.
            captured_at: Time its frame was read.
        """
        with self._lock:
            self._presented.add(at)
            self._present_age_ms.add((at - captured_at) * 1000)

    def count(self, drop: str) -> None:
        """Increment one counter of ``DROPS``.

        Args:
            drop: Counter name.
        """
        with self._lock:
            self._drops[drop] += 1

    def snapshot(self, now: float) -> Dict[str, object]:
        """Return the current values as plain data.

        Args:
            now: Current ``time.perf_counter()``.

        Returns:
            Rates (``None`` when unknown), latency median and p95, stage
            medians, totals, drops, last detection count and frame size.
        """
        with self._lock:
            snapshot = {
                "source_fps": self._read.per_second(now),
                "processed_fps": self._completed.per_second(now),
                "output_fps": self._presented.per_second(now),
                "capture_to_result_ms": _spread(self._result_age_ms),
                "capture_to_present_ms": _spread(self._present_age_ms),
                "stage_median_ms": {
                    stage: samples.median() for stage, samples in self._stages.items()
                },
                "frames_read": self._read.total,
                "results_completed": self._completed.total,
                "results_presented": self._presented.total,
                "drops": dict(self._drops),
                "detections": self._detections,
                "size_hw": self._size_hw,
            }

        return snapshot


def overlay_lines(snapshot: Mapping[str, object], *, header: str) -> List[str]:
    """Format a snapshot as the lines drawn over the live image.

    Args:
        snapshot: Result of ``LiveStats.snapshot``.
        header: Mode, backend and device line.

    Returns:
        Text lines, top to bottom.
    """
    present = snapshot["capture_to_present_ms"]
    stages = snapshot["stage_median_ms"]
    drops = snapshot["drops"]
    height, width = snapshot["size_hw"] or (0, 0)
    lines = [
        header,
        f"output {_number(snapshot['output_fps'])} FPS | "
        f"processed {_number(snapshot['processed_fps'])} FPS | "
        f"camera {_number(snapshot['source_fps'])} FPS",
        f"capture->present {_number(present['median'])} ms "
        f"(p95 {_number(present['p95'])})",
        "median ms: "
        + " | ".join(f"{stage[:-3]} {_number(stages[stage])}" for stage in STAGES),
        f"{width}x{height} | detections {snapshot['detections']} | "
        f"dropped: camera {drops['capture_overwritten']} "
        f"result {drops['result_replaced']}",
    ]

    return lines


def _spread(samples: RollingSamples) -> Dict[str, Optional[float]]:
    spread = {"median": samples.median(), "p95": samples.p95()}

    return spread


def _number(value: Optional[float]) -> str:
    if value is None:
        return "-"

    text = f"{value:.1f}"

    return text
