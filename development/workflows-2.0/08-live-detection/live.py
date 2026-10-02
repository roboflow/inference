"""Live loop: capture thread, graph worker thread, display on the main thread.

    capture thread   VideoCapture.read() -> frames slot (1 item)
                     camera: a newer frame replaces an untaken one (counted)
                     video/image file: waits until the slot is free (lossless)
    graph worker     frames slot -> HWC BGR to CHW RGB ImageData (one copy)
                     serial:   session.run, one frame at a time
                     pipeline: session.pipeline(max_in_flight=N); a frame is
                               taken only while fewer than N runs are pending,
                               results delivered in submission order
                     -> results slot (1 item; a newer result replaces an
                        unshown one, counted)
    main thread      results slot -> CHW RGB to HWC BGR (one copy) -> overlay
                     -> imshow / waitKey; q, Esc, window close, Ctrl-C exit

The main thread never runs the graph. Shutdown sets ``stop``, closes the
frames slot, closes the window, then joins the worker (a running inference
finishes; it cannot be interrupted) and the reader (which releases the
capture it owns). Shutdown also runs when startup or closing the window
fails; threads that never started are not joined.
"""

import threading
import time
from collections import deque
from concurrent.futures import Future, wait
from dataclasses import dataclass
from typing import Any, Deque, Dict, Generic, List, Optional, Tuple, TypeVar

import cv2
import numpy as np
import torch
from roboflow_workflows.execution_engine.v2.blocks.image_data import ImageData
from roboflow_workflows.execution_engine.v2.pipelining import (
    PipelineFullError,
    PipelineOptions,
)
from stats import STAGES, LiveStats, overlay_lines

WINDOW = "V2 live detection"
POLL_SECONDS = 0.1
# Below max_in_flight pending runs a pool worker is idle or about to be; a
# submit that still waits this long means a stalled pipeline, not a full one.
STALLED_SUBMIT_SECONDS = 1.0
READER_JOIN_SECONDS = 2.0
EXIT_KEYS = (ord("q"), 27)

T = TypeVar("T")


@dataclass(frozen=True)
class Frame:
    """One frame as the source delivered it.

    Attributes:
        index: Position in the source, from 0.
        pixels_bgr: ``(height, width, 3)`` uint8 BGR array, owned by the frame.
        captured_at: ``time.perf_counter()`` right after the read returned.
    """

    index: int
    pixels_bgr: np.ndarray
    captured_at: float


@dataclass(frozen=True)
class LiveResult:
    """One run result ready for the display.

    Attributes:
        frame_index: Index of the frame it was computed from.
        captured_at: Capture time of that frame.
        annotated: ``(3, height, width)`` uint8 RGB tensor with boxes and labels.
        predictions: Native detections of the frame.
    """

    frame_index: int
    captured_at: float
    annotated: torch.Tensor
    predictions: Any


@dataclass(frozen=True)
class LiveSummary:
    """How a live run ended.

    Attributes:
        reason: Why the loop stopped, for example ``q/Esc`` or ``end of input``.
        snapshot: Final ``LiveStats.snapshot``.
        last_result: The last result the main thread received, if any.
        error: Failure of the worker or reader, if any.
        reader_still_blocked: The capture read did not return in time; the
            capture is released when it does.
    """

    reason: str
    snapshot: Dict[str, object]
    last_result: Optional[LiveResult]
    error: Optional[BaseException]
    reader_still_blocked: bool


class LatestSlot(Generic[T]):
    """Holds at most one item between two threads.

    ``put`` replaces an untaken item (and says so) unless it waits for the
    slot to be free. ``take`` also returns, empty-handed, after ``wake``.
    After ``close`` puts are ignored and the remaining item can be taken.
    """

    def __init__(self):
        self._condition = threading.Condition()
        self._item: Optional[T] = None
        self._closed = False
        self._woken = False

    def put(self, item: T, *, wait_until_free: bool = False) -> bool:
        """Store ``item``.

        Args:
            item: The new item.
            wait_until_free: Wait until the previous item was taken (or the
                slot closed) instead of replacing it.

        Returns:
            True when an untaken item was replaced.
        """
        with self._condition:
            if wait_until_free:
                self._condition.wait_for(lambda: self._item is None or self._closed)
            if self._closed:
                return False

            replaced = self._item is not None
            self._item = item
            self._condition.notify_all()

        return replaced

    def take(self, *, timeout: float) -> Optional[T]:
        """Remove and return the item, waiting at most ``timeout`` seconds.

        Args:
            timeout: Longest wait; 0 does not wait.

        Returns:
            The item, or None on timeout, ``wake`` or a closed empty slot.
        """
        with self._condition:
            self._condition.wait_for(
                lambda: self._item is not None or self._closed or self._woken,
                timeout=timeout,
            )
            item = self._item
            self._item = None
            self._woken = False
            self._condition.notify_all()

        return item

    def wake(self) -> None:
        """Make a waiting ``take`` return now."""
        with self._condition:
            self._woken = True
            self._condition.notify_all()

    def close(self) -> None:
        """Ignore further puts and release every waiter."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    @property
    def exhausted(self) -> bool:
        """Closed and nothing left to take."""
        with self._condition:
            exhausted = self._closed and self._item is None

        return exhausted


class StillImage:
    """A ``VideoCapture``-like source delivering one image once.

    Args:
        pixels_bgr: ``(height, width, 3)`` uint8 BGR image.
    """

    def __init__(self, pixels_bgr: np.ndarray):
        self._pixels: Optional[np.ndarray] = pixels_bgr

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return the image on the first call and end of input afterwards."""
        pixels = self._pixels
        self._pixels = None

        return pixels is not None, pixels

    def release(self) -> None:
        """Nothing to release."""


class FrameReader(threading.Thread):
    """Reads the source on its own thread; the only user of the capture.

    Args:
        capture: ``cv2.VideoCapture`` or ``StillImage``; released on exit.
        frames: Slot the frames go to; closed on exit.
        stats: Records reads and overwritten frames.
        stop: Set by the main thread to end the run.
        lossless: Wait for a free slot (files) instead of replacing (camera).
    """

    def __init__(
        self,
        capture: Any,
        *,
        frames: LatestSlot[Frame],
        stats: LiveStats,
        stop: threading.Event,
        lossless: bool,
    ):
        super().__init__(name="live-capture", daemon=True)
        self.error: Optional[BaseException] = None
        self._capture = capture
        self._frames = frames
        self._stats = stats
        self._stop_event = stop
        self._lossless = lossless

    def run(self) -> None:
        index = 0
        try:
            while not self._stop_event.is_set():
                ok, pixels = self._capture.read()
                captured_at = time.perf_counter()
                if not ok:
                    return

                self._stats.frame_read(at=captured_at, size_hw=pixels.shape[:2])
                frame = Frame(index=index, pixels_bgr=pixels, captured_at=captured_at)
                replaced = self._frames.put(frame, wait_until_free=self._lossless)
                if replaced:
                    self._stats.count("capture_overwritten")
                index += 1
        except BaseException as error:
            self.error = error
        finally:
            try:
                self._capture.release()
            except BaseException as error:
                self.error = self.error or error
            finally:
                self._frames.close()


class GraphWorker(threading.Thread):
    """Runs the compiled workflow off the GUI thread, in either mode.

    Args:
        session: Session of the compiled live workflow.
        frames: Slot the frames come from.
        results: Slot the results go to; closed on exit.
        stats: Records completed runs and drops.
        stop: Set by the main thread to end the run.
        confidence: Workflow input ``confidence``.
        max_in_flight: Pipeline bound; None runs ``session.run`` serially.
    """

    def __init__(
        self,
        session: Any,
        *,
        frames: LatestSlot[Frame],
        results: LatestSlot[LiveResult],
        stats: LiveStats,
        stop: threading.Event,
        confidence: float,
        max_in_flight: Optional[int],
    ):
        super().__init__(name="live-graph", daemon=True)
        self.error: Optional[BaseException] = None
        self._session = session
        self._frames = frames
        self._results = results
        self._stats = stats
        self._stop_event = stop
        self._confidence = confidence
        self._max_in_flight = max_in_flight

    def run(self) -> None:
        try:
            if self._max_in_flight is None:
                self._run_serial()
            else:
                self._run_pipelined(max_in_flight=self._max_in_flight)
        except BaseException as error:
            self.error = error
        finally:
            self._results.close()

    def _run_serial(self) -> None:
        while not self._stop_event.is_set():
            frame = self._frames.take(timeout=POLL_SECONDS)
            if frame is None:
                if self._frames.exhausted:
                    return
                continue

            result = self._session.run(self._inputs(frame))
            self._deliver(frame, result=result)

    def _run_pipelined(self, *, max_in_flight: int) -> None:
        # Admission: take a frame only while fewer than max_in_flight runs are
        # pending. Frames that wait meanwhile are replaced in the frames slot
        # and counted there as capture_overwritten.
        pending: Deque[Tuple[Frame, Future]] = deque()
        options = PipelineOptions(max_in_flight=max_in_flight)
        with self._session.pipeline(options=options) as pipeline:
            while not self._stop_event.is_set():
                while pending and pending[0][1].done():
                    frame, future = pending.popleft()
                    self._deliver(frame, result=future.result())

                if len(pending) == max_in_flight or (
                    pending and self._frames.exhausted
                ):
                    wait([pending[0][1]], timeout=POLL_SECONDS)
                    continue
                if self._frames.exhausted:
                    return

                frame = self._frames.take(timeout=POLL_SECONDS)
                if frame is None:
                    continue

                try:
                    future = pipeline.submit(
                        self._inputs(frame), timeout=STALLED_SUBMIT_SECONDS
                    )
                except PipelineFullError as error:
                    raise RuntimeError(
                        f"Pipeline stalled: no worker accepted frame {frame.index} "
                        f"within {STALLED_SUBMIT_SECONDS} s with {len(pending)} of "
                        f"{max_in_flight} runs pending"
                    ) from error
                future.add_done_callback(lambda _: self._frames.wake())
                pending.append((frame, future))
        # Leaving the block waits for runs in progress; their results are dropped.

    def _inputs(self, frame: Frame) -> Dict[str, Any]:
        inputs = {"image": frame_to_image(frame), "confidence": self._confidence}

        return inputs

    def _deliver(self, frame: Frame, *, result: Any) -> None:
        completed_at = time.perf_counter()
        predictions = output_value(result, "predictions")
        timings = {stage: float(output_value(result, stage)) for stage in STAGES}
        self._stats.result_completed(
            at=completed_at,
            captured_at=frame.captured_at,
            timings=timings,
            detections=len(predictions),
        )

        live_result = LiveResult(
            frame_index=frame.index,
            captured_at=frame.captured_at,
            annotated=output_value(result, "annotated").tensor_image,
            predictions=predictions,
        )
        if self._results.put(live_result):
            self._stats.count("result_replaced")


def run_live(
    session: Any,
    *,
    capture: Any,
    lossless: bool,
    confidence: float,
    max_in_flight: Optional[int],
    header: str,
    display: bool,
    max_results: Optional[int] = None,
    duration_seconds: Optional[float] = None,
) -> LiveSummary:
    """Run the live loop until the input ends or the user exits.

    Args:
        session: Session of the compiled live workflow, already warmed up.
        capture: Opened ``cv2.VideoCapture`` or ``StillImage``; the reader
            thread owns and releases it.
        lossless: Process every frame (files) instead of the newest (camera).
        confidence: Workflow input ``confidence``.
        max_in_flight: Pipeline bound; None runs serially.
        header: First overlay line (mode, backend, device).
        display: Show a window; otherwise only count results (headless).
        max_results: Stop after this many presented (or, headless, received)
            results.
        duration_seconds: Stop after this many seconds.

    Returns:
        Why the loop ended, the final statistics and the last result.
    """
    stats = LiveStats()
    stop = threading.Event()
    frames: LatestSlot[Frame] = LatestSlot()
    results: LatestSlot[LiveResult] = LatestSlot()
    reader = FrameReader(
        capture, frames=frames, stats=stats, stop=stop, lossless=lossless
    )
    worker = GraphWorker(
        session,
        frames=frames,
        results=results,
        stats=stats,
        stop=stop,
        confidence=confidence,
        max_in_flight=max_in_flight,
    )

    loop = _DisplayLoop if display else _HeadlessLoop
    main_loop = loop(
        results,
        stats=stats,
        header=header,
        max_results=max_results,
        duration_seconds=duration_seconds,
    )
    try:
        reader.start()
        worker.start()
        reason = main_loop.run()
    except KeyboardInterrupt:
        reason = "Ctrl-C"
    finally:
        stop.set()
        frames.close()
        try:
            main_loop.close()
        finally:
            _join(worker=worker, reader=reader, capture=capture)

    error = worker.error or reader.error
    summary = LiveSummary(
        reason="error" if error is not None else reason,
        snapshot=stats.snapshot(time.perf_counter()),
        last_result=main_loop.last_result,
        error=error,
        reader_still_blocked=reader.is_alive(),
    )

    return summary


def _join(*, worker: "GraphWorker", reader: "FrameReader", capture: Any) -> None:
    # ``ident`` is None for a thread that never started; such threads are not
    # joined, and the capture of a reader that never started is released here.
    if worker.ident is not None:
        worker.join()
    if reader.ident is not None:
        reader.join(timeout=READER_JOIN_SECONDS)
    else:
        capture.release()


class _HeadlessLoop:
    """Main thread without a window: receive results and count them."""

    def __init__(
        self,
        results: LatestSlot[LiveResult],
        *,
        stats: LiveStats,
        header: str,
        max_results: Optional[int],
        duration_seconds: Optional[float],
    ):
        self.last_result: Optional[LiveResult] = None
        self._results = results
        self._stats = stats
        self._header = header
        self._max_results = max_results
        self._duration_seconds = duration_seconds
        self._received = 0
        self._started = time.perf_counter()

    def run(self) -> str:
        while True:
            result = self._results.take(timeout=POLL_SECONDS)
            if result is not None:
                self._received += 1
                self.last_result = result

            reason = self._limit_reached()
            if reason is not None:
                return reason

    def close(self) -> None:
        pass

    def _limit_reached(self) -> Optional[str]:
        if self._max_results is not None and self._received >= self._max_results:
            return "frame limit"
        if (
            self._duration_seconds is not None
            and time.perf_counter() - self._started >= self._duration_seconds
        ):
            return "duration"
        if self._results.exhausted:
            return "end of input"

        return None


class _DisplayLoop(_HeadlessLoop):
    """Main thread with a window: show new results, poll keys and the window."""

    def run(self) -> str:
        # A placeholder makes the window visible, so it can be closed before
        # the first result arrives.
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.imshow(WINDOW, np.zeros((360, 640, 3), dtype=np.uint8))
        shown: Optional[LiveResult] = None
        while True:
            # HighGUI paints the last imshow inside waitKey, so a result counts
            # as presented only after it.
            key = cv2.waitKey(1) & 0xFF
            if shown is not None:
                self._stats.result_presented(
                    at=time.perf_counter(), captured_at=shown.captured_at
                )
                self._received += 1
                self.last_result = shown
                shown = None

            if key in EXIT_KEYS:
                return "q/Esc"
            # Checked before the next imshow, which would reopen a closed window.
            try:
                visible = cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE)
            except cv2.error as error:
                # Cocoa removes closed windows instead of reporting visibility 0.
                if error.code != cv2.Error.StsNullPtr:
                    raise
                visible = 0
            if visible < 1:
                return "window closed"
            reason = self._limit_reached()
            if reason is not None:
                return reason

            shown = self._results.take(timeout=0)
            if shown is not None:
                self._show(shown)

    def close(self) -> None:
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def _show(self, result: LiveResult) -> None:
        pixels_bgr = image_to_bgr(result.annotated)
        lines = overlay_lines(
            self._stats.snapshot(time.perf_counter()), header=self._header
        )
        draw_overlay(pixels_bgr, lines=lines)
        cv2.imshow(WINDOW, pixels_bgr)


def frame_to_image(frame: Frame) -> ImageData:
    """Camera boundary: HWC BGR array to a CHW RGB CPU ImageData, one copy.

    Args:
        frame: Frame from the source.

    Returns:
        A contiguous CPU image named ``frame-<index>``.
    """
    chw_rgb = np.ascontiguousarray(frame.pixels_bgr.transpose(2, 0, 1)[::-1])
    image = ImageData.from_tensor(
        torch.from_numpy(chw_rgb), image_id=f"frame-{frame.index}"
    )

    return image


def image_to_bgr(tensor_image: torch.Tensor) -> np.ndarray:
    """Display boundary: CHW RGB CPU tensor to an HWC BGR array, one copy.

    Args:
        tensor_image: ``(3, height, width)`` uint8 tensor on the CPU.

    Returns:
        A new contiguous ``(height, width, 3)`` BGR array.
    """
    pixels_bgr = np.ascontiguousarray(tensor_image.numpy()[::-1].transpose(1, 2, 0))

    return pixels_bgr


def draw_overlay(pixels_bgr: np.ndarray, *, lines: List[str]) -> None:
    """Draw the statistics in the top-left corner, in place.

    Args:
        pixels_bgr: Display copy of the frame.
        lines: Text lines from ``overlay_lines``.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, pixels_bgr.shape[1] / 2000)
    thickness = max(1, round(scale * 1.5))
    line_height = int(30 * scale)
    padding = line_height // 2
    text_width = max(
        cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines
    )

    # Darken only the text box, so the statistics stay readable.
    box = pixels_bgr[: line_height * len(lines) + padding, : text_width + 2 * padding]
    box //= 3
    for position, line in enumerate(lines, start=1):
        cv2.putText(
            pixels_bgr,
            line,
            (padding, line_height * position),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def output_value(result: Any, name: str) -> Any:
    """Return the value of a single-port workflow output.

    Args:
        result: ``RunResult`` of the live workflow.
        name: Workflow output name.

    Returns:
        The value as the engine delivered it.
    """
    (entry,) = result.selections[name].values()
    value = result.outputs.data[entry]

    return value
