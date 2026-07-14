"""Micro-batch tensor-native tracker updates across independent video streams."""

from __future__ import annotations

import atexit
import os
import threading
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

import supervision as sv
import torch


@dataclass(slots=True)
class _TrackerResult:
    detections: sv.Detections
    completion_event: torch.cuda.Event | None


@dataclass(slots=True)
class _TrackerRequest:
    tracker: Any
    detections: sv.Detections
    frame: Any | None
    timestamp: float | None
    ready_event: torch.cuda.Event | None
    future: Future[_TrackerResult]

    @property
    def compatibility_key(self) -> tuple[type, str, str, bool]:
        boxes = self.detections.xyxy
        return (
            type(self.tracker),
            str(boxes.device),
            str(boxes.dtype),
            self.frame is not None,
        )


class TrackerBatchScheduler:
    """Collect compatible tracker calls and execute one Tracktors batch."""

    def __init__(
        self, *, batch_window_ms: float = 0.0, max_batch_size: int = 32
    ) -> None:
        if not 0.0 <= batch_window_ms <= 2.0:
            raise ValueError("batch_window_ms must be between 0 and 2")
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        self.batch_window_seconds = batch_window_ms / 1_000.0
        self.max_batch_size = max_batch_size
        self._condition = threading.Condition()
        self._pending: deque[_TrackerRequest] = deque()
        self._closed = False
        self._executor: Any | None = None
        self._execution_lock = threading.Lock()
        self._worker = threading.Thread(
            target=self._run,
            name="tracktors-batch-scheduler",
            daemon=True,
        )
        self._worker.start()

    def update(
        self,
        tracker: Any,
        detections: sv.Detections,
        *,
        frame: Any | None = None,
        timestamp: float | None = None,
    ) -> sv.Detections:
        """Submit one stream update and wait for its ordered batch result."""
        future: Future[_TrackerResult] = Future()
        request = _TrackerRequest(
            tracker=tracker,
            detections=detections,
            frame=frame,
            timestamp=timestamp,
            ready_event=self._record_ready_event(detections),
            future=future,
        )
        with self._condition:
            if self._closed:
                raise RuntimeError("tracker batch scheduler is closed")
            self._pending.append(request)
            self._condition.notify()
        result = future.result()
        self._prepare_consumer_handoff(result)
        return result.detections

    def execute_batch(
        self,
        trackers: list[Any],
        detections: list[sv.Detections],
        *,
        frames: list[Any | None],
        timestamps: list[float | None],
    ) -> list[sv.Detections]:
        """Execute one explicit SIMD batch with the persistent CUDA executor."""
        import tracktors

        with self._execution_lock:
            if self._executor is None:
                executor_type = getattr(tracktors, "CUDABatchExecutor", None)
                if executor_type is not None:
                    self._executor = executor_type(
                        max_workers=self.max_batch_size,
                    )
            return tracktors.update_batch(
                trackers,
                detections,
                frames=frames,
                timestamps=timestamps,
                executor=self._executor,
            )

    def close(self) -> None:
        """Stop accepting work and fail requests that have not started."""
        with self._condition:
            if self._closed:
                return
            self._closed = True
            pending = list(self._pending)
            self._pending.clear()
            self._condition.notify_all()
        for request in pending:
            request.future.set_exception(RuntimeError("tracker batch scheduler closed"))
        if threading.current_thread() is not self._worker:
            self._worker.join()
        with self._execution_lock:
            if self._executor is not None:
                self._executor.close()
                self._executor = None

    def _take_batch(self) -> list[_TrackerRequest]:
        with self._condition:
            while not self._pending and not self._closed:
                self._condition.wait()
            if not self._pending:
                return []

            first = self._pending.popleft()
            if self.batch_window_seconds:
                deadline = time.monotonic() + self.batch_window_seconds
                while len(self._pending) + 1 < self.max_batch_size:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(timeout=remaining)

            batch = [first]
            tracker_ids = {id(first.tracker)}
            retained: deque[_TrackerRequest] = deque()
            while self._pending:
                candidate = self._pending.popleft()
                compatible = candidate.compatibility_key == first.compatibility_key
                unique_tracker = id(candidate.tracker) not in tracker_ids
                if compatible and unique_tracker and len(batch) < self.max_batch_size:
                    batch.append(candidate)
                    tracker_ids.add(id(candidate.tracker))
                else:
                    retained.append(candidate)
            self._pending = retained
            return batch

    def _run(self) -> None:
        while True:
            batch = self._take_batch()
            if not batch:
                if self._closed:
                    return
                continue
            try:
                self._prepare_worker_handoff(batch)
                outputs = self.execute_batch(
                    [request.tracker for request in batch],
                    [request.detections for request in batch],
                    frames=[request.frame for request in batch],
                    timestamps=[request.timestamp for request in batch],
                )
                if len(outputs) != len(batch):
                    raise RuntimeError(
                        "tracktors.update_batch returned the wrong number of results"
                    )
                completion_events = [
                    self._record_completion_event(output) for output in outputs
                ]
            except BaseException as error:
                for request in batch:
                    request.future.set_exception(error)
            else:
                for request, output, completion_event in zip(
                    batch,
                    outputs,
                    completion_events,
                ):
                    request.future.set_result(
                        _TrackerResult(
                            detections=output,
                            completion_event=completion_event,
                        )
                    )

    @staticmethod
    def _record_ready_event(
        detections: sv.Detections,
    ) -> torch.cuda.Event | None:
        """Record when the submitting stream has finished producing detections."""
        boxes = getattr(detections, "xyxy", None)
        if not isinstance(boxes, torch.Tensor) or boxes.device.type != "cuda":
            return None
        with torch.cuda.device(boxes.device):
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(boxes.device))
        return event

    @classmethod
    def _prepare_worker_handoff(cls, batch: list[_TrackerRequest]) -> None:
        """Make scheduler streams wait for submitting streams without host syncs."""
        for request in batch:
            if request.ready_event is None:
                continue
            device = request.detections.xyxy.device
            with torch.cuda.device(device):
                worker_stream = torch.cuda.current_stream(device)
                worker_stream.wait_event(request.ready_event)
                cls._record_detection_stream(request.detections, worker_stream)

    @staticmethod
    def _record_completion_event(
        detections: sv.Detections,
    ) -> torch.cuda.Event | None:
        """Record when scheduler-owned CUDA work is visible to consumers."""
        boxes = getattr(detections, "xyxy", None)
        if not isinstance(boxes, torch.Tensor) or boxes.device.type != "cuda":
            return None
        with torch.cuda.device(boxes.device):
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(boxes.device))
        return event

    @classmethod
    def _prepare_consumer_handoff(cls, result: _TrackerResult) -> None:
        """Order caller CUDA work after the scheduler result and retain storage."""
        if result.completion_event is None:
            return
        device = result.detections.xyxy.device
        with torch.cuda.device(device):
            consumer_stream = torch.cuda.current_stream(device)
            consumer_stream.wait_event(result.completion_event)
            cls._record_detection_stream(result.detections, consumer_stream)

    @classmethod
    def _record_value_stream(cls, value: Any, stream: torch.cuda.Stream) -> None:
        """Record nested tensor storage on a CUDA consumer stream."""
        if isinstance(value, torch.Tensor):
            value.record_stream(stream)
            return
        if isinstance(value, dict):
            for nested_value in value.values():
                cls._record_value_stream(nested_value, stream)
            return
        if isinstance(value, (list, tuple)):
            for nested_value in value:
                cls._record_value_stream(nested_value, stream)

    @classmethod
    def _record_detection_stream(
        cls,
        detections: sv.Detections,
        stream: torch.cuda.Stream,
    ) -> None:
        """Record every tensor-backed detection field on a CUDA stream."""
        for field_name in (
            "xyxy",
            "mask",
            "confidence",
            "class_id",
            "tracker_id",
            "data",
        ):
            cls._record_value_stream(getattr(detections, field_name, None), stream)


_GLOBAL_SCHEDULER: TrackerBatchScheduler | None = None
_GLOBAL_LOCK = threading.Lock()


def get_tracker_batch_scheduler() -> TrackerBatchScheduler:
    """Return the process-wide scheduler configured from environment variables."""
    global _GLOBAL_SCHEDULER
    with _GLOBAL_LOCK:
        if _GLOBAL_SCHEDULER is None:
            window = min(
                2.0,
                max(0.0, float(os.getenv("TRACKTORS_BATCH_WINDOW_MS", "0"))),
            )
            maximum = max(1, int(os.getenv("TRACKTORS_MAX_BATCH_SIZE", "32")))
            _GLOBAL_SCHEDULER = TrackerBatchScheduler(
                batch_window_ms=window,
                max_batch_size=maximum,
            )
        return _GLOBAL_SCHEDULER


def _close_global_scheduler() -> None:
    if _GLOBAL_SCHEDULER is not None:
        _GLOBAL_SCHEDULER.close()


atexit.register(_close_global_scheduler)
