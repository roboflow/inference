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

HOST_TRACKER_IDS_KEY = "__tracktors_host_tracker_ids__"


@dataclass(slots=True)
class _TrackerRequest:
    tracker: Any
    detections: sv.Detections
    frame: Any | None
    timestamp: float | None
    future: Future[sv.Detections]

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

    def __init__(self, *, batch_window_ms: float = 0.0, max_batch_size: int = 32) -> None:
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
        future: Future[sv.Detections] = Future()
        request = _TrackerRequest(
            tracker=tracker,
            detections=detections,
            frame=frame,
            timestamp=timestamp,
            future=future,
        )
        with self._condition:
            if self._closed:
                raise RuntimeError("tracker batch scheduler is closed")
            self._pending.append(request)
            self._condition.notify()
        return future.result()

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
                import tracktors

                if self._executor is None:
                    executor_type = getattr(tracktors, "CUDABatchExecutor", None)
                    if executor_type is not None:
                        self._executor = executor_type(
                            max_workers=self.max_batch_size,
                        )

                outputs = tracktors.update_batch(
                    [request.tracker for request in batch],
                    [request.detections for request in batch],
                    frames=[request.frame for request in batch],
                    timestamps=[request.timestamp for request in batch],
                    executor=self._executor,
                )
                if len(outputs) != len(batch):
                    raise RuntimeError(
                        "tracktors.update_batch returned the wrong number of results"
                    )
                self._materialize_tracker_ids(outputs)
            except BaseException as error:
                for request in batch:
                    request.future.set_exception(error)
            else:
                for request, output in zip(batch, outputs):
                    request.future.set_result(output)

    @staticmethod
    def _materialize_tracker_ids(outputs: list[sv.Detections]) -> None:
        """Transfer tracker IDs once for the whole batch's object metadata."""
        tensors: list[torch.Tensor] = []
        lengths: list[int] = []
        for output in outputs:
            tracker_ids = getattr(output, "tracker_id", None)
            if not isinstance(tracker_ids, torch.Tensor):
                return
            tensors.append(tracker_ids.reshape(-1))
            lengths.append(tracker_ids.numel())
        if not tensors:
            return
        packed = torch.cat(tensors) if len(tensors) > 1 else tensors[0]
        host_values = packed.detach().to("cpu").tolist()
        offset = 0
        for output, length in zip(outputs, lengths):
            data = getattr(output, "data", None)
            if data is None:
                data = {}
                output.data = data
            data[HOST_TRACKER_IDS_KEY] = host_values[offset : offset + length]
            offset += length


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
