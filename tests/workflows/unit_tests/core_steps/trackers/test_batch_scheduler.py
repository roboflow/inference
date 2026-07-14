"""Tests for tensor-native tracker micro-batching."""

from __future__ import annotations

import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor

import torch

from inference.core.workflows.core_steps.trackers.batch_scheduler import (
    TrackerBatchScheduler,
)


class _Tracker:
    pass


def _detections(value: float):
    return types.SimpleNamespace(
        xyxy=torch.tensor([[value, 0.0, value + 1.0, 1.0]]),
        tracker_id=torch.tensor([int(value)]),
        data={},
    )


def test_scheduler_batches_compatible_independent_trackers(monkeypatch) -> None:
    calls: list[list[_Tracker]] = []
    lock = threading.Lock()
    module = types.ModuleType("tracktors")

    def update_batch(trackers, detections, **kwargs):
        with lock:
            calls.append(list(trackers))
        return detections

    module.update_batch = update_batch
    monkeypatch.setitem(sys.modules, "tracktors", module)
    scheduler = TrackerBatchScheduler(batch_window_ms=2.0, max_batch_size=8)
    trackers = [_Tracker() for _ in range(4)]
    detections = [_detections(float(index)) for index in range(4)]
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(scheduler.update, tracker, detection)
                for tracker, detection in zip(trackers, detections)
            ]
            outputs = [future.result() for future in futures]
    finally:
        scheduler.close()

    assert all(output is detection for output, detection in zip(outputs, detections))
    assert [len(call) for call in calls] == [4]
    assert [output.tracker_id.tolist() for output in outputs] == [[0], [1], [2], [3]]
    assert all(output.data == {} for output in outputs)


def test_scheduler_never_batches_two_frames_for_the_same_tracker(monkeypatch) -> None:
    batch_sizes: list[int] = []
    module = types.ModuleType("tracktors")

    def update_batch(trackers, detections, **kwargs):
        batch_sizes.append(len(trackers))
        assert len({id(tracker) for tracker in trackers}) == len(trackers)
        return detections

    module.update_batch = update_batch
    monkeypatch.setitem(sys.modules, "tracktors", module)
    scheduler = TrackerBatchScheduler(batch_window_ms=2.0, max_batch_size=8)
    tracker = _Tracker()
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(scheduler.update, tracker, _detections(1.0))
            second = executor.submit(scheduler.update, tracker, _detections(2.0))
            first.result()
            second.result()
    finally:
        scheduler.close()

    assert batch_sizes == [1, 1]


def test_explicit_batches_reuse_and_close_cuda_executor(monkeypatch) -> None:
    """Synchronous SIMD calls share one executor and close it deterministically."""
    executors = []
    received_executors = []
    module = types.ModuleType("tracktors")

    class _Executor:
        """Record construction and deterministic scheduler-owned closure."""

        def __init__(self, max_workers):
            """Capture the configured maximum batch width."""
            self.max_workers = max_workers
            self.closed = False
            executors.append(self)

        def close(self):
            """Record release of executor-owned streams and worker pools."""
            self.closed = True

    def update_batch(trackers, detections, **kwargs):
        """Capture which executor was supplied to the Tracktors boundary."""
        received_executors.append(kwargs["executor"])
        return detections

    module.CUDABatchExecutor = _Executor
    module.update_batch = update_batch
    monkeypatch.setitem(sys.modules, "tracktors", module)
    scheduler = TrackerBatchScheduler(max_batch_size=8)
    trackers = [_Tracker(), _Tracker()]
    detections = [_detections(1.0), _detections(2.0)]

    try:
        scheduler.execute_batch(
            trackers,
            detections,
            frames=[None, None],
            timestamps=[None, None],
        )
        scheduler.execute_batch(
            trackers,
            detections,
            frames=[None, None],
            timestamps=[None, None],
        )
    finally:
        scheduler.close()

    assert len(executors) == 1
    assert received_executors == [executors[0], executors[0]]
    assert executors[0].max_workers == 8
    assert executors[0].closed
