import queue
import threading
from contextlib import nullcontext

import numpy as np
import torch

from inference_models.models.optimization.contracts import ExecutionContext
from inference_models.models.optimization.torch_readiness import TensorReadinessTracker
from inference_models.models.yolo26.optimization.preprocessors import (
    _PinnedImageSlot,
    _PinnedImageSlotPool,
)
from inference_models.models.yolo26.optimization.schedulers import (
    BaseYOLO26DepthExecutionScheduler,
    CUDAEventHandoffYOLO26DepthExecutionScheduler,
)


class _FakeStream:
    def __init__(self, calls):
        self._calls = calls

    def synchronize(self) -> None:
        self._calls.append("stream-synchronize")

    def wait_event(self, event) -> None:
        self._calls.append(("wait-event", event))


class _FakeEvent:
    def __init__(self, calls):
        self._calls = calls

    def record(self, stream) -> None:
        self._calls.append(("record-event", stream))

    def synchronize(self) -> None:
        self._calls.append("event-synchronize")


def _context(*, stream) -> ExecutionContext:
    return ExecutionContext(
        device_kind="gpu",
        device="cuda:0",
        current_stream=stream,
    )


def _event_scheduler(*, inference_stream):
    scheduler = object.__new__(CUDAEventHandoffYOLO26DepthExecutionScheduler)
    scheduler._engine_lock = threading.Lock()
    scheduler._inference_stream = inference_stream
    scheduler._preprocess_readiness = TensorReadinessTracker[torch.cuda.Event]()

    return scheduler


def test_base_scheduler_preserves_preprocess_synchronization(monkeypatch) -> None:
    calls = []
    stream = _FakeStream(calls)
    scheduler = object.__new__(BaseYOLO26DepthExecutionScheduler)
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda _: nullcontext())
    tensor = torch.zeros(1)

    result = scheduler.finalize_preprocess(
        tensor,
        context=_context(stream=stream),
        independent_stage_execution=False,
    )

    assert result is tensor
    assert calls == ["stream-synchronize"]


def test_event_scheduler_hands_exact_tensor_readiness_to_engine(
    monkeypatch,
) -> None:
    calls = []
    producer_stream = _FakeStream(calls)
    inference_stream = _FakeStream(calls)
    ready_event = _FakeEvent(calls)
    scheduler = _event_scheduler(inference_stream=inference_stream)
    monkeypatch.setattr(torch.cuda, "Event", lambda: ready_event)
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda _: nullcontext())
    tensor = torch.zeros(1)
    other = torch.zeros(1)

    result = scheduler.finalize_preprocess(
        tensor,
        context=_context(stream=producer_stream),
        independent_stage_execution=False,
    )
    scheduler.execute_engine(
        other,
        operation=lambda stream: calls.append(("execute-other", stream)) or other,
    )
    scheduler.execute_engine(
        result,
        operation=lambda stream: calls.append(("execute", stream)) or result,
    )
    scheduler.execute_engine(
        result,
        operation=lambda stream: calls.append(("execute-again", stream)) or result,
    )

    assert result is tensor
    assert calls == [
        ("record-event", producer_stream),
        ("execute-other", inference_stream),
        ("wait-event", ready_event),
        ("execute", inference_stream),
        ("execute-again", inference_stream),
    ]


def test_event_scheduler_keeps_direct_preprocess_ready_on_return(
    monkeypatch,
) -> None:
    calls = []
    producer_stream = _FakeStream(calls)
    ready_event = _FakeEvent(calls)
    scheduler = _event_scheduler(inference_stream=_FakeStream(calls))
    monkeypatch.setattr(torch.cuda, "Event", lambda: ready_event)
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda _: nullcontext())
    tensor = torch.zeros(1)

    result = scheduler.finalize_preprocess(
        tensor,
        context=_context(stream=producer_stream),
        independent_stage_execution=True,
    )

    assert result is tensor
    assert calls == [
        ("record-event", producer_stream),
        "event-synchronize",
    ]
    assert scheduler._preprocess_readiness.consume(tensor) is None


def test_pinned_slot_waits_for_h2d_before_host_reuse(monkeypatch) -> None:
    calls = []
    event = _FakeEvent(calls)
    slot = _PinnedImageSlot(
        tensor=torch.empty(0),
        array=np.empty((1, 1, 3), dtype=np.uint8),
        reuse_event=event,
        transfer_pending=True,
    )
    pool = object.__new__(_PinnedImageSlotPool)
    pool._slots = queue.LifoQueue(maxsize=1)
    pool._slots.put(slot)
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda _: nullcontext())

    acquired = pool.acquire()
    pool.release(acquired)

    assert acquired is slot
    assert not acquired.transfer_pending
    assert calls == ["event-synchronize"]
