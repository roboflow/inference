import sys
import threading
from contextlib import nullcontext
from types import SimpleNamespace

import torch

from inference_models.models.optimization.contracts import ExecutionContext
from inference_models.models.rfdetr.optimization.buffer_strategies import (
    BaseBufferStrategy,
)
from inference_models.models.rfdetr.optimization.contracts import (
    EngineExecutionRequest,
    EngineInputBuffer,
    PreprocessResult,
)
from inference_models.models.rfdetr.optimization.engine_plugins import (
    BaseEngineAdjacentPlugin,
)
from inference_models.models.rfdetr.optimization.readiness import (
    PreprocessReadinessTracker,
)
from inference_models.models.rfdetr.optimization.schedulers import (
    BaseExecutionScheduler,
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

    def synchronize(self) -> None:
        self._calls.append("event-synchronize")


class _FakeOutput:
    def __init__(self, calls):
        self._calls = calls

    def record_stream(self, stream) -> None:
        self._calls.append(("record-stream", stream))


def _context(*, stream=None) -> ExecutionContext:
    return ExecutionContext(
        device_kind="gpu",
        device="cuda:0",
        current_stream=stream,
    )


def _scheduler(*, inference_stream) -> BaseExecutionScheduler:
    scheduler = object.__new__(BaseExecutionScheduler)
    scheduler._engine_lock = threading.Lock()
    scheduler._inference_stream = inference_stream
    scheduler._preprocess_readiness = PreprocessReadinessTracker()
    scheduler._thread_local_storage = threading.local()

    return scheduler


def test_base_buffer_strategy_preserves_engine_input_and_readiness() -> None:
    tensor = torch.zeros((1, 3, 4, 4))
    ready_event = object()
    result = PreprocessResult(
        tensor=tensor,
        metadata=[],
        implementation_id="test-preprocessor",
        ready_event=ready_event,
        input_kind="test",
    )

    engine_input_buffer = BaseBufferStrategy().prepare_engine_input(
        result=result,
        context=_context(),
    )

    assert engine_input_buffer.tensor is tensor
    assert engine_input_buffer.ready_event is ready_event
    assert engine_input_buffer.preprocessor_implementation_id == "test-preprocessor"


def test_base_scheduler_hands_readiness_to_exact_engine_input() -> None:
    calls = []
    inference_stream = _FakeStream(calls)
    scheduler = _scheduler(inference_stream=inference_stream)
    ready_event = _FakeEvent(calls)
    tensor = torch.zeros((1, 3, 4, 4))
    engine_input_buffer = EngineInputBuffer(
        tensor=tensor,
        ready_event=ready_event,
        input_kind="test",
        preprocessor_implementation_id="test-preprocessor",
    )

    scheduled_tensor = scheduler.finalize_preprocess(
        engine_input_buffer,
        context=_context(stream=_FakeStream(calls)),
        independent_stage_execution=False,
    )
    model_results = scheduler.execute_engine(
        scheduled_tensor,
        operation=lambda stream: (
            calls.append(("execute", stream)) or torch.zeros(1),
            torch.zeros(1),
        ),
    )

    assert scheduled_tensor is tensor
    assert model_results[0].shape == (1,)
    assert calls == [
        ("wait-event", ready_event),
        ("execute", inference_stream),
    ]


def test_base_scheduler_synchronizes_independent_preprocessing() -> None:
    calls = []
    scheduler = _scheduler(inference_stream=_FakeStream(calls))
    ready_event = _FakeEvent(calls)
    tensor = torch.zeros((1, 3, 4, 4))

    scheduled_tensor = scheduler.finalize_preprocess(
        EngineInputBuffer(
            tensor=tensor,
            ready_event=ready_event,
            input_kind="test",
            preprocessor_implementation_id="test-preprocessor",
        ),
        context=_context(stream=_FakeStream(calls)),
        independent_stage_execution=True,
    )

    assert scheduled_tensor is tensor
    assert calls == ["event-synchronize"]
    assert scheduler._preprocess_readiness.consume(tensor) is None


def test_base_scheduler_records_output_lifetime_and_synchronizes(monkeypatch) -> None:
    calls = []
    postprocess_stream = _FakeStream(calls)
    scheduler = _scheduler(inference_stream=_FakeStream(calls))
    scheduler._thread_local_storage.postprocess_stream = postprocess_stream
    monkeypatch.setattr(
        torch.cuda,
        "stream",
        lambda stream: nullcontext(stream),
    )
    model_results = (_FakeOutput(calls), _FakeOutput(calls))

    results = scheduler.execute_postprocess(
        model_results,
        operation=lambda stream: calls.append(("postprocess", stream)) or ["result"],
    )

    assert results == ["result"]
    assert calls == [
        ("record-stream", postprocess_stream),
        ("record-stream", postprocess_stream),
        ("postprocess", postprocess_stream),
        "stream-synchronize",
    ]


def test_base_engine_plugin_delegates_to_existing_trt_boundary(monkeypatch) -> None:
    calls = []

    def infer_from_trt_engine(**kwargs):
        calls.append(kwargs)

        return torch.zeros(1), torch.ones(1)

    monkeypatch.setitem(
        sys.modules,
        "inference_models.models.common.trt",
        SimpleNamespace(infer_from_trt_engine=infer_from_trt_engine),
    )
    stream = object()
    request = EngineExecutionRequest(
        pre_processed_images=torch.zeros((1, 3, 4, 4)),
        trt_config=object(),
        engine=object(),
        execution_context=object(),
        device=torch.device("cuda:0"),
        input_name="images",
        output_names=["dets", "labels"],
        trt_cuda_graph_cache=None,
    )

    detections, labels = BaseEngineAdjacentPlugin().execute(
        request=request,
        context=_context(stream=stream),
    )

    assert detections.item() == 0
    assert labels.item() == 1
    assert calls[0]["pre_processed_images"] is request.pre_processed_images
    assert calls[0]["stream"] is stream
    assert calls[0]["outputs"] == ["dets", "labels"]
