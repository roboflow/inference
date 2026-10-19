"""Exercise the production graph helper with real CUDA streams and graphs.

Only TensorRT's metadata/enqueue interface is doubled: its enqueue performs a
real Torch CUDA multiply into the addresses registered by the production helper.
This isolates stream ordering from TensorRT installation and model downloads;
it does not qualify TensorRT engine numerics or Jetson behavior.
"""

import importlib.util
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from inference_models.models.optimization.contracts import ExecutionContext
from inference_models.models.rfdetr.optimization.contracts import EngineInputBuffer
from inference_models.models.rfdetr.optimization.readiness import (
    PreprocessReadinessTracker,
)
from inference_models.models.rfdetr.optimization.schedulers import (
    BaseExecutionScheduler,
)

pytestmark = [pytest.mark.gpu_only, pytest.mark.torch_models]


class _Context:
    def __init__(self, engine):
        self.engine = engine
        self.addresses = {}
        self.shape = None

    def set_input_shape(self, name, shape):
        self.shape = shape
        return True

    def set_tensor_address(self, name, address):
        self.addresses[name] = address
        return True

    def get_tensor_shape(self, name):
        return self.shape

    def execute_async_v3(self, *, stream_handle):
        assert torch.cuda.current_stream().cuda_stream == stream_handle
        source = self.engine.tensors[self.addresses["images"]]
        for name in ("dets", "labels"):
            destination = self.engine.tensors[self.addresses[name]]
            torch.mul(source, 2, out=destination)
        return True


class _Engine:
    def __init__(self):
        self.tensors = {}

    def create_execution_context(self):
        context = _Context(self)
        return context

    def get_tensor_dtype(self, name):
        return "float"


@pytest.fixture
def graph_runtime(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA GPU is required for stream-ordering coverage")

    class _Logger:
        class Severity:
            pass

    fake_trt = ModuleType("tensorrt")
    fake_trt.ILogger = _Logger
    fake_trt.ICudaEngine = _Engine
    fake_trt.IExecutionContext = _Context
    fake_trt.DataType = SimpleNamespace(
        FLOAT="float", HALF="half", INT32="int32", INT8="int8", BOOL="bool"
    )
    fake_pycuda = ModuleType("pycuda")
    fake_pycuda.__path__ = []
    fake_pycuda.driver = ModuleType("pycuda.driver")
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)
    monkeypatch.setitem(sys.modules, "pycuda", fake_pycuda)
    monkeypatch.setitem(sys.modules, "pycuda.driver", fake_pycuda.driver)

    import inference_models.models.common

    source = Path(inference_models.models.common.__file__).parent / "trt.py"
    module_name = "_trt_graph_readiness_test_runtime"
    spec = importlib.util.spec_from_file_location(module_name, source)
    runtime = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, runtime)
    spec.loader.exec_module(runtime)

    engine = _Engine()
    original_empty = torch.empty
    original_empty_like = torch.empty_like

    def tracked_empty(*args, **kwargs):
        tensor = original_empty(*args, **kwargs)
        engine.tensors[tensor.data_ptr()] = tensor
        return tensor

    def tracked_empty_like(*args, **kwargs):
        tensor = original_empty_like(*args, **kwargs)
        engine.tensors[tensor.data_ptr()] = tensor
        return tensor

    monkeypatch.setattr(torch, "empty", tracked_empty)
    monkeypatch.setattr(torch, "empty_like", tracked_empty_like)
    # Initialize CUDA and the multiply kernel before introducing the race.
    warmup = torch.ones(16, device="cuda")
    torch.mul(warmup, 2, out=warmup)
    torch.cuda._sleep(1)
    torch.cuda.synchronize()

    yield runtime, engine

    torch.cuda.synchronize()


def _scheduler(*, stream):
    scheduler = object.__new__(BaseExecutionScheduler)
    scheduler._engine_lock = threading.Lock()
    scheduler._inference_stream = stream
    scheduler._preprocess_readiness = PreprocessReadinessTracker()
    scheduler._thread_local_storage = threading.local()
    return scheduler


def _infer(runtime, engine, tensor, *, stream, cache, synchronize):
    results = runtime.infer_from_trt_engine(
        pre_processed_images=tensor,
        trt_config=SimpleNamespace(static_batch_size=1),
        engine=engine,
        context=engine.create_execution_context(),
        device=tensor.device,
        input_name="images",
        outputs=["dets", "labels"],
        stream=stream,
        trt_cuda_graph_cache=cache,
        synchronize=synchronize,
    )
    return results


@pytest.mark.parametrize("warm_cache", [False, True], ids=["capture", "replay"])
@pytest.mark.parametrize("synchronize", [False, True], ids=["async", "sync"])
@pytest.mark.parametrize("readiness", ["tracker", "legacy_event"])
@pytest.mark.parametrize("reuse_input", [False, True], ids=["copy", "external-input"])
def test_graph_waits_for_asynchronous_input(
    graph_runtime, warm_cache, synchronize, readiness, reuse_input
):
    runtime, engine = graph_runtime
    caller_stream = torch.cuda.Stream()
    producer_stream = torch.cuda.Stream()
    scheduler = _scheduler(stream=caller_stream)
    tensor = torch.full((1, 3, 4, 4), -113.0, device="cuda")
    tensor._trt_reuse_as_input_buffer = reuse_input
    engine.tensors[tensor.data_ptr()] = tensor
    cache = runtime.TRTCudaGraphCache(capacity=1)
    torch.cuda.synchronize()

    if warm_cache:
        _infer(
            runtime, engine, tensor, stream=caller_stream, cache=cache, synchronize=True
        )

    with torch.cuda.stream(producer_stream):
        # A single sleeping GPU thread leaves the device free to execute an
        # incorrectly unordered copy/replay. No CPU sleep or stream mock is used.
        torch.cuda._sleep(1_000_000_000)
        tensor.fill_(37.0)
        ready_event = torch.cuda.Event()
        ready_event.record(producer_stream)
    assert not ready_event.query(), "The delayed producer completed before the probe"

    if readiness == "tracker":
        scheduled_tensor = scheduler.finalize_preprocess(
            EngineInputBuffer(
                tensor=tensor,
                ready_event=ready_event,
                input_kind="test",
                preprocessor_implementation_id="delayed-cuda-producer",
            ),
            context=ExecutionContext(
                device_kind="gpu", device="cuda:0", current_stream=producer_stream
            ),
            independent_stage_execution=False,
        )
        assert not hasattr(scheduled_tensor, "_trt_ready_event")
    else:
        tensor._trt_ready_event = ready_event
        scheduled_tensor = tensor

    results = scheduler.execute_engine(
        scheduled_tensor,
        operation=lambda stream: _infer(
            runtime,
            engine,
            scheduled_tensor,
            stream=stream,
            cache=cache,
            synchronize=synchronize,
        ),
    )
    for result in results:
        result._trt_produce_event.synchronize()
        torch.testing.assert_close(result, torch.full_like(result, 74.0))
    assert cache.get_current_size() == 1
    if warm_cache:
        assert tensor._trt_consumed_event.query()


def test_graph_disabled_preserves_scheduler_readiness(graph_runtime):
    runtime, engine = graph_runtime
    caller_stream = torch.cuda.Stream()
    producer_stream = torch.cuda.Stream()
    scheduler = _scheduler(stream=caller_stream)
    tensor = torch.full((1, 3, 4, 4), -113.0, device="cuda")
    engine.tensors[tensor.data_ptr()] = tensor
    torch.cuda.synchronize()
    with torch.cuda.stream(producer_stream):
        torch.cuda._sleep(1_000_000_000)
        tensor.fill_(37.0)
        ready_event = torch.cuda.Event()
        ready_event.record(producer_stream)
    assert not ready_event.query()
    scheduled_tensor = scheduler.finalize_preprocess(
        EngineInputBuffer(
            tensor=tensor,
            ready_event=ready_event,
            input_kind="test",
            preprocessor_implementation_id="delayed-cuda-producer",
        ),
        context=ExecutionContext(
            device_kind="gpu", device="cuda:0", current_stream=producer_stream
        ),
        independent_stage_execution=False,
    )
    results = scheduler.execute_engine(
        scheduled_tensor,
        operation=lambda stream: _infer(
            runtime,
            engine,
            scheduled_tensor,
            stream=stream,
            cache=None,
            synchronize=True,
        ),
    )
    for result in results:
        torch.testing.assert_close(result, torch.full_like(result, 74.0))
