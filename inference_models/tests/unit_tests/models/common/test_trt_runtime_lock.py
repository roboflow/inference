import importlib
import sys
import threading
from importlib.machinery import ModuleSpec
from types import ModuleType

import pytest

_ISOLATED_MODULES = [
    "inference_models.models.common.cuda",
    "inference_models.models.common.trt",
]
_OVERLAP_WAIT_SECONDS = 0.3
_JOIN_TIMEOUT_SECONDS = 10.0


class _FakeSeverity:
    pass


class _FakeLogger:
    Severity = _FakeSeverity
    VERBOSE = "VERBOSE"
    INFO = "INFO"
    WARNING = "WARNING"


class _FakeILogger:
    Severity = _FakeSeverity

    def __init__(self):
        pass


class _FakeEngine:
    pass


class _ConcurrencyTracker:
    def __init__(self):
        self._state_lock = threading.Lock()
        self._inside = 0
        self.max_inside = 0
        self.overlap_observed = threading.Event()

    def enter(self) -> None:
        with self._state_lock:
            self._inside += 1
            self.max_inside = max(self.max_inside, self._inside)
            if self._inside > 1:
                self.overlap_observed.set()
        self.overlap_observed.wait(timeout=_OVERLAP_WAIT_SECONDS)

    def leave(self) -> None:
        with self._state_lock:
            self._inside -= 1


def _build_fake_tensorrt(tracker: _ConcurrencyTracker) -> ModuleType:
    class FakeRuntime:
        def __init__(self, logger):
            tracker.enter()
            self.engine_host_code_allowed = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            tracker.leave()
            return False

        def deserialize_cuda_engine(self, data):
            return _FakeEngine()

    module = ModuleType("tensorrt")
    module.__spec__ = ModuleSpec("tensorrt", loader=None)
    module.ILogger = _FakeILogger
    module.Logger = _FakeLogger
    module.IExecutionContext = type("IExecutionContext", (), {})
    module.ICudaEngine = type("ICudaEngine", (), {})
    module.Runtime = FakeRuntime
    return module


def _fake_pycuda_modules() -> dict:
    fake_cuda = ModuleType("pycuda.driver")
    fake_cuda.__spec__ = ModuleSpec("pycuda.driver", loader=None)
    fake_cuda.Context = type("Context", (), {})
    fake_cuda.Device = type("Device", (), {})
    fake_cuda.init = lambda: None
    fake_pycuda = ModuleType("pycuda")
    fake_pycuda.__spec__ = ModuleSpec("pycuda", loader=None, is_package=True)
    fake_pycuda.__path__ = []
    fake_pycuda.driver = fake_cuda
    return {"pycuda": fake_pycuda, "pycuda.driver": fake_cuda}


@pytest.fixture
def trt_module_with_fake_tensorrt(monkeypatch):
    tracker = _ConcurrencyTracker()
    monkeypatch.setitem(sys.modules, "tensorrt", _build_fake_tensorrt(tracker))
    for name, module in _fake_pycuda_modules().items():
        monkeypatch.setitem(sys.modules, name, module)
    previous_modules = {name: sys.modules.pop(name, None) for name in _ISOLATED_MODULES}
    parents = {
        name: importlib.import_module(name.rsplit(".", 1)[0])
        for name in _ISOLATED_MODULES
    }
    previous_attributes = {
        name: getattr(parents[name], name.rsplit(".", 1)[1], None)
        for name in _ISOLATED_MODULES
    }
    try:
        yield importlib.import_module("inference_models.models.common.trt"), tracker
    finally:
        for name in _ISOLATED_MODULES:
            sys.modules.pop(name, None)
            if previous_modules[name] is not None:
                sys.modules[name] = previous_modules[name]
            attribute = name.rsplit(".", 1)[1]
            if previous_attributes[name] is not None:
                setattr(parents[name], attribute, previous_attributes[name])
            elif hasattr(parents[name], attribute):
                delattr(parents[name], attribute)


def test_load_trt_model_never_creates_runtimes_concurrently(
    trt_module_with_fake_tensorrt, tmp_path
) -> None:
    # given
    trt_module, tracker = trt_module_with_fake_tensorrt
    engine_path = tmp_path / "model.plan"
    engine_path.write_bytes(b"engine")
    errors = []

    def load() -> None:
        try:
            trt_module.load_trt_model(str(engine_path))
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=load, daemon=True) for _ in range(2)]

    # when
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=_JOIN_TIMEOUT_SECONDS)

    # then
    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert tracker.max_inside == 1
