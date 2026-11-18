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
_COORDINATION_TIMEOUT_SECONDS = 5.0
_JOIN_TIMEOUT_SECONDS = 15.0


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


class _InstrumentedLock:
    def __init__(self, on_contention):
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._attempts = 0
        self._owner = None
        self._generation = 0
        self._on_contention = on_contention

    def held_by_current_thread(self) -> bool:
        return self._owner == threading.get_ident()

    def held_generation(self) -> int:
        return self._generation

    def __enter__(self):
        with self._state_lock:
            self._attempts += 1
            if self._attempts > 1:
                self._on_contention()
        self._lock.acquire()
        self._owner = threading.get_ident()
        self._generation += 1
        return self

    def __exit__(self, *args):
        self._owner = None
        self._lock.release()
        return False


class _ConcurrencyTracker:
    def __init__(self):
        self._state_lock = threading.Lock()
        self._inside = 0
        self.max_inside = 0
        self.second_arrival = threading.Event()
        self.lock = _InstrumentedLock(on_contention=self.second_arrival.set)
        self.unguarded_calls = []
        self._runtime_generations = {}

    def check_lock_held(self, call_name: str) -> None:
        if not self.lock.held_by_current_thread():
            self.unguarded_calls.append(call_name)

    def check_same_acquisition(self) -> None:
        self.check_lock_held("deserialize_cuda_engine")
        created_in = self._runtime_generations.get(threading.get_ident())
        if created_in != self.lock.held_generation():
            self.unguarded_calls.append("deserialize_cuda_engine_split_lock")

    def enter(self) -> None:
        self.check_lock_held("Runtime")
        self._runtime_generations[threading.get_ident()] = self.lock.held_generation()
        with self._state_lock:
            self._inside += 1
            self.max_inside = max(self.max_inside, self._inside)
            if self._inside > 1:
                self.second_arrival.set()
        self.second_arrival.wait(timeout=_COORDINATION_TIMEOUT_SECONDS)

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
            tracker.check_same_acquisition()
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
        trt_module = importlib.import_module("inference_models.models.common.trt")
        monkeypatch.setattr(trt_module, "_TRT_RUNTIME_CREATION_LOCK", tracker.lock)
        yield trt_module, tracker
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
    assert tracker.unguarded_calls == []
    assert tracker.max_inside == 1
