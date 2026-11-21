import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from inference_models.errors import CorruptedModelPackageError


class _ILogger:
    class Severity:
        pass


class _Logger(_ILogger):
    VERBOSE = "VERBOSE"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

    def __init__(self, min_severity=None):
        self.min_severity = min_severity


class _Runtime:
    created_with = []
    engine = object()

    def __init__(self, logger):
        type(self).created_with.append(logger)
        self.engine_host_code_allowed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def deserialize_cuda_engine(self, data):
        return type(self).engine


@pytest.fixture
def trt_module(monkeypatch):
    runtime_class = type(
        "_FreshRuntime", (_Runtime,), {"created_with": [], "engine": object()}
    )
    fake_trt = ModuleType("tensorrt")
    fake_trt.ILogger = _ILogger
    fake_trt.Logger = _Logger
    fake_trt.Runtime = runtime_class
    fake_trt.ICudaEngine = type("ICudaEngine", (), {})
    fake_trt.IExecutionContext = type("IExecutionContext", (), {})
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
    module_name = "_trt_native_logger_test_runtime"
    spec = importlib.util.spec_from_file_location(module_name, source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)

    return module, fake_trt


@pytest.fixture
def engine_file(tmp_path):
    path = tmp_path / "engine.plan"
    path.write_bytes(b"engine")

    return str(path)


def test_load_trt_model_uses_native_logger(trt_module, engine_file) -> None:
    # given
    module, fake_trt = trt_module

    # when
    engine = module.load_trt_model(engine_file)

    # then
    assert engine is fake_trt.Runtime.engine
    (logger,) = fake_trt.Runtime.created_with
    assert type(logger) is fake_trt.Logger
    assert not isinstance(logger, module.InferenceTRTLogger)


def test_load_trt_model_reuses_logger_across_loads(trt_module, engine_file) -> None:
    # given
    module, fake_trt = trt_module

    # when
    module.load_trt_model(engine_file)
    module.load_trt_model(engine_file)

    # then
    first_logger, second_logger = fake_trt.Runtime.created_with
    assert first_logger is second_logger


def test_load_trt_model_raises_when_engine_cannot_be_deserialized(
    trt_module, engine_file
) -> None:
    # given
    module, fake_trt = trt_module
    fake_trt.Runtime.engine = None

    # when
    with pytest.raises(CorruptedModelPackageError):
        module.load_trt_model(engine_file)


def test_trt_module_imports_without_native_logger_class(monkeypatch) -> None:
    # given
    fake_trt = ModuleType("tensorrt")
    fake_trt.ILogger = _ILogger
    fake_trt.ICudaEngine = type("ICudaEngine", (), {})
    fake_trt.IExecutionContext = type("IExecutionContext", (), {})
    fake_pycuda = ModuleType("pycuda")
    fake_pycuda.__path__ = []
    fake_pycuda.driver = ModuleType("pycuda.driver")
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)
    monkeypatch.setitem(sys.modules, "pycuda", fake_pycuda)
    monkeypatch.setitem(sys.modules, "pycuda.driver", fake_pycuda.driver)

    import inference_models.models.common

    source = Path(inference_models.models.common.__file__).parent / "trt.py"
    module_name = "_trt_native_logger_import_test_runtime"
    spec = importlib.util.spec_from_file_location(module_name, source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)

    # when
    spec.loader.exec_module(module)

    # then
    assert hasattr(module, "load_trt_model")
