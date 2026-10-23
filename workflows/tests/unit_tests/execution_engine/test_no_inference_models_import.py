import importlib
import sys


def test_core_module_does_not_import_performance_profiler() -> None:
    module_name = "roboflow_workflows.execution_engine.v1.executor.core"
    sys.modules.pop(module_name, None)
    sys.modules.pop("inference_models.utils.performance", None)

    importlib.import_module(module_name)

    assert "inference_models.utils.performance" not in sys.modules
    assert "inference_models" in sys.modules
