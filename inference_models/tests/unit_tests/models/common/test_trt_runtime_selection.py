import importlib
import sys
from importlib.machinery import ModuleSpec
from types import ModuleType

import pytest

_TRT_MODULES = [
    "inference_models.models.common.trt",
    "inference_models.models.deep_lab_v3_plus.deep_lab_v3_plus_segmentation_trt",
    "inference_models.models.resnet.resnet_classification_trt",
    "inference_models.models.rfdetr.rfdetr_instance_segmentation_trt",
    "inference_models.models.rfdetr.rfdetr_key_points_detection_trt",
    "inference_models.models.rfdetr.rfdetr_object_detection_trt",
    "inference_models.models.rfdetr.rfdetr_semantic_segmentation_trt",
    "inference_models.models.vit.vit_classification_trt",
    "inference_models.models.yolact.yolact_instance_segmentation_trt",
    "inference_models.models.yolo26.yolo26_depth_estimation_trt",
    "inference_models.models.yolo26.yolo26_instance_segmentation_trt",
    "inference_models.models.yolo26.yolo26_key_points_detection_trt",
    "inference_models.models.yolo26.yolo26_object_detection_trt",
    "inference_models.models.yolo26.yolo26_semantic_segmentation_trt",
    "inference_models.models.yolonas.yolonas_object_detection_trt",
    "inference_models.models.yolov10.yolov10_object_detection_trt",
    "inference_models.models.yolov5.yolov5_instance_segmentation_trt",
    "inference_models.models.yolov5.yolov5_object_detection_trt",
    "inference_models.models.yolov7.yolov7_instance_segmentation_trt",
    "inference_models.models.yolov8.yolov8_instance_segmentation_trt",
    "inference_models.models.yolov8.yolov8_key_points_detection_trt",
    "inference_models.models.yolov8.yolov8_object_detection_trt",
]
_ISOLATED_MODULES = ["inference_models.models.common.cuda", *_TRT_MODULES]
_MISSING = object()


class _FakeTensorRT(ModuleType):
    def __getattr__(self, name: str) -> type:
        if name.startswith("__"):
            raise AttributeError(name)
        value = type(name, (), {"Severity": type("Severity", (), {})})
        setattr(self, name, value)
        return value


def _fake_module(name: str) -> ModuleType:
    module = _FakeTensorRT(name)
    module.__spec__ = ModuleSpec(name, loader=None)
    return module


@pytest.fixture
def isolated_trt_imports(monkeypatch):
    fake_cuda = ModuleType("pycuda.driver")
    fake_cuda.__spec__ = ModuleSpec("pycuda.driver", loader=None)
    fake_cuda.Context = type("Context", (), {})
    fake_cuda.Device = type("Device", (), {})
    fake_cuda.init = lambda: None
    fake_pycuda = ModuleType("pycuda")
    fake_pycuda.__spec__ = ModuleSpec("pycuda", loader=None, is_package=True)
    fake_pycuda.__path__ = []
    fake_pycuda.driver = fake_cuda
    monkeypatch.setitem(sys.modules, "pycuda", fake_pycuda)
    monkeypatch.setitem(sys.modules, "pycuda.driver", fake_cuda)

    previous_parent_attributes = {}
    for name in _ISOLATED_MODULES:
        parent_name, attribute = name.rsplit(".", 1)
        parent = importlib.import_module(parent_name)
        previous_parent_attributes[name] = (
            parent,
            attribute,
            getattr(parent, attribute, _MISSING),
        )
    previous_modules = {name: sys.modules.pop(name, None) for name in _ISOLATED_MODULES}

    try:
        yield monkeypatch
    finally:
        for name in _ISOLATED_MODULES:
            sys.modules.pop(name, None)
            previous = previous_modules[name]
            if previous is not None:
                sys.modules[name] = previous
            parent, attribute, previous_attribute = previous_parent_attributes[name]
            if previous_attribute is _MISSING:
                if hasattr(parent, attribute):
                    delattr(parent, attribute)
            else:
                setattr(parent, attribute, previous_attribute)


@pytest.mark.parametrize("module_name", _TRT_MODULES)
def test_trt_module_selects_full_runtime_when_both_runtimes_are_installed(
    isolated_trt_imports, module_name: str
) -> None:
    # given
    full_runtime = _fake_module("tensorrt")
    lean_runtime = _fake_module("tensorrt_lean")
    isolated_trt_imports.setitem(sys.modules, "tensorrt", full_runtime)
    isolated_trt_imports.setitem(sys.modules, "tensorrt_lean", lean_runtime)

    # when
    module = importlib.import_module(module_name)

    # then
    assert module.trt is full_runtime


@pytest.mark.parametrize("module_name", _TRT_MODULES)
def test_trt_module_falls_back_to_lean_runtime_when_full_runtime_is_missing(
    isolated_trt_imports, module_name: str
) -> None:
    # given
    lean_runtime = _fake_module("tensorrt_lean")
    isolated_trt_imports.setitem(sys.modules, "tensorrt", None)
    isolated_trt_imports.setitem(sys.modules, "tensorrt_lean", lean_runtime)

    # when
    module = importlib.import_module(module_name)

    # then
    assert module.trt is lean_runtime
