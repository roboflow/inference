import importlib
import json
import sys
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from inference_models.entities import ImageDimensions
from inference_models.errors import CorruptedModelPackageError, ModelRuntimeError
from inference_models.models.auto_loaders.entities import BackendType
from inference_models.models.auto_loaders.models_registry import (
    KEYPOINT_DETECTION_TASK,
    REGISTERED_MODELS,
    RegistryEntry,
)
from inference_models.models.common.roboflow.model_packages import (
    PreProcessingMetadata,
    StaticCropOffset,
)
from inference_models.models.rfdetr.class_remapping import ClassesReMapping

_MODEL_MODULE = "inference_models.models.rfdetr.rfdetr_key_points_detection_trt"
_TRT_DEPENDENCY_MODULES = (
    "inference_models.models.common.cuda",
    "inference_models.models.common.trt",
    _MODEL_MODULE,
)
_MISSING = object()

_INFERENCE_CONFIG = {
    "image_pre_processing": {"auto-orient": {"enabled": True}},
    "network_input": {
        "training_input_size": {"height": 560, "width": 560},
        "dynamic_spatial_size_supported": False,
        "color_mode": "rgb",
        "resize_mode": "stretch",
        "input_channels": 3,
    },
}
_KEYPOINTS_METADATA = [
    {
        "object_class": "person",
        "object_class_id": 0,
        "keypoints": {"0": "left", "1": "right"},
        "edges": [{"from": 0, "to": 1}],
    }
]


@pytest.fixture
def rfdetr_keypoint_trt_module(monkeypatch):
    """Import the TRT model with inert annotation-only dependency doubles.

    The model eagerly imports TensorRT and PyCUDA, while these tests must run in
    the CPU-only unit-test job. Real dependency coverage lives in the
    ``trt_extras`` integration suite.
    """

    class ILogger:
        class Severity:
            pass

    fake_trt = ModuleType("tensorrt")
    fake_trt.__spec__ = ModuleSpec("tensorrt", loader=None)
    fake_trt.ILogger = ILogger
    fake_trt.ICudaEngine = type("ICudaEngine", (), {})
    fake_trt.IExecutionContext = type("IExecutionContext", (), {})

    fake_cuda = ModuleType("pycuda.driver")
    fake_cuda.__spec__ = ModuleSpec("pycuda.driver", loader=None)
    fake_cuda.Context = type("Context", (), {})
    fake_cuda.Device = type("Device", (), {})
    fake_cuda.init = lambda: None
    fake_pycuda = ModuleType("pycuda")
    fake_pycuda.__spec__ = ModuleSpec(
        "pycuda",
        loader=None,
        is_package=True,
    )
    fake_pycuda.__path__ = []
    fake_pycuda.driver = fake_cuda

    previous_parent_attributes = {}
    for name in _TRT_DEPENDENCY_MODULES:
        parent_name, attribute = name.rsplit(".", 1)
        parent = importlib.import_module(parent_name)
        previous_parent_attributes[name] = (
            parent,
            attribute,
            getattr(parent, attribute, _MISSING),
        )
    previous_modules = {
        name: sys.modules.pop(name, None) for name in _TRT_DEPENDENCY_MODULES
    }
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)
    monkeypatch.setitem(sys.modules, "pycuda", fake_pycuda)
    monkeypatch.setitem(sys.modules, "pycuda.driver", fake_cuda)

    try:
        module = importlib.import_module(_MODEL_MODULE)
        yield module
    finally:
        for name in _TRT_DEPENDENCY_MODULES:
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


def _write_keypoint_trt_package(package_dir) -> None:
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "class_names.txt").write_text("person\n")
    (package_dir / "inference_config.json").write_text(json.dumps(_INFERENCE_CONFIG))
    (package_dir / "trt_config.json").write_text(json.dumps({"static_batch_size": 1}))
    (package_dir / "engine.plan").write_bytes(b"fake-engine")
    (package_dir / "keypoints_metadata.json").write_text(
        json.dumps(_KEYPOINTS_METADATA)
    )


def _preprocessing_meta(height: int = 64, width: int = 64) -> PreProcessingMetadata:
    size = ImageDimensions(height=height, width=width)
    return PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=size,
        size_after_pre_processing=size,
        inference_size=size,
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=width,
            crop_height=height,
        ),
    )


def test_rfdetr_keypoint_trt_is_registered() -> None:
    entry = REGISTERED_MODELS[("rfdetr", KEYPOINT_DETECTION_TASK, BackendType.TRT)]

    assert isinstance(entry, RegistryEntry)
    assert (
        entry.model_class._module_name
        == "inference_models.models.rfdetr.rfdetr_key_points_detection_trt"
    )
    assert entry.model_class._class_name == "RFDetrForKeyPointsTRT"
    assert entry.supported_model_features == {
        "resolution",
        "patch_size",
        "num_windows",
        "dec_layers",
        "num_queries",
        "num_select",
    }


def test_import_rfdetr_keypoint_trt_class(rfdetr_keypoint_trt_module) -> None:
    assert rfdetr_keypoint_trt_module.RFDetrForKeyPointsTRT.__name__ == (
        "RFDetrForKeyPointsTRT"
    )


def test_from_pretrained_rejects_non_cuda_device(rfdetr_keypoint_trt_module) -> None:
    with pytest.raises(ModelRuntimeError):
        rfdetr_keypoint_trt_module.RFDetrForKeyPointsTRT.from_pretrained(
            model_name_or_path="/tmp/missing-package",
            device=torch.device("cpu"),
        )


def test_from_pretrained_rejects_missing_package_files(
    rfdetr_keypoint_trt_module, tmp_path
) -> None:
    with pytest.raises(CorruptedModelPackageError):
        rfdetr_keypoint_trt_module.RFDetrForKeyPointsTRT.from_pretrained(
            model_name_or_path=str(tmp_path),
            device=torch.device("cuda"),
        )


def test_from_pretrained_rejects_wrong_output_count(
    rfdetr_keypoint_trt_module, tmp_path, monkeypatch
) -> None:
    _write_keypoint_trt_package(tmp_path)
    fake_engine = SimpleNamespace()
    fake_engine.create_execution_context = MagicMock(return_value=object())

    monkeypatch.setattr(rfdetr_keypoint_trt_module.cuda, "init", lambda: None)
    monkeypatch.setattr(
        rfdetr_keypoint_trt_module.cuda, "Device", lambda index: object()
    )
    monkeypatch.setattr(
        rfdetr_keypoint_trt_module,
        "use_primary_cuda_context",
        lambda cuda_device: _NullContext(object()),
    )
    monkeypatch.setattr(
        rfdetr_keypoint_trt_module,
        "load_trt_model",
        lambda model_path, engine_host_code_allowed=False: fake_engine,
    )
    monkeypatch.setattr(
        rfdetr_keypoint_trt_module,
        "get_trt_engine_inputs_and_outputs",
        lambda engine: (["images"], ["dets", "labels"]),
    )

    with pytest.raises(CorruptedModelPackageError, match="3 model outputs"):
        rfdetr_keypoint_trt_module.RFDetrForKeyPointsTRT.from_pretrained(
            model_name_or_path=str(tmp_path),
            device=torch.device("cuda"),
        )


def test_from_pretrained_rejects_wrong_input_count(
    rfdetr_keypoint_trt_module, tmp_path, monkeypatch
) -> None:
    _write_keypoint_trt_package(tmp_path)
    fake_engine = SimpleNamespace()
    fake_engine.create_execution_context = MagicMock(return_value=object())

    monkeypatch.setattr(rfdetr_keypoint_trt_module.cuda, "init", lambda: None)
    monkeypatch.setattr(
        rfdetr_keypoint_trt_module.cuda, "Device", lambda index: object()
    )
    monkeypatch.setattr(
        rfdetr_keypoint_trt_module,
        "use_primary_cuda_context",
        lambda cuda_device: _NullContext(object()),
    )
    monkeypatch.setattr(
        rfdetr_keypoint_trt_module,
        "load_trt_model",
        lambda model_path, engine_host_code_allowed=False: fake_engine,
    )
    monkeypatch.setattr(
        rfdetr_keypoint_trt_module,
        "get_trt_engine_inputs_and_outputs",
        lambda engine: (["images", "extra"], ["dets", "labels", "keypoints"]),
    )

    with pytest.raises(CorruptedModelPackageError, match="single model input"):
        rfdetr_keypoint_trt_module.RFDetrForKeyPointsTRT.from_pretrained(
            model_name_or_path=str(tmp_path),
            device=torch.device("cuda"),
        )


def test_resolve_outputs_by_export_names(rfdetr_keypoint_trt_module) -> None:
    names = rfdetr_keypoint_trt_module._resolve_rfdetr_keypoint_trt_output_names(
        engine=SimpleNamespace(),
        output_names=["keypoints", "dets", "labels"],
    )

    assert names == ["dets", "labels", "keypoints"]


def test_resolve_outputs_by_rank_and_shape(rfdetr_keypoint_trt_module) -> None:
    engine = SimpleNamespace(
        get_tensor_shape=lambda name: {
            "output0": (1, 300, 2),
            "output1": (1, 300, 4),
            "output2": (1, 300, 2, 8),
        }[name]
    )

    names = rfdetr_keypoint_trt_module._resolve_rfdetr_keypoint_trt_output_names(
        engine=engine,
        output_names=["output0", "output1", "output2"],
    )

    assert names == ["output1", "output0", "output2"]


def test_post_process_returns_keypoints(rfdetr_keypoint_trt_module) -> None:
    model = object.__new__(rfdetr_keypoint_trt_module.RFDetrForKeyPointsTRT)
    device = torch.device("cpu")
    model._class_names = ["person"]
    model._classes_re_mapping = ClassesReMapping(
        remaining_class_ids=torch.tensor([0], dtype=torch.int64, device=device),
        class_mapping=torch.tensor([0], dtype=torch.int64, device=device),
    )
    model._device = device
    model.recommended_parameters = None
    model._parsed_key_points_metadata = [["left", "right"]]
    model._key_points_classes_for_instances = torch.tensor([2], device=device)
    model._key_points_slots_in_prediction = 2

    bboxes = torch.tensor([[[0.5, 0.5, 0.4, 0.4]]], dtype=torch.float32)
    logits = torch.tensor([[[8.0]]], dtype=torch.float32)
    keypoints = torch.zeros((1, 1, 2, 8), dtype=torch.float32)
    keypoints[..., 0] = 0.4
    keypoints[..., 1] = 0.6
    keypoints[..., 2] = 5.0
    keypoints[..., 4] = 0.0
    keypoints[..., 6] = 0.0

    key_points_list, detections_list = model.post_process(
        model_results=(bboxes, logits, keypoints),
        pre_processing_meta=[_preprocessing_meta()],
        confidence=0.1,
        key_points_threshold=0.1,
    )

    assert len(key_points_list) == 1
    assert key_points_list[0].xy.shape[0] == 1
    assert key_points_list[0].xy.shape[1] == 2
    assert key_points_list[0].class_id.tolist() == [0]
    assert detections_list is not None
    assert len(detections_list) == 1
    assert detections_list[0].class_id.tolist() == [0]


class _NullContext:
    def __init__(self, value):
        self._value = value

    def __enter__(self):
        return self._value

    def __exit__(self, exc_type, exc, tb):
        return False
