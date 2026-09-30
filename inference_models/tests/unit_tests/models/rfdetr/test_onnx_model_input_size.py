import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip(
    "onnxruntime",
    reason="onnxruntime is not installed (requires the onnx-* extra)",
)

from inference_models.errors import ModelPackageRestrictedError
from inference_models.models.rfdetr import (
    rfdetr_instance_segmentation_onnx,
    rfdetr_key_points_detection_onnx,
    rfdetr_object_detection_onnx,
)

LOADERS = [
    pytest.param(
        rfdetr_object_detection_onnx,
        rfdetr_object_detection_onnx.RFDetrForObjectDetectionONNX,
        id="object-detection",
    ),
    pytest.param(
        rfdetr_instance_segmentation_onnx,
        rfdetr_instance_segmentation_onnx.RFDetrForInstanceSegmentationOnnx,
        id="instance-segmentation",
    ),
    pytest.param(
        rfdetr_key_points_detection_onnx,
        rfdetr_key_points_detection_onnx.RFDetrForKeyPointsONNX,
        id="keypoints",
    ),
]


def _package(tmp_path, configured_size: int) -> str:
    (tmp_path / "class_names.txt").write_text("egg\n")
    (tmp_path / "weights.onnx").write_bytes(b"not used")
    (tmp_path / "keypoints_metadata.json").write_text(
        json.dumps(
            [
                {
                    "object_class_id": 0,
                    "keypoints": {"0": "top", "1": "bottom"},
                    "edges": [{"from": 0, "to": 1}],
                }
            ]
        )
    )
    (tmp_path / "inference_config.json").write_text(
        json.dumps(
            {
                "network_input": {
                    "training_input_size": {
                        "height": configured_size,
                        "width": configured_size,
                    },
                    "dynamic_spatial_size_supported": False,
                    "color_mode": "rgb",
                    "resize_mode": "stretch",
                    "input_channels": 3,
                    "scaling_factor": 255,
                    "normalization": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
                }
            }
        )
    )
    return str(tmp_path)


def _stub_session(monkeypatch, module, input_shape: list) -> None:
    session = Mock()
    session.get_inputs.return_value = [SimpleNamespace(shape=input_shape, name="input")]
    monkeypatch.setattr(
        module.onnxruntime, "InferenceSession", Mock(return_value=session)
    )
    monkeypatch.setattr(
        module,
        "align_device_with_onnx_session",
        lambda **kwargs: torch.device("cpu"),
    )
    monkeypatch.setattr(
        module,
        "set_onnx_execution_provider_defaults",
        lambda **kwargs: kwargs["providers"],
    )


def _load(model_class, package_dir: str, **kwargs):
    return model_class.from_pretrained(
        package_dir,
        device=torch.device("cpu"),
        onnx_execution_providers=["CPUExecutionProvider"],
        **kwargs,
    )


def _training_input_size(model) -> tuple:
    size = model._inference_config.network_input.training_input_size
    return size.height, size.width


@pytest.mark.parametrize("module, model_class", LOADERS)
def test_onnx_loader_uses_the_model_input_size_over_the_config(
    tmp_path, monkeypatch, module, model_class
) -> None:
    _stub_session(monkeypatch, module, input_shape=[1, 3, 384, 384])

    model = _load(model_class, _package(tmp_path, configured_size=640))

    assert _training_input_size(model) == (384, 384)


@pytest.mark.parametrize("module, model_class", LOADERS)
def test_onnx_loader_keeps_a_matching_config(
    tmp_path, monkeypatch, module, model_class
) -> None:
    _stub_session(monkeypatch, module, input_shape=[1, 3, 384, 384])

    model = _load(model_class, _package(tmp_path, configured_size=384))

    assert _training_input_size(model) == (384, 384)


@pytest.mark.parametrize("module, model_class", LOADERS)
def test_onnx_loader_keeps_the_config_for_dynamic_inputs(
    tmp_path, monkeypatch, module, model_class
) -> None:
    _stub_session(monkeypatch, module, input_shape=["batch", 3, "height", "width"])

    model = _load(model_class, _package(tmp_path, configured_size=640))

    assert _training_input_size(model) == (640, 640)


@pytest.mark.parametrize("module, model_class", LOADERS)
def test_onnx_loader_limits_the_model_input_size_not_the_config(
    tmp_path, monkeypatch, module, model_class
) -> None:
    _stub_session(monkeypatch, module, input_shape=[1, 3, 384, 384])

    model = _load(
        model_class,
        _package(tmp_path, configured_size=640),
        rf_detr_max_input_resolution=512,
    )

    assert _training_input_size(model) == (384, 384)


@pytest.mark.parametrize("module, model_class", LOADERS)
def test_onnx_loader_rejects_a_model_input_over_the_limit(
    tmp_path, monkeypatch, module, model_class
) -> None:
    _stub_session(monkeypatch, module, input_shape=[1, 3, 1024, 1024])

    with pytest.raises(ModelPackageRestrictedError):
        _load(
            model_class,
            _package(tmp_path, configured_size=384),
            rf_detr_max_input_resolution=800,
        )
