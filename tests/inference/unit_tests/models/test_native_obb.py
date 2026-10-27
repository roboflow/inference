"""Execute an exported ONNX artifact through native model and request consumers."""

import json
import math
import os
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper

from inference_models.models.rfdetr.rfdetr_oriented_object_detection_onnx import (
    RFDetrForOrientedObjectDetectionONNX,
    _corners,
)


def metadata(size=64):
    return dict(
        schemaVersion=1,
        taskType="oriented-object-detection",
        annotationFormat="oriented-box-v1",
        transformPolicy="obb-front-axis-v1",
        classHeading={"plane": True, "stone": False},
        classes=[
            dict(categoryId=7, outputIndex=1, name="plane", heading=True),
            dict(categoryId=9, outputIndex=2, name="stone", heading=False),
        ],
        versionSchema={},
        resolvedRecipe={},
        modelRevision="a" * 40,
        preprocessing=dict(
            type="rfdetr-obb-letterbox-v1",
            inputSize=size,
            paddingValue=114,
            scale="uniform-max-side",
            paddingOffset="continuous-centered",
            interpolation="pillow-bilinear-affine",
            angle="unchanged",
            colorMode="rgb",
            scalingFactor=255,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    )


def package(tmp_path, *, side=0, box=(0.5, 0.5, 0.4, 0.2, 0.3), malformed=None):
    dets = np.array([[box]], np.float32)
    labels = np.array([[[100.0, 12.0, -12.0]]], np.float32)
    heading = np.full((1, 1, 4), -10.0, np.float32)
    heading[0, 0, side] = 10.0
    values = {"dets": dets, "labels": labels, "heading": heading}
    if malformed:
        values[malformed][...] = np.nan
    nodes = [
        helper.make_node("Constant", [], [name], value=numpy_helper.from_array(v))
        for name, v in values.items()
    ]
    graph = helper.make_graph(
        nodes,
        "native-obb",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])],
        [
            helper.make_tensor_value_info(n, TensorProto.FLOAT, v.shape)
            for n, v in values.items()
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9
    )
    onnx.save(model, tmp_path / "weights.onnx")
    (tmp_path / "obb_metadata.json").write_text(json.dumps(metadata()))
    (tmp_path / "inference_config.json").write_text(
        json.dumps({"image_pre_processing": {}})
    )
    return RFDetrForOrientedObjectDetectionONNX.from_pretrained(str(tmp_path))


@pytest.mark.parametrize("side", range(4))
@pytest.mark.parametrize("wh", [(0.4, 0.2), (0.2, 0.4), (0.3, 0.3)])
def test_all_front_sides_preserve_footprint_and_directed_squares(tmp_path, side, wh):
    model = package(tmp_path, side=side, box=(0.5, 0.5, *wh, 0.3))
    result = model(np.zeros((35, 73, 3), np.uint8), confidence=0.5)[0]
    assert result.class_id.tolist() == [
        7
    ]  # category identity, never background output0
    assert result.class_names == ["plane"]
    assert result.heading.tolist() == [True]
    assert result.detection_id == ["q0:c7"]
    scale = 64 / 73
    expected = torch.tensor([[36.5, 17.5, wh[0] * 64 / scale, wh[1] * 64 / scale, 0.3]])
    actual = _corners(result.obb)[0].numpy()
    reference = _corners(expected)[0].numpy()
    np.testing.assert_allclose(
        sorted(map(tuple, actual)), sorted(map(tuple, reference)), atol=1e-5
    )
    expected_angle = (0.3 + side * math.pi / 2 + math.pi) % (2 * math.pi) - math.pi
    assert result.obb[0, 4].item() == pytest.approx(expected_angle, abs=1e-6)
    assert result.obb[0, 2:4].tolist() == pytest.approx(
        [wh[side % 2] * 73, wh[1 - side % 2] * 73]
    )
    assert result.obb[0, :2].tolist() == pytest.approx([36.5, 17.5])
    assert result.xyxy.dtype.is_floating_point


def test_realized_preprocessing_matches_model_similarity_on_odd_image(tmp_path):
    model = package(tmp_path)
    image = np.zeros((35, 73, 3), np.uint8)
    image[17:19, 35:38] = 255
    arrays, metas = model.pre_process([image])
    meta = metas[0]
    assert meta.scale_width == meta.scale_height == 64 / 73
    assert meta.pad_top == pytest.approx((64 - 35 * 64 / 73) / 2)
    assert arrays.shape == (1, 3, 64, 64)
    assert arrays.dtype == np.float32
    assert (
        model([image, image], confidence=0.5)[1].obb.tolist()
        == model(image, confidence=0.5)[0].obb.tolist()
    )


@pytest.mark.parametrize("name", ["dets", "labels", "heading"])
def test_nonfinite_predictions_are_rejected(tmp_path, name):
    model = package(tmp_path, malformed=name)
    with pytest.raises(ValueError, match="finite"):
        model(np.zeros((64, 64, 3), np.uint8))


@pytest.mark.parametrize(
    "field", ["classHeading", "classes", "modelRevision", "preprocessing"]
)
def test_missing_frozen_metadata_rejects(tmp_path, field):
    package(tmp_path)
    value = metadata()
    del value[field]
    (tmp_path / "obb_metadata.json").write_text(json.dumps(value))
    with pytest.raises(ValueError):
        RFDetrForOrientedObjectDetectionONNX.from_pretrained(str(tmp_path))


def test_empty_confidence_result_keeps_geometry_rank(tmp_path):
    result = package(tmp_path)(np.zeros((64, 64, 3), np.uint8), confidence=1.0)[0]
    assert result.obb.shape == (0, 5)
    assert result.to_supervision().data["xyxyxyxy"].shape == (0, 4, 2)


def test_optional_actual_exported_model(tmp_path):
    fixture = os.environ.get("OBB_EXPORTED_ONNX")
    if fixture is None:
        pytest.skip("Set OBB_EXPORTED_ONNX to the actual RF-DETR export fixture")
    (tmp_path / "weights.onnx").symlink_to(Path(fixture))
    (tmp_path / "obb_metadata.json").write_text(json.dumps(metadata()))
    (tmp_path / "inference_config.json").write_text(
        json.dumps({"image_pre_processing": {}})
    )
    model = RFDetrForOrientedObjectDetectionONNX.from_pretrained(str(tmp_path))
    result = model(np.zeros((35, 73, 3), np.uint8), confidence=0, max_detections=4)[0]
    assert result.obb.shape == (4, 5)
    assert torch.isfinite(result.obb).all()
    assert set(result.class_id.tolist()) <= {7, 9}
    assert len(set(result.detection_id)) == 4


def test_auto_loader_and_request_consumer(tmp_path, monkeypatch):
    package(tmp_path, side=3)
    (tmp_path / "model_config.json").write_text(
        json.dumps(
            {
                "model_architecture": "rfdetr",
                "task_type": "oriented-object-detection",
                "backend_type": "onnx",
            }
        )
    )
    from inference_models import AutoModel

    loaded = AutoModel.from_pretrained(
        str(tmp_path), allow_direct_local_storage_loading=True
    )
    assert isinstance(loaded, RFDetrForOrientedObjectDetectionONNX)
    from inference.core.entities.requests.inference import (
        InferenceRequestImage,
        OrientedObjectDetectionInferenceRequest,
    )
    from inference.core.models import inference_models_adapters as adapters

    monkeypatch.setattr(
        adapters, "ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES", True
    )
    adapter = adapters.InferenceModelsOrientedObjectDetectionAdapter(str(tmp_path))
    request = OrientedObjectDetectionInferenceRequest(
        model_id="native/1",
        image=InferenceRequestImage(
            type="numpy_object", value=np.zeros((35, 73, 3), np.uint8)
        ),
        confidence=0.5,
        disable_active_learning=True,
    )
    response = adapter.infer_from_request(request)
    assert response.task_type == "oriented-object-detection"
    assert response.image.width == 73
    assert response.image.height == 35
    row = response.predictions[0]
    assert row.class_id == 7 and row.heading
    assert row.obb[:2] == pytest.approx([36.5, 17.5])
    assert row.obb[4] == pytest.approx(0.3 - math.pi / 2, abs=1e-6)
    assert row.detection_id == "q0:c7"
    assert "x" not in row.model_dump()


def test_sdk_uses_native_route_and_skips_client_resize():
    from inference_sdk.http.client import (
        NEW_INFERENCE_ENDPOINTS,
        _determine_client_downsizing_parameters,
    )
    from inference_sdk.http.entities import (
        HTTPClientMode,
        InferenceConfiguration,
        ModelDescription,
    )

    assert (
        NEW_INFERENCE_ENDPOINTS["oriented-object-detection"]
        == "/infer/oriented_object_detection"
    )
    description = ModelDescription(
        model_id="native/1",
        task_type="oriented-object-detection",
        batch_size=1,
        input_height=64,
        input_width=64,
    )
    assert _determine_client_downsizing_parameters(False, description, 1024) == (
        None,
        None,
    )
    params = InferenceConfiguration(confidence_threshold=0.25).to_api_call_parameters(
        HTTPClientMode.V1, "oriented-object-detection"
    )
    assert params["confidence"] == 0.25


def test_static_crop_preprocessing_is_inverted_before_response(tmp_path):
    package(tmp_path)
    (tmp_path / "inference_config.json").write_text(
        json.dumps(
            {
                "image_pre_processing": {
                    "static-crop": {
                        "enabled": True,
                        "x_min": 10,
                        "x_max": 60,
                        "y_min": 10,
                        "y_max": 30,
                    }
                }
            }
        )
    )
    model = RFDetrForOrientedObjectDetectionONNX.from_pretrained(str(tmp_path))
    result = model(np.zeros((35, 73, 3), np.uint8), confidence=0.5)[0]
    assert result.obb[0, :2].tolist() == pytest.approx([25, 6.5])


@pytest.mark.parametrize("tensor", [False, True])
def test_workflow_block_executes_installed_onnx_request_and_native_transport(
    tmp_path, monkeypatch, tensor
):
    package(tmp_path, side=2)
    (tmp_path / "model_config.json").write_text(
        json.dumps(
            {
                "model_architecture": "rfdetr",
                "task_type": "oriented-object-detection",
                "backend_type": "onnx",
            }
        )
    )
    from roboflow_workflows.core_steps.common.entities import StepExecutionMode
    from roboflow_workflows.core_steps.common.oriented import serialise_native_obb
    from roboflow_workflows.core_steps.models.roboflow.oriented_object_detection import (
        v1,
    )
    from roboflow_workflows.execution_engine.entities.base import (
        Batch,
        ImageParentMetadata,
        WorkflowImageData,
    )

    from inference.core.interfaces.workflows_models_provider import (
        ModelManagerModelsProvider,
    )
    from inference.core.models import inference_models_adapters as adapters

    monkeypatch.setattr(
        adapters, "ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES", True
    )
    monkeypatch.setattr(v1, "ENABLE_TENSOR_DATA_REPRESENTATION", tensor)
    adapter = adapters.InferenceModelsOrientedObjectDetectionAdapter(str(tmp_path))

    class LocalManager:
        def add_model(self, **kwargs):
            pass

        def infer_from_request_sync(self, model_id, request, **kwargs):
            return adapter.infer_from_request(request)

    block = v1.RoboflowOrientedObjectDetectionModelBlockV1(
        ModelManagerModelsProvider(LocalManager()), None, StepExecutionMode.LOCAL
    )
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="image0"),
        numpy_image=np.zeros((35, 73, 3), np.uint8),
    )
    result = block.run(
        Batch([image], indices=[(0,)]),
        model_id="native/1",
        confidence=0.5,
        max_detections=3,
        class_filter=["plane"],
    )
    native = serialise_native_obb(result[0]["predictions"])
    assert native["task_type"] == "oriented-object-detection"
    assert native["predictions"][0]["obb"] == pytest.approx(
        [36.5, 17.5, 0.4 * 73, 0.2 * 73, 0.3 - math.pi], abs=1e-5
    )
    assert native["predictions"][0]["parent_id"] == "image0"
    assert native["predictions"][0]["heading"]


def test_unused_frozen_classes_are_retained_without_output_mapping(tmp_path):
    package(tmp_path)
    frozen = metadata()
    frozen["classHeading"]["unused"] = True
    (tmp_path / "obb_metadata.json").write_text(json.dumps(frozen))
    loaded = RFDetrForOrientedObjectDetectionONNX.from_pretrained(str(tmp_path))
    result = loaded(np.zeros((64, 64, 3), np.uint8), confidence=0.5)[0]
    assert result.image_metadata["class_heading"]["unused"] is True
