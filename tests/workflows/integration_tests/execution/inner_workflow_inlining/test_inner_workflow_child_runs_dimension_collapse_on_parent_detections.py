"""
Equivalence for ``dimension_collapse`` inside ``inner_workflow`` with detection on the parent.

Inverse of ``test_inner_workflow_parent_runs_dimension_collapse_on_inner_detections``: the
parent runs object detection and the child inlines only ``roboflow_core/dimension_collapse@v1``
on the parent's predictions (passed via ``parameter_bindings``).

With a runtime list of two images, the flattened ``collapsed`` output matches the parent-only
case: one workflow result row whose ``collapsed`` value lists two ``sv.Detections``.
"""

from typing import Any
from unittest import mock

import numpy as np
import pytest
import supervision as sv
import torch

from inference.core.entities.requests.inference import ObjectDetectionInferenceRequest
from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
)
from inference.core.env import ENABLE_TENSOR_DATA_REPRESENTATION
from inference.core.managers.base import ModelManager
from inference_models.models.base.object_detection import Detections as NativeDetections
from tests.workflows.integration_tests.execution.inner_workflow_inlining._common import (
    child_dimension_collapse_from_parent_detections,
    execution_engine,
)

# Under ENABLE_TENSOR_DATA_REPRESENTATION the OD block runs inference through
# `run_tensor_native_inference` (+ `get_class_names`), not `infer_from_request_sync`,
# and the collapsed predictions are native `inference_models.Detections` (no
# `__getitem__` / value `__eq__`). The numpy test skips when the flag is on; the
# `*_tensor_native` parity test (skipped when off) drives the same scenario natively.
_NUMPY_ONLY = pytest.mark.skipif(
    ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="numpy ModelManager mock; OD block is native under the flag — see *_tensor_native",
)
_TENSOR_ONLY = pytest.mark.skipif(
    not ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="tensor-native variant; runs only with ENABLE_TENSOR_DATA_REPRESENTATION=True",
)


def _mock_od_response(h: int, w: int) -> ObjectDetectionInferenceResponse:
    return ObjectDetectionInferenceResponse(
        image=InferenceResponseImage(width=w, height=h),
        predictions=[
            ObjectDetectionPrediction(
                x=60,
                y=60,
                width=60,
                height=60,
                confidence=0.99,
                class_id=0,
                detection_id="mock-d0",
                **{"class": "x"},
            ),
            ObjectDetectionPrediction(
                x=250,
                y=60,
                width=80,
                height=80,
                confidence=0.95,
                class_id=1,
                detection_id="mock-d1",
                **{"class": "y"},
            ),
            ObjectDetectionPrediction(
                x=450,
                y=60,
                width=100,
                height=100,
                confidence=0.90,
                class_id=2,
                detection_id="mock-d2",
                **{"class": "z"},
            ),
        ],
    )


def _hw_from_od_request_image(img: Any) -> tuple[int, int]:
    if isinstance(img, dict):
        arr = img["value"]
    else:
        arr = img.value
    assert isinstance(arr, np.ndarray), type(arr)
    return int(arr.shape[0]), int(arr.shape[1])


def _mock_infer_object_detection_from_request_sync(
    model_id: str,
    request: ObjectDetectionInferenceRequest,
) -> ObjectDetectionInferenceResponse | list[ObjectDetectionInferenceResponse]:
    imgs = request.image if isinstance(request.image, list) else [request.image]
    responses = [_mock_od_response(*_hw_from_od_request_image(im)) for im in imgs]
    if len(responses) == 1:
        return responses[0]
    return responses


def _assert_collapsed_two_image_batch(collapsed: list) -> None:
    assert isinstance(collapsed, list)
    assert len(collapsed) == 2
    for det in collapsed:
        assert isinstance(det, sv.Detections)
        assert len(det) == 3


def _nested_workflow(inner: dict) -> dict:
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": "roboflow_core/roboflow_object_detection_model@v3",
                "name": "general_detection",
                "image": "$inputs.image",
                "model_id": "yolov8n-640",
            },
            {
                "type": "roboflow_core/inner_workflow@v1",
                "name": "nested",
                "workflow_definition": inner,
                "parameter_bindings": {
                    "predictions": "$steps.general_detection.predictions",
                },
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "collapsed",
                "selector": "$steps.nested.collapsed",
            },
        ],
    }


def _flat_workflow() -> dict:
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": "roboflow_core/roboflow_object_detection_model@v3",
                "name": "general_detection",
                "image": "$inputs.image",
                "model_id": "yolov8n-640",
            },
            {
                "type": "roboflow_core/dimension_collapse@v1",
                "name": "collapse",
                "data": "$steps.general_detection.predictions",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "collapsed",
                "selector": "$steps.collapse.output",
            },
        ],
    }


@_NUMPY_ONLY
def test_inlined_child_dimension_collapse_matches_flat_workflow(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    infer_mock = mock.MagicMock(
        side_effect=_mock_infer_object_detection_from_request_sync,
    )
    inner = child_dimension_collapse_from_parent_detections()

    with mock.patch.object(ModelManager, "add_model"), mock.patch.object(
        ModelManager,
        "infer_from_request_sync",
        new=infer_mock,
    ):
        nested_engine = execution_engine(model_manager, _nested_workflow(inner))
        flat_engine = execution_engine(model_manager, _flat_workflow())
        nested_result = nested_engine.run(runtime_parameters={"image": dogs_image})
        flat_result = flat_engine.run(runtime_parameters={"image": dogs_image})

    assert infer_mock.call_count == 2
    for call in infer_mock.call_args_list:
        call_model_id, call_request = call.args
        assert call_model_id == "yolov8n-640"
        assert isinstance(call_request, ObjectDetectionInferenceRequest)
        imgs = (
            call_request.image
            if isinstance(call_request.image, list)
            else [call_request.image]
        )
        assert len(imgs) == 1

    assert nested_result == flat_result
    assert len(nested_result) == 1


@_NUMPY_ONLY
def test_inlined_child_dimension_collapse_matches_flat_workflow_runtime_image_list(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    infer_mock = mock.MagicMock(
        side_effect=_mock_infer_object_detection_from_request_sync,
    )
    inner = child_dimension_collapse_from_parent_detections()
    images = [dogs_image, dogs_image.copy()]

    with mock.patch.object(ModelManager, "add_model"), mock.patch.object(
        ModelManager,
        "infer_from_request_sync",
        new=infer_mock,
    ):
        nested_engine = execution_engine(model_manager, _nested_workflow(inner))
        flat_engine = execution_engine(model_manager, _flat_workflow())
        nested_result = nested_engine.run(runtime_parameters={"image": images})
        flat_result = flat_engine.run(runtime_parameters={"image": images})

    assert infer_mock.call_count == 2
    for call in infer_mock.call_args_list:
        call_model_id, call_request = call.args
        assert call_model_id == "yolov8n-640"
        assert isinstance(call_request, ObjectDetectionInferenceRequest)
        imgs = (
            call_request.image
            if isinstance(call_request.image, list)
            else [call_request.image]
        )
        assert len(imgs) == 2

    assert nested_result == flat_result
    assert len(nested_result) == 1
    _assert_collapsed_two_image_batch(nested_result[0]["collapsed"])


# --- tensor-native parity --------------------------------------------------


def _one_native_detections() -> NativeDetections:
    # Native equivalent of _mock_od_response: same three boxes as raw `Detections`
    # (xyxy from the numpy x/y/w/h centres). The OD block attaches the image lineage;
    # detection_id rides in bboxes_metadata, class names via get_class_names.
    return NativeDetections(
        xyxy=torch.tensor(
            [[30, 30, 90, 90], [210, 20, 290, 100], [400, 10, 500, 110]],
            dtype=torch.float32,
        ),
        class_id=torch.tensor([0, 1, 2], dtype=torch.long),
        confidence=torch.tensor([0.99, 0.95, 0.90], dtype=torch.float32),
        bboxes_metadata=[
            {"detection_id": "mock-d0"},
            {"detection_id": "mock-d1"},
            {"detection_id": "mock-d2"},
        ],
    )


def _run_tensor_native_inference(model_id: str, images, **kwargs):
    # Native equivalent of _mock_infer_object_detection_from_request_sync: one raw
    # `Detections` per input image (the OD block zips images with this list). A fresh
    # object per image keeps the two-image batch from aliasing one tensor.
    imgs = images if isinstance(images, list) else [images]
    return [_one_native_detections() for _ in imgs]


_EXPECTED_BOXES = [
    ("mock-d0", "x", 0, 0.99, [30, 30, 90, 90]),
    ("mock-d1", "y", 1, 0.95, [210, 20, 290, 100]),
    ("mock-d2", "z", 2, 0.90, [400, 10, 500, 110]),
]


def _assert_native_three_box_detections(det: NativeDetections) -> None:
    # Native parity of the per-detection value checks: Detections has no __getitem__,
    # so detection_id reads from bboxes_metadata and class_name from image_metadata.
    assert isinstance(det, NativeDetections)
    assert len(det) == 3
    for i, (det_id, class_name, class_id, conf, expected_xyxy) in enumerate(
        _EXPECTED_BOXES
    ):
        assert det.bboxes_metadata[i]["detection_id"] == det_id
        assert int(det.class_id[i]) == class_id
        assert float(det.confidence[i]) == pytest.approx(conf)
        assert det.image_metadata["class_names"][int(det.class_id[i])] == class_name
        np.testing.assert_allclose(
            det.xyxy[i].cpu().numpy(), expected_xyxy, rtol=0, atol=1e-3
        )


def _assert_collapsed_native(collapsed: list, expected_images: int) -> None:
    # Native parity of _assert_collapsed_two_image_batch (generalised to N images):
    # the collapsed list holds one native Detections per image, each with 3 boxes.
    assert isinstance(collapsed, list)
    assert len(collapsed) == expected_images
    for det in collapsed:
        _assert_native_three_box_detections(det)


@_TENSOR_ONLY
def test_inlined_child_dimension_collapse_matches_flat_workflow_tensor_native(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    run_mock = mock.MagicMock(side_effect=_run_tensor_native_inference)
    inner = child_dimension_collapse_from_parent_detections()

    with mock.patch.object(ModelManager, "add_model"), mock.patch.object(
        ModelManager, "get_class_names", return_value=["x", "y", "z"]
    ), mock.patch.object(
        ModelManager,
        "run_tensor_native_inference",
        new=run_mock,
    ):
        nested_engine = execution_engine(model_manager, _nested_workflow(inner))
        flat_engine = execution_engine(model_manager, _flat_workflow())
        nested_result = nested_engine.run(runtime_parameters={"image": dogs_image})
        flat_result = flat_engine.run(runtime_parameters={"image": dogs_image})

    assert run_mock.call_count == 2
    for call in run_mock.call_args_list:
        model_id = call.kwargs.get("model_id", call.args[0] if call.args else None)
        assert model_id == "yolov8n-640"
        imgs = call.kwargs.get("images", call.args[1] if len(call.args) > 1 else None)
        assert imgs is not None and len(imgs) == 1

    # Native Detections has no value __eq__; assert the values on BOTH runs instead of
    # comparing nested_result == flat_result.
    assert len(nested_result) == 1
    assert len(flat_result) == 1
    _assert_collapsed_native(nested_result[0]["collapsed"], expected_images=1)
    _assert_collapsed_native(flat_result[0]["collapsed"], expected_images=1)


@_TENSOR_ONLY
def test_inlined_child_dimension_collapse_matches_flat_workflow_runtime_image_list_tensor_native(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    run_mock = mock.MagicMock(side_effect=_run_tensor_native_inference)
    inner = child_dimension_collapse_from_parent_detections()
    images = [dogs_image, dogs_image.copy()]

    with mock.patch.object(ModelManager, "add_model"), mock.patch.object(
        ModelManager, "get_class_names", return_value=["x", "y", "z"]
    ), mock.patch.object(
        ModelManager,
        "run_tensor_native_inference",
        new=run_mock,
    ):
        nested_engine = execution_engine(model_manager, _nested_workflow(inner))
        flat_engine = execution_engine(model_manager, _flat_workflow())
        nested_result = nested_engine.run(runtime_parameters={"image": images})
        flat_result = flat_engine.run(runtime_parameters={"image": images})

    assert run_mock.call_count == 2
    for call in run_mock.call_args_list:
        model_id = call.kwargs.get("model_id", call.args[0] if call.args else None)
        assert model_id == "yolov8n-640"
        imgs = call.kwargs.get("images", call.args[1] if len(call.args) > 1 else None)
        assert imgs is not None and len(imgs) == 2

    # Native Detections has no value __eq__; assert the values on BOTH runs instead of
    # comparing nested_result == flat_result.
    assert len(nested_result) == 1
    assert len(flat_result) == 1
    _assert_collapsed_native(nested_result[0]["collapsed"], expected_images=2)
    _assert_collapsed_native(flat_result[0]["collapsed"], expected_images=2)
