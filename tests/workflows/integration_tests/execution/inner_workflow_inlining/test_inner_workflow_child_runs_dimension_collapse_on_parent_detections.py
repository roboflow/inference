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
import supervision as sv

from inference.core.entities.requests.inference import ObjectDetectionInferenceRequest
from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
)
from inference.core.managers.base import ModelManager
from tests.workflows.integration_tests.execution.inner_workflow_inlining._common import (
    child_dimension_collapse_from_parent_detections,
    execution_engine,
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
