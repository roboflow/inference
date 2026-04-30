"""
Equivalence for ``test_inner_workflow_child_runs_dynamic_crop_on_parent_detections``.
"""

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
    child_dynamic_crop_from_parent_detections,
    execution_engine,
)


def _infer_from_request_sync_factory(h: int, w: int):
    def infer_from_request_sync(
        model_id: str,
        request: ObjectDetectionInferenceRequest,
    ):
        imgs = request.image if isinstance(request.image, list) else [request.image]
        assert len(imgs) == 1, f"Mock Expected 1 image, got {len(imgs)}"

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

    return infer_from_request_sync


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
                    "image": "$inputs.image",
                    "predictions": "$steps.general_detection.predictions",
                },
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "from_child",
                "selector": "$steps.nested.crop_predictions",
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
                "type": "roboflow_core/dynamic_crop@v1",
                "name": "cropping",
                "image": "$inputs.image",
                "predictions": "$steps.general_detection.predictions",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "from_child",
                "selector": "$steps.cropping.predictions",
            },
        ],
    }


def _assert_crop_predictions_equal(crop_preds: list) -> None:
    expected = [
        ("mock-d0", "x", 0, 0.99, [0, 0, 60, 60]),
        ("mock-d1", "y", 1, 0.95, [0, 0, 80, 80]),
        ("mock-d2", "z", 2, 0.90, [0, 0, 100, 100]),
    ]
    assert isinstance(crop_preds, list)
    assert len(crop_preds) == 3
    for det, (det_id, class_name, class_id, conf, expected_xyxy) in zip(
        crop_preds, expected
    ):
        assert isinstance(det, sv.Detections)
        assert len(det) == 1
        assert det["detection_id"][0] == det_id
        assert det.class_id[0] == class_id
        assert det.confidence[0] == conf
        assert det["class_name"][0] == class_name
        np.testing.assert_allclose(det.xyxy[0], expected_xyxy, rtol=0, atol=1e-3)


def test_inlined_dynamic_crop_matches_inner_workflow(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    h, w = dogs_image.shape[:2]
    infer_mock = mock.MagicMock(
        side_effect=_infer_from_request_sync_factory(h, w),
    )
    inner = child_dynamic_crop_from_parent_detections()

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
    _assert_crop_predictions_equal(nested_result[0]["from_child"])
