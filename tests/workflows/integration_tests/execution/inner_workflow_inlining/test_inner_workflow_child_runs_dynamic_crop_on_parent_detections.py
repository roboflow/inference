"""
Equivalence for ``test_inner_workflow_child_runs_dynamic_crop_on_parent_detections``.
"""

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
    child_dynamic_crop_from_parent_detections,
    execution_engine,
)

# Under ENABLE_TENSOR_DATA_REPRESENTATION the OD block runs inference through
# `run_tensor_native_inference` (+ `get_class_names`), not `infer_from_request_sync`,
# and downstream crop predictions are native `inference_models.Detections` (no
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


@_NUMPY_ONLY
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


def _run_tensor_native_inference_factory(h: int, w: int):
    # Native equivalent of _infer_from_request_sync_factory: same three boxes as raw
    # `Detections` (xyxy from the numpy x/y/w/h centres). The OD block attaches the
    # image lineage; detection_id rides in bboxes_metadata, class names via get_class_names.
    def run_tensor_native_inference(model_id: str, images, **kwargs):
        imgs = images if isinstance(images, list) else [images]
        assert len(imgs) == 1, f"Mock Expected 1 image, got {len(imgs)}"
        return [
            NativeDetections(
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
        ]

    return run_tensor_native_inference


def _assert_crop_predictions_equal_native(crop_preds: list) -> None:
    # Native parity of _assert_crop_predictions_equal: Detections has no __getitem__,
    # so detection_id reads from bboxes_metadata and class_name from image_metadata.
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
        assert isinstance(det, NativeDetections)
        assert len(det) == 1
        assert det.bboxes_metadata[0]["detection_id"] == det_id
        assert int(det.class_id[0]) == class_id
        assert float(det.confidence[0]) == pytest.approx(conf)
        assert det.image_metadata["class_names"][int(det.class_id[0])] == class_name
        np.testing.assert_allclose(
            det.xyxy[0].cpu().numpy(), expected_xyxy, rtol=0, atol=1e-3
        )


@_TENSOR_ONLY
def test_inlined_dynamic_crop_matches_inner_workflow_tensor_native(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    h, w = dogs_image.shape[:2]
    run_mock = mock.MagicMock(
        side_effect=_run_tensor_native_inference_factory(h, w),
    )
    inner = child_dynamic_crop_from_parent_detections()

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
        model_id = call.kwargs.get(
            "model_id", call.args[0] if call.args else None
        )
        assert model_id == "yolov8n-640"
        imgs = call.kwargs.get(
            "images", call.args[1] if len(call.args) > 1 else None
        )
        assert imgs is not None and len(imgs) == 1

    # Native Detections has no value __eq__; assert the values on BOTH runs instead of
    # comparing nested_result == flat_result.
    assert len(nested_result) == 1
    assert len(flat_result) == 1
    _assert_crop_predictions_equal_native(nested_result[0]["from_child"])
    _assert_crop_predictions_equal_native(flat_result[0]["from_child"])
