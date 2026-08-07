"""
Equivalence for ``test_inner_workflow_parent_detection_offset_after_nested_crop``:
nested ``dynamic_crop`` vs inlined crop, including ``detection_offset`` on crop outputs.
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
# and downstream crop/offset predictions are native `inference_models.Detections` (no
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
        assert len(imgs) == 1

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


def _nested_parent_workflow_crop_only(inner: dict) -> dict:
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
                "name": "crop_preds",
                "selector": "$steps.nested.crop_predictions",
            },
        ],
    }


def _nested_parent_workflow_with_detection_offset(
    inner: dict,
    offset_x: int,
    offset_y: int,
) -> dict:
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
            {
                "type": "roboflow_core/detection_offset@v1",
                "name": "padded",
                "predictions": "$steps.nested.crop_predictions",
                "offset_width": offset_x,
                "offset_height": offset_y,
                "units": "Pixels",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "padded_preds",
                "selector": "$steps.padded.predictions",
            },
        ],
    }


def _flat_parent_workflow_crop_only() -> dict:
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
                "name": "crop_preds",
                "selector": "$steps.cropping.predictions",
            },
        ],
    }


def _flat_parent_workflow_with_detection_offset(
    offset_x: int,
    offset_y: int,
) -> dict:
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
            {
                "type": "roboflow_core/detection_offset@v1",
                "name": "padded",
                "predictions": "$steps.cropping.predictions",
                "offset_width": offset_x,
                "offset_height": offset_y,
                "units": "Pixels",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "padded_preds",
                "selector": "$steps.padded.predictions",
            },
        ],
    }


def _assert_offset_detection_lists_equal(
    a_list: list,
    b_list: list,
) -> None:
    """``detection_offset`` issues new UUIDs; compare geometry and class metadata only."""
    assert len(a_list) == len(b_list)
    for det_a, det_b in zip(a_list, b_list):
        assert isinstance(det_a, sv.Detections) and isinstance(det_b, sv.Detections)
        assert len(det_a) == len(det_b)
        np.testing.assert_allclose(
            det_a.xyxy.astype(np.float32),
            det_b.xyxy.astype(np.float32),
            rtol=0,
            atol=1e-3,
        )
        assert det_a.data["class_name"][0] == det_b.data["class_name"][0]
        assert det_a.class_id[0] == det_b.class_id[0]
        np.testing.assert_allclose(
            det_a.confidence[0], det_b.confidence[0], rtol=0, atol=1e-6
        )


@_NUMPY_ONLY
def test_inlined_crop_and_offset_match_nested_inner_workflow(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    h, w = dogs_image.shape[:2]
    infer_mock = mock.MagicMock(
        side_effect=_infer_from_request_sync_factory(h, w),
    )
    inner = child_dynamic_crop_from_parent_detections()
    offset_x = 10
    offset_y = 10

    with mock.patch.object(ModelManager, "add_model"), mock.patch.object(
        ModelManager,
        "infer_from_request_sync",
        new=infer_mock,
    ):
        nested_crop = execution_engine(
            model_manager,
            _nested_parent_workflow_crop_only(inner),
        )
        flat_crop = execution_engine(
            model_manager,
            _flat_parent_workflow_crop_only(),
        )
        nested_off = execution_engine(
            model_manager,
            _nested_parent_workflow_with_detection_offset(
                inner,
                offset_x,
                offset_y,
            ),
        )
        flat_off = execution_engine(
            model_manager,
            _flat_parent_workflow_with_detection_offset(offset_x, offset_y),
        )

        res_nested_crop = nested_crop.run(runtime_parameters={"image": dogs_image})
        res_flat_crop = flat_crop.run(runtime_parameters={"image": dogs_image})
        res_nested_off = nested_off.run(runtime_parameters={"image": dogs_image})
        res_flat_off = flat_off.run(runtime_parameters={"image": dogs_image})

    assert res_nested_crop == res_flat_crop

    nested_off_preds = res_nested_off[0]["padded_preds"]
    flat_off_preds = res_flat_off[0]["padded_preds"]
    _assert_offset_detection_lists_equal(nested_off_preds, flat_off_preds)

    infer_mock.assert_called()
    assert infer_mock.call_count == 4


def _run_tensor_native_inference_factory(h: int, w: int):
    # Native equivalent of _infer_from_request_sync_factory: same three boxes as raw
    # `Detections` (xyxy from the numpy x/y/w/h centres). The OD block attaches the
    # image lineage; detection_id rides in bboxes_metadata, class names via get_class_names.
    def run_tensor_native_inference(model_id: str, images, **kwargs):
        imgs = images if isinstance(images, list) else [images]
        assert len(imgs) == 1
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


def _native_class_name(det: NativeDetections, index: int) -> str:
    return det.image_metadata["class_names"][int(det.class_id[index])]


def _assert_crop_predictions_equal_native(crop_preds: list) -> None:
    # Native parity of the numpy crop-only comparison: each crop yields one detection at
    # the crop's local origin. Detections has no __getitem__, so detection_id reads from
    # bboxes_metadata and class_name from image_metadata.
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
        assert _native_class_name(det, 0) == class_name
        np.testing.assert_allclose(
            det.xyxy[0].cpu().numpy(), expected_xyxy, rtol=0, atol=1e-3
        )


def _assert_offset_detection_lists_equal_native(
    a_list: list,
    b_list: list,
) -> None:
    """Native parity of _assert_offset_detection_lists_equal.

    ``detection_offset`` issues new UUIDs, so geometry and class metadata are compared
    nested-vs-flat only. Native ``Detections`` has no value ``__eq__`` / ``.data`` —
    xyxy is a torch tensor and class names resolve through ``image_metadata``.
    """
    assert len(a_list) == len(b_list)
    for det_a, det_b in zip(a_list, b_list):
        assert isinstance(det_a, NativeDetections) and isinstance(
            det_b, NativeDetections
        )
        assert len(det_a) == len(det_b)
        np.testing.assert_allclose(
            det_a.xyxy.cpu().numpy().astype(np.float32),
            det_b.xyxy.cpu().numpy().astype(np.float32),
            rtol=0,
            atol=1e-3,
        )
        assert _native_class_name(det_a, 0) == _native_class_name(det_b, 0)
        assert int(det_a.class_id[0]) == int(det_b.class_id[0])
        np.testing.assert_allclose(
            float(det_a.confidence[0]),
            float(det_b.confidence[0]),
            rtol=0,
            atol=1e-6,
        )


@_TENSOR_ONLY
def test_inlined_crop_and_offset_match_nested_inner_workflow_tensor_native(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    h, w = dogs_image.shape[:2]
    run_mock = mock.MagicMock(
        side_effect=_run_tensor_native_inference_factory(h, w),
    )
    inner = child_dynamic_crop_from_parent_detections()
    offset_x = 10
    offset_y = 10

    with mock.patch.object(ModelManager, "add_model"), mock.patch.object(
        ModelManager, "get_class_names", return_value=["x", "y", "z"]
    ), mock.patch.object(
        ModelManager,
        "run_tensor_native_inference",
        new=run_mock,
    ):
        nested_crop = execution_engine(
            model_manager,
            _nested_parent_workflow_crop_only(inner),
        )
        flat_crop = execution_engine(
            model_manager,
            _flat_parent_workflow_crop_only(),
        )
        nested_off = execution_engine(
            model_manager,
            _nested_parent_workflow_with_detection_offset(
                inner,
                offset_x,
                offset_y,
            ),
        )
        flat_off = execution_engine(
            model_manager,
            _flat_parent_workflow_with_detection_offset(offset_x, offset_y),
        )

        res_nested_crop = nested_crop.run(runtime_parameters={"image": dogs_image})
        res_flat_crop = flat_crop.run(runtime_parameters={"image": dogs_image})
        res_nested_off = nested_off.run(runtime_parameters={"image": dogs_image})
        res_flat_off = flat_off.run(runtime_parameters={"image": dogs_image})

    # Native Detections has no value __eq__; assert the crop-only values on BOTH runs
    # instead of comparing res_nested_crop == res_flat_crop.
    assert len(res_nested_crop) == 1
    assert len(res_flat_crop) == 1
    _assert_crop_predictions_equal_native(res_nested_crop[0]["crop_preds"])
    _assert_crop_predictions_equal_native(res_flat_crop[0]["crop_preds"])

    nested_off_preds = res_nested_off[0]["padded_preds"]
    flat_off_preds = res_flat_off[0]["padded_preds"]
    _assert_offset_detection_lists_equal_native(nested_off_preds, flat_off_preds)

    run_mock.assert_called()
    assert run_mock.call_count == 4
