"""
Equivalence for ``test_inner_workflow_parent_detection_offset_after_nested_crop``:
nested ``dynamic_crop`` vs inlined crop, including ``detection_offset`` on crop outputs.
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
        model_id: str, request: ObjectDetectionInferenceRequest,
    ):
        imgs = request.image if isinstance(request.image, list) else [request.image]
        assert len(imgs) == 1

        return ObjectDetectionInferenceResponse(
            image=InferenceResponseImage(width=w, height=h),
            predictions=[
                ObjectDetectionPrediction(
                    x=60,
                    y=60,
                    width=80,
                    height=80,
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
                    confidence=0.99,
                    class_id=1,
                    detection_id="mock-d1",
                    **{"class": "y"},
                ),
                ObjectDetectionPrediction(
                    x=450,
                    y=60,
                    width=80,
                    height=80,
                    confidence=0.99,
                    class_id=2,
                    detection_id="mock-d2",
                    **{"class": "z"},
                ),
            ],
        )

    return infer_from_request_sync


def _nested_parent_workflow(
    inner: dict, *, apply_detection_offset: bool, offset_x: int, offset_y: int,
) -> dict:
    steps: list = [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v2",
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
    ]
    if apply_detection_offset:
        steps.append(
            {
                "type": "roboflow_core/detection_offset@v1",
                "name": "padded",
                "predictions": "$steps.nested.crop_predictions",
                "offset_width": offset_x,
                "offset_height": offset_y,
                "units": "Pixels",
            },
        )
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": steps,
        "outputs": [
            {
                "type": "JsonField",
                "name": (
                    "padded_preds" if apply_detection_offset else "crop_preds"
                ),
                "selector": (
                    "$steps.padded.predictions"
                    if apply_detection_offset
                    else "$steps.nested.crop_predictions"
                ),
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


def _flat_parent_workflow(
    *, apply_detection_offset: bool, offset_x: int, offset_y: int,
) -> dict:
    steps: list = [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v2",
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
    ]
    if apply_detection_offset:
        steps.append(
            {
                "type": "roboflow_core/detection_offset@v1",
                "name": "padded",
                "predictions": "$steps.cropping.predictions",
                "offset_width": offset_x,
                "offset_height": offset_y,
                "units": "Pixels",
            },
        )
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": steps,
        "outputs": [
            {
                "type": "JsonField",
                "name": (
                    "padded_preds" if apply_detection_offset else "crop_preds"
                ),
                "selector": (
                    "$steps.padded.predictions"
                    if apply_detection_offset
                    else "$steps.cropping.predictions"
                ),
            },
        ],
    }


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
            _nested_parent_workflow(
                inner,
                apply_detection_offset=False,
                offset_x=offset_x,
                offset_y=offset_y,
            ),
        )
        flat_crop = execution_engine(
            model_manager,
            _flat_parent_workflow(
                apply_detection_offset=False,
                offset_x=offset_x,
                offset_y=offset_y,
            ),
        )
        nested_off = execution_engine(
            model_manager,
            _nested_parent_workflow(
                inner,
                apply_detection_offset=True,
                offset_x=offset_x,
                offset_y=offset_y,
            ),
        )
        flat_off = execution_engine(
            model_manager,
            _flat_parent_workflow(
                apply_detection_offset=True,
                offset_x=offset_x,
                offset_y=offset_y,
            ),
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

    before = res_flat_crop[0]["crop_preds"]
    after = res_flat_off[0]["padded_preds"]
    assert len(before) == len(after) == 3

    dx, dy = offset_x // 2, offset_y // 2
    for det_b, det_a in zip(before, after):
        assert isinstance(det_b, sv.Detections) and isinstance(det_a, sv.Detections)
        assert len(det_b) == len(det_a) == 1

        hid, wid = det_b["image_dimensions"][0]
        x1, y1, x2, y2 = det_b.xyxy[0]
        expected = np.array(
            [
                max(0.0, x1 - dx),
                max(0.0, y1 - dy),
                min(float(wid), x2 + dx),
                min(float(hid), y2 + dy),
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            det_a.xyxy[0].astype(np.float32),
            expected,
            rtol=0,
            atol=1e-3,
        )

        assert str(det_a.data["detection_id"][0]) != str(det_b.data["detection_id"][0])
        assert len(str(det_a.data["detection_id"][0])) == 36
        assert det_a.data["class_name"][0] == det_b.data["class_name"][0]
        assert det_a.class_id[0] == det_b.class_id[0]
        np.testing.assert_allclose(
            det_a.confidence[0], det_b.confidence[0], rtol=0, atol=1e-6
        )
