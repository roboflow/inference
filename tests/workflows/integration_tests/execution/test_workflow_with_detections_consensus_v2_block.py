import numpy as np
import supervision as sv
from supervision.config import ORIENTED_BOX_COORDINATES

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

IMAGE_HEIGHT, IMAGE_WIDTH = 100, 120


def _consensus_workflow(prediction_kind: str) -> dict:
    return {
        "version": "1.3.0",
        "inputs": [
            {
                "type": "WorkflowBatchInput",
                "name": "predictions_a",
                "kind": [prediction_kind],
            },
            {
                "type": "WorkflowBatchInput",
                "name": "predictions_b",
                "kind": [prediction_kind],
            },
        ],
        "steps": [
            {
                "type": "roboflow_core/detections_consensus@v2",
                "name": "consensus",
                "predictions_batches": [
                    "$inputs.predictions_a",
                    "$inputs.predictions_b",
                ],
                "required_votes": 2,
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "predictions",
                "selector": "$steps.consensus.predictions",
            }
        ],
    }


def _metadata(class_name: str) -> dict:
    dims = np.array([[IMAGE_HEIGHT, IMAGE_WIDTH]])
    return {
        "detection_id": np.array(["det"]),
        "class_name": np.array([class_name]),
        "parent_id": np.array(["image"]),
        "parent_coordinates": np.array([[0, 0]]),
        "parent_dimensions": dims,
        "root_parent_id": np.array(["image"]),
        "root_parent_coordinates": np.array([[0, 0]]),
        "root_parent_dimensions": dims,
        "image_dimensions": dims,
    }


def _mask_detection(mask: np.ndarray, class_name: str) -> sv.Detections:
    masks = mask[np.newaxis, ...]
    return sv.Detections(
        xyxy=sv.mask_to_xyxy(masks).astype(np.float64),
        confidence=np.array([0.9], dtype=np.float64),
        class_id=np.array([0]),
        mask=masks,
        data=_metadata(class_name),
    )


def _rotated_rect(cx: float, cy: float, w: float, h: float, angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    cos, sin = np.cos(angle), np.sin(angle)
    rot = np.array([[cos, -sin], [sin, cos]])
    corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    return (corners @ rot.T + [cx, cy]).astype(np.float32)


def _obb_detection(corners: np.ndarray, class_name: str) -> sv.Detections:
    xyxy = np.array(
        [
            [
                corners[:, 0].min(),
                corners[:, 1].min(),
                corners[:, 0].max(),
                corners[:, 1].max(),
            ]
        ],
        dtype=np.float64,
    )
    data = _metadata(class_name)
    data[ORIENTED_BOX_COORDINATES] = corners[np.newaxis, ...]
    return sv.Detections(
        xyxy=xyxy,
        confidence=np.array([0.9], dtype=np.float64),
        class_id=np.array([0]),
        data=data,
    )


def _run(workflow: dict, model_manager: ModelManager, runtime_parameters: dict) -> list:
    execution_engine = ExecutionEngine.init(
        workflow_definition=workflow,
        init_parameters={
            "workflows_core.model_manager": model_manager,
            "workflows_core.api_key": None,
            "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
        },
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    return execution_engine.run(runtime_parameters=runtime_parameters)


def test_detections_consensus_v2_merges_masks_with_default_union(
    model_manager: ModelManager,
) -> None:
    # given - two segmentation sources agreeing on one object with overlapping masks
    mask_a = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=bool)
    mask_a[10:60, 10:60] = True
    mask_b = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=bool)
    mask_b[20:70, 20:70] = True

    # when
    result = _run(
        _consensus_workflow("instance_segmentation_prediction"),
        model_manager,
        {
            "predictions_a": [_mask_detection(mask_a, "person")],
            "predictions_b": [_mask_detection(mask_b, "person")],
        },
    )

    # then - one consensus detection carrying the boolean union of both masks
    predictions: sv.Detections = result[0]["predictions"]
    assert len(predictions) == 1
    merged_mask = predictions.mask[0]
    assert merged_mask.dtype == bool
    assert np.array_equal(merged_mask, mask_a | mask_b)


def test_detections_consensus_v2_matches_overlapping_oriented_boxes(
    model_manager: ModelManager,
) -> None:
    # given - two near-identical oriented boxes (high oriented-box IoU)
    quad_a = _rotated_rect(60, 50, 60, 30, 30)
    quad_b = _rotated_rect(62, 52, 60, 30, 30)

    # when
    result = _run(
        _consensus_workflow("object_detection_prediction"),
        model_manager,
        {
            "predictions_a": [_obb_detection(quad_a, "thing")],
            "predictions_b": [_obb_detection(quad_b, "thing")],
        },
    )

    # then - they reach the required 2 votes and merge into one detection
    assert len(result[0]["predictions"]) == 1


def test_detections_consensus_v2_uses_oriented_box_iou_for_matching(
    model_manager: ModelManager,
) -> None:
    # given - crossed thin rectangles: near-identical axis-aligned envelopes but
    # oriented bodies that barely overlap
    quad_a = _rotated_rect(60, 50, 90, 8, 45)
    quad_b = _rotated_rect(60, 50, 90, 8, -45)

    # when
    result = _run(
        _consensus_workflow("object_detection_prediction"),
        model_manager,
        {
            "predictions_a": [_obb_detection(quad_a, "thing")],
            "predictions_b": [_obb_detection(quad_b, "thing")],
        },
    )

    # then - oriented-box IoU (not the overlapping envelope) drives matching, so
    # neither reaches the required 2 votes and no consensus detection is produced
    assert len(result[0]["predictions"]) == 0
