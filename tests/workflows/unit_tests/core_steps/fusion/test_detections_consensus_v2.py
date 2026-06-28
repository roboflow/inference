import numpy as np
import pytest
import supervision as sv
from pydantic import ValidationError
from supervision.config import ORIENTED_BOX_COORDINATES

from inference.core.workflows.core_steps.fusion.detections_consensus.v2 import (
    AggregationMode,
    BlockManifest,
    calculate_iou,
)


def test_detections_consensus_v2_validation_when_valid_specification_given() -> None:
    # given
    specification = {
        "type": "roboflow_core/detections_consensus@v2",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "image_metadata": "$steps.detection.image",
        "required_votes": 3,
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result == BlockManifest(
        type="roboflow_core/detections_consensus@v2",
        name="some",
        predictions_batches=[
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        image_metadata="$steps.detection.image",
        required_votes=3,
        class_aware=True,
        iou_threshold=0.3,
        confidence=0.0,
        classes_to_consider=None,
        required_objects=None,
        presence_confidence_aggregation=AggregationMode.MAX,
        detections_merge_confidence_aggregation=AggregationMode.AVERAGE,
        detections_merge_coordinates_aggregation=AggregationMode.AVERAGE,
    )


def test_detections_consensus_v2_validation_rejects_v1_type_alias() -> None:
    # given
    specification = {
        "type": "roboflow_core/detections_consensus@v1",
        "name": "some",
        "predictions": ["$steps.detection.predictions"],
        "image_metadata": "$steps.detection.image",
        "required_votes": 1,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def _rotated_rect(
    cx: float, cy: float, w: float, h: float, angle_deg: float
) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    cos, sin = np.cos(angle), np.sin(angle)
    rot = np.array([[cos, -sin], [sin, cos]])
    corners = np.array(
        [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]
    )
    return (corners @ rot.T + [cx, cy]).astype(np.float32)


def test_calculate_iou_uses_box_iou_when_no_richer_geometry_present() -> None:
    # given - partially overlapping plain boxes, same expectation as the v1
    # behaviour: box A size = box B size = 800, intersection = 200,
    # expected result = 200 / 1400 = 1 / 7
    detections = sv.Detections(
        xyxy=np.array([[80, 190, 120, 210], [100, 200, 140, 220]], dtype=np.float64),
        confidence=np.array([0.5, 0.6], dtype=np.float64),
        class_id=np.array([1, 2]),
        data={"class_name": np.array(["a", "b"])},
    )
    detection_a = detections[0]
    detection_b = detections[1]

    # when
    result = calculate_iou(
        detection_a=detection_a,
        detection_b=detection_b,
    )

    # then
    assert abs(result - 1 / 7) < 1e-5


def test_calculate_iou_uses_mask_iou_when_both_have_masks() -> None:
    # given - identical AABB, disjoint masks. AABB IoU would be 1.0;
    # mask IoU is 0.
    mask_a = np.zeros((1, 10, 10), dtype=bool)
    mask_a[0, 0:5, 0:5] = True
    mask_b = np.zeros((1, 10, 10), dtype=bool)
    mask_b[0, 5:10, 5:10] = True
    xyxy = np.array([[0, 0, 10, 10]], dtype=np.float32)

    detection_a = sv.Detections(xyxy=xyxy, mask=mask_a)
    detection_b = sv.Detections(xyxy=xyxy, mask=mask_b)

    # when
    result = calculate_iou(
        detection_a=detection_a,
        detection_b=detection_b,
    )

    # then
    assert result == 0.0


def test_calculate_iou_uses_obb_iou_when_both_have_oriented_boxes() -> None:
    # given - crossed thin rectangles at +/-45 deg: near-identical AABBs but
    # barely-overlapping oriented bodies.
    quad_a = _rotated_rect(50, 50, 100, 10, +45)
    quad_b = _rotated_rect(50, 50, 100, 10, -45)
    aabb = np.array(
        [
            [
                quad_a[:, 0].min(),
                quad_a[:, 1].min(),
                quad_a[:, 0].max(),
                quad_a[:, 1].max(),
            ]
        ],
        dtype=np.float32,
    )

    detection_a = sv.Detections(
        xyxy=aabb, data={ORIENTED_BOX_COORDINATES: quad_a[None, ...]}
    )
    detection_b = sv.Detections(
        xyxy=aabb, data={ORIENTED_BOX_COORDINATES: quad_b[None, ...]}
    )

    # when
    result = calculate_iou(
        detection_a=detection_a,
        detection_b=detection_b,
    )

    # then
    assert result < 0.2
