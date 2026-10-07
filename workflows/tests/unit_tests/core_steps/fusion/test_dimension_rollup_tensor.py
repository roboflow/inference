"""Tensor-native sibling of the keypoint rollup tests in ``test_dimension_rollup.py``.

The tensor rollup rebuilds the native ``KeyPoints`` of the rolled-up prediction
through ``build_native_key_points``; with holes in the child keypoints the slots
must follow the keypoint class ids, not the packed positions.
"""

import numpy as np
import supervision as sv
from roboflow_workflows.core_steps.fusion.detections_list_rollup.v1_tensor import (
    merge_crop_predictions,
)
from roboflow_workflows.execution_engine.v1.dynamic_blocks.representation_boundary import (
    sv_detections_to_native,
    sv_detections_to_native_key_point_prediction,
)


def _parent_crop(xyxy: list) -> sv.Detections:
    detections = sv.Detections(
        xyxy=np.array([xyxy], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0], dtype=int),
    )
    detections.data["root_parent_dimensions"] = [(480, 640)]
    return detections


def _child_with_keypoint_holes() -> sv.Detections:
    # One person with nose (0), right_eye (2) and right_ear (4); the eye and ear
    # on the left were below the keypoint confidence threshold.
    return sv.Detections(
        xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
        confidence=np.array([0.85], dtype=np.float32),
        class_id=np.array([0], dtype=int),
        data={
            "keypoints_xy": np.array(
                [[[20.0, 25.0], [15.0, 35.0], [5.0, 35.0]]], dtype=np.float32
            ),
            "keypoints_confidence": np.array([[0.9, 0.8, 0.7]], dtype=np.float32),
            "keypoints_class_name": np.array(
                [["nose", "right_eye", "right_ear"]], dtype=object
            ),
            "keypoints_class_id": np.array([[0, 2, 4]], dtype=int),
            "prediction_type": np.array(["keypoint-detection"]),
        },
    )


def test_tensor_rollup_rebuilds_key_points_at_skeleton_slots() -> None:
    # given
    parent = sv_detections_to_native(_parent_crop([100, 200, 300, 400]))
    child = sv_detections_to_native_key_point_prediction(_child_with_keypoint_holes())

    # when
    (key_points, detections), crop_zones = merge_crop_predictions(
        parent, [child], "max", 0.0, 10.0
    )

    # then: COCO slots, shifted into parent coordinates by the crop offset
    assert len(detections) == 1
    assert key_points.xy.shape == (1, 17, 2)
    assert key_points.xy[0, [0, 2, 4]].tolist() == [
        [120.0, 225.0],
        [115.0, 235.0],
        [105.0, 235.0],
    ]
    assert key_points.xy[0, [1, 3]].abs().sum().item() == 0.0
    visible = key_points.to_supervision().visible
    assert visible[0].tolist() == [i in (0, 2, 4) for i in range(17)]
    assert len(crop_zones) == 1
