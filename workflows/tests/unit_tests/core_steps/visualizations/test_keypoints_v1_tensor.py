"""Tensor-native sibling of the partial-skeleton tests in ``test_keypoints.py``.

The tensor Keypoint Visualization renders ``KeyPoints.to_supervision()`` directly, so
it depends on the native ``KeyPoints`` keeping skeleton slots. Locally executed models
emit the full skeleton; remotely executed model steps, the rollup block and the
dynamic-block boundary rebuild ``KeyPoints`` through ``build_native_key_points``.
"""

import numpy as np
from roboflow_workflows.core_steps.common.keypoints import COCO_KEYPOINT_NAMES
from roboflow_workflows.core_steps.models.roboflow.keypoint_detection.v1_tensor import (
    _native_key_points_from_inference_predictions,
)
from roboflow_workflows.core_steps.visualizations.keypoint.v1_tensor import (
    KeypointVisualizationBlockV1,
)
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


def _remote_pose_response(keypoints: list) -> list:
    return [
        {
            "x": 250,
            "y": 250,
            "width": 400,
            "height": 400,
            "confidence": 0.9,
            "class": "person",
            "class_id": 0,
            "keypoints": [
                {
                    "x": 100.0 + 10 * class_id,
                    "y": 100.0 + 20 * class_id,
                    "confidence": 0.9,
                    "class": class_name,
                    "class_id": class_id,
                }
                for class_id, class_name in keypoints
            ],
        }
    ]


def _render_edges(response: list) -> np.ndarray:
    key_points = _native_key_points_from_inference_predictions(
        detection_dicts=response, image_metadata={}
    )
    output = KeypointVisualizationBlockV1().run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=np.zeros((500, 500, 3), dtype=np.uint8),
        ),
        predictions=(key_points, None),
        copy_image=True,
        annotator_type="edge",
        color="#A351FB",
        text_color="black",
        text_scale=0.5,
        text_thickness=1,
        text_padding=10,
        thickness=2,
        radius=10,
    )
    return output["image"].numpy_image


def test_tensor_keypoint_visualization_draws_edges_for_upper_body_only() -> None:
    # given: a remotely executed model step returned only COCO keypoints 0-10
    response = _remote_pose_response([(i, COCO_KEYPOINT_NAMES[i]) for i in range(11)])

    # when
    rendered = _render_edges(response)

    # then
    assert rendered.any(), "upper-body edges must be drawn"


def test_tensor_keypoint_visualization_does_not_draw_coco_bones_for_other_order() -> (
    None
):
    # given: a custom skeleton that reuses COCO names but numbers them differently
    response = _remote_pose_response([(0, "left_eye"), (1, "right_eye"), (2, "nose")])

    # when
    rendered = _render_edges(response)

    # then: no default skeleton for three keypoints, so nothing is drawn
    assert not rendered.any()
