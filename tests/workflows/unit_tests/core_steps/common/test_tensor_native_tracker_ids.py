"""Tests for first-class tracker IDs on tensor-native predictions."""

import torch

from inference.core.workflows.core_steps.common.serializers_tensor import (
    serialise_sv_detections,
)
from inference.core.workflows.core_steps.common.tensor_native import (
    take_prediction_by_indices,
)
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAMES_KEY,
    DETECTION_ID_KEY,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections


def _detections() -> Detections:
    """Build tracked object detections with serializable metadata."""
    return Detections(
        xyxy=torch.arange(12, dtype=torch.float32).reshape(3, 4),
        class_id=torch.tensor([0, 1, 0]),
        confidence=torch.tensor([0.9, 0.8, 0.7]),
        image_metadata={CLASS_NAMES_KEY: {0: "zero", 1: "one"}},
        bboxes_metadata=[
            {DETECTION_ID_KEY: "a"},
            {DETECTION_ID_KEY: "b"},
            {DETECTION_ID_KEY: "c"},
        ],
        tracker_id=torch.tensor([10, 11, 12]),
    )


def test_object_selection_preserves_same_device_tracker_ids() -> None:
    """Object gathers keep IDs aligned on their existing tensor device."""
    detections = _detections()
    indices = torch.tensor([2, 0])

    selected = take_prediction_by_indices(detections, indices)

    assert selected.tracker_id.device == detections.xyxy.device
    assert selected.tracker_id.tolist() == [12, 10]
    assert [item[DETECTION_ID_KEY] for item in selected.bboxes_metadata] == ["c", "a"]


def test_instance_selection_preserves_tracker_ids_and_masks() -> None:
    """Instance gathers carry IDs and dense masks through the same indexing."""
    base = _detections()
    detections = InstanceDetections(
        xyxy=base.xyxy,
        class_id=base.class_id,
        confidence=base.confidence,
        mask=torch.arange(48).reshape(3, 4, 4),
        image_metadata=base.image_metadata,
        bboxes_metadata=base.bboxes_metadata,
        tracker_id=base.tracker_id,
    )

    selected = take_prediction_by_indices(detections, torch.tensor([1]))

    assert selected.tracker_id.tolist() == [11]
    assert torch.equal(selected.mask, detections.mask[1:2])


def test_keypoint_bbox_selection_preserves_tracker_ids() -> None:
    """Keypoint prediction tuples keep IDs on the bounding-box component."""
    detections = _detections()
    key_points = KeyPoints(
        xy=torch.zeros((3, 2, 2)),
        class_id=detections.class_id,
        confidence=torch.ones((3, 2)),
    )

    selected_key_points, selected_boxes = take_prediction_by_indices(
        (key_points, detections),
        torch.tensor([2, 1]),
    )

    assert selected_key_points.xy.shape[0] == 2
    assert selected_boxes.tracker_id.tolist() == [12, 11]


def test_serializer_prefers_first_class_tracker_ids() -> None:
    """The host serializer emits tensor IDs without requiring metadata patches."""
    detections = _detections()
    detections.bboxes_metadata[0]["tracker_id"] = 999

    result = serialise_sv_detections(detections)

    assert [prediction["tracker_id"] for prediction in result["predictions"]] == [
        10,
        11,
        12,
    ]
