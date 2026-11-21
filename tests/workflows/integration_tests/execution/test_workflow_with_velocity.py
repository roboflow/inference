"""Exercise velocity after a detections difference in a tracked workflow."""

import datetime
from typing import TYPE_CHECKING

import numpy as np
import supervision as sv
from roboflow_workflows.environment import ENABLE_TENSOR_DATA_REPRESENTATION
from roboflow_workflows.execution_engine.core import ExecutionEngine
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)

if TYPE_CHECKING:
    from inference_models.models.base.object_detection import Detections


WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {
            "type": "WorkflowBatchInput",
            "name": "detections",
            "kind": ["object_detection_prediction"],
            "dimensionality": 1,
        },
        {
            "type": "WorkflowBatchInput",
            "name": "reference",
            "kind": ["object_detection_prediction"],
            "dimensionality": 1,
        },
    ],
    "steps": [
        {
            "type": "roboflow_core/byte_tracker@v3",
            "name": "tracker",
            "image": "$inputs.image",
            "detections": "$inputs.detections",
        },
        {
            "type": "roboflow_core/detections_difference@v1",
            "name": "difference",
            "reference_predictions": "$inputs.reference",
            "candidate_predictions": "$steps.tracker.tracked_detections",
        },
        {
            "type": "roboflow_core/velocity@v1",
            "name": "velocity",
            "image": "$inputs.image",
            "detections": "$steps.difference.persisted_detections",
            "smoothing_alpha": 1.0,
            "pixels_per_meter": 2.0,
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "detections",
            "coordinates_system": "own",
            "selector": "$steps.velocity.velocity_detections",
        }
    ],
}


def _detections(x: float, *, empty: bool = False) -> "sv.Detections | Detections":
    boxes = np.array([[x, 8.0, x + 30.0, 38.0]], dtype=np.float32)
    if empty:
        boxes = boxes[:0]

    class_ids = np.zeros(len(boxes), dtype=int)
    confidences = np.full(len(boxes), 0.95, dtype=np.float32)
    image_metadata = {"image_dimensions": np.array([48, 48])}
    if ENABLE_TENSOR_DATA_REPRESENTATION:
        import torch

        from inference_models.models.base.object_detection import Detections

        return Detections(
            xyxy=torch.from_numpy(boxes),
            class_id=torch.from_numpy(class_ids),
            confidence=torch.from_numpy(confidences),
            bboxes_metadata=(
                [{"detection_id": "box", "class_name": "object"}] if len(boxes) else []
            ),
            image_metadata=image_metadata,
        )

    return sv.Detections(
        xyxy=boxes,
        class_id=class_ids,
        confidence=confidences,
        data={
            "detection_id": np.full(len(boxes), "box"),
            "class_name": np.full(len(boxes), "object"),
        },
        metadata=image_metadata,
    )


def test_workflow_with_velocity_accepts_frame_without_persisted_detections() -> None:
    """Preserve the track across a frame for which the difference output is empty."""
    engine = ExecutionEngine.init(workflow_definition=WORKFLOW)
    outputs = []
    try:
        for frame in range(1, 4):
            image = WorkflowImageData(
                parent_metadata=ImageParentMetadata(parent_id="source"),
                numpy_image=np.zeros((48, 48, 3), dtype=np.uint8),
                video_metadata=VideoMetadata(
                    video_identifier="video",
                    frame_number=frame,
                    frame_timestamp=datetime.datetime.fromtimestamp(
                        frame, tz=datetime.timezone.utc
                    ),
                    fps=1.0,
                    comes_from_video_file=True,
                ),
            )
            x = 8.0 + 2.0 * frame
            (output,) = engine.run(
                runtime_parameters={
                    "image": image,
                    "detections": [_detections(x)],
                    "reference": [_detections(x, empty=frame == 2)],
                },
                serialize_results=False,
            )
            outputs.append(output["detections"])
    finally:
        engine.close()

    first, empty, last = outputs
    assert [len(first), len(empty), len(last)] == [1, 0, 1]
    if ENABLE_TENSOR_DATA_REPRESENTATION:
        assert (
            last.bboxes_metadata[0]["tracker_id"]
            == first.bboxes_metadata[0]["tracker_id"]
        )
        assert last.bboxes_metadata[0]["speed"] == 1.0
    else:
        np.testing.assert_array_equal(last.tracker_id, first.tracker_id)
        np.testing.assert_array_equal(last.data["speed"], [1.0])
