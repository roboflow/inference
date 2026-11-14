import numpy as np
import pytest
import supervision as sv
import torch

from inference.core.env import ENABLE_TENSOR_DATA_REPRESENTATION
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAMES_KEY,
    DETECTION_ID_KEY,
)
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference_models.models.base.object_detection import Detections as NativeDetections


def _predictions(boxes: list[list[int]]) -> sv.Detections | NativeDetections:
    labels = ["first", "second", "reference"]
    detection_ids = ["a", "b", "ref"]
    if ENABLE_TENSOR_DATA_REPRESENTATION:
        return NativeDetections(
            xyxy=torch.tensor(boxes, dtype=torch.float32),
            class_id=torch.arange(3),
            confidence=torch.ones(3),
            image_metadata={CLASS_NAMES_KEY: dict(enumerate(labels))},
            bboxes_metadata=[{DETECTION_ID_KEY: value} for value in detection_ids],
        )

    predictions = sv.Detections(
        xyxy=np.array(boxes, dtype=np.float32),
        class_id=np.arange(3),
        confidence=np.ones(3),
        data={
            "class_name": np.array(labels),
            DETECTION_ID_KEY: np.array(detection_ids),
        },
    )
    return predictions


@pytest.mark.parametrize("axis", ["horizontal", "vertical"])
@pytest.mark.parametrize("calibration", ["pixel to centimeter", "reference object"])
def test_distance_measurement_workflow_at_projected_edge(
    axis: str, calibration: str
) -> None:
    workflow = {
        "version": "1.0",
        "inputs": [
            {
                "type": "WorkflowBatchInput",
                "name": "predictions",
                "kind": ["object_detection_prediction"],
            }
        ],
        "steps": [
            {
                "type": "roboflow_core/distance_measurement@v1",
                "name": "distance",
                "predictions": "$inputs.predictions",
                "object_1_class_name": "first",
                "object_2_class_name": "second",
                "reference_axis": axis,
                "calibration_method": calibration,
                "reference_object_class_name": "reference",
                "reference_width": 10.0,
                "reference_height": 10.0,
                "pixel_ratio": 2.0,
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": name,
                "selector": f"$steps.distance.{name}",
            }
            for name in ("distance_pixel", "distance_cm")
        ],
    }
    # The boxes stay separated on the other axis as the measured gap goes
    # from zero to positive and then becomes a projected overlap.
    second_boxes = [[10, 30, 20, 40], [15, 30, 25, 40], [5, 30, 15, 40]]
    if axis == "vertical":
        second_boxes = [[y1, x1, y2, x2] for x1, y1, x2, y2 in second_boxes]

    predictions = [
        _predictions([[0, 0, 10, 10], second_box, [50, 50, 70, 70]])
        for second_box in second_boxes
    ]

    engine = ExecutionEngine.init(workflow_definition=workflow, init_parameters={})
    try:
        result = engine.run(
            runtime_parameters={"predictions": predictions}, serialize_results=False
        )
    finally:
        engine.close()

    assert result == [
        {"distance_pixel": 0, "distance_cm": 0},
        {"distance_pixel": 5, "distance_cm": 2.5},
        {"distance_pixel": 0, "distance_cm": 0},
    ]
