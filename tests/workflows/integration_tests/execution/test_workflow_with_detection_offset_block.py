import numpy as np
import pytest
import supervision as sv
import torch

from inference.core.env import ENABLE_TENSOR_DATA_REPRESENTATION
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAMES_KEY,
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
)
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference_models.models.base.object_detection import Detections as NativeDetections


def _prediction(
    box: list[int], dimensions: list[int]
) -> sv.Detections | NativeDetections:
    if ENABLE_TENSOR_DATA_REPRESENTATION:
        return NativeDetections(
            xyxy=torch.tensor([box], dtype=torch.uint8),
            class_id=torch.tensor([0]),
            confidence=torch.tensor([0.5]),
            image_metadata={
                IMAGE_DIMENSIONS_KEY: dimensions,
                CLASS_NAMES_KEY: {0: "object"},
            },
            bboxes_metadata=[{DETECTION_ID_KEY: "source-box"}],
        )

    prediction = sv.Detections(
        xyxy=np.array([box], dtype=np.uint8),
        class_id=np.array([0]),
        confidence=np.array([0.5]),
        data={
            IMAGE_DIMENSIONS_KEY: np.array([dimensions]),
            "class_name": np.array(["object"]),
            DETECTION_ID_KEY: np.array(["source-box"]),
        },
    )
    return prediction


EDGE_BOXES = [
    ([2, 3, 20, 30], [255, 255]),
    ([230, 240, 250, 253], [300, 320]),
]


@pytest.mark.parametrize(
    "units, offset_width, offset_height, inputs, expected",
    [
        ("Pixels", 20, 10, EDGE_BOXES, [[0, 0, 30, 35], [220, 235, 260, 258]]),
        ("Percent (%)", 100, 100, EDGE_BOXES, [[0, 0, 29, 43], [220, 234, 260, 259]]),
        ("Pixels", 10**20, 10**20, EDGE_BOXES, [[0, 0, 255, 255], [0, 0, 320, 300]]),
        (
            "Percent (%)",
            10**18,
            10**18,
            EDGE_BOXES,
            [[0, 0, 255, 255], [0, 0, 320, 300]],
        ),
        (
            "Pixels",
            np.uint8(20),
            np.uint8(10),
            EDGE_BOXES,
            [[0, 0, 30, 35], [220, 235, 260, 258]],
        ),
        (
            "Percent (%)",
            np.uint8(100),
            np.uint8(100),
            EDGE_BOXES,
            [[0, 0, 29, 43], [220, 234, 260, 259]],
        ),
        # 253 * 66483 / 200 is 84100.995. Round toward zero before adding the edge.
        ("Percent (%)", 66483, 1, [([0, 3, 253, 8], [30, 84355])], [[0, 3, 84353, 8]]),
    ],
    ids=[
        "pixels",
        "percent",
        "oversized-pixels",
        "oversized-percent",
        "selector-pixels",
        "selector-percent",
        "exact-percent",
    ],
)
def test_detection_offset_workflow_uint8_boxes(
    units: str,
    offset_width: int | np.uint8,
    offset_height: int | np.uint8,
    inputs: list[tuple[list[int], list[int]]],
    expected: list[list[int]],
) -> None:
    workflow = {
        "version": "1.0",
        "inputs": [
            {
                "type": "WorkflowBatchInput",
                "name": "detections",
                "kind": ["object_detection_prediction"],
            }
        ],
        "steps": [
            {
                "type": "roboflow_core/detection_offset@v1",
                "name": "pad",
                "predictions": "$inputs.detections",
                "offset_width": offset_width,
                "offset_height": offset_height,
                "units": units,
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "boxes",
                "selector": "$steps.pad.predictions",
                "coordinates_system": "own",
            }
        ],
    }
    detections = [_prediction(box, dimensions) for box, dimensions in inputs]
    runtime_parameters = {"detections": detections}
    for name, value in (
        ("offset_width", offset_width),
        ("offset_height", offset_height),
    ):
        if isinstance(value, np.uint8):
            workflow["inputs"].append(
                {"type": "WorkflowParameter", "name": name, "kind": ["integer"]}
            )
            workflow["steps"][0][name] = f"$inputs.{name}"
            runtime_parameters[name] = value

    engine = ExecutionEngine.init(workflow_definition=workflow, init_parameters={})
    try:
        result = engine.run(
            runtime_parameters=runtime_parameters, serialize_results=False
        )
    finally:
        engine.close()

    assert [row["boxes"].xyxy[0].tolist() for row in result] == expected
