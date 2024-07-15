import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

FUNCTION_TO_GET_OVERLAP_OF_BBOXES = """
def run(predictions: sv.Detections, class_x: str, class_y: str) -> BlockResult:
    bboxes_class_x = predictions[predictions.data["class_name"] == class_x]
    bboxes_class_y = predictions[predictions.data["class_name"] == class_y]
    overlap = []
    for bbox_x in bboxes_class_x:
        bbox_x_coords = bbox_x[0]
        bbox_overlaps = []
        for bbox_y in bboxes_class_y:
            if bbox_y[-1]["detection_id"] == bbox_x[-1]["detection_id"]:
                continue
            bbox_y_coords = bbox_y[0]
            x_min = max(bbox_x_coords[0], bbox_y_coords[0])
            y_min = max(bbox_x_coords[1], bbox_y_coords[1])
            x_max = min(bbox_x_coords[2], bbox_y_coords[2])
            y_max = min(bbox_x_coords[3], bbox_y_coords[3])
            # compute the area of intersection rectangle
            intersection_area = max(0, x_max - x_min + 1) * max(0, y_max - y_min + 1)
            box_x_area = (bbox_x_coords[2] - bbox_x_coords[0] + 1) * (bbox_x_coords[3] - bbox_x_coords[1] + 1)
            local_overlap = intersection_area / (box_x_area + 1e-5)
            bbox_overlaps.append(local_overlap)
        overlap.append(bbox_overlaps)
    return  {"overlap": overlap}
"""


FUNCTION_TO_GET_MAXIMUM_OVERLAP = """
def run(overlaps: List[List[float]]) -> BlockResult:
    max_value = -1
    for overlap in overlaps:
        for overlap_value in overlap:
            if not max_value:
                max_value = overlap_value
            else:
                max_value = max(max_value, overlap_value)
    return {"max_value": max_value}
"""

WORKFLOW_WITH_OVERLAP_MEASUREMENT = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "dynamic_blocks_definitions": [
        {
            "type": "DynamicBlockDefinition",
            "manifest": {
                "type": "ManifestDescription",
                "block_type": "OverlapMeasurement",
                "inputs": {
                    "predictions": {
                        "type": "DynamicInputDefinition",
                        "selector_types": ["step_output"],
                    },
                    "class_x": {
                        "type": "DynamicInputDefinition",
                        "value_types": ["string"],
                    },
                    "class_y": {
                        "type": "DynamicInputDefinition",
                        "value_types": ["string"],
                    },
                },
                "outputs": {"overlap": {"type": "DynamicOutputDefinition", "kind": []}},
            },
            "code": {
                "type": "PythonCode",
                "function_code": FUNCTION_TO_GET_OVERLAP_OF_BBOXES,
            },
        },
        {
            "type": "DynamicBlockDefinition",
            "manifest": {
                "type": "ManifestDescription",
                "block_type": "MaximumOverlap",
                "inputs": {
                    "overlaps": {
                        "type": "DynamicInputDefinition",
                        "selector_types": ["step_output"],
                    },
                },
                "outputs": {
                    "max_value": {"type": "DynamicOutputDefinition", "kind": []}
                },
            },
            "code": {
                "type": "PythonCode",
                "function_code": FUNCTION_TO_GET_MAXIMUM_OVERLAP,
            },
        },
    ],
    "steps": [
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "model",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
        },
        {
            "type": "OverlapMeasurement",
            "name": "overlap_measurement",
            "predictions": "$steps.model.predictions",
            "class_x": "dog",
            "class_y": "dog",
        },
        {
            "type": "ContinueIf",
            "name": "continue_if",
            "condition_statement": {
                "type": "StatementGroup",
                "statements": [
                    {
                        "type": "BinaryStatement",
                        "left_operand": {
                            "type": "DynamicOperand",
                            "operand_name": "overlaps",
                            "operations": [{"type": "SequenceLength"}],
                        },
                        "comparator": {"type": "(Number) >="},
                        "right_operand": {
                            "type": "StaticOperand",
                            "value": 1,
                        },
                    }
                ],
            },
            "evaluation_parameters": {"overlaps": "$steps.overlap_measurement.overlap"},
            "next_steps": ["$steps.maximum_overlap"],
        },
        {
            "type": "MaximumOverlap",
            "name": "maximum_overlap",
            "overlaps": "$steps.overlap_measurement.overlap",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "overlaps",
            "selector": "$steps.overlap_measurement.overlap",
        },
        {
            "type": "JsonField",
            "name": "max_overlap",
            "selector": "$steps.maximum_overlap.max_value",
        },
    ],
}


@pytest.mark.asyncio
async def test_workflow_with_custom_python_blocks_measuring_overlap(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
        "workflows_core.allow_custom_python_execution": True,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_OVERLAP_MEASUREMENT,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = await execution_engine.run_async(
        runtime_parameters={
            "image": [dogs_image, crowd_image],
        }
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 2, "Expected 2 elements in the output for two input images"
    assert set(result[0].keys()) == {
        "overlaps",
        "max_overlap",
    }, "Expected all declared outputs to be delivered"
    assert set(result[1].keys()) == {
        "overlaps",
        "max_overlap",
    }, "Expected all declared outputs to be delivered"
    assert (
        len(result[0]["overlaps"]) == 2
    ), "Expected 2 instances of dogs found, each overlap with another for first image"
    assert (
        abs(result[0]["max_overlap"] - 0.177946) < 1e-3
    ), "Expected max overlap to be calculated properly"
    assert (
        len(result[1]["overlaps"]) == 0
    ), "Expected no instances of dogs found for second image"
    assert (
        result[1]["max_overlap"] is None
    ), "Expected `max_overlap` not to be calculated for second image due to conditional execution"
