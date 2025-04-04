import numpy as np
import pytest
import supervision as sv

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

DETECTIONS_MERGE_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "ObjectDetectionModel",
            "name": "detection",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
        },
        {
            "type": "roboflow_core/detections_merge@v1",
            "name": "detections_merge",
            "predictions": "$steps.detection.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.detections_merge.predictions",
        }
    ],
}


@add_to_workflows_gallery(
    category="Basic Workflows",
    use_case_title="Workflow with detections merge",
    use_case_description="""
This workflow demonstrates how to merge multiple object detections into a single bounding box.
This is useful when you want to:
- Combine overlapping detections of the same object
- Create a single region that contains multiple detected objects
- Simplify multiple detections into one larger detection
    """,
    workflow_definition=DETECTIONS_MERGE_WORKFLOW,
    workflow_name_in_app="merge-detections",
)
def test_detections_merge_workflow(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=DETECTIONS_MERGE_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image],
        }
    )

    # then
    assert len(result) == 1, "One set of outputs expected"
    assert "result" in result[0], "Output must contain key 'result'"
    assert isinstance(
        result[0]["result"], sv.Detections
    ), "Output must be instance of sv.Detections"

    # Check that we have exactly one merged detection
    assert len(result[0]["result"]) == 1, "Should have exactly one merged detection"

    # Check that the merged detection has all required fields
    assert "class_name" in result[0]["result"].data, "Should have class_name in data"
    assert (
        "detection_id" in result[0]["result"].data
    ), "Should have detection_id in data"

    # Check that the bounding box has reasonable dimensions
    merged_bbox = result[0]["result"].xyxy[0]
    image_height, image_width = dogs_image.shape[:2]

    # Check that coordinates are within image bounds
    assert 0 <= merged_bbox[0] <= image_width, "x1 should be within image bounds"
    assert 0 <= merged_bbox[1] <= image_height, "y1 should be within image bounds"
    assert 0 <= merged_bbox[2] <= image_width, "x2 should be within image bounds"
    assert 0 <= merged_bbox[3] <= image_height, "y2 should be within image bounds"

    # Check that the box has reasonable dimensions
    assert merged_bbox[2] > merged_bbox[0], "x2 should be greater than x1"
    assert merged_bbox[3] > merged_bbox[1], "y2 should be greater than y1"

    # Check that the box is large enough to likely contain the dogs
    box_width = merged_bbox[2] - merged_bbox[0]
    box_height = merged_bbox[3] - merged_bbox[1]
    assert box_width > 100, "Merged box should be reasonably wide"
    assert box_height > 100, "Merged box should be reasonably tall"
