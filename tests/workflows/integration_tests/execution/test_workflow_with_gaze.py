import sys

import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

GAZE_DETECTION_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {
            "type": "WorkflowParameter",
            "name": "do_run_face_detection",
            "default_value": True,
        },
    ],
    "steps": [
        {
            "type": "roboflow_core/gaze@v1",
            "name": "gaze",
            "images": "$inputs.image",
            "do_run_face_detection": "$inputs.do_run_face_detection",
        },
        {
            "type": "roboflow_core/keypoint_visualization@v1",
            "name": "visualization",
            "predictions": "$steps.gaze.face_predictions",
            "image": "$inputs.image",
            "annotator_type": "vertex",
            "color": "#A351FB",
            "text_color": "black",
            "text_scale": 0.5,
            "text_thickness": 1,
            "text_padding": 10,
            "thickness": 2,
            "radius": 10,
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "face_predictions",
            "selector": "$steps.gaze.face_predictions",
        },
        {
            "type": "JsonField",
            "name": "yaw_degrees",
            "selector": "$steps.gaze.yaw_degrees",
        },
        {
            "type": "JsonField",
            "name": "pitch_degrees",
            "selector": "$steps.gaze.pitch_degrees",
        },
        {
            "type": "JsonField",
            "name": "visualization",
            "selector": "$steps.visualization.image",
        },
    ],
}


@add_to_workflows_gallery(
    category="Workflows with foundation models",
    use_case_title="Gaze Detection Workflow",
    use_case_description="""
This workflow uses L2CS-Net to detect faces and estimate their gaze direction.
The output includes:
- Face detections with facial landmarks
- Gaze angles (yaw and pitch) in degrees
- Visualization of facial landmarks
""",
    workflow_definition=GAZE_DETECTION_WORKFLOW,
    workflow_name_in_app="gaze-detection",
)
@pytest.mark.skip(
    reason="Test not supported on Python 3.12+, skipping due to dependencies conflict when building CI"
)
def test_gaze_workflow_with_face_detection(
    model_manager: ModelManager,
    face_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=GAZE_DETECTION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [face_image],
            "do_run_face_detection": True,
        }
    )

    # then
    assert len(result) == 1, "Single image given, expected single output"
    assert set(result[0].keys()) == {
        "face_predictions",
        "yaw_degrees",
        "pitch_degrees",
        "visualization",
    }, "Expected all outputs to be registered"

    # Check face predictions
    assert len(result[0]["face_predictions"]) > 0, "Expected at least one face detected"
    assert result[0]["face_predictions"].data["prediction_type"][0] == "facial-landmark"

    # Check angles
    assert len(result[0]["yaw_degrees"]) == len(result[0]["face_predictions"])
    assert len(result[0]["pitch_degrees"]) == len(result[0]["face_predictions"])

    # Check visualization
    assert not np.array_equal(
        face_image, result[0]["visualization"].numpy_image
    ), "Expected visualization to modify the image"


@pytest.mark.skip(
    reason="Test not supported on Python 3.12+, skipping due to dependencies conflict when building CI"
)
def test_gaze_workflow_batch_processing(
    model_manager: ModelManager,
    face_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=GAZE_DETECTION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [face_image, face_image],  # Process same image twice
            "do_run_face_detection": True,
        }
    )

    # then
    assert len(result) == 2, "Expected results for both images"
    # Results should be identical since we used the same image
    assert (
        result[0]["face_predictions"].box_area == result[1]["face_predictions"].box_area
    )
    assert result[0]["yaw_degrees"] == result[1]["yaw_degrees"]
    assert result[0]["pitch_degrees"] == result[1]["pitch_degrees"]
