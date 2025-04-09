import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.transformations.camera_calibration.v1 import (
    OUTPUT_CALIBRATED_IMAGE_KEY,
)
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

WORKFLOW_WITH_CAMERA_CALIBRATION = {
    "version": "1.0",
    "inputs": [
        {"type": "InferenceImage", "name": "images"},
        {"type": "WorkflowParameter", "name": "fx"},
        {"type": "WorkflowParameter", "name": "fy"},
        {"type": "WorkflowParameter", "name": "cx"},
        {"type": "WorkflowParameter", "name": "cy"},
        {"type": "WorkflowParameter", "name": "k1"},
        {"type": "WorkflowParameter", "name": "k2"},
        {"type": "WorkflowParameter", "name": "k3"},
        {"type": "WorkflowParameter", "name": "p1"},
        {"type": "WorkflowParameter", "name": "p2"},
    ],
    "steps": [
        {
            "type": "roboflow_core/camera-calibration@v1",
            "name": "camera_calibration",
            "images": "$inputs.images",
            "fx": "$inputs.fx",
            "fy": "$inputs.fy",
            "cx": "$inputs.cx",
            "cy": "$inputs.cy",
            "k1": "$inputs.k1",
            "k2": "$inputs.k2",
            "k3": "$inputs.k3",
            "p1": "$inputs.p1",
            "p2": "$inputs.p2",
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "camera_calibration_image",
            "coordinates_system": "own",
            "selector": f"$steps.camera_calibration.{OUTPUT_CALIBRATED_IMAGE_KEY}",
        }
    ],
}


@add_to_workflows_gallery(
    category="Workflows with classical Computer Vision methods",
    use_case_title="Workflow removing camera distortions",
    use_case_description="""
In this example, we demonstrate how to remove distortions from the camera based on coefficients provided.
    """,
    workflow_definition=WORKFLOW_WITH_CAMERA_CALIBRATION,
    workflow_name_in_app="camera-calibration",
)
def test_workflow_with_camera_calibration(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_CAMERA_CALIBRATION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "images": [dogs_image],
            "fx": 1.48052348e03,
            "fy": 1.62041507e03,
            "cx": 7.76228486e02,
            "cy": 5.09102914e02,
            "k1": -0.67014685,
            "k2": 0.84140975,
            "k3": -0.40499778,
            "p1": -0.00559933,
            "p2": 0.00425916,
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "One image provided, so one output expected"
    assert set(result[0].keys()) == {
        "camera_calibration_image",
    }, "Expected all declared outputs to be delivered"
