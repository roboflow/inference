import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

WORKFLOW_WITH_CAMERA_FOCUS_V1 = {
    "version": "1.0",
    "inputs": [{"type": "InferenceImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/camera_focus@v1",
            "name": "camera_focus",
            "image": "$inputs.image",
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "camera_focus_image",
            "coordinates_system": "own",
            "selector": "$steps.camera_focus.image",
        },
        {
            "type": "JsonField",
            "name": "camera_focus_measure",
            "selector": "$steps.camera_focus.focus_measure",
        },
    ],
}

WORKFLOW_WITH_CAMERA_FOCUS_V2 = {
    "version": "1.0",
    "inputs": [{"type": "InferenceImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/camera_focus@v2",
            "name": "camera_focus",
            "image": "$inputs.image",
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "camera_focus_image",
            "coordinates_system": "own",
            "selector": "$steps.camera_focus.image",
        },
        {
            "type": "JsonField",
            "name": "camera_focus_measure",
            "selector": "$steps.camera_focus.focus_measure",
        },
        {
            "type": "JsonField",
            "name": "bbox_focus_measures",
            "selector": "$steps.camera_focus.bbox_focus_measures",
        },
    ],
}


def test_workflow_with_camera_focus_v1(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_CAMERA_FOCUS_V1,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "One image provided, so one output expected"
    assert set(result[0].keys()) == {
        "camera_focus_image",
        "camera_focus_measure",
    }, "Expected all declared outputs to be delivered"
    assert isinstance(
        result[0]["camera_focus_measure"], float
    ), "Expected camera focus output to be a float"
    assert (
        abs(result[0]["camera_focus_measure"] - 131.16) < 1e-2
    ), "Expected focus score to be close to 131.16"


@add_to_workflows_gallery(
    category="Workflows with classical Computer Vision methods",
    use_case_title="Workflow generating camera focus measure",
    use_case_description="""
In this example, we demonstrate how to evaluate camera focus using the Tenengrad focus measure
with visualization overlays including zebra warnings, focus peaking, HUD, and composition grid.
    """,
    workflow_definition=WORKFLOW_WITH_CAMERA_FOCUS_V2,
    workflow_name_in_app="camera-focus",
)
def test_workflow_with_camera_focus_v2(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_CAMERA_FOCUS_V2,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "One image provided, so one output expected"
    assert set(result[0].keys()) == {
        "camera_focus_image",
        "camera_focus_measure",
        "bbox_focus_measures",
    }, "Expected all declared outputs to be delivered"
    assert isinstance(
        result[0]["camera_focus_measure"], float
    ), "Expected camera focus output to be a float"
    assert result[0]["camera_focus_measure"] > 0, "Expected focus score to be positive"
    assert (
        result[0]["bbox_focus_measures"] == []
    ), "Expected empty bbox_focus_measures when no detections provided"
