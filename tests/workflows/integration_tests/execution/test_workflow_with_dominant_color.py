import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

MINIMAL_DOMINANT_COLOR_WORKFLOW = {
    "version": "1.0",
    "inputs": [{"type": "InferenceImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/dominant_color@v1",
            "name": "dominant_color",
            "image": "$inputs.image",
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "color",
            "coordinates_system": "own",
            "selector": "$steps.dominant_color.rgb_color",
        }
    ],
}


@add_to_workflows_gallery(
    category="Workflows with classical Computer Vision methods",
    use_case_title="Workflow calculating dominant color",
    use_case_description="""
This example shows how Dominant Color block can be used against input image. 
    """,
    workflow_definition=MINIMAL_DOMINANT_COLOR_WORKFLOW,
    workflow_name_in_app="dominant-color",
)
def test_dominant_color_workflow_when_minimal_valid_input_provided(
    model_manager: ModelManager,
    red_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=MINIMAL_DOMINANT_COLOR_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": red_image,
        }
    )
    print(result)
    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "color",
    }, "Expected all declared outputs to be delivered"
    assert len(result[0]["color"]) == 3, "Expected 3 RGB values in the color field"
    assert result[0]["color"] == (255, 0, 0), "Expected red dominant color in the image"


DOMINANT_COLOR_WORKFLOW = {
    "version": "1.0",
    "inputs": [{"type": "InferenceImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/dominant_color@v1",
            "name": "dominant_color",
            "image": "$inputs.image",
            "color_clusters": 4,
            "max_iterations": 100,
            "target_size": 100,
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "color",
            "coordinates_system": "own",
            "selector": "$steps.dominant_color.rgb_color",
        }
    ],
}


def test_dominant_color_workflow_when_additional_input_provided(
    model_manager: ModelManager,
    red_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=DOMINANT_COLOR_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": red_image,
        }
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "color",
    }, "Expected all declared outputs to be delivered"
    assert len(result[0]["color"]) == 3, "Expected 3 RGB values in the color field"
    assert result[0]["color"] == (255, 0, 0), "Expected red dominant color in the image"
