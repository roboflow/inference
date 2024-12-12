import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)


WORKFLOW_DEFINITION = {
  "version": "1.0",
  "inputs": [
    {
      "type": "InferenceImage",
      "name": "image"
    }
  ],
  "steps": [
    {
      "type": "roboflow_core/depth_anything_v2@v1",
      "name": "depth_anything_v2",
      "image": "$inputs.image",
      "model_size": "Large",
      "colormap": "viridis",
      "min_depth": 0,
      "max_depth": 1
    }
  ],
  "outputs": [
    {
      "type": "JsonField",
      "name": "normalized_depth",
      "coordinates_system": "own",
      "selector": "$steps.depth_anything_v2.normalized_depth"
    },
    {
      "type": "JsonField",
      "name": "image",
      "coordinates_system": "own",
      "selector": "$steps.depth_anything_v2.image"
    }
  ],
}



@add_to_workflows_gallery(
    category="Workflows with model blocks",
    use_case_title="Workflow with depth anything v2",
    workflow_name_in_app="depth-anything-v2",
    use_case_description="""
This workflow demonstrates how to visualize the predictions of a depth anything v2 model. 
    """,
    workflow_definition=WORKFLOW_DEFINITION,
)
def test_depth_anything_v2_workflow_when_valid_input_provided(
    model_manager: ModelManager,
    fruit_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_DEFINITION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,    
    )

    # when
    result = execution_engine.run(runtime_parameters={"image": fruit_image})

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "normalized_depth",
        "image",
    }, "Expected all declared outputs to be delivered"
