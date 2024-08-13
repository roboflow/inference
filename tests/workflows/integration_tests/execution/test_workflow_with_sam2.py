import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

SAM_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"}
    ],
    "steps": [
        {
            "type": "roboflow_core/segment_anything@v1",
            "name": "segment_anything",
            "images": "$inputs.image"
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.segment_anything.predictions",
        },
    ],
}


@pytest.mark.asyncio
async def test_sam2_workflow_when_minimal_valid_input_provided(
    model_manager: ModelManager,
    dogs_image: np.ndarray
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=SAM_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image]
        }
    )


    print("result", result)

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 2 elements in the output for two input images"
    assert set(result[0].keys()) == {
        "predictions",
    }, "Expected all declared outputs to be delivered"
    assert result[0]["predictions"].mask is not None, "Expected mask to be delivered"
    assert result[0]["predictions"].mask.shape == (1, 427, 640)
    assert result[0]["predictions"].data['prediction_type'][0] == 'instance-segmentation'