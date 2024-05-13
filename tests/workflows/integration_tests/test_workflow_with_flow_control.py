from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.entities.base import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.introspection import blocks_loader

FLOW_CONTROL_WORKFLOW = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "ABTest",
            "name": "ab_test",
            "a_step": "$steps.a",
            "b_step": "$steps.b",
        },
        {
            "type": "ObjectDetectionModel",
            "name": "a",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
        },
        {
            "type": "ObjectDetectionModel",
            "name": "b",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions_a",
            "selector": "$steps.a.predictions",
        },
        {
            "type": "JsonField",
            "name": "predictions_b",
            "selector": "$steps.b.predictions",
        },
    ],
}


@pytest.mark.asyncio
@mock.patch.object(blocks_loader, "get_plugin_modules")
async def test_flow_control_model(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.flow_control_plugin"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=FLOW_CONTROL_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    # when
    result = await execution_engine.run_async(runtime_parameters={"image": crowd_image})

    # then
    assert set(result.keys()) == {
        "predictions_a",
        "predictions_b",
    }, "Expected all declared outputs to be delivered"
    assert (len(result["predictions_a"]) > 0) != (
        len(result["predictions_b"]) > 0
    ), "Expected only one of two outputs to be filled with data"
