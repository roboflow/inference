from datetime import datetime
from unittest import mock
from unittest.mock import MagicMock

import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.introspection import blocks_loader

TIME_WORKFLOW = {
    "version": "1.0",
    "inputs": [],
    "steps": [
        {
            "type": "CurrentTime",
            "name": "time_step",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "time",
            "selector": "$steps.time_step.time",
        }
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_flow_control_step_not_operating_on_batches(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.input_free_blocks_plugin"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=TIME_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(runtime_parameters={})

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided, so one output element expected"
    assert set(result[0].keys()) == {
        "time",
    }, "Expected all declared outputs to be delivered"
    reported_time = datetime.fromisoformat(result[0]["time"])
    assert isinstance(reported_time, datetime)
