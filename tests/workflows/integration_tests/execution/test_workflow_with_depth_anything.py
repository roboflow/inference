import copy
import json
import os

import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from tests.workflows.integration_tests.execution.conftest import bool_env
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

DEPTH_ESTIMATION_WORKFLOW_DEFINITION = {
    "version": "1.0",
    "inputs": [
        {"type": "InferenceImage", "name": "image"},
    ],
    "steps": [
        {
        "type": "roboflow_core/depth_estimation@v1",
        "name": "depth_estimation",
        "images": "$inputs.image"
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "model_predictions",
            "coordinates_system": "own",
            "selector": "$steps.depth_estimation.*",
        }
    ],
}


@add_to_workflows_gallery(
    category="Workflows with Depth Estimation",
    use_case_title="Depth Estimation",
    use_case_description="""
**THIS EXAMPLE CAN ONLY BE RUN LOCALLY OR USING DEDICATED DEPLOYMENT**

Use Depth Estimation to estimate the depth of an image.
    """,
    workflow_definition=DEPTH_ESTIMATION_WORKFLOW_DEFINITION,
    workflow_name_in_app="depth_estimation",
)
@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_DEPTH_ESTIMATION_TEST", True)), reason="Skipping Depth Estimation test"
)
def test_depth_estimation_inference(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=DEPTH_ESTIMATION_WORKFLOW_DEFINITION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
        }
    )

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "model_predictions",
    }, "Expected all declared outputs to be delivered"
