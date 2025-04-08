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

SMOLVLM2_WORKFLOW_DEFINITION = {
    "version": "1.0",
    "inputs": [
        {"type": "InferenceImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "confidence", "default_value": 0.4},
    ],
    "steps": [
        {
            "type": "roboflow_core/smolvlm2@v1",
            "name": "model",
            "images": "$inputs.image",
            "task_type": "lmm",
            "prompt": "What is in this image?",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "model_predictions",
            "coordinates_system": "own",
            "selector": "$steps.model.*",
        }
    ],
}


@add_to_workflows_gallery(
    category="Workflows with Visual Language Models",
    use_case_title="SmolVLM2",
    use_case_description="""
**THIS EXAMPLE CAN ONLY BE RUN LOCALLY OR USING DEDICATED DEPLOYMENT**

Use SmolVLM2 to ask questions about images, including documents and photos, and get answers in natural language.
    """,
    workflow_definition=SMOLVLM2_WORKFLOW_DEFINITION,
    workflow_name_in_app="smolvlm2",
)
@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_SMOLVLM2_TEST", True)), reason="Skipping SmolVLM 2 test"
)
def test_smolvlm2_inference(
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
        workflow_definition=SMOLVLM2_WORKFLOW_DEFINITION,
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
