"""
This test module requires Stability AI image generation API key passed via env variable WORKFLOWS_TEST_STABILITY_AI_KEY.
This is supposed to be used only locally, as that would be too much of a cost in CI
"""

import os

import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

STABILITY_AI_API_KEY = os.getenv("WORKFLOWS_TEST_STABILITY_AI_KEY")

UNCONSTRAINED_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowParameter", "name": "api_key"},
        {
            "type": "WorkflowParameter",
            "name": "prompt",
            "default_value": "Raccoon in space suit",
        },
        {"type": "InferenceImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "roboflow_core/stability_ai_image_gen@v1",
            "name": "stability_ai_image_generation",
            "prompt": "$inputs.prompt",
            "api_key": "$inputs.api_key",
            "image": "$inputs.image",
            "strength": 0.3,
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "stability_ai_image_generation",
            "coordinates_system": "own",
            "selector": "$steps.stability_ai_image_generation.image",
        }
    ],
}


@add_to_workflows_gallery(
    category="Workflows with Visual Language Models",
    use_case_title="Prompting Stability-AI with arbitrary prompt",
    use_case_description="""
In this example, Stability-AI image generation model is prompted with arbitrary text from user 
    """,
    workflow_definition=UNCONSTRAINED_WORKFLOW,
    workflow_name_in_app="stability-ai-arbitrary-prompt",
)
@pytest.mark.skipif(
    condition=STABILITY_AI_API_KEY is None, reason="Stability-AI API key not provided"
)
def test_stabilit_ai_image_generation_workflow_with_unconstrained_prompt(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=UNCONSTRAINED_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
            "api_key": STABILITY_AI_API_KEY,
            "prompt": "Raccoon in space suit",
        }
    )

    # then
    assert len(result) == 1, "Single image given, expected single output"
    assert set(result[0].keys()) == {
        "stability_ai_image_generation"
    }, "Expected all outputs to be delivered"
