"""
This test module requires an OpenRouter API key passed via env variable
WORKFLOWS_TEST_OPEN_ROUTER_API_KEY. This is supposed to be used only locally,
as that would be too much of a cost in CI.
"""

import os

import numpy as np
import pytest
import supervision as sv

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

OPEN_ROUTER_API_KEY = os.getenv("WORKFLOWS_TEST_OPEN_ROUTER_API_KEY")

OBJECT_DETECTION_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "api_key"},
        {"type": "WorkflowParameter", "name": "classes"},
    ],
    "steps": [
        {
            "type": "roboflow_core/qwen_vlm@v2",
            "name": "qwen",
            "images": "$inputs.image",
            "backend": "openrouter",
            "openrouter_model_version": "Qwen 3.7 Plus",
            "task_type": "object-detection",
            "classes": "$inputs.classes",
            "api_key": "$inputs.api_key",
        },
        {
            "type": "roboflow_core/vlm_as_detector@v2",
            "name": "parser",
            "vlm_output": "$steps.qwen.output",
            "image": "$inputs.image",
            "classes": "$steps.qwen.classes",
            "model_type": "qwen",
            "task_type": "object-detection",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "qwen_result",
            "selector": "$steps.qwen.output",
        },
        {
            "type": "JsonField",
            "name": "parsed_prediction",
            "selector": "$steps.parser.predictions",
        },
    ],
}


def test_object_detection_workflow_compiles(model_manager: ModelManager) -> None:
    """Ungated compile-only check: the live tests below are API-key-gated and
    never run in CI, so without this nothing in CI would compile the
    qwen_vlm@v2 -> vlm_as_detector@v2 block pair."""
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }

    execution_engine = ExecutionEngine.init(
        workflow_definition=OBJECT_DETECTION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    assert execution_engine is not None


@pytest.mark.skipif(
    condition=OPEN_ROUTER_API_KEY is None, reason="OpenRouter API key not provided"
)
def test_workflow_with_object_detection_prompt(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=OBJECT_DETECTION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image],
            "api_key": OPEN_ROUTER_API_KEY,
            "classes": ["dog"],
        }
    )

    # then
    assert len(result) == 1, "Single image given, expected single output"
    assert set(result[0].keys()) == {
        "qwen_result",
        "parsed_prediction",
    }, "Expected all outputs to be delivered"
    assert isinstance(
        result[0]["parsed_prediction"], sv.Detections
    ), "Expected parsed detections"
    assert len(result[0]["parsed_prediction"]) > 0, "Expected dogs to be detected"
    assert set(result[0]["parsed_prediction"]["class_name"].tolist()) == {
        "dog"
    }, "Expected only dogs to be detected"


UNCONSTRAINED_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "api_key"},
        {"type": "WorkflowParameter", "name": "prompt"},
    ],
    "steps": [
        {
            "type": "roboflow_core/qwen_vlm@v2",
            "name": "qwen",
            "images": "$inputs.image",
            "backend": "openrouter",
            "openrouter_model_version": "Qwen 3.7 Flash",
            "task_type": "unconstrained",
            "prompt": "$inputs.prompt",
            "api_key": "$inputs.api_key",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.qwen.output",
        },
    ],
}


@pytest.mark.skipif(
    condition=OPEN_ROUTER_API_KEY is None, reason="OpenRouter API key not provided"
)
def test_workflow_with_unconstrained_prompt(
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
            "image": [dogs_image],
            "api_key": OPEN_ROUTER_API_KEY,
            "prompt": "What animals are in this image?",
        }
    )

    # then
    assert len(result) == 1, "Single image given, expected single output"
    assert set(result[0].keys()) == {"result"}, "Expected all outputs to be delivered"
    assert (
        isinstance(result[0]["result"], str) and len(result[0]["result"]) > 0
    ), "Expected non-empty string generated"
