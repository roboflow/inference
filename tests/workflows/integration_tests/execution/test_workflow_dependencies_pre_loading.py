"""
Integration tests for pre-loading of declared dependent resources
(`ExecutionEngine.init(..., dependencies_pre_init=...)`) against a true
model manager.

Requires the `ROBOFLOW_API_KEY` environment variable — models are actually
registered in the model manager (weights downloaded) at engine init / on the
first run.
"""

import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

OBJECT_DETECTION_WORKFLOW_WITH_STATIC_MODEL_ID = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v3",
            "name": "general_detection",
            "images": "$inputs.image",
            "model_id": "yolov8n-640",
            "class_filter": ["dog"],
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.general_detection.predictions",
        },
    ],
}


def test_pre_loading_of_dependencies_when_static_model_id_used(
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

    # when
    execution_engine = ExecutionEngine.init(
        workflow_definition=OBJECT_DETECTION_WORKFLOW_WITH_STATIC_MODEL_ID,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
        dependencies_pre_init=["roboflow_platform_model"],
    )

    # then - model must be registered at init time, before any run
    assert (
        "yolov8n-640" in model_manager
    ), "Expected declared model to be pre-loaded into model manager at init"
    assert (
        len(model_manager.models()) == 1
    ), "Expected exactly the declared model to be registered at init"

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
        }
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "predictions",
    }, "Expected all declared outputs to be delivered"
    assert (
        len(result[0]["predictions"]) == 2
    ), "Expected 2 dogs detected on input image, as measured in reference run"


OBJECT_DETECTION_WORKFLOW_WITH_INPUT_FED_MODEL_ID = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "model"},
    ],
    "steps": [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v3",
            "name": "general_detection",
            "images": "$inputs.image",
            "model_id": "$inputs.model",
            "class_filter": ["dog"],
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.general_detection.predictions",
        },
    ],
}


def test_pre_loading_of_dependencies_when_model_id_fed_by_input_parameter(
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

    # when
    execution_engine = ExecutionEngine.init(
        workflow_definition=OBJECT_DETECTION_WORKFLOW_WITH_INPUT_FED_MODEL_ID,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
        dependencies_pre_init=["roboflow_platform_model"],
    )

    # then - nothing can be pre-loaded at init, dependency awaits first run
    assert (
        len(model_manager.models()) == 0
    ), "Expected no model to be registered at init when model id is input-fed"
    assert [
        dependency.metadata.model_id
        for dependency in execution_engine._engine._pending_runtime_dependencies
    ] == ["$inputs.model"], "Expected input-fed dependency to await first run"

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
            "model": "yolov8n-640",
        }
    )

    # then - first run resolved the input value and registered the model
    assert (
        "yolov8n-640" in model_manager
    ), "Expected input-fed model to be registered during the first run"
    assert (
        execution_engine._engine._pending_dependencies_resolution_attempted is True
    ), "Expected first-run resolution to be marked as attempted"
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "predictions",
    }, "Expected all declared outputs to be delivered"
    assert (
        len(result[0]["predictions"]) == 2
    ), "Expected 2 dogs detected on input image, as measured in reference run"
