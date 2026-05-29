import pytest

from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.errors import DuplicatedNameError
from inference.core.workflows.execution_engine.v1.compiler.core import compile_workflow

DEFINITION_WITH_DUPLICATED_INPUTS = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "detection",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
            "class_filter": ["car"],
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "detections",
            "selector": "$steps.detection.*",
        },
    ],
}


def test_compilation_of_workflow_with_duplicated_inputs(
    model_manager: ModelManager,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }

    # when
    with pytest.raises(DuplicatedNameError):
        _ = compile_workflow(
            workflow_definition=DEFINITION_WITH_DUPLICATED_INPUTS,
            init_parameters=workflow_init_parameters,
        )


DEFINITION_WITH_DUPLICATED_STEPS = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "detection",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
            "class_filter": ["car"],
        },
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "detection",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
            "class_filter": ["car"],
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "detections",
            "selector": "$steps.detection.*",
        },
    ],
}


def test_compilation_of_workflow_with_duplicated_steps(
    model_manager: ModelManager,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }

    # when
    with pytest.raises(DuplicatedNameError):
        _ = compile_workflow(
            workflow_definition=DEFINITION_WITH_DUPLICATED_STEPS,
            init_parameters=workflow_init_parameters,
        )


DEFINITION_WITH_DUPLICATED_OUTPUTS = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "detection",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
            "class_filter": ["car"],
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "detections",
            "selector": "$steps.detection.*",
        },
        {
            "type": "JsonField",
            "name": "detections",
            "selector": "$steps.detection.predictions",
        },
    ],
}


def test_compilation_of_workflow_with_duplicated_outputs(
    model_manager: ModelManager,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }

    # when
    with pytest.raises(DuplicatedNameError):
        _ = compile_workflow(
            workflow_definition=DEFINITION_WITH_DUPLICATED_OUTPUTS,
            init_parameters=workflow_init_parameters,
        )
