from copy import deepcopy

import pytest
from pydantic import ValidationError

from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.errors import WorkflowSyntaxError
from inference.core.workflows.execution_engine.v1.compiler.core import compile_workflow

VALID_DEFINITION = {
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
            "type": "Crop",
            "name": "crops",
            "image": "$inputs.image",
            "predictions": "$steps.detection.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "crops",
            "selector": "$steps.crops.crops",
        },
    ],
}


def test_compilation_of_workflow_where_definition_does_not_specify_fields(
    model_manager: ModelManager,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }

    # when
    with pytest.raises(WorkflowSyntaxError) as error:
        _ = compile_workflow(
            workflow_definition={},
            init_parameters=workflow_init_parameters,
        )

    # then
    assert isinstance(error.value.inner_error, ValidationError)


@pytest.mark.parametrize("field_to_remove", ["version", "inputs", "steps", "outputs"])
def test_compilation_of_workflow_where_definition_does_not_specify_all_fields(
    model_manager: ModelManager,
    field_to_remove: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    workflow_definition = deepcopy(VALID_DEFINITION)
    del workflow_definition[field_to_remove]

    # when
    with pytest.raises(WorkflowSyntaxError) as error:
        _ = compile_workflow(
            workflow_definition=workflow_definition,
            init_parameters=workflow_init_parameters,
        )

    # then
    assert isinstance(error.value.inner_error, ValidationError)


DEFINITION_WITH_NON_EXISTING_STEP = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "NotExistingBlock",
            "name": "invalid",
            "image": "$inputs.image",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "invalid",
            "selector": "$steps.crops.crops",
        },
    ],
}


def test_compilation_of_workflow_where_non_existing_step_is_requested(
    model_manager: ModelManager,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }

    # when
    with pytest.raises(WorkflowSyntaxError) as error:
        _ = compile_workflow(
            workflow_definition=DEFINITION_WITH_NON_EXISTING_STEP,
            init_parameters=workflow_init_parameters,
        )

    # then
    assert isinstance(error.value.inner_error, ValidationError)


DEFINITION_WITH_STEP_SYNTAX_ERROR = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "detection",
            # missing required image field
            "model_id": "yolov8n-640",
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


def test_compilation_of_workflow_where_existing_step_is_defined_incorrectly(
    model_manager: ModelManager,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }

    # when
    with pytest.raises(WorkflowSyntaxError) as error:
        _ = compile_workflow(
            workflow_definition=DEFINITION_WITH_STEP_SYNTAX_ERROR,
            init_parameters=workflow_init_parameters,
        )

    # then
    assert isinstance(error.value.inner_error, ValidationError)
