import pytest

from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.errors import ReferenceTypeError
from inference.core.workflows.execution_engine.v1.compiler.core import compile_workflow

KINDS_CONFLICTING_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "OCRModel",
            "name": "ocr",
            "image": "$inputs.image",
        },
        {
            "type": "Crop",
            "name": "crops",
            "image": "$inputs.image",
            "predictions": "$steps.ocr.result",
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


def test_compilation_of_workflow_with_conflicting_kinds(
    model_manager: ModelManager,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }

    # when
    with pytest.raises(ReferenceTypeError) as error:
        _ = compile_workflow(
            workflow_definition=KINDS_CONFLICTING_WORKFLOW,
            init_parameters=workflow_init_parameters,
        )

    # then
    assert (
        "Failed to validate reference provided for step: $steps.crops regarding property: "
        "predictions with value: $steps.ocr.result" in str(error.value)
    )
