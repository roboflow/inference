import pytest

from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.errors import ExecutionGraphStructureError
from inference.core.workflows.execution_engine.v1.compiler.core import compile_workflow

CYCLIC_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "detection",
            "image": "$steps.crops.crops",
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


def test_compilation_of_cyclic_workflow(
    model_manager: ModelManager,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }

    # when
    with pytest.raises(ExecutionGraphStructureError) as error:
        _ = compile_workflow(
            workflow_definition=CYCLIC_WORKFLOW,
            init_parameters=workflow_init_parameters,
        )

    # then
    assert "Detected cycle in execution graph" in str(error.value)
