import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

PROBLEMATIC_WORKFLOW = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "ObjectDetectionModel",
            "name": "general_detection",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
            "class_filter": ["dog"],
        },
        {
            "type": "ContinueIf",
            "name": "continue_if",
            "condition_statement": {
                "type": "StatementGroup",
                "statements": [
                    {
                        "type": "BinaryStatement",
                        "left_operand": {
                            "type": "DynamicOperand",
                            "operand_name": "prediction",
                            "operations": [{"type": "SequenceLength"}],
                        },
                        "comparator": {"type": "(Number) =="},
                        "right_operand": {
                            "type": "StaticOperand",
                            "value": 5,
                        },
                    }
                ],
            },
            "evaluation_parameters": {
                "prediction": "$steps.general_detection.predictions"
            },
            "next_steps": ["$steps.cropping"],
        },
        {
            "type": "Crop",
            "name": "cropping",
            "image": "$inputs.image",
            "predictions": "$steps.general_detection.predictions",
        },
        {
            "type": "ClassificationModel",
            "name": "breds_classification",
            "image": "$steps.cropping.crops",
            "model_id": "dog-breed-xpaq6/1",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.breds_classification.predictions",
        },
    ],
}


def test_workflow_with_flow_control_eliminating_step_changing_lineage(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    """
    This test case covers bug that in Execution Engine versions <=1.1.0.
    The bug was not registering outputs from steps given no inputs provided
    (inputs may have been filtered by flow-control steps). As a result,
    given that step changes dimensionality (registers new data lineage) -
    registration could not happen and downstream steps was raising error:
    "Lineage ['<workflow_input>', 'XXX'] not found. [...]"
    """
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=PROBLEMATIC_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image],
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "1 image provided, so 1 output elements expected"
    assert result[0].keys() == {
        "predictions"
    }, "Expected all declared outputs to be delivered for first result"
    assert len([e for e in result[0]["predictions"] if e]) == 0, (
        "Expected no predictions, due to conditional execution applied, effectively preventing "
        "`cropping` step from running"
    )
