import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

TWO_STAGE_WORKFLOW_WITH_FLOW_CONTROL = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v2",
            "name": "general_detection",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
            "class_filter": ["dog"],
        },
        {
            "type": "roboflow_core/dynamic_crop@v1",
            "name": "cropping",
            "image": "$inputs.image",
            "predictions": "$steps.general_detection.predictions",
        },
        {
            "type": "roboflow_core/roboflow_classification_model@v2",
            "name": "breds_classification",
            "image": "$steps.cropping.crops",
            "model_id": "dog-breed-xpaq6/1",
            "confidence": 0.09,
        },
        {
            "type": "roboflow_core/continue_if@v1",
            "name": "continue_if",
            "condition_statement": {
                "type": "StatementGroup",
                "statements": [
                    {
                        "type": "BinaryStatement",
                        "left_operand": {
                            "type": "DynamicOperand",
                            "operand_name": "predictions",
                            "operations": [
                                {
                                    "type": "ClassificationPropertyExtract",
                                    "property_name": "top_class_confidence",
                                }
                            ],
                        },
                        "comparator": {"type": "(Number) >="},
                        "right_operand": {"type": "StaticOperand", "value": 0.35},
                    }
                ],
            },
            "evaluation_parameters": {
                "predictions": "$steps.breds_classification.predictions"
            },
            "next_steps": ["$steps.property_definition"],
        },
        {
            "type": "roboflow_core/property_definition@v1",
            "name": "property_definition",
            "operations": [
                {"type": "ClassificationPropertyExtract", "property_name": "top_class"}
            ],
            "data": "$steps.breds_classification.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "class_name",
            "selector": "$steps.property_definition.output",
        },
    ],
}


def test_two_stage_workflow_with_flow_control_when_there_is_nothing_predicted_from_first_stage_model(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=TWO_STAGE_WORKFLOW_WITH_FLOW_CONTROL,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": crowd_image,
        }
    )

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "class_name",
    }, "Expected all declared outputs to be delivered"
    assert (
        len(result[0]["class_name"]) == 0
    ), "Expected no prediction from 2nd model, as no dogs detected"


def test_two_stage_workflow_with_flow_control_when_there_is_something_predicted_from_first_stage_model(
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
        workflow_definition=TWO_STAGE_WORKFLOW_WITH_FLOW_CONTROL,
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
        "class_name",
    }, "Expected all declared outputs to be delivered"
    assert result[0]["class_name"] == [
        "116.Parson_russell_terrier",
        None,
    ], "Expected one crop to be passed by continue_if block and the other failed"
