import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

WORKFLOW_WITH_DETECTIONS_SPECIALISED_CLASSIFICATION_AND_CUSTOM_EXPRESSION = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "reference"},
    ],
    "steps": [
        {
            "type": "ObjectDetectionModel",
            "name": "general_detection",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
            "class_filter": ["dog"],
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
            "confidence": 0.09,
        },
        {
            "type": "DetectionsClassesReplacement",
            "name": "classes_replacement",
            "object_detection_predictions": "$steps.general_detection.predictions",
            "classification_predictions": "$steps.breds_classification.predictions",
        },
        {
            "type": "Expression",
            "name": "expression",
            "data": {
                "reference_value": "$inputs.reference",
                "predictions": "$steps.classes_replacement.predictions",
            },
            "data_operations": {
                "predictions": [
                    {"type": "DetectionsPropertyExtract", "property_name": "class_name"}
                ]
            },
            "switch": {
                "type": "CasesDefinition",
                "cases": [
                    {
                        "type": "CaseDefinition",
                        "condition": {
                            "type": "StatementGroup",
                            "statements": [
                                {
                                    "type": "BinaryStatement",
                                    "left_operand": {
                                        "type": "DynamicOperand",
                                        "operand_name": "reference_value",
                                    },
                                    "comparator": {"type": "in (Sequence)"},
                                    "right_operand": {
                                        "type": "DynamicOperand",
                                        "operand_name": "predictions",
                                    },
                                }
                            ],
                        },
                        "result": {"type": "StaticCaseResult", "value": "FOUND"},
                    }
                ],
                "default": {"type": "StaticCaseResult", "value": "NOT FOUND"},
            },
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "verdict",
            "selector": "$steps.expression.output",
        }
    ],
}


def test_detection_plus_classification_workflow_when_reference_found_at_least_in_one_image(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
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
        workflow_definition=WORKFLOW_WITH_DETECTIONS_SPECIALISED_CLASSIFICATION_AND_CUSTOM_EXPRESSION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image, crowd_image],
            "reference": "116.Parson_russell_terrier",
        }
    )

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 2, "Expected 2 element in the output for two input images"
    assert set(result[0].keys()) == {
        "verdict",
    }, "Expected all declared outputs to be delivered for first output"
    assert set(result[1].keys()) == {
        "verdict",
    }, "Expected all declared outputs to be delivered for second output"
    assert result[0]["verdict"] == "FOUND"
    assert result[1]["verdict"] == "NOT FOUND"


def test_detection_plus_classification_workflow_when_reference_not_found(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
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
        workflow_definition=WORKFLOW_WITH_DETECTIONS_SPECIALISED_CLASSIFICATION_AND_CUSTOM_EXPRESSION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image, crowd_image],
            "reference": "NON-EXISTING",
        }
    )

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 2, "Expected 2 element in the output for two input images"
    assert set(result[0].keys()) == {
        "verdict",
    }, "Expected all declared outputs to be delivered for first output"
    assert set(result[1].keys()) == {
        "verdict",
    }, "Expected all declared outputs to be delivered for second output"
    assert result[0]["verdict"] == "NOT FOUND"
    assert result[1]["verdict"] == "NOT FOUND"
