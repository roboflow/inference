import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.errors import ReferenceTypeError
from inference.core.workflows.execution_engine.core import ExecutionEngine

CLASSIFICATION_FILTER_OPERATION = {
    "type": "ClassificationFilter",
    "filter_operation": {
        "type": "StatementGroup",
        "statements": [
            {
                "type": "BinaryStatement",
                "left_operand": {
                    "type": "DynamicOperand",
                    "operations": [
                        {
                            "type": "ExtractClassificationPredictionProperty",
                            "property_name": "confidence",
                        }
                    ],
                },
                "comparator": {"type": "(Number) >="},
                "right_operand": {
                    "type": "DynamicOperand",
                    "operand_name": "threshold",
                },
            }
        ],
    },
}

CLASSIFICATION_FILTER_WORKFLOW = {
    "version": "1.3.0",
    "inputs": [
        {
            "type": "WorkflowBatchInput",
            "name": "predictions",
            "kind": ["classification_prediction"],
        },
        {
            "type": "WorkflowParameter",
            "name": "threshold",
            "default_value": 0.5,
        },
    ],
    "steps": [
        {
            "type": "roboflow_core/detections_filter@v2",
            "name": "filter",
            "predictions": "$inputs.predictions",
            "operations": [CLASSIFICATION_FILTER_OPERATION],
            "operations_parameters": {"threshold": "$inputs.threshold"},
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "filtered",
            "selector": "$steps.filter.predictions",
        }
    ],
}


def _init_engine(workflow: dict, model_manager: ModelManager) -> ExecutionEngine:
    return ExecutionEngine.init(
        workflow_definition=workflow,
        init_parameters={
            "workflows_core.model_manager": model_manager,
            "workflows_core.api_key": None,
            "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
        },
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )


def test_classification_detections_filter_end_to_end(
    model_manager: ModelManager,
) -> None:
    engine = _init_engine(
        workflow=CLASSIFICATION_FILTER_WORKFLOW,
        model_manager=model_manager,
    )
    prediction = {
        "image": {"width": 100, "height": 80},
        "predictions": [
            {"class": "cat", "class_id": 0, "confidence": 0.9},
            {"class": "dog", "class_id": 1, "confidence": 0.4},
        ],
        "top": "cat",
        "confidence": 0.9,
    }

    result = engine.run(
        runtime_parameters={
            "predictions": [prediction],
            "threshold": 0.5,
        }
    )

    assert len(result) == 1
    assert result[0]["filtered"]["predictions"] == [
        {"class": "cat", "class_id": 0, "confidence": 0.9}
    ]
    assert result[0]["filtered"]["top"] == "cat"
    assert result[0]["filtered"]["confidence"] == 0.9


def test_classification_configured_output_cannot_feed_detection_only_block(
    model_manager: ModelManager,
) -> None:
    invalid_workflow = {
        **CLASSIFICATION_FILTER_WORKFLOW,
        "steps": [
            *CLASSIFICATION_FILTER_WORKFLOW["steps"],
            {
                "type": "roboflow_core/detections_filter@v1",
                "name": "detection_only_filter",
                "predictions": "$steps.filter.predictions",
                "operations": [],
                "operations_parameters": {},
            },
        ],
    }

    with pytest.raises(ReferenceTypeError):
        _init_engine(workflow=invalid_workflow, model_manager=model_manager)
