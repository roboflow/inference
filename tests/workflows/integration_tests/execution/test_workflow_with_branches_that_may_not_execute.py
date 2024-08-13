import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

WORKFLOW_WITH_BRANCHES_THAT_MAY_NOT_EXECUTE = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "model_id"},
    ],
    "steps": [
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "detection",
            "image": "$inputs.image",
            "model_id": "$inputs.model_id",
            "confidence": 0.5,
        },
        {
            "type": "DetectionsFilter",
            "name": "people_filter",
            "predictions": "$steps.detection.predictions",
            "operations": [
                {
                    "type": "DetectionsFilter",
                    "filter_operation": {
                        "type": "StatementGroup",
                        "operator": "and",
                        "statements": [
                            {
                                "type": "BinaryStatement",
                                "negate": False,
                                "left_operand": {
                                    "type": "DynamicOperand",
                                    "operations": [
                                        {
                                            "type": "ExtractDetectionProperty",
                                            "property_name": "class_name",
                                        }
                                    ],
                                },
                                "comparator": {"type": "in (Sequence)"},
                                "right_operand": {
                                    "type": "StaticOperand",
                                    "value": ["person"],
                                },
                            }
                        ],
                    },
                }
            ],
        },
        {
            "type": "DetectionsFilter",
            "name": "dogs_filter",
            "predictions": "$steps.detection.predictions",
            "operations": [
                {
                    "type": "DetectionsFilter",
                    "filter_operation": {
                        "type": "StatementGroup",
                        "operator": "and",
                        "statements": [
                            {
                                "type": "BinaryStatement",
                                "negate": False,
                                "left_operand": {
                                    "type": "DynamicOperand",
                                    "operations": [
                                        {
                                            "type": "ExtractDetectionProperty",
                                            "property_name": "class_name",
                                        }
                                    ],
                                },
                                "comparator": {"type": "in (Sequence)"},
                                "right_operand": {
                                    "type": "StaticOperand",
                                    "value": ["dog"],
                                },
                            }
                        ],
                    },
                }
            ],
        },
        {
            "type": "Crop",
            "name": "people_cropper",
            "image": "$inputs.image",
            "predictions": "$steps.people_filter.predictions",
        },
        {
            "type": "Crop",
            "name": "dogs_cropper",
            "image": "$inputs.image",
            "predictions": "$steps.dogs_filter.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.detection.predictions",
        },
        {
            "type": "JsonField",
            "name": "people",
            "selector": "$steps.people_cropper.crops",
        },
        {"type": "JsonField", "name": "dogs", "selector": "$steps.dogs_cropper.crops"},
    ],
}


def test_workflow_with_optional_execution_of_branches_impacting_results_in_batch_mode(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
    dogs_image: np.ndarray,
    license_plate_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_BRANCHES_THAT_MAY_NOT_EXECUTE,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [crowd_image, dogs_image, license_plate_image],
            "model_id": "yolov8n-640",
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 3, "Three images provided - three outputs expected"
    assert (
        len(result[0]["people"]) == 11
    ), "Expected 11 crops of person class instance for crowd image"
    assert (
        len(result[0]["dogs"]) == 0
    ), "Expected 0 crops of dogs class instance for crowd image"
    assert (
        len(result[1]["people"]) == 0
    ), "Expected 0 crops of person class instance for dogs image"
    assert (
        len(result[1]["dogs"]) == 2
    ), "Expected 2 crops of dogs class instance for dogs image"
    assert (
        len(result[2]["people"]) == 0
    ), "Expected 0 crops of person class instance for cars image"
    assert (
        len(result[2]["dogs"]) == 0
    ), "Expected 0 crops of dogs class instance for cars image"


def test_workflow_with_optional_execution_of_branches_impacting_results_when_both_alternative_outputs_should_not_be_created(
    model_manager: ModelManager,
    license_plate_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_BRANCHES_THAT_MAY_NOT_EXECUTE,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": license_plate_image,
            "model_id": "yolov8n-640",
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "One image provided - one output expected"
    assert (
        len(result[0]["people"]) == 0
    ), "Expected 0 crops of person class instance for cars image"
    assert (
        len(result[0]["dogs"]) == 0
    ), "Expected 0 crops of dogs class instance for cars image"


def test_workflow_with_optional_execution_of_branches_impacting_results_when_only_dogs_related_outputs_to_be_created(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_BRANCHES_THAT_MAY_NOT_EXECUTE,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
            "model_id": "yolov8n-640",
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "One image provided - one output expected"
    assert (
        len(result[0]["people"]) == 0
    ), "Expected 0 crops of person class instance for cars image"
    assert (
        len(result[0]["dogs"]) == 2
    ), "Expected 2 crops of dogs class instance for dogs image"


def test_workflow_with_optional_execution_of_branches_impacting_results_when_only_people_related_outputs_to_be_created(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_BRANCHES_THAT_MAY_NOT_EXECUTE,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": crowd_image,
            "model_id": "yolov8n-640",
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "One image provided - one output expected"
    assert (
        len(result[0]["people"]) == 11
    ), "Expected 11 crops of person class instance for crowd image"
    assert (
        len(result[0]["dogs"]) == 0
    ), "Expected 0 crops of dogs class instance for crowd image"
