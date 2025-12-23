from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.introspection import blocks_loader
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

AB_TEST_WORKFLOW = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "ABTest",
            "name": "ab_test",
            "a_step": "$steps.a",
            "b_step": "$steps.b",
        },
        {
            "type": "ObjectDetectionModel",
            "name": "a",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
        },
        {
            "type": "ObjectDetectionModel",
            "name": "b",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions_a",
            "selector": "$steps.a.predictions",
        },
        {
            "type": "JsonField",
            "name": "predictions_b",
            "selector": "$steps.b.predictions",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_flow_control_step_not_operating_on_batches(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    """
    In this test scenario, we run step (ABTest) which is not running in
    SIMD mode, as it only accepts non-batch parameters - hence
    single decision made at start will affect all downstream execution
    paths.

    We expect, based on flip of coin to execute either step "a" or step "b".

    What is verified from EE standpoint:
    * Creating execution branches for all batch elements, when input batch size is 1
    """
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.flow_control_plugin"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=AB_TEST_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(runtime_parameters={"image": crowd_image})

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided, so one output element expected"
    assert set(result[0].keys()) == {
        "predictions_a",
        "predictions_b",
    }, "Expected all declared outputs to be delivered"
    assert (result[0]["predictions_a"] and not result[0]["predictions_b"]) or (
        not result[0]["predictions_a"] and result[0]["predictions_b"]
    ), "Expected only one of the results provided, mutually exclusive based on random choice"


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_flow_control_step_not_operating_on_batches_affecting_batch_of_inputs(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    """
    In this test scenario, we run step (ABTest) which is not running in
    SIMD mode, as it only accepts non-batch parameters - hence
    single decision made at start will affect all downstream execution
    paths.

    We expect, based on flip of coin to execute either step "a" or step "b".

    What is verified from EE standpoint:
    * Creating execution branches for all batch elements, when input batch size is 4
    """
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.flow_control_plugin"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=AB_TEST_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(runtime_parameters={"image": [crowd_image] * 4})

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 4, "4 images provided, so 4 output elements expected"
    empty_element = (
        "predictions_a" if not result[0]["predictions_a"] else "predictions_b"
    )
    for i in range(4):
        assert set(result[i].keys()) == {
            "predictions_a",
            "predictions_b",
        }, "Expected all declared outputs to be delivered"
        assert (result[i]["predictions_a"] and not result[i]["predictions_b"]) or (
            not result[i]["predictions_a"] and result[i]["predictions_b"]
        ), "Expected only one of the results provided, mutually exclusive based on random choice"
        assert not result[i][
            empty_element
        ], f"Expected `{empty_element}` to be empty for each output, as ABTest takes only non-batch parameters and should decide once for all batch elements"


FILTERING_OPERATION = {
    "type": "DetectionsFilter",
    "filter_operation": {
        "type": "StatementGroup",
        "operator": "and",
        "statements": [
            {
                "type": "BinaryStatement",
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
                    "type": "DynamicOperand",
                    "operand_name": "classes",
                },
            },
            {
                "type": "BinaryStatement",
                "left_operand": {
                    "type": "DynamicOperand",
                    "operations": [
                        {
                            "type": "ExtractDetectionProperty",
                            "property_name": "size",
                        },
                    ],
                },
                "comparator": {"type": "(Number) >="},
                "right_operand": {
                    "type": "DynamicOperand",
                    "operand_name": "image",
                    "operations": [
                        {
                            "type": "ExtractImageProperty",
                            "property_name": "size",
                        },
                        {"type": "Multiply", "other": 0.02},
                    ],
                },
            },
        ],
    },
}

WORKFLOW_WITH_CONDITION_DEPENDENT_ON_MODEL_PREDICTION = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "classes"},
        {"type": "WorkflowParameter", "name": "detections_meeting_condition"},
    ],
    "steps": [
        {
            "type": "ObjectDetectionModel",
            "name": "a",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
        },
        {
            "type": "Condition",
            "name": "condition",
            "condition_statement": {
                "type": "StatementGroup",
                "statements": [
                    {
                        "type": "BinaryStatement",
                        "left_operand": {
                            "type": "DynamicOperand",
                            "operand_name": "prediction",
                            "operations": [
                                FILTERING_OPERATION,
                                {"type": "SequenceLength"},
                            ],
                        },
                        "comparator": {"type": "(Number) >="},
                        "right_operand": {
                            "type": "DynamicOperand",
                            "operand_name": "detections_meeting_condition",
                        },
                    }
                ],
            },
            "evaluation_parameters": {
                "image": "$inputs.image",
                "prediction": "$steps.a.predictions",
                "classes": "$inputs.classes",
                "detections_meeting_condition": "$inputs.detections_meeting_condition",
            },
            "steps_if_true": ["$steps.b"],
            "steps_if_false": ["$steps.c"],
        },
        {
            "type": "ObjectDetectionModel",
            "name": "b",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
        },
        {
            "type": "ObjectDetectionModel",
            "name": "c",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions_b",
            "selector": "$steps.b.predictions",
        },
        {
            "type": "JsonField",
            "name": "predictions_c",
            "selector": "$steps.c.predictions",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_flow_control_step_affecting_batches(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
    crowd_image: np.ndarray,
    dogs_image: np.ndarray,
) -> None:
    """
    Inn this test scenario, we make predictions from object detection model.
    Then we make if-else statement for each image form input batch checking if
    model found at least 2 big instances of classes {car, person}. If that is
    the case we run $steps.b otherwise $steps.c.

    What is verified from EE standpoint:
    * Creating execution branches for each batch element separately, and then
    executing downstream step according to decision made at previous step -
    with execution branches being independent
    * proper behavior of steps expecting non-empty inputs w.r.t. masks for
    execution branches
    * proper broadcasting of non-batch parameters for execution branches
    """
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.flow_control_plugin"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_CONDITION_DEPENDENT_ON_MODEL_PREDICTION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [crowd_image, dogs_image],
            "classes": ["person", "car"],
            "detections_meeting_condition": 2,
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 2, "2 images provided, so 2 output elements expected"
    assert result[0].keys() == {
        "predictions_b",
        "predictions_c",
    }, "Expected all declared outputs to be delivered for first result"
    assert result[1].keys() == {
        "predictions_b",
        "predictions_c",
    }, "Expected all declared outputs to be delivered for second result"
    assert (
        result[0]["predictions_b"] and not result[0]["predictions_c"]
    ), "At crowd image it is expected to spot 2 big instances of classes person, car - hence model b should fire"
    assert (
        not result[1]["predictions_b"] and result[1]["predictions_c"]
    ), "At dogs image it is not expected to spot people nor cars - hence model c should fire"


WORKFLOW_WITH_CONDITION_DEPENDENT_ON_CROPS = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "ObjectDetectionModel",
            "name": "first_detection",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
        },
        {
            "type": "DetectionsTransformation",
            "name": "enlarging_boxes",
            "predictions": "$steps.first_detection.predictions",
            "operations": [
                {"type": "DetectionsOffset", "offset_x": 50, "offset_y": 50}
            ],
        },
        {
            "type": "Crop",
            "name": "first_crop",
            "image": "$inputs.image",
            "predictions": "$steps.enlarging_boxes.predictions",
        },
        {
            "type": "ObjectDetectionModel",
            "name": "second_detection",
            "image": "$steps.first_crop.crops",
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
                            "value": 1,
                        },
                    }
                ],
            },
            "evaluation_parameters": {
                "prediction": "$steps.second_detection.predictions"
            },
            "next_steps": ["$steps.classification"],
        },
        {
            "type": "ClassificationModel",
            "name": "classification",
            "image": "$steps.first_crop.crops",
            "model_id": "dog-breed-xpaq6/1",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "dog_classification",
            "selector": "$steps.classification.predictions",
        }
    ],
}


@add_to_workflows_gallery(
    category="Workflows with flow control",
    use_case_title="Workflow with if statement applied on nested batches",
    use_case_description="""
In this test scenario we verify if we can successfully apply conditional
branching when data dimensionality increases.
We first make detections on input images and perform crop increasing
dimensionality to 2. Then we make another detections on cropped images
and check if inside crop we only see one instance of class dog (very naive
way of making sure that bboxes contain only single objects).
Only if that condition is true, we run classification model - to
classify dog breed.
    """,
    workflow_definition=WORKFLOW_WITH_CONDITION_DEPENDENT_ON_CROPS,
    workflow_name_in_app="flow-control-nested-batches",
)
def test_flow_control_step_affecting_data_with_increased_dimensionality(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
    dogs_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_CONDITION_DEPENDENT_ON_CROPS,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [crowd_image, dogs_image],
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 2, "2 images provided, so 2 output elements expected"
    assert result[0].keys() == {
        "dog_classification"
    }, "Expected all declared outputs to be delivered for first result"
    assert result[0].keys() == {
        "dog_classification"
    }, "Expected all declared outputs to be delivered for second result"
    assert (
        result[0]["dog_classification"] == [None] * 12
    ), "There is 12 crops for first image, but none got dogs classification results due to not meeting condition"
    assert (
        len([e for e in result[1]["dog_classification"] if e]) == 2
    ), "Expected 2 bboxes of dogs detected"


WORKFLOW_WITH_NON_BATCH_CONDITION_BASED_ON_INPUT_AFFECTING_FURTHER_EXECUTION = {
    "version": "1.0",
    "inputs": [
        {"type": "InferenceImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "some", "default_value": 1},
    ],
    "steps": [
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
                            "operand_name": "left",
                        },
                        "comparator": {"type": "(Number) =="},
                        "right_operand": {"type": "StaticOperand", "value": 1},
                    }
                ],
            },
            "next_steps": ["$steps.dependent_model"],
            "evaluation_parameters": {"left": "$inputs.some"},
        },
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "dependent_model",
            "images": "$inputs.image",
            "model_id": "yolov8n-640",
        },
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "independent_model",
            "images": "$inputs.image",
            "model_id": "yolov8n-640",
        },
        {
            "type": "Expression",
            "name": "dependent_expression",
            "data": {
                "predictions": "$steps.dependent_model.predictions",
            },
            "switch": {
                "type": "CasesDefinition",
                "cases": [],
                "default": {"type": "StaticCaseResult", "value": "EXECUTED DEPENDENT!"},
            },
        },
        {
            "type": "Expression",
            "name": "independent_expression",
            "data": {
                "predictions": "$steps.independent_model.predictions",
            },
            "switch": {
                "type": "CasesDefinition",
                "cases": [],
                "default": {
                    "type": "StaticCaseResult",
                    "value": "EXECUTED INDEPENDENT!",
                },
            },
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "dependent_predictions",
            "selector": "$steps.dependent_model.predictions",
        },
        {
            "type": "JsonField",
            "name": "dependent_expression",
            "selector": "$steps.dependent_expression.output",
        },
        {
            "type": "JsonField",
            "name": "independent_predictions",
            "selector": "$steps.independent_model.predictions",
        },
        {
            "type": "JsonField",
            "name": "independent_expression",
            "selector": "$steps.independent_expression.output",
        },
    ],
}


@add_to_workflows_gallery(
    category="Workflows with flow control",
    use_case_title="Workflow with if statement applied on non batch-oriented input",
    use_case_description="""
In this test scenario we show that we can use non-batch oriented conditioning (ContinueIf block).

If statement is effectively applied on input parameter that would determine path of execution for
all data passed in `image` input. When the value matches expectation - all dependent steps
will be executed, otherwise only the independent ones.
    """,
    workflow_definition=WORKFLOW_WITH_CONDITION_DEPENDENT_ON_CROPS,
    workflow_name_in_app="flow-control-on-parameter",
)
def test_flow_control_workflow_where_non_batch_nested_parameter_affects_further_execution_when_condition_is_met(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
    dogs_image: np.ndarray,
) -> None:
    """
    In this test scenario we verify if we can successfully use non-simd
    conditioning in case of ContinueIf block, for which Execution Engine
    must unwrap compound non-simd input parameters (which was the bug
    prior to v0.16.0).
    We take input directly, compare to 1 and if value matches - we execute
    steps prefixed with "dependent_" in names. Independently - steps with names
    prefixed with "independent_" should be executed.
    Scenario checks what happens when condition is met.
    """
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_NON_BATCH_CONDITION_BASED_ON_INPUT_AFFECTING_FURTHER_EXECUTION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [crowd_image, dogs_image],
            "some": 1,
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 2, "2 images provided, so 2 output elements expected"
    assert result[0].keys() == {
        "dependent_predictions",
        "independent_predictions",
        "dependent_expression",
        "independent_expression",
    }, "Expected all declared outputs to be delivered for first result"
    assert result[0].keys() == {
        "dependent_predictions",
        "independent_predictions",
        "dependent_expression",
        "independent_expression",
    }, "Expected all declared outputs to be delivered for second result"
    assert (
        len(result[0]["dependent_predictions"]) == 12
    ), "Expected 12 detections in crowd image"
    assert (
        result[0]["dependent_expression"] == "EXECUTED DEPENDENT!"
    ), "Expected dependent expression to execute"
    assert (
        len(result[0]["independent_predictions"]) == 12
    ), "Expected 12 detections in crowd image"
    assert (
        result[0]["independent_expression"] == "EXECUTED INDEPENDENT!"
    ), "Expected independent expression to execute"
    assert (
        len(result[1]["dependent_predictions"]) == 2
    ), "Expected 2 detections in dogs image"
    assert (
        result[1]["dependent_expression"] == "EXECUTED DEPENDENT!"
    ), "Expected dependent expression to execute"
    assert (
        len(result[1]["independent_predictions"]) == 2
    ), "Expected 2 detections in dogs image"
    assert (
        result[1]["independent_expression"] == "EXECUTED INDEPENDENT!"
    ), "Expected independent expression to execute"


def test_flow_control_workflow_where_non_batch_nested_parameter_affects_further_execution_when_condition_is_not_met(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
    dogs_image: np.ndarray,
) -> None:
    """
    In this test scenario we verify if we can successfully use non-simd
    conditioning in case of ContinueIf block, for which Execution Engine
    must unwrap compound non-simd input parameters (which was the bug
    prior to v0.16.0).
    We take input directly, compare to 1 and if value matches - we execute
    steps prefixed with "dependent_" in names. Independently - steps with names
    prefixed with "independent_" should be executed.
    Scenario checks what happens when condition is not met.
    """
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_NON_BATCH_CONDITION_BASED_ON_INPUT_AFFECTING_FURTHER_EXECUTION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [crowd_image, dogs_image],
            "some": 2,
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 2, "2 images provided, so 2 output elements expected"
    assert result[0].keys() == {
        "dependent_predictions",
        "independent_predictions",
        "dependent_expression",
        "independent_expression",
    }, "Expected all declared outputs to be delivered for first result"
    assert result[0].keys() == {
        "dependent_predictions",
        "independent_predictions",
        "dependent_expression",
        "independent_expression",
    }, "Expected all declared outputs to be delivered for second result"
    assert (
        result[0]["dependent_predictions"] is None
    ), "Expected dependent model not to execute"
    assert (
        result[0]["dependent_expression"] is None
    ), "Expected dependent expression not to execute"
    assert (
        len(result[0]["independent_predictions"]) == 12
    ), "Expected 12 detections in crowd image"
    assert (
        result[0]["independent_expression"] == "EXECUTED INDEPENDENT!"
    ), "Expected independent expression to execute"
    assert (
        result[1]["dependent_predictions"] is None
    ), "Expected dependent model not to execute"
    assert (
        result[1]["dependent_expression"] is None
    ), "Expected dependent expression not to execute"
    assert (
        len(result[1]["independent_predictions"]) == 2
    ), "Expected 2 detections in dogs image"
    assert (
        result[1]["independent_expression"] == "EXECUTED INDEPENDENT!"
    ), "Expected independent expression to execute"


WORKFLOW_WITH_NON_BATCH_CONDITION_BASED_ON_STEP_OUTPUT_AFFECTING_FURTHER_EXECUTION = {
    "version": "1.0",
    "inputs": [
        {"type": "InferenceImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "some", "default_value": "1"},
    ],
    "steps": [
        {
            "type": "Expression",
            "name": "input_processing_expression",
            "data": {
                "some": "$inputs.some",
            },
            "switch": {
                "type": "CasesDefinition",
                "cases": [],
                "default": {
                    "type": "DynamicCaseResult",
                    "parameter_name": "some",
                    "operations": [{"type": "ToNumber", "cast_to": "int"}],
                },
            },
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
                            "operand_name": "left",
                        },
                        "comparator": {"type": "(Number) =="},
                        "right_operand": {"type": "StaticOperand", "value": 1},
                    }
                ],
            },
            "next_steps": ["$steps.dependent_model"],
            "evaluation_parameters": {
                "left": "$steps.input_processing_expression.output"
            },
        },
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "dependent_model",
            "images": "$inputs.image",
            "model_id": "yolov8n-640",
        },
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "independent_model",
            "images": "$inputs.image",
            "model_id": "yolov8n-640",
        },
        {
            "type": "Expression",
            "name": "dependent_expression",
            "data": {
                "predictions": "$steps.dependent_model.predictions",
            },
            "switch": {
                "type": "CasesDefinition",
                "cases": [],
                "default": {"type": "StaticCaseResult", "value": "EXECUTED DEPENDENT!"},
            },
        },
        {
            "type": "Expression",
            "name": "independent_expression",
            "data": {
                "predictions": "$steps.independent_model.predictions",
            },
            "switch": {
                "type": "CasesDefinition",
                "cases": [],
                "default": {
                    "type": "StaticCaseResult",
                    "value": "EXECUTED INDEPENDENT!",
                },
            },
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "dependent_predictions",
            "selector": "$steps.dependent_model.predictions",
        },
        {
            "type": "JsonField",
            "name": "dependent_expression",
            "selector": "$steps.dependent_expression.output",
        },
        {
            "type": "JsonField",
            "name": "independent_predictions",
            "selector": "$steps.independent_model.predictions",
        },
        {
            "type": "JsonField",
            "name": "independent_expression",
            "selector": "$steps.independent_expression.output",
        },
    ],
}


def test_flow_control_workflow_where_non_batch_nested_parameter_produced_by_step_affects_further_execution_when_condition_is_met(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
    dogs_image: np.ndarray,
) -> None:
    """
    In this test scenario we verify if we can successfully use non-simd
    conditioning in case of ContinueIf block, for which Execution Engine
    must unwrap compound non-simd input parameters (which was the bug
    prior to v0.16.0).
    We take input, pass it to expression block (casting value to int), and
    this expression block output is compared to 1 and if value matches - we execute
    steps prefixed with "dependent_" in names. Independently - steps with names
    prefixed with "independent_" should be executed.
    Scenario checks what happens when condition is met.
    """
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_NON_BATCH_CONDITION_BASED_ON_STEP_OUTPUT_AFFECTING_FURTHER_EXECUTION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [crowd_image, dogs_image],
            "some": "1",
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 2, "2 images provided, so 2 output elements expected"
    assert result[0].keys() == {
        "dependent_predictions",
        "independent_predictions",
        "dependent_expression",
        "independent_expression",
    }, "Expected all declared outputs to be delivered for first result"
    assert result[0].keys() == {
        "dependent_predictions",
        "independent_predictions",
        "dependent_expression",
        "independent_expression",
    }, "Expected all declared outputs to be delivered for second result"
    assert (
        len(result[0]["dependent_predictions"]) == 12
    ), "Expected 12 detections in crowd image"
    assert (
        result[0]["dependent_expression"] == "EXECUTED DEPENDENT!"
    ), "Expected dependent expression to execute"
    assert (
        len(result[0]["independent_predictions"]) == 12
    ), "Expected 12 detections in crowd image"
    assert (
        result[0]["independent_expression"] == "EXECUTED INDEPENDENT!"
    ), "Expected independent expression to execute"
    assert (
        len(result[1]["dependent_predictions"]) == 2
    ), "Expected 2 detections in dogs image"
    assert (
        result[1]["dependent_expression"] == "EXECUTED DEPENDENT!"
    ), "Expected dependent expression to execute"
    assert (
        len(result[1]["independent_predictions"]) == 2
    ), "Expected 2 detections in dogs image"
    assert (
        result[1]["independent_expression"] == "EXECUTED INDEPENDENT!"
    ), "Expected independent expression to execute"


def test_flow_control_workflow_where_non_batch_nested_parameter_produced_by_step_affects_further_execution_when_condition_is_not_met(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
    dogs_image: np.ndarray,
) -> None:
    """
    In this test scenario we verify if we can successfully use non-simd
    conditioning in case of ContinueIf block, for which Execution Engine
    must unwrap compound non-simd input parameters (which was the bug
    prior to v0.16.0).
    We take input, pass it to expression block (casting value to int), and
    this expression block output is compared to 1 and if value matches - we execute
    steps prefixed with "dependent_" in names. Independently - steps with names
    prefixed with "independent_" should be executed.
    Scenario checks what happens when condition is not met.
    """
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_NON_BATCH_CONDITION_BASED_ON_STEP_OUTPUT_AFFECTING_FURTHER_EXECUTION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [crowd_image, dogs_image],
            "some": "2",
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 2, "2 images provided, so 2 output elements expected"
    assert result[0].keys() == {
        "dependent_predictions",
        "independent_predictions",
        "dependent_expression",
        "independent_expression",
    }, "Expected all declared outputs to be delivered for first result"
    assert result[0].keys() == {
        "dependent_predictions",
        "independent_predictions",
        "dependent_expression",
        "independent_expression",
    }, "Expected all declared outputs to be delivered for second result"
    assert (
        result[0]["dependent_predictions"] is None
    ), "Expected dependent model not to execute"
    assert (
        result[0]["dependent_expression"] is None
    ), "Expected dependent expression not to execute"
    assert (
        len(result[0]["independent_predictions"]) == 12
    ), "Expected 12 detections in crowd image"
    assert (
        result[0]["independent_expression"] == "EXECUTED INDEPENDENT!"
    ), "Expected independent expression to execute"
    assert (
        result[1]["dependent_predictions"] is None
    ), "Expected dependent model not to execute"
    assert (
        result[1]["dependent_expression"] is None
    ), "Expected dependent expression not to execute"
    assert (
        len(result[1]["independent_predictions"]) == 2
    ), "Expected 2 detections in dogs image"
    assert (
        result[1]["independent_expression"] == "EXECUTED INDEPENDENT!"
    ), "Expected independent expression to execute"


WORKFLOW_WITH_CONTINUE_IF_AND_STOP_DELAY = {
    "version": "1.0",
    "inputs": [
        {"type": "InferenceImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "condition_value", "default_value": 1},
    ],
    "steps": [
        {
            "type": "ContinueIf",
            "name": "continue_if_with_delay",
            "condition_statement": {
                "type": "StatementGroup",
                "statements": [
                    {
                        "type": "BinaryStatement",
                        "left_operand": {
                            "type": "DynamicOperand",
                            "operand_name": "condition",
                        },
                        "comparator": {"type": "(Number) =="},
                        "right_operand": {"type": "StaticOperand", "value": 1},
                    }
                ],
            },
            "next_steps": ["$steps.dependent_model"],
            "evaluation_parameters": {"condition": "$inputs.condition_value"},
            "stop_delay": 0.5,
        },
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "dependent_model",
            "images": "$inputs.image",
            "model_id": "yolov8n-640",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.dependent_model.predictions",
        }
    ],
}


@add_to_workflows_gallery(
    category="Workflows with flow control",
    use_case_title="Workflow with continue_if block using stop_delay",
    use_case_description="""
In this test scenario we verify the stop_delay functionality of the continue_if block.
The stop_delay parameter allows the conditional branch to continue executing for a
specified duration after the condition becomes false, enabling graceful degradation
and delayed termination scenarios.
""",
    workflow_definition=WORKFLOW_WITH_CONTINUE_IF_AND_STOP_DELAY,
    workflow_name_in_app="continue-if-stop-delay",
)
def test_continue_if_with_stop_delay_true_condition(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    """
    Test continue_if block with stop_delay when condition is true.
    When condition is true, the next steps should execute immediately.
    """
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_CONTINUE_IF_AND_STOP_DELAY,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when - condition is true (condition_value=1)
    result = execution_engine.run(
        runtime_parameters={
            "image": crowd_image,
            "condition_value": 1,
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Expected 1 result"
    assert (
        len(result[0]["predictions"]) > 0
    ), "Expected detections when condition is true"


def test_continue_if_with_stop_delay_false_condition(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    """
    Test continue_if block with stop_delay when condition is false.
    When condition is false, execution should persist within the stop_delay window.
    """
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_CONTINUE_IF_AND_STOP_DELAY,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when - condition is false (condition_value=2), first execution
    result = execution_engine.run(
        runtime_parameters={
            "image": crowd_image,
            "condition_value": 2,
        }
    )

    # then - First execution with false condition and stop_delay still active
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Expected 1 result"
    # With stop_delay > 0 and condition false, should not execute on first false
    assert (
        result[0]["predictions"] is None
    ), "Expected no detections on first false condition"


def test_continue_if_with_multiple_calls_respects_stop_delay(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    """
    Test that continue_if block respects stop_delay across multiple executions.
    When condition becomes true, stop_delay timer should start.
    When condition is later false but within stop_delay window, execution continues.
    After stop_delay expires, execution should terminate.
    """
    import time

    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_CONTINUE_IF_AND_STOP_DELAY,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # First execution with condition true
    result1 = execution_engine.run(
        runtime_parameters={
            "image": crowd_image,
            "condition_value": 1,
        }
    )

    # Check that condition true causes execution
    assert (
        len(result1[0]["predictions"]) > 0
    ), "Expected detections when condition is true"

    # Second execution with condition false but within stop_delay
    result2 = execution_engine.run(
        runtime_parameters={
            "image": crowd_image,
            "condition_value": 2,
        }
    )

    # Should execute within stop_delay window
    assert (
        len(result2[0]["predictions"]) > 0
    ), "Expected detections within stop_delay window after condition became false"

    # Wait for stop_delay to expire (0.5 seconds + buffer)
    time.sleep(0.6)

    # Third execution with condition false after stop_delay expires
    result3 = execution_engine.run(
        runtime_parameters={
            "image": crowd_image,
            "condition_value": 2,
        }
    )

    # Should not execute after stop_delay expires
    assert (
        result3[0]["predictions"] is None
    ), "Expected no detections after stop_delay window expires"
