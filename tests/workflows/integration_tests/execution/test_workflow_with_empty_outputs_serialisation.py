import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

WORKFLOW_DEFINITION = {
    "version": "1.0",
    "inputs": [{"type": "InferenceImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/roboflow_multi_label_classification_model@v1",
            "name": "multi_label",
            "images": "$inputs.image",
            "model_id": "animal-classification-9lufm/1",
            "confidence": 0.2,
        },
        {
            "type": "roboflow_core/property_definition@v1",
            "name": "multi_classes",
            "data": "$steps.multi_label.predictions",
            "operations": [
                {"type": "ClassificationPropertyExtract", "property_name": "top_class"}
            ],
        },
        {
            "type": "roboflow_core/continue_if@v1",
            "name": "includes_any",
            "condition_statement": {
                "type": "StatementGroup",
                "statements": [
                    {
                        "type": "BinaryStatement",
                        "left_operand": {
                            "type": "StaticOperand",
                            "value": ["cat", "dog"],
                        },
                        "comparator": {"type": "any in (Sequence)"},
                        "right_operand": {
                            "type": "DynamicOperand",
                            "operand_name": "left",
                            "operations": [
                                {
                                    "type": "ClassificationPropertyExtract",
                                    "property_name": "top_class",
                                }
                            ],
                        },
                    }
                ],
            },
            "next_steps": ["$steps.resize"],
            "evaluation_parameters": {"left": "$steps.multi_label.predictions"},
        },
        {
            "type": "roboflow_core/continue_if@v1",
            "name": "includes_all",
            "condition_statement": {
                "type": "StatementGroup",
                "statements": [
                    {
                        "type": "BinaryStatement",
                        "left_operand": {
                            "type": "StaticOperand",
                            "value": ["cat", "dog"],
                        },
                        "comparator": {"type": "all in (Sequence)"},
                        "right_operand": {
                            "type": "DynamicOperand",
                            "operand_name": "left",
                            "operations": [
                                {
                                    "type": "ClassificationPropertyExtract",
                                    "property_name": "top_class",
                                }
                            ],
                        },
                    }
                ],
            },
            "next_steps": ["$steps.rotate"],
            "evaluation_parameters": {"left": "$steps.multi_label.predictions"},
        },
        {
            "type": "roboflow_core/image_preprocessing@v1",
            "name": "rotate",
            "image": "$inputs.image",
            "task_type": "rotate",
            "rotation_degrees": 45,
        },
        {
            "type": "roboflow_core/continue_if@v1",
            "name": "exact_match",
            "condition_statement": {
                "type": "StatementGroup",
                "statements": [
                    {
                        "type": "BinaryStatement",
                        "left_operand": {
                            "type": "StaticOperand",
                            "value": ["dog", "cat"],
                        },
                        "comparator": {"type": "=="},
                        "right_operand": {
                            "type": "DynamicOperand",
                            "operand_name": "left",
                            "operations": [
                                {
                                    "type": "ClassificationPropertyExtract",
                                    "property_name": "top_class",
                                }
                            ],
                        },
                    }
                ],
            },
            "next_steps": ["$steps.flip"],
            "evaluation_parameters": {"left": "$steps.multi_label.predictions"},
        },
        {
            "type": "roboflow_core/image_preprocessing@v1",
            "name": "flip",
            "image": "$inputs.image",
            "task_type": "flip",
        },
        {
            "type": "roboflow_core/image_preprocessing@v1",
            "name": "resize",
            "image": "$inputs.image",
            "task_type": "resize",
            "width": 50,
            "height": 50,
        },
        {
            "type": "roboflow_core/roboflow_classification_model@v1",
            "name": "single_label",
            "images": "$inputs.image",
            "model_id": "animals-2kk5l/1",
        },
        {
            "type": "roboflow_core/continue_if@v1",
            "name": "top_class",
            "condition_statement": {
                "type": "StatementGroup",
                "statements": [
                    {
                        "type": "BinaryStatement",
                        "left_operand": {
                            "type": "DynamicOperand",
                            "operand_name": "left",
                            "operations": [
                                {
                                    "type": "ClassificationPropertyExtract",
                                    "property_name": "top_class",
                                }
                            ],
                        },
                        "comparator": {"type": "=="},
                        "right_operand": {"type": "StaticOperand", "value": "cat"},
                    }
                ],
            },
            "next_steps": ["$steps.grayscale"],
            "evaluation_parameters": {"left": "$steps.single_label.predictions"},
        },
        {
            "type": "roboflow_core/convert_grayscale@v1",
            "name": "grayscale",
            "image": "$inputs.image",
        },
        {
            "type": "roboflow_core/property_definition@v1",
            "name": "single_classes",
            "data": "$steps.single_label.predictions",
            "operations": [
                {"type": "ClassificationPropertyExtract", "property_name": "top_class"}
            ],
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "multi",
            "coordinates_system": "own",
            "selector": "$steps.multi_label.predictions",
        },
        {
            "type": "JsonField",
            "name": "single_classes",
            "coordinates_system": "own",
            "selector": "$steps.single_classes.output",
        },
        {
            "type": "JsonField",
            "name": "multi_classes",
            "coordinates_system": "own",
            "selector": "$steps.multi_classes.output",
        },
        {
            "type": "JsonField",
            "name": "single_class_matched",
            "coordinates_system": "own",
            "selector": "$steps.grayscale.image",
        },
        {
            "type": "JsonField",
            "name": "includes_any_passed",
            "coordinates_system": "own",
            "selector": "$steps.resize.*",
        },
        {
            "type": "JsonField",
            "name": "includes_all_passed",
            "coordinates_system": "own",
            "selector": "$steps.rotate.image",
        },
        {
            "type": "JsonField",
            "name": "exact_match_passed",
            "coordinates_system": "own",
            "selector": "$steps.flip.image",
        },
    ],
}


def test_workflow_which_requires_empty_outputs_serialisation(
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
        workflow_definition=WORKFLOW_DEFINITION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
        },
        serialize_results=True,
    )

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "includes_any_passed",
        "multi",
        "single_class_matched",
        "multi_classes",
        "includes_all_passed",
        "single_classes",
        "exact_match_passed",
    }, "Expected all declared outputs to be delivered"
    assert (
        result[0]["single_class_matched"] is None
    ), "Expected empty result to serialize properly"
    assert result[0]["includes_any_passed"] == {
        "image": None
    }, "Expect compound empty result to be serialized properly"
    assert (
        result[0]["includes_all_passed"] is None
    ), "Expected empty result to serialize properly"
    assert (
        result[0]["exact_match_passed"] is None
    ), "Expected empty result to serialize properly"
