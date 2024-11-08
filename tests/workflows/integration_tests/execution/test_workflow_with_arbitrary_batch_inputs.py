from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest
import supervision as sv

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.utils.image_utils import load_image
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.errors import (
    AssumptionError,
    ExecutionGraphStructureError,
    RuntimeInputError,
)
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.introspection import blocks_loader

TWO_STAGE_WORKFLOW = {
    "version": "1.3.0",
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


OBJECT_DETECTION_WORKFLOW = {
    "version": "1.3.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "ObjectDetectionModel",
            "name": "general_detection",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
            "class_filter": ["dog"],
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.general_detection.*",
        },
    ],
}


CROP_WORKFLOW = {
    "version": "1.3.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {
            "type": "WorkflowBatchInput",
            "name": "predictions",
            "kind": ["object_detection_prediction"],
        },
    ],
    "steps": [
        {
            "type": "Crop",
            "name": "cropping",
            "image": "$inputs.image",
            "predictions": "$inputs.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.cropping.*",
        },
    ],
}

CLASSIFICATION_WORKFLOW = {
    "version": "1.3.0",
    "inputs": [
        {
            "type": "WorkflowBatchInput",
            "name": "crops",
            "kind": ["image"],
            "dimensionality": 2,
        },
    ],
    "steps": [
        {
            "type": "ClassificationModel",
            "name": "breds_classification",
            "image": "$inputs.crops",
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


def test_debug_execution_of_workflow_for_single_image_without_conditional_evaluation(
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
    end_to_end_execution_engine = ExecutionEngine.init(
        workflow_definition=TWO_STAGE_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    first_step_execution_engine = ExecutionEngine.init(
        workflow_definition=OBJECT_DETECTION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    second_step_execution_engine = ExecutionEngine.init(
        workflow_definition=CROP_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    third_step_execution_engine = ExecutionEngine.init(
        workflow_definition=CLASSIFICATION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    e2e_results = end_to_end_execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
        }
    )
    detection_results = first_step_execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
        }
    )
    cropping_results = second_step_execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
            "predictions": detection_results[0]["result"]["predictions"],
        }
    )
    classification_results = third_step_execution_engine.run(
        runtime_parameters={
            "crops": [[e["crops"] for e in cropping_results[0]["result"]]],
        }
    )

    # then
    e2e_top_classes = [p["top"] for p in e2e_results[0]["predictions"]]
    debug_top_classes = [p["top"] for p in classification_results[0]["predictions"]]
    assert (
        e2e_top_classes == debug_top_classes
    ), "Expected top class prediction from step-by-step execution to match e2e execution"
    e2e_confidence = [p["confidence"] for p in e2e_results[0]["predictions"]]
    debug_confidence = [
        p["confidence"] for p in classification_results[0]["predictions"]
    ]
    assert np.allclose(
        e2e_confidence, debug_confidence, atol=1e-4
    ), "Expected confidences from step-by-step execution to match e2e execution"


def test_debug_execution_of_workflow_for_single_image_without_conditional_evaluation_when_serialization_is_requested(
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
    end_to_end_execution_engine = ExecutionEngine.init(
        workflow_definition=TWO_STAGE_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    first_step_execution_engine = ExecutionEngine.init(
        workflow_definition=OBJECT_DETECTION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    second_step_execution_engine = ExecutionEngine.init(
        workflow_definition=CROP_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    third_step_execution_engine = ExecutionEngine.init(
        workflow_definition=CLASSIFICATION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    e2e_results = end_to_end_execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
        },
        serialize_results=True,
    )
    detection_results = first_step_execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
        },
        serialize_results=True,
    )
    detection_results_not_serialized = first_step_execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
        },
    )
    cropping_results = second_step_execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
            "predictions": detection_results[0]["result"]["predictions"],
        },
        serialize_results=True,
    )
    cropping_results_not_serialized = second_step_execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
            "predictions": detection_results_not_serialized[0]["result"]["predictions"],
        },
        serialize_results=False,
    )
    classification_results = third_step_execution_engine.run(
        runtime_parameters={
            "crops": [[e["crops"] for e in cropping_results[0]["result"]]],
        },
        serialize_results=True,
    )

    # then
    assert isinstance(
        detection_results[0]["result"]["predictions"], dict
    ), "Expected sv.Detections to be serialized"
    assert isinstance(
        detection_results_not_serialized[0]["result"]["predictions"], sv.Detections
    ), "Expected sv.Detections not to be serialized"
    deserialized_detections = sv.Detections.from_inference(
        detection_results[0]["result"]["predictions"]
    )
    assert np.allclose(
        deserialized_detections.confidence,
        detection_results_not_serialized[0]["result"]["predictions"].confidence,
        atol=1e-4,
    ), "Expected confidence match when serialized detections are deserialized"
    intermediate_crop = cropping_results[0]["result"][0]["crops"]
    assert (
        intermediate_crop["type"] == "base64"
    ), "Expected crop to be serialized to base64"
    decoded_image, _ = load_image(intermediate_crop)
    number_of_pixels = (
        decoded_image.shape[0] * decoded_image.shape[1] * decoded_image.shape[2]
    )
    assert (
        decoded_image.shape
        == cropping_results_not_serialized[0]["result"][0]["crops"].numpy_image.shape
    ), "Expected deserialized crop to match in size with not serialized one"
    assert (
        abs(
            (decoded_image.sum() / number_of_pixels)
            - (
                cropping_results_not_serialized[0]["result"][0][
                    "crops"
                ].numpy_image.sum()
                / number_of_pixels
            )
        )
        < 1e-1
    ), "Content of serialized and not serialized crop should roughly match (up to compression)"
    e2e_top_classes = [p["top"] for p in e2e_results[0]["predictions"]]
    debug_top_classes = [p["top"] for p in classification_results[0]["predictions"]]
    assert (
        e2e_top_classes == debug_top_classes
    ), "Expected top class prediction from step-by-step execution to match e2e execution"
    e2e_confidence = [p["confidence"] for p in e2e_results[0]["predictions"]]
    debug_confidence = [
        p["confidence"] for p in classification_results[0]["predictions"]
    ]
    assert np.allclose(
        e2e_confidence, debug_confidence, atol=1e-1
    ), "Expected confidences from step-by-step execution to match e2e execution"


def test_debug_execution_of_workflow_for_batch_of_images_without_conditional_evaluation(
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
    end_to_end_execution_engine = ExecutionEngine.init(
        workflow_definition=TWO_STAGE_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    first_step_execution_engine = ExecutionEngine.init(
        workflow_definition=OBJECT_DETECTION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    second_step_execution_engine = ExecutionEngine.init(
        workflow_definition=CROP_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    third_step_execution_engine = ExecutionEngine.init(
        workflow_definition=CLASSIFICATION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    e2e_results = end_to_end_execution_engine.run(
        runtime_parameters={
            "image": [dogs_image, dogs_image],
        }
    )
    detection_results = first_step_execution_engine.run(
        runtime_parameters={
            "image": [dogs_image, dogs_image],
        }
    )
    cropping_results = second_step_execution_engine.run(
        runtime_parameters={
            "image": [dogs_image, dogs_image],
            "predictions": [
                detection_results[0]["result"]["predictions"],
                detection_results[1]["result"]["predictions"],
            ],
        }
    )
    classification_results = third_step_execution_engine.run(
        runtime_parameters={
            "crops": [
                [e["crops"] for e in cropping_results[0]["result"]],
                [e["crops"] for e in cropping_results[1]["result"]],
            ],
        }
    )

    # then
    e2e_top_classes = [p["top"] for r in e2e_results for p in r["predictions"]]
    debug_top_classes = [
        p["top"] for r in classification_results for p in r["predictions"]
    ]
    assert (
        e2e_top_classes == debug_top_classes
    ), "Expected top class prediction from step-by-step execution to match e2e execution"
    e2e_confidence = [p["confidence"] for r in e2e_results for p in r["predictions"]]
    debug_confidence = [
        p["confidence"] for r in classification_results for p in r["predictions"]
    ]
    assert np.allclose(
        e2e_confidence, debug_confidence, atol=1e-4
    ), "Expected confidences from step-by-step execution to match e2e execution"


TWO_STAGE_WORKFLOW_WITH_FLOW_CONTROL = {
    "version": "1.3.0",
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
            "type": "Crop",
            "name": "cropping",
            "image": "$inputs.image",
            "predictions": "$steps.general_detection.predictions",
        },
        {
            "type": "roboflow_core/continue_if@v1",
            "name": "verify_crop_size",
            "condition_statement": {
                "type": "StatementGroup",
                "statements": [
                    {
                        "type": "BinaryStatement",
                        "left_operand": {
                            "type": "DynamicOperand",
                            "operand_name": "crops",
                            "operations": [
                                {
                                    "type": "ExtractImageProperty",
                                    "property_name": "size",
                                },
                            ],
                        },
                        "comparator": {"type": "(Number) >="},
                        "right_operand": {
                            "type": "StaticOperand",
                            "value": 48000,
                        },
                    }
                ],
            },
            "next_steps": ["$steps.breds_classification"],
            "evaluation_parameters": {"crops": "$steps.cropping.crops"},
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


def test_debug_execution_of_workflow_for_batch_of_images_with_conditional_evaluation(
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
    end_to_end_execution_engine = ExecutionEngine.init(
        workflow_definition=TWO_STAGE_WORKFLOW_WITH_FLOW_CONTROL,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    first_step_execution_engine = ExecutionEngine.init(
        workflow_definition=OBJECT_DETECTION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    second_step_execution_engine = ExecutionEngine.init(
        workflow_definition=CROP_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    third_step_execution_engine = ExecutionEngine.init(
        workflow_definition=CLASSIFICATION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    e2e_results = end_to_end_execution_engine.run(
        runtime_parameters={
            "image": [dogs_image, dogs_image],
        }
    )
    detection_results = first_step_execution_engine.run(
        runtime_parameters={
            "image": [dogs_image, dogs_image],
        }
    )
    cropping_results = second_step_execution_engine.run(
        runtime_parameters={
            "image": [dogs_image, dogs_image],
            "predictions": [
                detection_results[0]["result"]["predictions"],
                detection_results[1]["result"]["predictions"],
            ],
        }
    )
    classification_results = third_step_execution_engine.run(
        runtime_parameters={
            "crops": [
                [cropping_results[0]["result"][0]["crops"], None],
                [cropping_results[1]["result"][0]["crops"], None],
            ],
        }
    )

    # then
    assert (
        e2e_results[0]["predictions"][0] is not None
    ), "Expected first dog crop not to be excluded by conditional eval"
    assert (
        e2e_results[0]["predictions"][1] is None
    ), "Expected second dog crop to be excluded by conditional eval"
    assert (
        e2e_results[1]["predictions"][0] is not None
    ), "Expected first dog crop not to be excluded by conditional eval"
    assert (
        e2e_results[1]["predictions"][1] is None
    ), "Expected second dog crop to be excluded by conditional eval"
    e2e_top_classes = [
        p["top"] if p else None for r in e2e_results for p in r["predictions"]
    ]
    debug_top_classes = [
        p["top"] if p else None
        for r in classification_results
        for p in r["predictions"]
    ]
    assert (
        e2e_top_classes == debug_top_classes
    ), "Expected top class prediction from step-by-step execution to match e2e execution"
    e2e_confidence = [
        p["confidence"] if p else -1000.0 for r in e2e_results for p in r["predictions"]
    ]
    debug_confidence = [
        p["confidence"] if p else -1000.0
        for r in classification_results
        for p in r["predictions"]
    ]
    assert np.allclose(
        e2e_confidence, debug_confidence, atol=1e-4
    ), "Expected confidences from step-by-step execution to match e2e execution"


def test_debug_execution_when_empty_batch_oriented_input_provided(
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
        workflow_definition=CROP_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    with pytest.raises(RuntimeInputError):
        _ = execution_engine.run(
            runtime_parameters={"image": [dogs_image, dogs_image], "predictions": None}
        )


WORKFLOW_WITH_BATCH_ORIENTED_CONFIDENCE = {
    "version": "1.3.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {
            "type": "WorkflowBatchInput",
            "name": "confidence",
        },
    ],
    "steps": [
        {
            "type": "ObjectDetectionModel",
            "name": "general_detection",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
            "class_filter": ["dog"],
            "confidence": "$inputs.confidence",
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.general_detection.*",
        },
    ],
}


def test_workflow_run_which_hooks_up_batch_oriented_input_into_non_batch_oriented_parameters(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }

    # when
    with pytest.raises(ExecutionGraphStructureError):
        _ = ExecutionEngine.init(
            workflow_definition=WORKFLOW_WITH_BATCH_ORIENTED_CONFIDENCE,
            init_parameters=workflow_init_parameters,
            max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
        )


WORKFLOW_WITH_NON_BATCH_ORIENTED_STEP_FEEDING_NON_BATCH_ORIENTED_STEP = {
    "version": "1.3.0",
    "inputs": [
        {"type": "WorkflowParameter", "name": "non_batch_parameter"},
    ],
    "steps": [
        {
            "type": "NonBatchInputBlock",
            "name": "step_one",
            "non_batch_parameter": "$inputs.non_batch_parameter",
        },
        {
            "type": "MixedInputWithBatchesBlock",
            "name": "step_two",
            "mixed_parameter": "$steps.step_one.float_value",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.step_two.float_value",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_when_non_batch_oriented_step_feeds_non_batch_oriented_step(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.mixed_input_characteristic_plugin",
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_NON_BATCH_ORIENTED_STEP_FEEDING_NON_BATCH_ORIENTED_STEP,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # then
    result = execution_engine.run(
        runtime_parameters={
            "non_batch_parameter": "some",
        }
    )

    # then
    assert len(result) == 1, "Expected singular result"
    assert result[0]["result"] == 0.4, "Expected hardcoded result"


WORKFLOW_WITH_NON_BATCH_ORIENTED_STEP_FEEDING_BATCH_ORIENTED_STEP_NOT_OPERATING_BATCH_WISE = {
    "version": "1.3.0",
    "inputs": [
        {"type": "WorkflowParameter", "name": "non_batch_parameter"},
    ],
    "steps": [
        {
            "type": "NonBatchInputBlock",
            "name": "step_one",
            "non_batch_parameter": "$inputs.non_batch_parameter",
        },
        {
            "type": "BatchInputBlockNotProcessingBatches",
            "name": "step_two",
            "batch_parameter": "$steps.step_one.float_value",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.step_two.float_value",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_when_non_batch_oriented_step_feeds_batch_oriented_step_not_operating_batch_wise(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.mixed_input_characteristic_plugin",
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_NON_BATCH_ORIENTED_STEP_FEEDING_BATCH_ORIENTED_STEP_NOT_OPERATING_BATCH_WISE,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # then
    result = execution_engine.run(
        runtime_parameters={
            "non_batch_parameter": "some",
        }
    )

    # then
    assert len(result) == 1, "Expected singular result"
    assert result[0]["result"] == 0.4, "Expected hardcoded result"


WORKFLOW_WITH_NON_BATCH_ORIENTED_STEP_FEEDING_BATCH_ORIENTED_STEP_OPERATING_BATCH_WISE = {
    "version": "1.3.0",
    "inputs": [
        {"type": "WorkflowParameter", "name": "non_batch_parameter"},
    ],
    "steps": [
        {
            "type": "NonBatchInputBlock",
            "name": "step_one",
            "non_batch_parameter": "$inputs.non_batch_parameter",
        },
        {
            "type": "BatchInputBlockProcessingBatches",
            "name": "step_two",
            "batch_parameter": "$steps.step_one.float_value",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.step_two.float_value",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_when_non_batch_oriented_step_feeds_batch_oriented_step_operating_batch_wise(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.mixed_input_characteristic_plugin",
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }

    # when
    with pytest.raises(ExecutionGraphStructureError):
        _ = ExecutionEngine.init(
            workflow_definition=WORKFLOW_WITH_NON_BATCH_ORIENTED_STEP_FEEDING_BATCH_ORIENTED_STEP_OPERATING_BATCH_WISE,
            init_parameters=workflow_init_parameters,
            max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
        )


WORKFLOW_WITH_NON_BATCH_ORIENTED_STEP_FEEDING_MIXED_INPUT_STEP = {
    "version": "1.3.0",
    "inputs": [
        {"type": "WorkflowParameter", "name": "non_batch_parameter"},
    ],
    "steps": [
        {
            "type": "NonBatchInputBlock",
            "name": "step_one",
            "non_batch_parameter": "$inputs.non_batch_parameter",
        },
        {
            "type": "MixedInputWithBatchesBlock",
            "name": "step_two",
            "mixed_parameter": "$steps.step_one.float_value",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.step_two.float_value",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_when_non_batch_oriented_step_feeds_mixed_input_step(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.mixed_input_characteristic_plugin",
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_NON_BATCH_ORIENTED_STEP_FEEDING_MIXED_INPUT_STEP,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # then
    result = execution_engine.run(
        runtime_parameters={
            "non_batch_parameter": "some",
        }
    )

    # then
    assert len(result) == 1, "Expected singular result"
    assert result[0]["result"] == 0.4, "Expected hardcoded result"


WORKFLOW_WITH_BATCH_ORIENTED_STEP_FEEDING_MIXED_INPUT_STEP = {
    "version": "1.3.0",
    "inputs": [
        {"type": "WorkflowParameter", "name": "non_batch_parameter"},
    ],
    "steps": [
        {
            "type": "NonBatchInputBlock",
            "name": "step_one",
            "non_batch_parameter": "$inputs.non_batch_parameter",
        },
        {
            "type": "BatchInputBlockNotProcessingBatches",
            "name": "step_two",
            "batch_parameter": "$steps.step_one.float_value",
        },
        {
            "type": "MixedInputWithBatchesBlock",
            "name": "step_three",
            "mixed_parameter": "$steps.step_two.float_value",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.step_three.float_value",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_when_batch_oriented_step_feeds_mixed_input_step(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.mixed_input_characteristic_plugin",
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_BATCH_ORIENTED_STEP_FEEDING_MIXED_INPUT_STEP,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # then
    result = execution_engine.run(
        runtime_parameters={
            "non_batch_parameter": "some",
        }
    )

    # then
    assert len(result) == 1, "Expected singular result"
    assert result[0]["result"] == 0.4, "Expected hardcoded result"


WORKFLOW_WITH_BATCH_ORIENTED_INPUT_FEEDING_INTO_BATCH_ORIENTED_STEP = {
    "version": "1.3.0",
    "inputs": [
        {
            "type": "WorkflowBatchInput",
            "name": "data",
        },
    ],
    "steps": [
        {
            "type": "BatchInputBlockProcessingBatches",
            "name": "step_one",
            "batch_parameter": "$inputs.data",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.step_one.float_value",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_when_batch_oriented_input_feeds_batch_input_step(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.mixed_input_characteristic_plugin",
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_BATCH_ORIENTED_INPUT_FEEDING_INTO_BATCH_ORIENTED_STEP,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # then
    result = execution_engine.run(
        runtime_parameters={
            "data": ["some", "other"],
        }
    )

    # then
    assert len(result) == 2, "Expected singular result"
    assert result[0]["result"] == 0.4, "Expected hardcoded result"
    assert result[1]["result"] == 0.4, "Expected hardcoded result"


WORKFLOW_WITH_BATCH_ORIENTED_INPUT_FEEDING_INTO_MIXED_INPUT_STEP = {
    "version": "1.3.0",
    "inputs": [
        {
            "type": "WorkflowBatchInput",
            "name": "data",
        },
    ],
    "steps": [
        {
            "type": "MixedInputWithBatchesBlock",
            "name": "step_one",
            "mixed_parameter": "$inputs.data",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.step_one.float_value",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_when_batch_oriented_input_feeds_mixed_input_step(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.mixed_input_characteristic_plugin",
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_BATCH_ORIENTED_INPUT_FEEDING_INTO_MIXED_INPUT_STEP,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # then
    result = execution_engine.run(
        runtime_parameters={
            "data": ["some", "other"],
        }
    )

    # then
    assert len(result) == 2, "Expected singular result"
    assert result[0]["result"] == 0.4, "Expected hardcoded result"
    assert result[1]["result"] == 0.4, "Expected hardcoded result"


WORKFLOW_WITH_BATCH_ORIENTED_INPUT_FEEDING_INTO_NON_BATCH_ORIENTED_STEP = {
    "version": "1.3.0",
    "inputs": [
        {
            "type": "WorkflowBatchInput",
            "name": "data",
        },
    ],
    "steps": [
        {
            "type": "NonBatchInputBlock",
            "name": "step_one",
            "non_batch_parameter": "$inputs.data",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.step_one.float_value",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_when_batch_oriented_input_feeds_non_batch_input_step(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.mixed_input_characteristic_plugin",
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_BATCH_ORIENTED_INPUT_FEEDING_INTO_NON_BATCH_ORIENTED_STEP,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "data": ["some", "other"],
        }
    )

    # then
    assert len(result) == 2, "Expected two outputs for two input elements"
    assert result[0]["result"] == 0.4, "Expected hardcoded value"
    assert result[1]["result"] == 0.4, "Expected hardcoded value"


WORKFLOW_WITH_NON_BATCH_ORIENTED_STEP_FEEDING_COMPOUND_NON_BATCH_ORIENTED_STEP = {
    "version": "1.3.0",
    "inputs": [
        {"type": "WorkflowParameter", "name": "non_batch_parameter"},
    ],
    "steps": [
        {
            "type": "NonBatchInputBlock",
            "name": "step_one",
            "non_batch_parameter": "$inputs.non_batch_parameter",
        },
        {
            "type": "CompoundNonBatchInputBlock",
            "name": "step_two",
            "compound_parameter": {
                "some": "$steps.step_one.float_value",
            },
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.step_two.float_value",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_when_non_batch_oriented_step_feeds_compound_non_batch_oriented_step(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.mixed_input_characteristic_plugin",
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_NON_BATCH_ORIENTED_STEP_FEEDING_COMPOUND_NON_BATCH_ORIENTED_STEP,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # then
    result = execution_engine.run(
        runtime_parameters={
            "non_batch_parameter": "some",
        }
    )

    # then
    assert len(result) == 1, "Expected singular result"
    assert result[0]["result"] == 0.4, "Expected hardcoded result"


WORKFLOW_WITH_NON_BATCH_ORIENTED_STEP_FEEDING_COMPOUND_MIXED_ORIENTED_STEP = {
    "version": "1.3.0",
    "inputs": [
        {"type": "WorkflowParameter", "name": "non_batch_parameter"},
    ],
    "steps": [
        {
            "type": "NonBatchInputBlock",
            "name": "step_one",
            "non_batch_parameter": "$inputs.non_batch_parameter",
        },
        {
            "type": "CompoundMixedInputBlockManifestBlock",
            "name": "step_two",
            "compound_parameter": {
                "some": "$steps.step_one.float_value",
            },
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.step_two.float_value",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_when_non_batch_oriented_step_feeds_compound_mixed_oriented_step(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.mixed_input_characteristic_plugin",
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_NON_BATCH_ORIENTED_STEP_FEEDING_COMPOUND_MIXED_ORIENTED_STEP,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # then
    result = execution_engine.run(
        runtime_parameters={
            "non_batch_parameter": "some",
        }
    )

    # then
    assert len(result) == 1, "Expected singular result"
    assert result[0]["result"] == 0.4, "Expected hardcoded result"


WORKFLOW_WITH_NON_BATCH_ORIENTED_STEP_FEEDING_COMPOUND_LOOSELY_BATCH_ORIENTED_STEP = {
    "version": "1.3.0",
    "inputs": [
        {"type": "WorkflowParameter", "name": "non_batch_parameter"},
    ],
    "steps": [
        {
            "type": "NonBatchInputBlock",
            "name": "step_one",
            "non_batch_parameter": "$inputs.non_batch_parameter",
        },
        {
            "type": "CompoundNonStrictBatchBlock",
            "name": "step_two",
            "compound_parameter": {
                "some": "$steps.step_one.float_value",
            },
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.step_two.float_value",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_when_non_batch_oriented_step_feeds_compound_loosely_batch_oriented_step(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.mixed_input_characteristic_plugin",
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_NON_BATCH_ORIENTED_STEP_FEEDING_COMPOUND_LOOSELY_BATCH_ORIENTED_STEP,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # then
    result = execution_engine.run(
        runtime_parameters={
            "non_batch_parameter": "some",
        }
    )

    # then
    assert len(result) == 1, "Expected singular result"
    assert result[0]["result"] == 0.4, "Expected hardcoded result"


WORKFLOW_WITH_NON_BATCH_ORIENTED_STEP_FEEDING_COMPOUND_STRICTLY_BATCH_ORIENTED_STEP = {
    "version": "1.3.0",
    "inputs": [
        {"type": "WorkflowParameter", "name": "non_batch_parameter"},
    ],
    "steps": [
        {
            "type": "NonBatchInputBlock",
            "name": "step_one",
            "non_batch_parameter": "$inputs.non_batch_parameter",
        },
        {
            "type": "CompoundStrictBatchBlock",
            "name": "step_two",
            "compound_parameter": {
                "some": "$steps.step_one.float_value",
            },
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.step_two.float_value",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_when_non_batch_oriented_step_feeds_compound_strictly_batch_oriented_step(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.mixed_input_characteristic_plugin",
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }

    # then
    with pytest.raises(ExecutionGraphStructureError):
        _ = ExecutionEngine.init(
            workflow_definition=WORKFLOW_WITH_NON_BATCH_ORIENTED_STEP_FEEDING_COMPOUND_STRICTLY_BATCH_ORIENTED_STEP,
            init_parameters=workflow_init_parameters,
            max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
        )


WORKFLOW_WITH_BATCH_ORIENTED_STEP_FEEDING_COMPOUND_NON_BATCH_ORIENTED_STEP = {
    "version": "1.3.0",
    "inputs": [
        {"type": "WorkflowParameter", "name": "non_batch_parameter"},
    ],
    "steps": [
        {
            "type": "NonBatchInputBlock",
            "name": "step_one",
            "non_batch_parameter": "$inputs.non_batch_parameter",
        },
        {
            "type": "BatchInputBlockNotProcessingBatches",
            "name": "step_two",
            "batch_parameter": "$steps.step_one.float_value",
        },
        {
            "type": "CompoundNonBatchInputBlock",
            "name": "step_three",
            "compound_parameter": {
                "some": "$steps.step_two.float_value",
            },
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.step_three.float_value",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_when_batch_oriented_step_feeds_compound_non_batch_oriented_step(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.mixed_input_characteristic_plugin",
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_BATCH_ORIENTED_STEP_FEEDING_COMPOUND_NON_BATCH_ORIENTED_STEP,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # then
    result = execution_engine.run(
        runtime_parameters={
            "non_batch_parameter": "some",
        }
    )

    # then
    assert len(result) == 1, "Expected singular result"
    assert result[0]["result"] == 0.4, "Expected hardcoded result"


WORKFLOW_WITH_BATCH_ORIENTED_STEP_FEEDING_MIXED_ORIENTED_STEP = {
    "version": "1.3.0",
    "inputs": [
        {"type": "WorkflowParameter", "name": "non_batch_parameter"},
    ],
    "steps": [
        {
            "type": "NonBatchInputBlock",
            "name": "step_one",
            "non_batch_parameter": "$inputs.non_batch_parameter",
        },
        {
            "type": "BatchInputBlockNotProcessingBatches",
            "name": "step_two",
            "batch_parameter": "$steps.step_one.float_value",
        },
        {
            "type": "CompoundMixedInputBlockManifestBlock",
            "name": "step_three",
            "compound_parameter": {
                "some": "$steps.step_two.float_value",
            },
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.step_three.float_value",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_when_batch_oriented_step_feeds_compound_mixed_oriented_step(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.mixed_input_characteristic_plugin",
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_BATCH_ORIENTED_STEP_FEEDING_MIXED_ORIENTED_STEP,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # then
    result = execution_engine.run(
        runtime_parameters={
            "non_batch_parameter": "some",
        }
    )

    # then
    assert len(result) == 1, "Expected singular result"
    assert result[0]["result"] == 0.4, "Expected hardcoded result"


WORKFLOW_WITH_BATCH_ORIENTED_STEP_FEEDING_BATCH_ORIENTED_STEP = {
    "version": "1.3.0",
    "inputs": [
        {"type": "WorkflowParameter", "name": "non_batch_parameter"},
    ],
    "steps": [
        {
            "type": "NonBatchInputBlock",
            "name": "step_one",
            "non_batch_parameter": "$inputs.non_batch_parameter",
        },
        {
            "type": "BatchInputBlockNotProcessingBatches",
            "name": "step_two",
            "batch_parameter": "$steps.step_one.float_value",
        },
        {
            "type": "CompoundNonStrictBatchBlock",
            "name": "step_three",
            "compound_parameter": {
                "some": "$steps.step_two.float_value",
            },
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.step_three.float_value",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_when_batch_oriented_step_feeds_compound_batch_oriented_step(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.mixed_input_characteristic_plugin",
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_BATCH_ORIENTED_STEP_FEEDING_BATCH_ORIENTED_STEP,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # then
    result = execution_engine.run(
        runtime_parameters={
            "non_batch_parameter": "some",
        }
    )

    # then
    assert len(result) == 1, "Expected singular result"
    assert result[0]["result"] == 0.4, "Expected hardcoded result"


WORKFLOW_WITH_NON_BATCH_ORIENTED_INPUT_FEEDING_COMPOUND_NON_BATCH_ORIENTED_STEP = {
    "version": "1.3.0",
    "inputs": [
        {"type": "WorkflowParameter", "name": "data"},
    ],
    "steps": [
        {
            "type": "CompoundNonBatchInputBlock",
            "name": "step_one",
            "compound_parameter": {
                "some": "$inputs.data",
            },
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.step_one.float_value",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_when_non_batch_oriented_input_feeds_compound_non_batch_oriented_step(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.mixed_input_characteristic_plugin",
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_NON_BATCH_ORIENTED_INPUT_FEEDING_COMPOUND_NON_BATCH_ORIENTED_STEP,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # then
    result = execution_engine.run(
        runtime_parameters={
            "data": "some",
        }
    )

    # then
    assert len(result) == 1, "Expected singular result"
    assert result[0]["result"] == 0.4, "Expected hardcoded result"


WORKFLOW_WITH_NON_BATCH_ORIENTED_INPUT_FEEDING_COMPOUND_MIXED_ORIENTED_STEP = {
    "version": "1.3.0",
    "inputs": [
        {"type": "WorkflowParameter", "name": "data"},
    ],
    "steps": [
        {
            "type": "CompoundMixedInputBlockManifestBlock",
            "name": "step_one",
            "compound_parameter": {
                "some": "$inputs.data",
            },
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.step_one.float_value",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_when_non_batch_oriented_input_feeds_compound_mixed_oriented_step(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.mixed_input_characteristic_plugin",
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_NON_BATCH_ORIENTED_INPUT_FEEDING_COMPOUND_MIXED_ORIENTED_STEP,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # then
    result = execution_engine.run(
        runtime_parameters={
            "data": "some",
        }
    )

    # then
    assert len(result) == 1, "Expected singular result"
    assert result[0]["result"] == 0.4, "Expected hardcoded result"


WORKFLOW_WITH_NON_BATCH_ORIENTED_INPUT_FEEDING_COMPOUND_LOOSELY_BATCH_ORIENTED_STEP = {
    "version": "1.3.0",
    "inputs": [
        {"type": "WorkflowParameter", "name": "data"},
    ],
    "steps": [
        {
            "type": "CompoundNonStrictBatchBlock",
            "name": "step_one",
            "compound_parameter": {
                "some": "$inputs.data",
            },
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.step_one.float_value",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_when_non_batch_oriented_input_feeds_compound_loosely_batch_oriented_step(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.mixed_input_characteristic_plugin",
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_NON_BATCH_ORIENTED_INPUT_FEEDING_COMPOUND_LOOSELY_BATCH_ORIENTED_STEP,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # then
    result = execution_engine.run(
        runtime_parameters={
            "data": "some",
        }
    )

    # then
    assert len(result) == 1, "Expected singular result"
    assert result[0]["result"] == 0.4, "Expected hardcoded result"


WORKFLOW_WITH_NON_BATCH_ORIENTED_INPUT_FEEDING_COMPOUND_STRICTLY_BATCH_ORIENTED_STEP = {
    "version": "1.3.0",
    "inputs": [
        {"type": "WorkflowParameter", "name": "data"},
    ],
    "steps": [
        {
            "type": "CompoundStrictBatchBlock",
            "name": "step_one",
            "compound_parameter": {
                "some": "$inputs.data",
            },
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.step_one.float_value",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_when_non_batch_oriented_input_feeds_compound_strictly_batch_oriented_step(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.mixed_input_characteristic_plugin",
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }

    # then
    with pytest.raises(ExecutionGraphStructureError):
        _ = ExecutionEngine.init(
            workflow_definition=WORKFLOW_WITH_NON_BATCH_ORIENTED_INPUT_FEEDING_COMPOUND_STRICTLY_BATCH_ORIENTED_STEP,
            init_parameters=workflow_init_parameters,
            max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
        )


WORKFLOW_WITH_BATCH_ORIENTED_INPUT_FEEDING_COMPOUND_NON_BATCH_ORIENTED_STEP = {
    "version": "1.3.0",
    "inputs": [
        {
            "type": "WorkflowBatchInput",
            "name": "data",
        },
    ],
    "steps": [
        {
            "type": "CompoundNonBatchInputBlock",
            "name": "step_one",
            "compound_parameter": {
                "some": "$inputs.data",
            },
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.step_one.float_value",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_when_batch_oriented_input_feeds_compound_non_batch_oriented_step(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.mixed_input_characteristic_plugin",
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_BATCH_ORIENTED_INPUT_FEEDING_COMPOUND_NON_BATCH_ORIENTED_STEP,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "data": ["some", "other"],
        }
    )

    # then
    assert len(result) == 2, "Expected 2 outputs for 2 inputs"
    assert result[0]["result"] == 0.4, "Expected hardcoded value"
    assert result[1]["result"] == 0.4, "Expected hardcoded value"


WORKFLOW_WITH_BATCH_ORIENTED_INPUT_FEEDING_COMPOUND_MIXED_ORIENTED_STEP = {
    "version": "1.3.0",
    "inputs": [
        {
            "type": "WorkflowBatchInput",
            "name": "data",
        },
    ],
    "steps": [
        {
            "type": "CompoundMixedInputBlockManifestBlock",
            "name": "step_one",
            "compound_parameter": {
                "some": "$inputs.data",
            },
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.step_one.float_value",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_when_batch_oriented_input_feeds_compound_mixed_oriented_step(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.mixed_input_characteristic_plugin",
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_BATCH_ORIENTED_INPUT_FEEDING_COMPOUND_MIXED_ORIENTED_STEP,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "data": ["some", "other"],
        }
    )

    # then
    assert len(result) == 2, "Expected singular result"
    assert result[0]["result"] == 0.4, "Expected hardcoded result"
    assert result[1]["result"] == 0.4, "Expected hardcoded result"


WORKFLOW_WITH_BATCH_ORIENTED_INPUT_FEEDING_COMPOUND_LOOSELY_BATCH_ORIENTED_STEP = {
    "version": "1.3.0",
    "inputs": [
        {
            "type": "WorkflowBatchInput",
            "name": "data",
        },
    ],
    "steps": [
        {
            "type": "CompoundNonStrictBatchBlock",
            "name": "step_one",
            "compound_parameter": {
                "some": "$inputs.data",
            },
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.step_one.float_value",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_when_batch_oriented_input_feeds_compound_loosely_batch_oriented_step(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.mixed_input_characteristic_plugin",
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_BATCH_ORIENTED_INPUT_FEEDING_COMPOUND_LOOSELY_BATCH_ORIENTED_STEP,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "data": ["some", "other"],
        }
    )

    # then
    assert len(result) == 2, "Expected singular result"
    assert result[0]["result"] == 0.4, "Expected hardcoded result"
    assert result[1]["result"] == 0.4, "Expected hardcoded result"


WORKFLOW_WITH_BATCH_ORIENTED_INPUT_FEEDING_COMPOUND_STRICTLY_BATCH_ORIENTED_STEP = {
    "version": "1.3.0",
    "inputs": [
        {
            "type": "WorkflowBatchInput",
            "name": "data",
        },
    ],
    "steps": [
        {
            "type": "CompoundStrictBatchBlock",
            "name": "step_one",
            "compound_parameter": {
                "some": "$inputs.data",
            },
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.step_one.float_value",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_when_batch_oriented_input_feeds_compound_strictly_batch_oriented_step(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.mixed_input_characteristic_plugin",
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_BATCH_ORIENTED_INPUT_FEEDING_COMPOUND_STRICTLY_BATCH_ORIENTED_STEP,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "data": ["some", "other"],
        }
    )

    # then
    assert len(result) == 2, "Expected singular result"
    assert result[0]["result"] == 0.4, "Expected hardcoded result"
    assert result[1]["result"] == 0.4, "Expected hardcoded result"
