import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

WORKFLOW_WITH_CLASSICAL_PATTERN_MATCHING = {
    "version": "1.0",
    "inputs": [
        {"type": "InferenceImage", "name": "image"},
        {"type": "InferenceImage", "name": "template"},
    ],
    "steps": [
        {
            "type": "roboflow_core/template_matching@v1",
            "name": "template_matching",
            "image": "$inputs.image",
            "template": "$inputs.template",
            "matching_threshold": 0.8,
        },
        {
            "type": "roboflow_core/bounding_box_visualization@v1",
            "name": "bounding_box_visualization",
            "predictions": "$steps.template_matching.predictions",
            "image": "$inputs.image",
        },
        {
            "type": "roboflow_core/label_visualization@v1",
            "name": "label_visualization",
            "predictions": "$steps.template_matching.predictions",
            "image": "$steps.bounding_box_visualization.image",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "coordinates_system": "own",
            "selector": "$steps.template_matching.predictions",
        },
        {
            "type": "JsonField",
            "name": "visualization",
            "coordinates_system": "own",
            "selector": "$steps.label_visualization.image",
        },
    ],
}


def test_workflow_with_classical_pattern_matching(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    crowd_image: np.ndarray,
) -> None:
    """
    In this test set we check how classical pattern matching block integrates with
    other blocks that accept sv.Detections.

    Please point out that a single image is passed as template, and batch of images
    are passed as images to look for template. This workflow does also validate
    Execution Engine capabilities to broadcast batch-oriented inputs properly.
    """
    # given
    template = dogs_image[220:280, 310:410]  # dog's head
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_CLASSICAL_PATTERN_MATCHING,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [crowd_image, dogs_image],
            "template": template,
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 2, "Two images provided, so two outputs expected"
    assert set(result[0].keys()) == {
        "predictions",
        "visualization",
    }, "Expected all declared outputs to be delivered"
    assert set(result[1].keys()) == {
        "predictions",
        "visualization",
    }, "Expected all declared outputs to be delivered"
    assert (
        len(result[0]["predictions"]) == 0
    ), "Expected no patterns matched in first image"
    assert (
        len(result[1]["predictions"]) == 1
    ), "Expected single pattern match in second image"
    assert np.allclose(
        result[1]["predictions"].xyxy, np.array([312, 222, 412, 282])
    ), "Expected to find single template match"
    assert result[1]["predictions"].class_id.tolist() == [
        0
    ], "Expected fixed class id 0"
    assert result[1]["predictions"]["class_name"].tolist() == [
        "template_match"
    ], "Expected fixed class name"
    assert np.allclose(
        result[1]["predictions"].confidence, np.array([1.0])
    ), "Expected fixed confidence"


WORKFLOW_WITH_CLASSICAL_PATTERN_MATCHING_REFERRING_THE_SAME_AS_IMAGE_AND_TEMPLATE = {
    "version": "1.0",
    "inputs": [
        {"type": "InferenceImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "roboflow_core/template_matching@v1",
            "name": "template_matching",
            "image": "$inputs.image",
            "template": "$inputs.image",
            "matching_threshold": 0.8,
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "coordinates_system": "own",
            "selector": "$steps.template_matching.predictions",
        }
    ],
}


def test_workflow_with_classical_pattern_matching_using_the_same_images_as_templates(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    crowd_image: np.ndarray,
) -> None:
    """
    In this test set we check how classical pattern matching block integrates with
    other blocks that accept sv.Detections.

    In this case we are checking what happens when the same input is referred twice in step.
    We are doing it to compensate for bug detected in Execution engine while working on
    adding classical CV steps.
    """
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_CLASSICAL_PATTERN_MATCHING_REFERRING_THE_SAME_AS_IMAGE_AND_TEMPLATE,
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
    assert len(result) == 2, "Two images provided, so two outputs expected"
    assert set(result[0].keys()) == {
        "predictions",
    }, "Expected all declared outputs to be delivered"
    assert set(result[1].keys()) == {
        "predictions",
    }, "Expected all declared outputs to be delivered"
    assert (
        len(result[0]["predictions"]) == 1
    ), "Expected single pattern matched in first image"
    assert (
        len(result[1]["predictions"]) == 1
    ), "Expected single pattern match in second image"
