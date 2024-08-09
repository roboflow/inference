import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

DETECTION_CLASSES_REPLACEMENT_WORKFLOW = {
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
        {
            "type": "DetectionsClassesReplacement",
            "name": "classes_replacement",
            "object_detection_predictions": "$steps.general_detection.predictions",
            "classification_predictions": "$steps.breds_classification.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "original_predictions",
            "selector": "$steps.general_detection.predictions",
        },
        {
            "type": "JsonField",
            "name": "predictions_with_replaced_classes",
            "selector": "$steps.classes_replacement.predictions",
        },
    ],
}


def test_detection_plus_classification_workflow_when_minimal_valid_input_provided(
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
        workflow_definition=DETECTION_CLASSES_REPLACEMENT_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image, dogs_image, crowd_image],
        }
    )

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 3, "Expected 3 element in the output for three input images"
    assert set(result[0].keys()) == {
        "original_predictions",
        "predictions_with_replaced_classes",
    }, "Expected all declared outputs to be delivered for first output"
    assert set(result[1].keys()) == {
        "original_predictions",
        "predictions_with_replaced_classes",
    }, "Expected all declared outputs to be delivered for second output"
    assert set(result[2].keys()) == {
        "original_predictions",
        "predictions_with_replaced_classes",
    }, "Expected all declared outputs to be delivered for third output"
    assert (
        len(result[0]["original_predictions"])
        == len(result[0]["predictions_with_replaced_classes"])
        == 2
    ), "Expected 2 dogs detected for first image"
    assert (
        len(result[1]["original_predictions"])
        == len(result[1]["predictions_with_replaced_classes"])
        == 2
    ), "Expected 2 dogs detected for second image"
    assert (
        len(result[2]["original_predictions"])
        == len(result[2]["predictions_with_replaced_classes"])
        == 0
    ), "Expected 0 dogs detected for third image"
    assert (
        result[0]["predictions_with_replaced_classes"].confidence.tolist()
        != result[0]["original_predictions"].confidence.tolist()
    ), "Expected confidences to be altered"
    assert (
        result[0]["predictions_with_replaced_classes"].class_id.tolist()
        != result[0]["original_predictions"].class_id.tolist()
    ), "Expected class_id to be altered"
    assert result[0]["predictions_with_replaced_classes"]["class_name"].tolist() == [
        "116.Parson_russell_terrier",
        "131.Wirehaired_pointing_griffon",
    ], "Expected classes to be changed"
    assert result[0]["original_predictions"]["class_name"].tolist() == [
        "dog",
        "dog",
    ], "Expected classes not to be changed"
    assert result[1]["predictions_with_replaced_classes"]["class_name"].tolist() == [
        "116.Parson_russell_terrier",
        "131.Wirehaired_pointing_griffon",
    ], "Expected classes to be changed"
    assert result[1]["original_predictions"]["class_name"].tolist() == [
        "dog",
        "dog",
    ], "Expected classes not to be changed"
    assert (
        result[0]["predictions_with_replaced_classes"].xyxy
        is not result[0]["original_predictions"].xyxy
    ), "Expected copy of data to be created by step"
    assert (
        result[1]["predictions_with_replaced_classes"].xyxy
        is not result[1]["original_predictions"].xyxy
    ), "Expected copy of data to be created by step"
    assert np.allclose(
        result[0]["predictions_with_replaced_classes"].xyxy,
        result[0]["original_predictions"].xyxy,
    ), "Expected values of other fields in detections to be untouched"
    assert np.allclose(
        result[1]["predictions_with_replaced_classes"].xyxy,
        result[1]["original_predictions"].xyxy,
    ), "Expected values of other fields in detections to be untouched"
