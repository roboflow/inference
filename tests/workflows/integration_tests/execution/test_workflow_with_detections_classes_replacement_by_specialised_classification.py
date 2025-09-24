import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

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
            "confidence": 0.09,
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


@add_to_workflows_gallery(
    category="Workflows with multiple models",
    use_case_title="Workflow with classifier providing detailed labels for detected objects",
    use_case_description="""
This example illustrates how helpful Workflows could be when you have generic object detection model 
(capable of detecting common classes - like dogs) and specific classifier (capable of providing granular 
predictions for narrow high-level classes of objects - like dogs breed classifier). Having list
of classifier predictions for each detected dog is not handy way of dealing with output - 
as you kind of loose the information about location of specific dog. To avoid this problem, you
may want to replace class labels of original bounding boxes (from the first model localising dogs) with
classes predicted by classifier.

In this example, we use Detections Classes Replacement block which is also interesting from the 
perspective of difference of its inputs dimensionality levels. `object_detection_predictions` input
has level 1 (there is one prediction with bboxes for each input image) and `classification_predictions`
has level 2 (there are bunch of classification results for each input image). The block combines that
two inputs and produces result at dimensionality level 1 - exactly the same as predictions from 
object detection model.
    """,
    workflow_definition=DETECTION_CLASSES_REPLACEMENT_WORKFLOW,
    workflow_name_in_app="detections-classes-replacement",
)
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
