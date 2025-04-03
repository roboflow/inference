import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

ACTIVE_LEARNING_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "data_percentage", "default_value": 50.0},
        {
            "type": "WorkflowParameter",
            "name": "persist_predictions",
            "default_value": True,
        },
        {"type": "WorkflowParameter", "name": "tag", "default_value": "my_tag"},
        {"type": "WorkflowParameter", "name": "disable_sink", "default_value": False},
        {"type": "WorkflowParameter", "name": "fire_and_forget", "default_value": True},
        {
            "type": "WorkflowParameter",
            "name": "labeling_batch_prefix",
            "default_value": "some",
        },
    ],
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
            "type": "roboflow_core/roboflow_dataset_upload@v2",
            "name": "data_collection",
            "images": "$steps.cropping.crops",
            "predictions": "$steps.breds_classification.predictions",
            "target_project": "my_project",
            "usage_quota_name": "my_quota",
            "data_percentage": "$inputs.data_percentage",
            "persist_predictions": "$inputs.persist_predictions",
            "minutely_usage_limit": 10,
            "hourly_usage_limit": 100,
            "daily_usage_limit": 1000,
            "max_image_size": (100, 200),
            "compression_level": 85,
            "registration_tags": ["a", "b", "$inputs.tag"],
            "disable_sink": "$inputs.disable_sink",
            "fire_and_forget": "$inputs.fire_and_forget",
            "labeling_batch_prefix": "$inputs.labeling_batch_prefix",
            "labeling_batches_recreation_frequency": "never",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.breds_classification.predictions",
        },
        {
            "type": "JsonField",
            "name": "registration_message",
            "selector": "$steps.data_collection.message",
        },
    ],
}


@add_to_workflows_gallery(
    category="Workflows enhanced by Roboflow Platform",
    use_case_title="Data Collection for Active Learning",
    use_case_description="""
This example showcases how to stack models on top of each other - in this particular
case, we detect objects using object detection models, requesting only "dogs" bounding boxes
in the output of prediction. Additionally, we register cropped images in Roboflow dataset.

Thanks to this setup, we are able to collect production data and continuously train better models over time.
""",
    workflow_definition=ACTIVE_LEARNING_WORKFLOW,
    workflow_name_in_app="data-collection-active-learning",
)
def test_detection_plus_classification_workflow_when_minimal_valid_input_provided(
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
        workflow_definition=ACTIVE_LEARNING_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
            "data_percentage": 0.0,
        }
    )

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "predictions",
        "registration_message",
    }, "Expected all declared outputs to be delivered"
    assert (
        len(result[0]["predictions"]) == 2
    ), "Expected 2 dogs crops on input image, hence 2 nested classification results"
    assert [result[0]["predictions"][0]["top"], result[0]["predictions"][1]["top"]] == [
        "116.Parson_russell_terrier",
        "131.Wirehaired_pointing_griffon",
    ], "Expected predictions to be as measured in reference run"
    assert (
        result[0]["registration_message"]
        == ["Registration skipped due to sampling settings"] * 2
    ), "Expected data not registered due to sampling"


def test_detection_plus_classification_workflow_when_nothing_to_be_registered(
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
        workflow_definition=ACTIVE_LEARNING_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": crowd_image,
            "data_percentage": 0.0,
        }
    )

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "predictions",
        "registration_message",
    }, "Expected all declared outputs to be delivered"
    assert (
        len(result[0]["predictions"]) == 0
    ), "Expected 0 dogs crops on input image, hence 0 nested classification results"
    assert (
        len(result[0]["registration_message"]) == 0
    ), "Expected 0 dogs crops on input image, hence 0 nested statuses of registration"
