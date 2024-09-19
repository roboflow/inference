from copy import deepcopy

import pytest

from inference.core.entities.requests.workflows import DescribeInterfaceRequest
from inference.core.interfaces.http.handlers.workflows import (
    handle_describe_workflows_interface,
)
from inference.core.workflows.errors import WorkflowDefinitionError
from inference.core.workflows.execution_engine.v1.introspection.outputs_discovery import (
    describe_workflow_outputs,
)

VALID_WORKFLOW_DEFINITION = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
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
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "detections",
            "selector": "$steps.general_detection.predictions",
        },
        {
            "type": "JsonField",
            "name": "crops",
            "selector": "$steps.cropping.crops",
        },
        {
            "type": "JsonField",
            "name": "classification",
            "selector": "$steps.breds_classification.*",
        },
    ],
}


def test_handle_describe_workflow_outputs_when_valid_specification_provided() -> None:
    # when
    result = describe_workflow_outputs(definition=VALID_WORKFLOW_DEFINITION)

    # then
    assert result == {
        "detections": ["object_detection_prediction"],
        "crops": ["image"],
        "classification": {
            "inference_id": ["string"],
            "predictions": ["classification_prediction"],
        },
    }


def test_handle_describe_workflow_outputs_when_specification_without_steps_provided() -> (
    None
):
    # given
    definition = deepcopy(VALID_WORKFLOW_DEFINITION)
    del definition["steps"]

    # when
    with pytest.raises(WorkflowDefinitionError):
        _ = describe_workflow_outputs(definition=definition)


def test_handle_describe_workflow_outputs_when_specification_without_outputs_provided() -> (
    None
):
    # given
    definition = deepcopy(VALID_WORKFLOW_DEFINITION)
    del definition["outputs"]

    # when
    with pytest.raises(WorkflowDefinitionError):
        _ = describe_workflow_outputs(definition=definition)


def test_handle_describe_workflow_outputs_when_output_refers_non_existing_step() -> (
    None
):
    # given
    definition = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
        ],
        "steps": [
            {
                "type": "ObjectDetectionModel",
                "name": "general_detection",
                "image": "$inputs.image",
                "model_id": "yolov8n-640",
                "class_filter": ["dog"],
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "detections",
                "selector": "$steps.non_existing.predictions",
            },
        ],
    }

    # when
    with pytest.raises(WorkflowDefinitionError):
        _ = describe_workflow_outputs(definition=definition)


def test_handle_describe_workflow_outputs_when_invalid_step_oncountered() -> None:
    # given
    definition = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
        ],
        "steps": [
            {
                "type": "Invalid",
                "name": "general_detection",
                "image": "$inputs.image",
                "model_id": "yolov8n-640",
                "class_filter": ["dog"],
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "detections",
                "selector": "$steps.general_detection.predictions",
            },
        ],
    }

    # when
    with pytest.raises(WorkflowDefinitionError):
        _ = describe_workflow_outputs(definition=definition)
