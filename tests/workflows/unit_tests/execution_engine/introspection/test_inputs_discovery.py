import pytest

from inference.core.workflows.errors import WorkflowDefinitionError
from inference.core.workflows.execution_engine.v1.introspection.inputs_discovery import (
    describe_workflow_inputs,
)


def test_describe_workflow_inputs_when_simple_valid_workflow_provided() -> None:
    # given
    definition = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "model_id"},
        ],
        "steps": [
            {
                "type": "ObjectDetectionModel",
                "name": "general_detection",
                "image": "$inputs.image",
                "model_id": "$inputs.model_id",
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
    result = describe_workflow_inputs(definition=definition)

    # then
    assert result == {"image": ["image"], "model_id": ["roboflow_model_id"]}


def test_describe_workflow_inputs_when_declared_input_kind_does_not_match_actual() -> (
    None
):
    # given
    definition = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "model_id", "kind": ["float"]},
        ],
        "steps": [
            {
                "type": "ObjectDetectionModel",
                "name": "general_detection",
                "image": "$inputs.image",
                "model_id": "$inputs.model_id",
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
        _ = describe_workflow_inputs(definition=definition)


def test_describe_workflow_inputs_when_declared_input_kind_does_matches_actual() -> (
    None
):
    # given
    definition = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {
                "type": "WorkflowParameter",
                "name": "model_id",
                "kind": ["roboflow_model_id"],
            },
        ],
        "steps": [
            {
                "type": "ObjectDetectionModel",
                "name": "general_detection",
                "image": "$inputs.image",
                "model_id": "$inputs.model_id",
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
    result = describe_workflow_inputs(definition=definition)

    # then
    assert result == {"image": ["image"], "model_id": ["roboflow_model_id"]}


def test_describe_workflow_inputs_when_inputs_with_syntax_error_provided() -> None:
    # given
    definition = {
        "version": "1.0",
        "inputs": [
            {"type_": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "model_id"},
        ],
        "steps": [
            {
                "type": "ObjectDetectionModel",
                "name": "general_detection",
                "image": "$inputs.image",
                "model_id": "$inputs.model_id",
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
        _ = describe_workflow_inputs(definition=definition)


def test_describe_workflow_inputs_when_steps_with_syntax_error_provided() -> None:
    # given
    definition = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "model_id"},
        ],
        "steps": [
            {
                "type_": "ObjectDetectionModel",
                "name": "general_detection",
                "image": "$inputs.image",
                "model_id": "$inputs.model_id",
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
        _ = describe_workflow_inputs(definition=definition)


def test_describe_workflow_inputs_when_unknown_step_provided() -> None:
    # given
    definition = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "model_id"},
        ],
        "steps": [
            {
                "type": "Invalid",
                "name": "general_detection",
                "image": "$inputs.image",
                "model_id": "$inputs.model_id",
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
        _ = describe_workflow_inputs(definition=definition)


def test_describe_workflow_inputs_step_with_invalid_configuration_provided() -> None:
    # given
    definition = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "model_id"},
        ],
        "steps": [
            {
                "type": "ObjectDetectionModel",
                "name": "general_detection",
                "image": "$inputs.image",
                "model_id": "$inputs.model_id",
                "class_filter": "INVALID",
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
        _ = describe_workflow_inputs(definition=definition)


def test_describe_workflow_inputs_when_workflow_without_inputs_provided() -> None:
    # given
    definition = {
        "version": "1.0",
        "steps": [
            {
                "type": "ObjectDetectionModel",
                "name": "general_detection",
                "image": "$inputs.image",
                "model_id": "$inputs.model_id",
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
        _ = describe_workflow_inputs(definition=definition)


def test_describe_workflow_inputs_when_workflow_without_steps_provided() -> None:
    # given
    definition = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "model_id"},
        ],
        "steps": [],
        "outputs": [
            {
                "type": "JsonField",
                "name": "detections",
                "selector": "$steps.general_detection.predictions",
            },
        ],
    }

    # when
    result = describe_workflow_inputs(definition=definition)

    # then
    assert result == {
        "image": ["*"],
        "model_id": ["*"],
    }


def test_describe_workflow_inputs_when_inputs_are_shared_between_steps_with_kinds_matching() -> (
    None
):
    # given
    definition = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image_1"},
            {"type": "WorkflowImage", "name": "image_2"},
            {"type": "WorkflowParameter", "name": "model_id"},
        ],
        "steps": [
            {
                "type": "ObjectDetectionModel",
                "name": "general_detection_1",
                "image": "$inputs.image_1",
                "model_id": "$inputs.model_id",
                "class_filter": ["dog"],
            },
            {
                "type": "ObjectDetectionModel",
                "name": "general_detection_2",
                "image": "$inputs.image_2",
                "model_id": "$inputs.model_id",
                "class_filter": ["dog"],
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "detections_1",
                "selector": "$steps.general_detection_1.predictions",
            },
            {
                "type": "JsonField",
                "name": "detections_2",
                "selector": "$steps.general_detection_2.predictions",
            },
        ],
    }

    # when
    result = describe_workflow_inputs(definition=definition)

    # then
    assert result == {
        "image_1": ["image"],
        "image_2": ["image"],
        "model_id": ["roboflow_model_id"],
    }


def test_describe_workflow_inputs_when_some_inputs_are_not_used() -> None:
    # given
    definition = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image_1"},
            {"type": "WorkflowImage", "name": "image_2"},
            {"type": "WorkflowParameter", "name": "model_id"},
            {"type": "WorkflowParameter", "name": "confidence"},
        ],
        "steps": [
            {
                "type": "ObjectDetectionModel",
                "name": "general_detection_1",
                "image": "$inputs.image_1",
                "model_id": "$inputs.model_id",
                "class_filter": ["dog"],
            },
            {
                "type": "ObjectDetectionModel",
                "name": "general_detection_2",
                "image": "$inputs.image_2",
                "model_id": "$inputs.model_id",
                "class_filter": ["dog"],
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "detections_1",
                "selector": "$steps.general_detection_1.predictions",
            },
            {
                "type": "JsonField",
                "name": "detections_2",
                "selector": "$steps.general_detection_2.predictions",
            },
        ],
    }

    # when
    result = describe_workflow_inputs(definition=definition)

    # then
    assert result == {
        "image_1": ["image"],
        "image_2": ["image"],
        "model_id": ["roboflow_model_id"],
        "confidence": ["*"],
    }


def test_describe_workflow_inputs_when_inputs_are_shared_between_steps_with_kinds_not_matching_by_reference_type() -> (
    None
):
    # given
    definition = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image_1"},
            {"type": "WorkflowImage", "name": "image_2"},
            {"type": "WorkflowParameter", "name": "model_id"},
        ],
        "steps": [
            {
                "type": "ObjectDetectionModel",
                "name": "general_detection_1",
                "image": "$inputs.image_1",
                "model_id": "$inputs.model_id",
                "class_filter": ["dog"],
            },
            {
                "type": "ObjectDetectionModel",
                "name": "general_detection_2",
                "image": "$inputs.image_2",
                "model_id": "$inputs.image_2",
                "class_filter": ["dog"],
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "detections_1",
                "selector": "$steps.general_detection_1.predictions",
            },
            {
                "type": "JsonField",
                "name": "detections_2",
                "selector": "$steps.general_detection_2.predictions",
            },
        ],
    }

    # when
    with pytest.raises(WorkflowDefinitionError):
        _ = describe_workflow_inputs(definition=definition)


def test_describe_workflow_inputs_when_inputs_are_shared_between_steps_with_kinds_not_matching_by_kind() -> (
    None
):
    # given
    definition = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image_1"},
            {"type": "WorkflowParameter", "name": "model_id"},
        ],
        "steps": [
            {
                "type": "ObjectDetectionModel",
                "name": "general_detection_1",
                "image": "$inputs.image_1",
                "model_id": "$inputs.model_id",
                "confidence": "$inputs.model_id",
                "class_filter": ["dog"],
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "detections_1",
                "selector": "$steps.general_detection_1.predictions",
            },
        ],
    }

    # when
    with pytest.raises(WorkflowDefinitionError):
        _ = describe_workflow_inputs(definition=definition)
