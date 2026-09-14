from copy import deepcopy
from typing import Any, List, Literal

import pytest
from pydantic import AliasChoices, Field

from inference.core.workflows.errors import WorkflowDefinitionError
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    IMAGE_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    Selector,
)
from inference.core.workflows.execution_engine.introspection.schema_parser import (
    parse_block_manifest,
)
from inference.core.workflows.execution_engine.v1.introspection.inputs_discovery import (
    describe_workflow_inputs,
    retrieve_input_selectors_details,
    search_input_selectors_in_steps,
)
from inference.core.workflows.prototypes.block import WorkflowBlockManifest


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


DETECTION_WORKFLOW_DEFINITION = {
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


@pytest.mark.parametrize(
    "image_kind, model_id_kind",
    [
        (IMAGE_KIND.model_dump(), ROBOFLOW_MODEL_ID_KIND.model_dump()),
        ({"name": "image"}, {"name": "roboflow_model_id"}),
        (IMAGE_KIND, ROBOFLOW_MODEL_ID_KIND),
        ("image", ROBOFLOW_MODEL_ID_KIND.model_dump()),
    ],
)
def test_describe_workflow_inputs_when_declared_input_kinds_are_kind_definitions(
    image_kind: Any,
    model_id_kind: Any,
) -> None:
    # given
    definition = deepcopy(DETECTION_WORKFLOW_DEFINITION)
    definition["inputs"][0]["kind"] = [image_kind]
    definition["inputs"][1]["kind"] = [model_id_kind]

    # when
    result = describe_workflow_inputs(definition=definition)

    # then
    assert result == {"image": ["image"], "model_id": ["roboflow_model_id"]}


def test_describe_workflow_inputs_when_declared_input_kind_definition_does_not_match_actual() -> (
    None
):
    # given
    definition = deepcopy(DETECTION_WORKFLOW_DEFINITION)
    definition["inputs"][1]["kind"] = [FLOAT_KIND.model_dump()]

    # when
    with pytest.raises(WorkflowDefinitionError):
        _ = describe_workflow_inputs(definition=definition)


@pytest.mark.parametrize(
    "declared_kind",
    [
        "roboflow_model_id",
        {"name": "roboflow_model_id"},
        [{"description": "Kind definition without name"}],
        [{"name": ["roboflow_model_id"]}],
        [["roboflow_model_id"]],
        [5],
        [None],
    ],
)
def test_describe_workflow_inputs_when_declared_input_kind_is_malformed(
    declared_kind: Any,
) -> None:
    # given
    definition = deepcopy(DETECTION_WORKFLOW_DEFINITION)
    definition["inputs"][1]["kind"] = declared_kind

    # when
    with pytest.raises(WorkflowDefinitionError) as error:
        _ = describe_workflow_inputs(definition=definition)

    # then
    assert error.value.context == "describing_workflow_inputs"
    assert "model_id" in error.value.public_message


@pytest.mark.parametrize(
    "step_type", ["roboflow_core/roboflow_dataset_upload@v1", "RoboflowDatasetUpload"]
)
@pytest.mark.parametrize("image_property_name", ["image", "images"])
def test_describe_workflow_inputs_when_step_property_is_provided_with_validation_alias(
    step_type: str,
    image_property_name: str,
) -> None:
    # given
    definition = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "project"},
        ],
        "steps": [
            {
                "type": step_type,
                "name": "data_collection",
                image_property_name: "$inputs.image",
                "target_project": "$inputs.project",
                "usage_quota_name": "my_quota",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "message",
                "selector": "$steps.data_collection.message",
            },
        ],
    }

    # when
    result = describe_workflow_inputs(definition=definition)

    # then
    assert result == {"image": ["image"], "project": ["roboflow_project"]}


class ManifestWithSchemaPropertyNotMatchingField(WorkflowBlockManifest):
    type: Literal["ManifestWithSchemaPropertyNotMatchingField"]
    # JSON schema names the property after the first alias choice - `image`
    images: Selector(kind=[IMAGE_KIND]) = Field(
        validation_alias=AliasChoices("image", "images"),
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return []


def test_search_input_selectors_in_steps_when_manifest_does_not_expose_property_declared_in_schema() -> (
    None
):
    # given
    block_type = "ManifestWithSchemaPropertyNotMatchingField"
    input_selectors_details = retrieve_input_selectors_details(
        inputs=[{"type": "WorkflowImage", "name": "image"}]
    )

    # when
    with pytest.raises(WorkflowDefinitionError) as error:
        _ = search_input_selectors_in_steps(
            steps=[{"type": block_type, "name": "some", "image": "$inputs.image"}],
            input_selectors_details=input_selectors_details,
            block_type_to_manifest={
                block_type: ManifestWithSchemaPropertyNotMatchingField
            },
            block_type_to_metadata={
                block_type: parse_block_manifest(
                    manifest_type=ManifestWithSchemaPropertyNotMatchingField
                )
            },
        )

    # then
    assert error.value.context == "describing_workflow_inputs"
    assert "`image`" in error.value.public_message


@pytest.mark.parametrize(
    "property_name, property_value",
    [
        ("inputs", None),
        ("inputs", {"type": "WorkflowImage", "name": "image"}),
        ("inputs", ["image"]),
        ("inputs", [{"type": "WorkflowImage", "name": ["image"]}]),
        ("inputs", [{"type": ["WorkflowImage"], "name": "image"}]),
        ("steps", None),
        ("steps", {"type": "ObjectDetectionModel", "name": "general_detection"}),
        ("steps", ["general_detection"]),
        ("steps", [{"type": ["ObjectDetectionModel"], "name": "general_detection"}]),
    ],
)
def test_describe_workflow_inputs_when_definition_structure_is_malformed(
    property_name: str,
    property_value: Any,
) -> None:
    # given
    definition = deepcopy(DETECTION_WORKFLOW_DEFINITION)
    definition[property_name] = property_value

    # when
    with pytest.raises(WorkflowDefinitionError):
        _ = describe_workflow_inputs(definition=definition)
