import pytest
from pydantic import ValidationError

from inference.enterprise.workflows.core_steps.common.operators import Operator
from inference.enterprise.workflows.core_steps.transformations.detection_filter import (
    BlockManifest,
    CompoundDetectionFilterDefinition,
    DetectionFilterBlock,
    DetectionFilterDefinition,
)


def test_manifest_parsing_when_valid_data_provided() -> None:
    # given
    data = {
        "type": "DetectionFilter",
        "name": "some",
        "predictions": "$steps.some.predictions",
        "filter_definition": {
            "type": "DetectionFilterDefinition",
            "field_name": "confidence",
            "operator": ">",
            "reference_value": 0.3,
        },
        "image_metadata": "$steps.some.image",
        "prediction_type": "$steps.some.prediction_type",
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result == BlockManifest(
        type="DetectionFilter",
        name="some",
        predictions="$steps.some.predictions",
        image_metadata="$steps.some.image",
        prediction_type="$steps.some.prediction_type",
        filter_definition=DetectionFilterDefinition(
            type="DetectionFilterDefinition",
            field_name="confidence",
            operator=Operator.GREATER_THAN,
            reference_value=0.3,
        ),
    )


def test_manifest_parsing_when_invalid_type_provided() -> None:
    # given
    data = {
        "type": "invalid",
        "name": "some",
        "predictions": "$steps.some.predictions",
        "filter_definition": {
            "type": "DetectionFilterDefinition",
            "field_name": "confidence",
            "operator": ">",
            "reference_value": 0.3,
        },
        "image_metadata": "$steps.some.image",
        "prediction_type": "$steps.some.prediction_type",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_manifest_parsing_when_invalid_predictions_provided() -> None:
    # given
    data = {
        "type": "DetectionFilter",
        "name": "some",
        "predictions": "invalid",
        "filter_definition": {
            "type": "DetectionFilterDefinition",
            "field_name": "confidence",
            "operator": ">",
            "reference_value": 0.3,
        },
        "image_metadata": "$steps.some.image",
        "prediction_type": "$steps.some.prediction_type",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_manifest_parsing_when_invalid_filter_definition_provided() -> None:
    # given
    data = {
        "type": "DetectionFilter",
        "name": "some",
        "predictions": "$steps.some.predictions",
        "filter_definition": {
            "type": "invalid",
            "field_name": "confidence",
            "operator": ">",
            "reference_value": 0.3,
        },
        "image_metadata": "$steps.some.image",
        "prediction_type": "$steps.some.prediction_type",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_manifest_parsing_when_invalid_image_metadata_provided() -> None:
    # given
    data = {
        "type": "DetectionFilter",
        "name": "some",
        "predictions": "$steps.some.predictions",
        "filter_definition": {
            "type": "DetectionFilterDefinition",
            "field_name": "confidence",
            "operator": ">",
            "reference_value": 0.3,
        },
        "image_metadata": "invalid",
        "prediction_type": "$steps.some.prediction_type",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_manifest_parsing_when_predictions_type_provided() -> None:
    # given
    data = {
        "type": "DetectionFilter",
        "name": "some",
        "predictions": "$steps.some.predictions",
        "filter_definition": {
            "type": "DetectionFilterDefinition",
            "field_name": "confidence",
            "operator": ">",
            "reference_value": 0.3,
        },
        "image_metadata": "$steps.some.image",
        "prediction_type": "invalid",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


@pytest.mark.asyncio
async def test_run_detection_filter_step_when_batch_detections_given() -> None:
    # given
    filter_definition = CompoundDetectionFilterDefinition.model_validate(
        {
            "type": "CompoundDetectionFilterDefinition",
            "left": {
                "type": "DetectionFilterDefinition",
                "field_name": "class_name",
                "operator": "==",
                "reference_value": "car",
            },
            "operator": "and",
            "right": {
                "type": "DetectionFilterDefinition",
                "field_name": "confidence",
                "operator": ">=",
                "reference_value": 0.5,
            },
        }
    )
    predictions = [
        [
            {
                "x": 10,
                "y": 10,
                "width": 20,
                "height": 20,
                "parent_id": "p1",
                "detection_id": "one",
                "class_name": "car",
                "confidence": 0.2,
            },
            {
                "x": 10,
                "y": 10,
                "width": 20,
                "height": 20,
                "parent_id": "p2",
                "detection_id": "two",
                "class_name": "car",
                "confidence": 0.5,
            },
        ],
        [
            {
                "x": 10,
                "y": 10,
                "width": 20,
                "height": 20,
                "parent_id": "p3",
                "detection_id": "three",
                "class_name": "dog",
                "confidence": 0.2,
            },
            {
                "x": 10,
                "y": 10,
                "width": 20,
                "height": 20,
                "parent_id": "p4",
                "detection_id": "four",
                "class_name": "car",
                "confidence": 0.5,
            },
        ],
    ]
    block = DetectionFilterBlock()

    # when
    result = await block.run_locally(
        predictions=predictions,
        filter_definition=filter_definition,
        image_metadata=[{"height": 100, "width": 100}] * 2,
        prediction_type=["object-detection"] * 2,
    )

    # then
    assert (
        result[0]["prediction_type"] == "object-detection"
    ), "Prediction type must be preserved"
    assert (
        result[1]["prediction_type"] == "object-detection"
    ), "Prediction type must be preserved"
    assert result[0]["predictions"] == [
        {
            "x": 10,
            "y": 10,
            "width": 20,
            "height": 20,
            "parent_id": "p2",
            "detection_id": "two",
            "class_name": "car",
            "confidence": 0.5,
        }
    ], "Only second prediction in each batch should survive"
    assert result[1]["predictions"] == [
        {
            "x": 10,
            "y": 10,
            "width": 20,
            "height": 20,
            "parent_id": "p4",
            "detection_id": "four",
            "class_name": "car",
            "confidence": 0.5,
        }
    ], "Only second prediction in each batch should survive"
    assert result[0]["parent_id"] == [
        "p2"
    ], "Only second prediction in each batch should mark parent_id"
    assert result[1]["parent_id"] == [
        "p4"
    ], "Only second prediction in each batch should mark parent_id"
    assert result[0]["image"] == {
        "height": 100,
        "width": 100,
    }, "image metadata must be copied from input"
    assert result[1]["image"] == {
        "height": 100,
        "width": 100,
    }, "image metadata must be copied from input"
