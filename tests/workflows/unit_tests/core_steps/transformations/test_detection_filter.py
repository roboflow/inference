import pytest
from pydantic import ValidationError

from inference.enterprise.workflows.core_steps.common.operators import Operator
from inference.enterprise.workflows.core_steps.transformations.detection_filter import (
    BlockManifest,
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
