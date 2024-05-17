import numpy as np
import pytest
from pydantic import ValidationError
import supervision as sv

from inference.core.workflows.core_steps.common.operators import Operator
from inference.core.workflows.core_steps.transformations.detection_filter import (
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
    batch_1_detections = sv.Detections(
        xyxy=np.array([[0, 0, 20, 20], [0, 0, 20, 20]], dtype=np.float64),
        class_id=np.array([0, 0]),
        confidence=np.array([0.2, 0.5], dtype=np.float64),
        data={
            "parent_id": np.array(["p1", "p2"]),
            "detection_id": np.array(["one", "two"]),
            "class_name": np.array(["car", "car"])
        }
    )
    batch_2_detections = sv.Detections(
        xyxy=np.array([[0, 0, 20, 20], [0, 0, 20, 20]], dtype=np.float64),
        class_id=np.array([1, 0]),
        confidence=np.array([0.2, 0.5], dtype=np.float64),
        data={
            "parent_id": np.array(["p3", "p4"]),
            "detection_id": np.array(["three", "four"]),
            "class_name": np.array(["dog", "car"])
        }
)
    block = DetectionFilterBlock()

    # when
    result = await block.run_locally(
        predictions=[batch_1_detections, batch_2_detections],
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
    assert result[0]["predictions"] == batch_1_detections[1], "Only second prediction in each batch should survive"
    assert result[1]["predictions"] == batch_2_detections[1], "Only second prediction in each batch should survive"
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
