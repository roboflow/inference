from typing import Any

import pytest
from pydantic import ValidationError

from inference.enterprise.workflows.core_steps.fusion.detections_consensus import (
    AggregationMode,
    BlockManifest,
)


def test_detections_consensus_validation_when_valid_specification_given() -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "image_metadata": "$steps.detection.image",
        "required_votes": 3,
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result == BlockManifest(
        type="DetectionsConsensus",
        name="some",
        predictions=["$steps.detection.predictions", "$steps.detection_2.predictions"],
        image_metadata="$steps.detection.image",
        required_votes=3,
        class_aware=True,
        iou_threshold=0.3,
        confidence=0.0,
        classes_to_consider=None,
        required_objects=None,
        presence_confidence_aggregation=AggregationMode.MAX,
        detections_merge_confidence_aggregation=AggregationMode.AVERAGE,
        detections_merge_coordinates_aggregation=AggregationMode.AVERAGE,
    )


@pytest.mark.parametrize("value", [3, "3", True, 3.0, [], set(), {}, None])
def test_detections_consensus_validation_when_predictions_of_invalid_type_given(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": value,
        "image_metadata": "$steps.detection.image",
        "required_votes": 3,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


@pytest.mark.parametrize("value", [None, 0, -1, "some", []])
def test_detections_consensus_validation_when_required_votes_of_invalid_type_given(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "image_metadata": "$steps.detection.image",
        "required_votes": value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


@pytest.mark.parametrize("value", [3, "$inputs.some"])
def test_detections_consensus_validation_when_required_votes_of_valid_type_given(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "image_metadata": "$steps.detection.image",
        "required_votes": value,
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result.required_votes == value


@pytest.mark.parametrize("value", [None, "some"])
def test_detections_consensus_validation_when_class_aware_of_invalid_type_given(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "image_metadata": "$steps.detection.image",
        "required_votes": 3,
        "class_aware": value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


@pytest.mark.parametrize("value", [True, False])
def test_detections_consensus_validation_when_class_aware_of_valid_type_given(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "image_metadata": "$steps.detection.image",
        "required_votes": 3,
        "class_aware": value,
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result.class_aware == value


@pytest.mark.parametrize(
    "field,value",
    [
        ("iou_threshold", None),
        ("iou_threshold", -1),
        ("iou_threshold", 2.0),
        ("iou_threshold", "some"),
        ("confidence", None),
        ("confidence", -1),
        ("confidence", 2.0),
        ("confidence", "some"),
    ],
)
def test_detections_consensus_validation_when_range_field_of_invalid_type_given(
    field: str,
    value: Any,
) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "image_metadata": "$steps.detection.image",
        "required_votes": 3,
        field: value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


@pytest.mark.parametrize(
    "field,value",
    [
        ("iou_threshold", 0.0),
        ("iou_threshold", 1.0),
        ("iou_threshold", 0.5),
        ("iou_threshold", "$inputs.some"),
        ("confidence", 0.0),
        ("confidence", 1.0),
        ("confidence", 0.5),
        ("confidence", "$inputs.some"),
    ],
)
def test_detections_consensus_validation_when_range_field_of_valid_type_given(
    field: str,
    value: Any,
) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "image_metadata": "$steps.detection.image",
        "required_votes": 3,
        field: value,
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert getattr(result, field) == value


@pytest.mark.parametrize("value", ["some", 1, 2.0, True, {}])
def test_detections_consensus_validation_when_classes_to_consider_of_invalid_type_given(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "image_metadata": "$steps.detection.image",
        "required_votes": 3,
        "classes_to_consider": value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


@pytest.mark.parametrize("value", ["$inputs.some", [], ["1", "2", "3"]])
def test_detections_consensus_validation_when_classes_to_consider_of_valid_type_given(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "image_metadata": "$steps.detection.image",
        "required_votes": 3,
        "classes_to_consider": value,
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result.classes_to_consider == value


@pytest.mark.parametrize(
    "value", ["some", -1, 0, {"some": None}, {"some": 1, "other": -1}]
)
def test_detections_consensus_validation_when_required_objects_of_invalid_type_given(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "image_metadata": "$steps.detection.image",
        "required_votes": 3,
        "required_objects": value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


@pytest.mark.parametrize(
    "value", [None, "$inputs.some", 1, 10, {"some": 1, "other": 10}]
)
def test_detections_consensus_validation_when_required_objects_of_valid_type_given(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "image_metadata": "$steps.detection.image",
        "required_votes": 3,
        "required_objects": value,
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result.required_objects == value
