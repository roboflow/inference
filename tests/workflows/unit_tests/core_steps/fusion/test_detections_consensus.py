from typing import Any
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import ValidationError
import supervision as sv

from inference.core.workflows.core_steps.fusion import detections_consensus
from inference.core.workflows.core_steps.fusion.detections_consensus import (
    AggregationMode,
    BlockManifest,
    aggregate_field_values,
    agree_on_consensus_for_all_detections_sources,
    calculate_iou,
    check_objects_presence_in_consensus_detections,
    does_not_detect_objects_in_any_source,
    enumerate_detections,
    filter_predictions,
    get_average_bounding_box,
    get_class_of_least_confident_detection,
    get_class_of_most_confident_detection,
    get_consensus_for_single_detection,
    get_detections_from_different_sources_with_max_overlap,
    get_largest_bounding_box,
    get_majority_class,
    get_parent_id_of_detections_from_sources,
    get_smallest_bounding_box,
    merge_detections,
)


@pytest.mark.parametrize(
    "predictions_batch_alias", ["predictions", "predictions_batches"]
)
def test_detections_consensus_validation_when_valid_specification_given(
    predictions_batch_alias: str,
) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        predictions_batch_alias: [
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
        predictions_batches=[
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
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
        "predictions_batches": value,
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
        "predictions_batches": [
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
        "predictions_batches": [
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
        "predictions_batches": [
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
        "predictions_batches": [
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
        "predictions_batches": [
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
        "predictions_batches": [
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
        "predictions_batches": [
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
        "predictions_batches": [
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
        "predictions_batches": [
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
        "predictions_batches": [
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


def test_aggregate_field_values_when_max_mode_is_chosen() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2]], dtype=np.float64),
        data={
            "a": np.array([0.3, 0.4, 0.7])
        }
    )

    # when
    result = aggregate_field_values(
        detections=detections,
        field="a",
        aggregation_mode=AggregationMode.MAX,
    )

    # then
    assert (result - 0.7) < 1e-5


def test_aggregate_field_values_when_min_mode_is_chosen() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2]], dtype=np.float64),
        data={
            "a": np.array([0.3, 0.4, 0.7])
        }
    )

    # when
    result = aggregate_field_values(
        detections=detections,
        field="a",
        aggregation_mode=AggregationMode.MIN,
    )

    # then
    assert (result - 0.3) < 1e-5


def test_aggregate_field_values_when_average_mode_is_chosen() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2]], dtype=np.float64),
        data={
            "a": np.array([0.3, 0.4, 0.5])
        }
    )

    # when
    result = aggregate_field_values(
        detections=detections,
        field="a",
        aggregation_mode=AggregationMode.AVERAGE,
    )

    # then
    assert (result - 0.4) < 1e-5


@pytest.mark.parametrize(
    "mode", [AggregationMode.MIN, AggregationMode.MAX, AggregationMode.AVERAGE]
)
def test_aggregate_field_values_when_empty_input_provided(
    mode: AggregationMode,
) -> None:
    with pytest.raises(ValueError):
        # when
        _ = aggregate_field_values(
            detections=sv.Detections(xyxy=np.array([[1, 1, 2, 2]])),
            field="a",
            aggregation_mode=mode,
        )


def test_get_largest_bounding_box_when_single_element_provided() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[80, 185, 120, 215]]),
        confidence=np.array([0.5]),
        class_id=np.array([1]),
        data={
            "detection_id": ["a"],
            "class_name": ["a"]
        }
    )

    # when
    result = get_largest_bounding_box(detections=detections)

    # then
    assert result == (80, 185, 120, 215)


def test_get_largest_bounding_box_when_multiple_elements_provided() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[80, 185, 120, 215], [85, 190, 135, 230]]),
        confidence=np.array([0.5, 0.5]),
        class_id=np.array([1, 1]),
        data={
            "detection_id": ["a", "a"],
            "class_name": ["a", "a"]
        }
    )

    # when
    result = get_largest_bounding_box(detections=detections)

    # then
    assert result == (85, 190, 135, 230)


def test_get_smallest_bounding_box_when_single_element_provided() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[80, 185, 120, 215]]),
        confidence=np.array([0.5]),
        class_id=np.array([1]),
        data={
            "detection_id": ["a"],
            "class_name": ["a"]
        }
    )

    # when
    result = get_smallest_bounding_box(detections=detections)

    # then
    assert result == (80, 185, 120, 215)


def test_get_smallest_bounding_box_when_multiple_elements_provided() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[80, 185, 120, 215], [85, 190, 135, 230]]),
        confidence=np.array([0.5, 0.5]),
        class_id=np.array([1, 1]),
        data={
            "detection_id": ["a", "a"],
            "class_name": ["a", "a"]
        }
    )

    # when
    result = get_smallest_bounding_box(detections=detections)

    # then
    assert result == (80, 185, 120, 215)


def test_get_average_bounding_box_when_single_element_provided() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[80, 185, 120, 215]]),
        confidence=np.array([0.5]),
        class_id=np.array([1]),
        data={
            "detection_id": ["a"],
            "class_name": ["a"]
        }
    )

    # when
    result = get_average_bounding_box(detections=detections)

    # then
    assert result == (80, 185, 120, 215)


def test_get_average_bounding_box_when_multiple_elements_provided() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[80, 185, 120, 215], [85, 190, 135, 230]]),
        confidence=np.array([0.5, 0.5]),
        class_id=np.array([1, 1]),
        data={
            "detection_id": ["a", "a"],
            "class_name": ["a", "a"]
        }
    )

    # when
    result = get_average_bounding_box(detections=detections)

    # then
    assert result == (82.5, 187.5, 127.5, 222.5)


def test_get_majority_class() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[80, 185, 120, 215], [85, 190, 135, 230], [1, 1, 2, 2]]),
        confidence=np.array([0.5, 0.5, 0.5]),
        class_id=np.array([0, 1, 0]),
        data={
            "detection_id": ["a", "b", "c"],
            "class_name": ["a", "b", "a"]
        }
    )

    # when
    result = get_majority_class(detections=detections)

    # then
    assert result == ("a", 0)


def test_get_class_of_most_confident_detection() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[80, 185, 120, 215], [85, 190, 135, 230], [1, 1, 2, 2]]),
        confidence=np.array([0.1, 0.3, 0.2]),
        class_id=np.array([0, 1, 0]),
        data={
            "detection_id": ["a", "b", "c"],
            "class_name": ["a", "b", "a"]
        }
    )

    # when
    result = get_class_of_most_confident_detection(detections=detections)

    # then
    assert result == ("b", 1)


def test_get_class_of_least_confident_detection() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[80, 185, 120, 215], [85, 190, 135, 230], [1, 1, 2, 2]]),
        confidence=np.array([0.1, 0.3, 0.2]),
        class_id=np.array([0, 1, 0]),
        data={
            "detection_id": ["a", "b", "c"],
            "class_name": ["a", "b", "a"]
        }
    )

    # when
    result = get_class_of_least_confident_detection(detections=detections)

    # then
    assert result == ("a", 0)


@mock.patch.object(detections_consensus, "uuid4")
def test_merge_detections(uuid4_mock: MagicMock) -> None:
    # given
    uuid4_mock.return_value = "some_uuid"
    detections = sv.Detections(
        xyxy=np.array([[80, 185, 120, 215], [85, 190, 135, 230]]),
        confidence=np.array([0.1, 0.3]),
        class_id=np.array([0, 1]),
        data={
            "detection_id": np.array(["a", "b"]),
            "class_name": np.array(["a", "b"]),
            "parent_id": np.array(["x", "x"])
        }
    )

    # when
    result = merge_detections(
        detections=detections,
        confidence_aggregation_mode=AggregationMode.AVERAGE,
        boxes_aggregation_mode=AggregationMode.MAX,
    )

    # then
    assert result == sv.Detections(
        xyxy=np.array([[85, 190, 135, 230]], dtype=np.float64),
        confidence=np.array([0.2], dtype=np.float64),
        class_id=np.array([0]),
        data={
            "detection_id": np.array(["some_uuid"]),
            "class_name": np.array(["a"]),
            "parent_id": np.array(["x"])
        }
    )


def test_calculate_iou_when_detections_are_zero_size() -> None:
    # given
    detections = sv.Detections.from_inference(
        {
            "image": {"height": 100, "width": 200},
            "predictions": [
                {"x": 100, "y": 200, "height": 0, "width": 1, "class_id": 1, "class": "a", "confidence": 0.5},
                {"x": 100, "y": 220, "height": 1, "width": 0, "class_id": 2, "class": "b", "confidence": 0.6},
            ],
        },
    )
    detection_a = detections[0]
    detection_b = detections[1]

    # when
    result = calculate_iou(
        detection_a=detection_a,
        detection_b=detection_b,
    )

    # then
    assert abs(result) < 1e-5


def test_calculate_iou_when_detections_do_not_overlap() -> None:
    # given
    detections = sv.Detections.from_inference(
        {
            "image": {"height": 100, "width": 200},
            "predictions": [
                {"x": 100, "y": 200, "height": 20, "width": 40, "class_id": 1, "class": "a", "confidence": 0.5},
                {"x": 100, "y": 220, "height": 20, "width": 40, "class_id": 2, "class": "b", "confidence": 0.6},
            ],
        },
    )
    detection_a = detections[0]
    detection_b = detections[1]

    # when
    result = calculate_iou(
        detection_a=detection_a,
        detection_b=detection_b,
    )

    # then
    assert abs(result) < 1e-5


def test_calculate_iou_when_detections_do_overlap_fully() -> None:
    # given
    detection_a = sv.Detections.from_inference(
        {
            "image": {"height": 100, "width": 200},
            "predictions": [
                {"x": 100, "y": 200, "height": 20, "width": 40, "class_id": 1, "class": "a", "confidence": 0.5},
            ],
        },
    )

    # when
    result = calculate_iou(
        detection_a=detection_a,
        detection_b=detection_a,
    )

    # then
    assert abs(result - 1.0) < 1e-5


def test_calculate_iou_when_detections_do_overlap_partially() -> None:
    # given
    detections = sv.Detections.from_inference(
        {
            "image": {"height": 100, "width": 200},
            "predictions": [
                {"x": 100, "y": 200, "height": 20, "width": 40, "class_id": 1, "class": "a", "confidence": 0.5},
                {"x": 120, "y": 210, "height": 20, "width": 40, "class_id": 2, "class": "b", "confidence": 0.6},
            ],
        },
    )
    detection_a = detections[0]
    detection_b = detections[1]

    # box A size = box B size = 800
    # intersection = (100, 200, 120, 210) -> size = 200
    # expected result = 200 / 1400 = 100 / 700 = 1 / 7

    # when
    result = calculate_iou(
        detection_a=detection_a,
        detection_b=detection_b,
    )

    # then
    assert abs(result - 1 / 7) < 1e-5


def test_enumerate_detections_when_no_predictions_given() -> None:
    # when
    result = list(enumerate_detections(detections_from_sources=[]))

    # then
    assert result == []


def test_enumerate_detections_when_source_with_no_predictions_given() -> None:
    # given
    source_a = []
    source_b = [{"a": 1}, {"b": 1}]

    # when
    result = list(enumerate_detections(detections_from_sources=[source_a, source_b]))

    # then
    assert result == [(1, {"a": 1}), (1, {"b": 1})]


def test_enumerate_detections_when_sources_with_predictions_given() -> None:
    # given
    source_a = [{"a": 1}, {"b": 1}]
    source_b = [{"c": 1}, {"d": 1}]

    # when
    result = list(enumerate_detections(detections_from_sources=[source_a, source_b]))

    # then
    assert result == [(0, {"a": 1}), (0, {"b": 1}), (1, {"c": 1}), (1, {"d": 1})]


def test_enumerate_detections_when_sources_with_predictions_given_and_source_to_be_excluded() -> (
    None
):
    # given
    source_a = [{"a": 1}, {"b": 1}]
    source_b = [{"c": 1}, {"d": 1}]

    # when
    result = list(
        enumerate_detections(
            detections_from_sources=[source_a, source_b],
            excluded_source_id=1,
        )
    )

    # then
    assert result == [(0, {"a": 1}), (0, {"b": 1})]


def test_get_detections_from_different_sources_with_max_overlap_when_candidate_already_considered() -> (
    None
):
    # given
    detections_from_sources = [[{"detection_id": "a"}], [{"detection_id": "b"}]]

    # when
    result = get_detections_from_different_sources_with_max_overlap(
        detection={"detection_id": "a"},
        source=0,
        detections_from_sources=detections_from_sources,
        iou_threshold=0.5,
        class_aware=True,
        detections_already_considered={"b"},
    )

    # then
    assert len(result) == 0


def test_get_detections_from_different_sources_with_max_overlap_when_candidate_overlap_is_to_small() -> (
    None
):
    # given
    predictions = [
        sv.Detections(
            xyxy=np.array([[80, 190, 120, 210], [80, 190, 120, 210]]),
            confidence=np.array([0.5, 0.5]),
            class_id=np.array([1, 1]),
            data={
                "detection_id": ["a", "c"],
                "class_name": ["a", "a"]
            }
        ),
        sv.Detections(
            xyxy=np.array([[100, 200, 140, 220]]),
            confidence=np.array([0.5]),
            class_id=np.array([1]),
            data={
                "detection_id": ["b"],
                "class_name": ["a"]
            }
        )
    ]

    # when
    result = get_detections_from_different_sources_with_max_overlap(
        detection=predictions[0][0],
        source=0,
        detections_from_sources=predictions,
        iou_threshold=0.5,
        class_aware=True,
        detections_already_considered=set(),
    )

    # then
    assert len(result) == 0


def test_get_detections_from_different_sources_with_max_overlap_when_class_does_not_match() -> (
    None
):
    # given
    predictions = [
        sv.Detections(
            xyxy=np.array([[80, 190, 120, 210], [80, 190, 120, 210]]),
            confidence=np.array([0.5, 0.5]),
            class_id=np.array([1, 1]),
            data={
                "detection_id": ["a", "b"],
                "class_name": ["a", "a"]
            }
        ),
        sv.Detections(
            xyxy=np.array([[80, 190, 120, 210]]),
            confidence=np.array([0.5]),
            class_id=np.array([1]),
            data={
                "detection_id": ["c"],
                "class_name": ["b"]
            }
        ),
        sv.Detections(
            xyxy=np.array([[80, 190, 120, 210]]),
            confidence=np.array([0.5]),
            class_id=np.array([1]),
            data={
                "detection_id": ["d"],
                "class_name": ["b"]
            }
        )
    ]

    # when
    result = get_detections_from_different_sources_with_max_overlap(
        detection=predictions[0][0],
        source=0,
        detections_from_sources=predictions,
        iou_threshold=0.5,
        class_aware=True,
        detections_already_considered=set(),
    )

    # then
    assert len(result) == 0


def test_get_detections_from_different_sources_with_max_overlap_when_class_does_not_match_but_class_unaware_mode_enabled() -> (
    None
):
    # given
    predictions = [
        sv.Detections(
            xyxy=np.array([[80, 190, 120, 210], [80, 190, 120, 210]]),
            confidence=np.array([0.5, 0.5]),
            class_id=np.array([1, 1]),
            data={
                "detection_id": ["a", "b"],
                "class_name": ["a", "a"]
            }
        ),
        sv.Detections(
            xyxy=np.array([[80, 190, 120, 210]]),
            confidence=np.array([0.5]),
            class_id=np.array([1]),
            data={
                "detection_id": ["c"],
                "class_name": ["b"]
            }
        ),
        sv.Detections(
            xyxy=np.array([[80, 190, 120, 210]]),
            confidence=np.array([0.5]),
            class_id=np.array([1]),
            data={
                "detection_id": ["d"],
                "class_name": ["b"]
            }
        )
    ]

    # when
    result = get_detections_from_different_sources_with_max_overlap(
        detection=predictions[0][0],
        source=0,
        detections_from_sources=predictions,
        iou_threshold=0.5,
        class_aware=False,
        detections_already_considered=set(),
    )

    # then
    assert result == {
        1: (
            sv.Detections(
                xyxy=np.array([[80, 190, 120, 210]]),
                confidence=np.array([0.5]),
                class_id=np.array([1]),
                data={
                    "detection_id": ["c"],
                    "class_name": ["b"]
                }
            ),
            1.0,
        ),
        2: (
            sv.Detections(
                xyxy=np.array([[80, 190, 120, 210]]),
                confidence=np.array([0.5]),
                class_id=np.array([1]),
                data={
                    "detection_id": ["d"],
                    "class_name": ["b"]
                }
            ),
            1.0,
        ),
    }, "In both sources other than source 0 it is expected to find fully overlapping prediction, but differ in class"


def test_get_detections_from_different_sources_with_max_overlap_when_multiple_candidates_can_be_found() -> (
    None
):
    # given
    predictions = [
        sv.Detections(
            xyxy=np.array([[80, 190, 120, 210], [80, 190, 120, 210]]),
            confidence=np.array([0.5, 0.5]),
            class_id=np.array([1, 1]),
            data={
                "detection_id": ["a", "b"],
                "class_name": ["a", "a"]
            }
        ),
        sv.Detections(
            xyxy=np.array([[100, 200, 140, 220], [80, 190, 120, 210]]),
            confidence=np.array([0.5, 0.5]),
            class_id=np.array([1, 1]),
            data={
                "detection_id": ["too_small", "c"],
                "class_name": ["a", "a"]
            }
        ),
        sv.Detections(
            xyxy=np.array([[100, 200, 140, 220], [80, 190, 120, 210]]),
            confidence=np.array([0.5, 0.5]),
            class_id=np.array([1, 1]),
            data={
                "detection_id": ["too_small", "d"],
                "class_name": ["a", "a"]
            }
        ),
    ]

    # when
    result = get_detections_from_different_sources_with_max_overlap(
        detection=predictions[0][0],
        source=0,
        detections_from_sources=predictions,
        iou_threshold=0.5,
        class_aware=True,
        detections_already_considered=set(),
    )

    # then
    assert result == {
        1: (
            sv.Detections(
                xyxy=np.array([[80, 190, 120, 210]]),
                confidence=np.array([0.5]),
                class_id=np.array([1]),
                data={
                    "detection_id": ["c"],
                    "class_name": ["a"]
                }
            ),
            1.0,
        ),
        2: (
            sv.Detections(
                xyxy=np.array([[80, 190, 120, 210]]),
                confidence=np.array([0.5]),
                class_id=np.array([1]),
                data={
                    "detection_id": ["d"],
                    "class_name": ["a"]
                }
            ),
            1.0,
        ),
    }, "In both sources other than source 0 it is expected to find fully overlapping prediction"


def test_filter_predictions_when_no_classes_to_consider_given() -> None:
    # given
    predictions = [[{"class": "a"}], [], [{"class": "b"}, {"class": "a"}]]

    # when
    result = filter_predictions(predictions=predictions, classes_to_consider=None)

    # then
    assert result == [[{"class": "a"}], [], [{"class": "b"}, {"class": "a"}]]


def test_filter_predictions_when_classes_to_consider_given() -> None:
    # given
    predictions = [
        sv.Detections(
            xyxy=np.array([[80, 190, 120, 210]]),
            confidence=np.array([0.5]),
            class_id=np.array([1]),
            data={
                "detection_id": ["d"],
                "class_name": ["a"]
            }
        ),
        sv.Detections(
            xyxy=np.array([[100, 200, 140, 220], [80, 190, 120, 210]]),
            confidence=np.array([0.5, 0.5]),
            class_id=np.array([1, 1]),
            data={
                "detection_id": ["e", "f"],
                "class_name": ["a", "b"]
            }
        )
    ]

    # when
    result = filter_predictions(predictions=predictions, classes_to_consider=["a", "c"])

    # then
    assert result == [
        sv.Detections(
            xyxy=np.array([[80, 190, 120, 210]]),
            confidence=np.array([0.5]),
            class_id=np.array([1]),
            data={
                "detection_id": ["d"],
                "class_name": ["a"]
            }
        ),
        sv.Detections(
            xyxy=np.array([[100, 200, 140, 220]]),
            confidence=np.array([0.5]),
            class_id=np.array([1]),
            data={
                "detection_id": ["e"],
                "class_name": ["a"]
            }
        )
    ]


def test_get_parent_id_of_detections_from_sources_when_parent_id_matches() -> None:
    # given
    predictions = [
        sv.Detections(
            xyxy=np.array([[80, 190, 120, 210]]),
            confidence=np.array([0.5]),
            class_id=np.array([1]),
            data={
                "detection_id": ["d"],
                "class_name": ["a"],
                "parent_id": ["a"]
            }
        ),
        sv.Detections(
            xyxy=np.array([[100, 200, 140, 220], [80, 190, 120, 210]]),
            confidence=np.array([0.5, 0.5]),
            class_id=np.array([1, 1]),
            data={
                "detection_id": ["e", "f"],
                "class_name": ["a", "b"],
                "parent_id": ["a", "a"]
            }
        )
    ]

    # when
    result = get_parent_id_of_detections_from_sources(
        detections_from_sources=predictions,
    )

    # then
    assert result == "a"


def test_get_parent_id_of_detections_from_sources_parent_id_does_not_match() -> None:
    # given
    predictions = [
        sv.Detections(
            xyxy=np.array([[80, 190, 120, 210]]),
            confidence=np.array([0.5]),
            class_id=np.array([1]),
            data={
                "detection_id": ["d"],
                "class_name": ["a"],
                "parent_id": ["b"]
            }
        ),
        sv.Detections(
            xyxy=np.array([[100, 200, 140, 220], [80, 190, 120, 210]]),
            confidence=np.array([0.5, 0.5]),
            class_id=np.array([1, 1]),
            data={
                "detection_id": ["e", "f"],
                "class_name": ["a", "b"],
                "parent_id": ["a", "a"]
            }
        )
    ]

    # when
    with pytest.raises(ValueError):
        _ = get_parent_id_of_detections_from_sources(
            detections_from_sources=predictions,
        )

def test_does_not_detect_objects_in_any_source_when_all_sources_give_empty_prediction() -> (
    None
):
    empty_detections = sv.Detections(
        xyxy=np.array([[1, 1, 2, 2]]),
        confidence=np.array([0.9]),
        class_id=np.array([1]),
        data={}
    )[[]]
    # given
    detections_from_sources = [
        empty_detections,
        empty_detections,
    ]

    # when
    result = does_not_detect_objects_in_any_source(
        detections_from_sources=detections_from_sources,
    )

    # then
    assert result is True


def test_does_not_detect_objects_in_any_source_when_no_source_registered() -> None:
    # given
    detections_from_sources = []

    # when
    result = does_not_detect_objects_in_any_source(
        detections_from_sources=detections_from_sources,
    )

    # then
    assert result is True


def test_does_not_detect_objects_in_any_source_when_not_all_sources_give_empty_prediction() -> (
    None
):
    # given
    detections_from_sources = [
        [],
        [{"parent_id": "b"}],
    ]

    # when
    result = does_not_detect_objects_in_any_source(
        detections_from_sources=detections_from_sources,
    )

    # then
    assert result is False


@mock.patch.object(detections_consensus, "uuid4")
def test_get_consensus_for_single_detection_when_only_single_source_and_single_source_is_enough(
    uuid_mock: MagicMock,
) -> None:
    # given
    uuid_mock.return_value = "xxx"
    detections = sv.Detections(
        xyxy=np.array([[80, 190, 120, 210]], dtype=np.float64),
        class_id=np.array([0]),
        confidence=np.array([0.9], dtype=np.float64),
        data={
            "parent_id": np.array(["some_parent"]),
            "detection_id": np.array(["c"]),
            "class_name": np.array(["a"])
        }
    )
    detections_from_sources = [
        [detections],
    ]
    detections_already_considered = set()

    # when
    (
        consensus_detections,
        detections_already_considered,
    ) = get_consensus_for_single_detection(
        detection=detections,
        source_id=0,
        detections_from_sources=detections_from_sources,
        iou_threshold=0.5,
        class_aware=True,
        required_votes=1,
        confidence=0.5,
        detections_merge_confidence_aggregation=AggregationMode.AVERAGE,
        detections_merge_coordinates_aggregation=AggregationMode.AVERAGE,
        detections_already_considered=detections_already_considered,
    )

    # then
    assert detections_already_considered == {"c"}
    assert consensus_detections == [sv.Detections(
        xyxy=np.array([[80, 190, 120, 210]], dtype=np.float64),
        confidence=np.array([0.9], dtype=np.float64),
        class_id=np.array([0]),
        data={
            "detection_id": np.array(["xxx"]),
            "class_name": np.array(["a"]),
            "parent_id": np.array(["some_parent"])
        }
    )]


def test_get_consensus_for_single_detection_when_only_single_source_and_single_source_is_not_enough() -> (
    None
):
    # given
    detections = sv.Detections(
        xyxy=np.array([[80, 190, 120, 210]], dtype=np.float64),
        class_id=np.array([0]),
        confidence=np.array([0.9], dtype=np.float64),
        data={
            "parent_id": np.array(["some_parent"]),
            "detection_id": np.array(["c"]),
            "class_name": np.array(["a"])
        }
    )
    detections_from_sources = [
        detections,
    ]
    detections_already_considered = set()

    # when
    (
        consensus_detections,
        detections_already_considered,
    ) = get_consensus_for_single_detection(
        detection=detections,
        source_id=0,
        detections_from_sources=detections_from_sources,
        iou_threshold=0.5,
        class_aware=True,
        required_votes=2,
        confidence=0.5,
        detections_merge_confidence_aggregation=AggregationMode.AVERAGE,
        detections_merge_coordinates_aggregation=AggregationMode.AVERAGE,
        detections_already_considered=detections_already_considered,
    )

    # then
    assert detections_already_considered == set()
    assert consensus_detections == []


@mock.patch.object(detections_consensus, "uuid4")
def test_get_consensus_for_single_detection_when_only_multiple_sources_matches_and_all_other_conditions_should_pass(
    uuid_mock: MagicMock,
) -> None:
    # given
    uuid_mock.return_value = "xxx"
    detections = sv.Detections(
        xyxy=np.array([[80, 190, 120, 210]], dtype=np.float64),
        class_id=np.array([0]),
        confidence=np.array([0.9], dtype=np.float64),
        data={
            "parent_id": np.array(["some_parent"]),
            "detection_id": np.array(["c"]),
            "class_name": np.array(["a"])
        }
    )
    detections_from_sources = [
        detections,
        sv.Detections(
            xyxy=np.array([[80, 185, 120, 215]], dtype=np.float64),
            class_id=np.array([0]),
            confidence=np.array([0.9], dtype=np.float64),
            data={
                "parent_id": np.array(["some_parent"]),
                "detection_id": np.array(["d"]),
                "class_name": np.array(["a"])
            }
        )
    ]
    detections_already_considered = set()

    # when
    (
        consensus_detections,
        detections_already_considered,
    ) = get_consensus_for_single_detection(
        detection=detections,
        source_id=0,
        detections_from_sources=detections_from_sources,
        iou_threshold=0.5,
        class_aware=True,
        required_votes=2,
        confidence=0.5,
        detections_merge_confidence_aggregation=AggregationMode.AVERAGE,
        detections_merge_coordinates_aggregation=AggregationMode.AVERAGE,
        detections_already_considered=detections_already_considered,
    )

    # then
    assert detections_already_considered == {"c", "d"}
    assert consensus_detections == [
        sv.Detections(
            xyxy=np.array([[80, 187.5, 120, 212.5]], dtype=np.float64),
            class_id=np.array([0]),
            confidence=np.array([0.9], dtype=np.float64),
            data={
                "parent_id": np.array(["some_parent"]),
                "detection_id": np.array(["xxx"]),
                "class_name": np.array(["a"])
            }
        )
    ]


def test_get_consensus_for_single_detection_when_only_multiple_sources_matches_but_not_enough_votes_collected() -> (
    None
):
    # given
    detections = sv.Detections(
        xyxy=np.array([[80, 190, 120, 210]], dtype=np.float64),
        class_id=np.array([0]),
        confidence=np.array([0.9], dtype=np.float64),
        data={
            "parent_id": np.array(["some_parent"]),
            "detection_id": np.array(["c"]),
            "class_name": np.array(["a"])
        }
    )
    empty_detections = sv.Detections(
        xyxy=np.array([[1, 1, 2, 2]]),
        confidence=np.array([0.9]),
        class_id=np.array([1]),
        data={}
    )[[]]
    detections_from_sources = [
        detections,
        sv.Detections(
            xyxy=np.array([[80, 185, 120, 215]], dtype=np.float64),
            class_id=np.array([0]),
            confidence=np.array([0.9], dtype=np.float64),
            data={
                "parent_id": np.array(["some_parent"]),
                "detection_id": np.array(["d"]),
                "class_name": np.array(["a"])
            }
        ),
        empty_detections,
    ]

    detections_already_considered = set()

    # when
    (
        consensus_detections,
        detections_already_considered,
    ) = get_consensus_for_single_detection(
        detection=detections,
        source_id=0,
        detections_from_sources=detections_from_sources,
        iou_threshold=0.5,
        class_aware=True,
        required_votes=3,
        confidence=0.5,
        detections_merge_confidence_aggregation=AggregationMode.AVERAGE,
        detections_merge_coordinates_aggregation=AggregationMode.AVERAGE,
        detections_already_considered=detections_already_considered,
    )

    # then
    assert detections_already_considered == set()
    assert consensus_detections == []


@mock.patch.object(detections_consensus, "uuid4")
def test_get_consensus_for_single_detection_when_only_multiple_sources_matches_but_confidence_is_not_enough(
    uuid_mock: MagicMock,
) -> None:
    # given
    uuid_mock.return_value = "xxx"
    detections = sv.Detections(
        xyxy=np.array([[80, 190, 120, 210]], dtype=np.float64),
        class_id=np.array([0]),
        confidence=np.array([0.7], dtype=np.float64),
        data={
            "parent_id": np.array(["some_parent"]),
            "detection_id": np.array(["c"]),
            "class_name": np.array(["a"])
        }
    )
    detections_from_sources = [
        detections,
        sv.Detections(
            xyxy=np.array([[80, 185, 120, 215]], dtype=np.float64),
            class_id=np.array([0]),
            confidence=np.array([0.7], dtype=np.float64),
            data={
                "parent_id": np.array(["some_parent"]),
                "detection_id": np.array(["d"]),
                "class_name": np.array(["a"])
            }
        ),
    ]
    detections_already_considered = set()

    # when
    (
        consensus_detections,
        detections_already_considered,
    ) = get_consensus_for_single_detection(
        detection=detections,
        source_id=0,
        detections_from_sources=detections_from_sources,
        iou_threshold=0.5,
        class_aware=True,
        required_votes=2,
        confidence=0.8,
        detections_merge_confidence_aggregation=AggregationMode.AVERAGE,
        detections_merge_coordinates_aggregation=AggregationMode.AVERAGE,
        detections_already_considered=detections_already_considered,
    )

    # then
    assert detections_already_considered == set()
    assert consensus_detections == []


@mock.patch.object(detections_consensus, "uuid4")
def test_get_consensus_for_single_detection_when_only_multiple_sources_matches_but_classes_do_not_match(
    uuid_mock: MagicMock,
) -> None:
    # given
    uuid_mock.return_value = "xxx"
    detections = sv.Detections(
        xyxy=np.array([[80, 190, 120, 210]], dtype=np.float64),
        class_id=np.array([0]),
        confidence=np.array([0.7], dtype=np.float64),
        data={
            "parent_id": np.array(["some_parent"]),
            "detection_id": np.array(["c"]),
            "class_name": np.array(["a"])
        }
    )
    detections_from_sources = [
        detections,
        sv.Detections(
            xyxy=np.array([[80, 185, 120, 215]], dtype=np.float64),
            class_id=np.array([0]),
            confidence=np.array([0.7], dtype=np.float64),
            data={
                "parent_id": np.array(["some_parent"]),
                "detection_id": np.array(["d"]),
                "class_name": np.array(["b"])
            }
        ),
    ]
    detections_already_considered = set()

    # when
    (
        consensus_detections,
        detections_already_considered,
    ) = get_consensus_for_single_detection(
        detection=detections,
        source_id=0,
        detections_from_sources=detections_from_sources,
        iou_threshold=0.5,
        class_aware=True,
        required_votes=2,
        confidence=0.5,
        detections_merge_confidence_aggregation=AggregationMode.AVERAGE,
        detections_merge_coordinates_aggregation=AggregationMode.AVERAGE,
        detections_already_considered=detections_already_considered,
    )

    # then
    assert detections_already_considered == set()
    assert consensus_detections == []


def test_check_objects_presence_in_consensus_detections_when_no_detections_provided() -> (
    None
):
    # given
    empty_detections = sv.Detections(
        xyxy=np.array([[1, 1, 2, 2]]),
        confidence=np.array([0.9]),
        class_id=np.array([1]),
        data={}
    )[[]]
    # when
    result = check_objects_presence_in_consensus_detections(
        consensus_detections=empty_detections,
        class_aware=True,
        aggregation_mode=AggregationMode.AVERAGE,
        required_objects=None,
    )

    # then
    assert result == (False, {})


def test_check_objects_presence_in_consensus_detections_when_no_detection_is_required_and_something_is_detected() -> (
    None
):
    # given
    consensus_detections = sv.Detections(
        xyxy=np.array([[80, 187.5, 120, 212.5]], dtype=np.float64),
        class_id=np.array([0]),
        confidence=np.array([0.9], dtype=np.float64),
        data={
            "parent_id": np.array(["some_parent"]),
            "detection_id": np.array(["xxx"]),
            "class_name": np.array(["a"])
        }
    )

    # when
    result = check_objects_presence_in_consensus_detections(
        consensus_detections=consensus_detections,
        class_aware=True,
        aggregation_mode=AggregationMode.AVERAGE,
        required_objects=None,
    )

    # then
    assert result == (True, {"a": 0.9})


def test_check_objects_presence_in_consensus_detections_when_one_detection_is_required_and_something_is_detected() -> (
    None
):
    # given
    consensus_detections = sv.Detections(
        xyxy=np.array([[80, 187.5, 120, 212.5]], dtype=np.float64),
        class_id=np.array([0]),
        confidence=np.array([0.9], dtype=np.float64),
        data={
            "parent_id": np.array(["some_parent"]),
            "detection_id": np.array(["xxx"]),
            "class_name": np.array(["a"])
        }
    )

    # when
    result = check_objects_presence_in_consensus_detections(
        consensus_detections=consensus_detections,
        class_aware=True,
        aggregation_mode=AggregationMode.AVERAGE,
        required_objects=1,
    )

    # then
    assert result == (True, {"a": 0.9})


def test_check_objects_presence_in_consensus_detections_when_specific_detection_is_required_and_something_is_detected_but_other_class() -> (
    None
):
    # given
    consensus_detections = sv.Detections(
        xyxy=np.array([[80, 187.5, 120, 212.5], [80, 187.5, 120, 212.5]], dtype=np.float64),
        class_id=np.array([0, 2]),
        confidence=np.array([0.9, 0.9], dtype=np.float64),
        data={
            "parent_id": np.array(["some_parent", "some_parent"]),
            "detection_id": np.array(["xxx", "yyy"]),
            "class_name": np.array(["a", "c"])
        }
    )

    # when
    result = check_objects_presence_in_consensus_detections(
        consensus_detections=consensus_detections,
        class_aware=True,
        aggregation_mode=AggregationMode.AVERAGE,
        required_objects={"b": 1},
    )

    # then
    assert result == (False, {})


def test_check_objects_presence_in_consensus_detections_when_specific_detection_is_required_and_something_is_detected_but_other_class_and_class_unaware_mode() -> (
    None
):
    # given
    consensus_detections = sv.Detections(
        xyxy=np.array([[80, 187.5, 120, 212.5], [80, 187.5, 120, 212.5]], dtype=np.float64),
        class_id=np.array([0, 2]),
        confidence=np.array([0.8, 1], dtype=np.float64),
        data={
            "parent_id": np.array(["some_parent", "some_parent"]),
            "detection_id": np.array(["xxx", "yyy"]),
            "class_name": np.array(["a", "c"])
        }
    )

    # when
    result = check_objects_presence_in_consensus_detections(
        consensus_detections=consensus_detections,
        class_aware=False,
        aggregation_mode=AggregationMode.AVERAGE,
        required_objects={"b": 2},
    )

    # then
    assert result == (True, {"any_object": 9 / 10})


def test_check_objects_presence_in_consensus_detections_when_specific_detection_is_required_and_something_is_detected_but_not_all_classes_satisfied() -> (
    None
):
    # given
    consensus_detections = sv.Detections(
        xyxy=np.array([[80, 187.5, 120, 212.5], [80, 187.5, 120, 212.5]], dtype=np.float64),
        class_id=np.array([0, 2]),
        confidence=np.array([0.9, 0.9], dtype=np.float64),
        data={
            "parent_id": np.array(["some_parent", "some_parent"]),
            "detection_id": np.array(["xxx", "yyy"]),
            "class_name": np.array(["a", "c"])
        }
    )

    # when
    result = check_objects_presence_in_consensus_detections(
        consensus_detections=consensus_detections,
        class_aware=True,
        aggregation_mode=AggregationMode.AVERAGE,
        required_objects={"a": 2, "b": 1},
    )

    # then
    assert result == (False, {})


def test_check_objects_presence_in_consensus_detections_when_specific_detection_is_required_and_something_is_detected_and_all_classes_satisfied() -> (
    None
):
    # given
    consensus_detections = sv.Detections(
        xyxy=np.array([[80, 187.5, 120, 212.5], [80, 187.5, 120, 212.5]], dtype=np.float64),
        class_id=np.array([0, 2]),
        confidence=np.array([0.9, 0.9], dtype=np.float64),
        data={
            "parent_id": np.array(["some_parent", "some_parent"]),
            "detection_id": np.array(["xxx", "yyy"]),
            "class_name": np.array(["a", "a"])
        }
    )

    # when
    result = check_objects_presence_in_consensus_detections(
        consensus_detections=consensus_detections,
        class_aware=True,
        aggregation_mode=AggregationMode.AVERAGE,
        required_objects={"a": 2},
    )

    # then
    assert result == (True, {"a": 0.9})


def test_agree_on_consensus_for_all_detections_sources_when_empty_predictions_given() -> (
    None
):
    # when
    result = agree_on_consensus_for_all_detections_sources(
        detections_from_sources=[],
        required_votes=2,
        class_aware=True,
        iou_threshold=0.5,
        confidence=0.5,
        classes_to_consider=None,
        required_objects=None,
        presence_confidence_aggregation=AggregationMode.MAX,
        detections_merge_confidence_aggregation=AggregationMode.AVERAGE,
        detections_merge_coordinates_aggregation=AggregationMode.AVERAGE,
    )

    # then
    assert result == ("undefined", False, {}, [])


def test_agree_on_consensus_for_all_detections_sources_when_predictions_do_not_match_classes() -> (
    None
):
    # given
    empty_detections = sv.Detections(
        xyxy=np.array([[1, 1, 2, 2]]),
        confidence=np.array([0.9]),
        class_id=np.array([1]),
        data={}
    )[[]]
    detections_from_sources = [
        sv.Detections(
            xyxy=np.array([[80, 185, 120, 215]], dtype=np.float64),
            class_id=np.array([0]),
            confidence=np.array([0.7], dtype=np.float64),
            data={
                "parent_id": np.array(["some_parent"]),
                "detection_id": np.array(["d"]),
                "class_name": np.array(["b"])
            }
        ),
        empty_detections,
    ]

    # when
    result = agree_on_consensus_for_all_detections_sources(
        detections_from_sources=detections_from_sources,
        required_votes=2,
        class_aware=True,
        iou_threshold=0.5,
        confidence=0.5,
        classes_to_consider=["y", "z"],
        required_objects=None,
        presence_confidence_aggregation=AggregationMode.MAX,
        detections_merge_confidence_aggregation=AggregationMode.AVERAGE,
        detections_merge_coordinates_aggregation=AggregationMode.AVERAGE,
    )

    # then
    assert result == ("some_parent", False, {}, empty_detections)


@mock.patch.object(detections_consensus, "uuid4")
def test_agree_on_consensus_for_all_detections_sources_when_predictions_from_single_source_given_but_thats_enough_for_consensus(
    uuid_mock: MagicMock,
) -> None:
    # given
    uuid_mock.return_value = "xxx"
    detections_from_sources = [sv.Detections(
        xyxy=np.array([[80, 185, 120, 215], [90, 185, 130, 215]], dtype=np.float64),
        class_id=np.array([0, 0]),
        confidence=np.array([0.7, 0.9], dtype=np.float64),
        data={
            "parent_id": np.array(["some_parent", "some_parent"]),
            "detection_id": np.array(["a", "b"]),
            "class_name": np.array(["b", "b"])
        }
    )]

    # when
    result = agree_on_consensus_for_all_detections_sources(
        detections_from_sources=detections_from_sources,
        required_votes=1,
        class_aware=True,
        iou_threshold=0.5,
        confidence=0.5,
        classes_to_consider=["b"],
        required_objects=None,
        presence_confidence_aggregation=AggregationMode.MAX,
        detections_merge_confidence_aggregation=AggregationMode.AVERAGE,
        detections_merge_coordinates_aggregation=AggregationMode.AVERAGE,
    )

    # then
    expected_consensus_detections = sv.Detections(
        xyxy=np.array([[80, 185, 120, 215], [90, 185, 130, 215]], dtype=np.float64),
        class_id=np.array([0, 0]),
        confidence=np.array([0.7, 0.9], dtype=np.float64),
        data={
            "parent_id": np.array(["some_parent", "some_parent"]),
            "detection_id": np.array(["xxx", "xxx"]),
            "class_name": np.array(["b", "b"])
        }
    )

    assert result == ("some_parent", True, {"b": 0.9}, expected_consensus_detections)


@mock.patch.object(detections_consensus, "uuid4")
def test_agree_on_consensus_for_all_detections_sources_when_predictions_from_multiple_sources_given_enough_for_consensus(
    uuid_mock: MagicMock,
) -> None:
    # given
    uuid_mock.return_value = "xxx"
    detections_from_sources = [
        sv.Detections(
            xyxy=np.array([[80, 185, 120, 215], [90, 185, 130, 215]], dtype=np.float64),
            class_id=np.array([0, 0]),
            confidence=np.array([0.7, 0.9], dtype=np.float64),
            data={
                "parent_id": np.array(["some_parent", "some_parent"]),
                "detection_id": np.array(["a", "b"]),
                "class_name": np.array(["b", "b"])
            }
        ),
        sv.Detections(
            xyxy=np.array([[80, 183, 120, 217], [90, 182, 130, 218]], dtype=np.float64),
            class_id=np.array([0, 0]),
            confidence=np.array([0.8, 1], dtype=np.float64),
            data={
                "parent_id": np.array(["some_parent", "some_parent"]),
                "detection_id": np.array(["c", "d"]),
                "class_name": np.array(["b", "b"])
            }
        )
    ]

    # when
    result = agree_on_consensus_for_all_detections_sources(
        detections_from_sources=detections_from_sources,
        required_votes=1,
        class_aware=True,
        iou_threshold=0.5,
        confidence=0.5,
        classes_to_consider=["b"],
        required_objects={"b": 2},
        presence_confidence_aggregation=AggregationMode.MAX,
        detections_merge_confidence_aggregation=AggregationMode.AVERAGE,
        detections_merge_coordinates_aggregation=AggregationMode.AVERAGE,
    )

    # then
    expected_consensus = sv.Detections(
        xyxy=np.array([[80, 184, 120, 216], [90, 183.5, 130, 216.5]], dtype=np.float64),
        class_id=np.array([0, 0]),
        confidence=np.array([0.75, 0.95], dtype=np.float64),
        data={
            "parent_id": np.array(["some_parent", "some_parent"]),
            "detection_id": np.array(["xxx", "xxx"]),
            "class_name": np.array(["b", "b"])
        }
    )

    assert result == ("some_parent", True, {"b": 0.95}, expected_consensus)
