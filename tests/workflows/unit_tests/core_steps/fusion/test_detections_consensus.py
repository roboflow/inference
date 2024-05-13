from typing import Any
from unittest import mock
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.common.utils import detection_to_xyxy
from inference.core.workflows.core_steps.fusion import detections_consensus
from inference.core.workflows.core_steps.fusion.detections_consensus import (
    AggregationMode,
    BlockManifest,
    aggregate_field_values,
    agree_on_consensus_for_all_detections_sources,
    calculate_iou,
    check_objects_presence_in_consensus_detections,
    does_not_detected_objects_in_any_source,
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
    detections = [{"a": 0.3}, {"a": 0.4}, {"a": 0.7}]

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
    detections = [{"a": 0.3}, {"a": 0.4}, {"a": 0.7}]

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
    detections = [{"a": 0.3}, {"a": 0.4}, {"a": 0.5}]

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
            detections=[],
            field="a",
            aggregation_mode=mode,
        )


def test_get_largest_bounding_box_when_single_element_provided() -> None:
    # given
    detections = [
        {"x": 100, "y": 200, "height": 30, "width": 40},
    ]

    # when
    result = get_largest_bounding_box(detections=detections)

    # then
    assert result == (100, 200, 40, 30)


def test_get_largest_bounding_box_when_multiple_elements_provided() -> None:
    # given
    detections = [
        {"x": 100, "y": 200, "height": 30, "width": 40},
        {"x": 110, "y": 210, "height": 40, "width": 50},
    ]

    # when
    result = get_largest_bounding_box(detections=detections)

    # then
    assert result == (110, 210, 50, 40)


def test_get_smallest_bounding_box_when_single_element_provided() -> None:
    # given
    detections = [
        {"x": 100, "y": 200, "height": 30, "width": 40},
    ]

    # when
    result = get_smallest_bounding_box(detections=detections)

    # then
    assert result == (100, 200, 40, 30)


def test_get_smallest_bounding_box_when_multiple_elements_provided() -> None:
    # given
    detections = [
        {"x": 100, "y": 200, "height": 30, "width": 40},
        {"x": 110, "y": 210, "height": 40, "width": 50},
    ]

    # when
    result = get_smallest_bounding_box(detections=detections)

    # then
    assert result == (100, 200, 40, 30)


def test_get_average_bounding_box_when_single_element_provided() -> None:
    # given
    detections = [
        {"x": 100, "y": 200, "height": 30, "width": 40},
    ]

    # when
    result = get_average_bounding_box(detections=detections)

    # then
    assert result == (100, 200, 40, 30)


def test_get_average_bounding_box_when_multiple_elements_provided() -> None:
    # given
    detections = [
        {"x": 100, "y": 200, "height": 30, "width": 40},
        {"x": 110, "y": 210, "height": 40, "width": 50},
    ]

    # when
    result = get_average_bounding_box(detections=detections)

    # then
    assert result == (105, 205, 45, 35)


def test_get_majority_class() -> None:
    # given
    detections = [
        {"class": "a", "class_id": 0},
        {"class": "b", "class_id": 1},
        {"class": "a", "class_id": 0},
    ]

    # when
    result = get_majority_class(detections=detections)

    # then
    assert result == ("a", 0)


def test_get_class_of_most_confident_detection() -> None:
    # given
    detections = [
        {"class": "a", "class_id": 0, "confidence": 0.1},
        {"class": "b", "class_id": 1, "confidence": 0.3},
        {"class": "a", "class_id": 0, "confidence": 0.2},
    ]

    # when
    result = get_class_of_most_confident_detection(detections=detections)

    # then
    assert result == ("b", 1)


def test_get_class_of_least_confident_detection() -> None:
    # given
    detections = [
        {"class": "a", "class_id": 0, "confidence": 0.1},
        {"class": "b", "class_id": 1, "confidence": 0.3},
        {"class": "a", "class_id": 0, "confidence": 0.2},
    ]

    # when
    result = get_class_of_least_confident_detection(detections=detections)

    # then
    assert result == ("a", 0)


@mock.patch.object(detections_consensus, "uuid4")
def test_merge_detections(uuid4_mock: MagicMock) -> None:
    # given
    uuid4_mock.return_value = "some_uuid"
    detections = [
        {
            "parent_id": "x",
            "class": "a",
            "class_id": 0,
            "confidence": 1 / 10,
            "x": 100,
            "y": 200,
            "height": 30,
            "width": 40,
        },
        {
            "parent_id": "x",
            "class": "a",
            "class_id": 0,
            "confidence": 3 / 10,
            "x": 110,
            "y": 210,
            "height": 40,
            "width": 50,
        },
    ]

    # when
    result = merge_detections(
        detections=detections,
        confidence_aggregation_mode=AggregationMode.AVERAGE,
        boxes_aggregation_mode=AggregationMode.MAX,
    )

    # then
    assert result == {
        "parent_id": "x",
        "detection_id": "some_uuid",
        "class": "a",
        "class_id": 0,
        "confidence": 2 / 10,
        "x": 110,
        "y": 210,
        "height": 40,
        "width": 50,
    }


def test_detection_to_xyxy() -> None:
    # given
    detection = {"x": 100, "y": 200, "height": 20, "width": 40}

    # when
    result = detection_to_xyxy(detection=detection)

    # then
    assert result == (80, 190, 120, 210)


def test_calculate_iou_when_detections_are_zero_size() -> None:
    # given
    detection_a = {"x": 100, "y": 200, "height": 0, "width": 1}
    detection_b = {"x": 100, "y": 220, "height": 1, "width": 0}

    # when
    result = calculate_iou(
        detection_a=detection_a,
        detection_b=detection_b,
    )

    # then
    assert abs(result) < 1e-5


def test_calculate_iou_when_detections_do_not_overlap() -> None:
    # given
    detection_a = {"x": 100, "y": 200, "height": 20, "width": 40}
    detection_b = {"x": 100, "y": 220, "height": 20, "width": 40}

    # when
    result = calculate_iou(
        detection_a=detection_a,
        detection_b=detection_b,
    )

    # then
    assert abs(result) < 1e-5


def test_calculate_iou_when_detections_do_overlap_fully() -> None:
    # given
    detection_a = {"x": 100, "y": 200, "height": 20, "width": 40}

    # when
    result = calculate_iou(
        detection_a=detection_a,
        detection_b=detection_a,
    )

    # then
    assert abs(result - 1.0) < 1e-5


def test_calculate_iou_when_detections_do_overlap_partially() -> None:
    # given
    detection_a = {"x": 100, "y": 200, "height": 20, "width": 40}
    detection_b = {"x": 120, "y": 210, "height": 20, "width": 40}

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
    detections_from_sources = [
        [
            {
                "detection_id": "a",
                "x": 100,
                "y": 200,
                "height": 20,
                "width": 40,
                "class": "a",
            },
            {
                "detection_id": "c",
                "x": 100,
                "y": 200,
                "height": 20,
                "width": 40,
                "class": "a",
            },
        ],
        [
            {
                "detection_id": "b",
                "x": 120,
                "y": 210,
                "height": 20,
                "width": 40,
                "class": "a",
            }
        ],
    ]

    # when
    result = get_detections_from_different_sources_with_max_overlap(
        detection=detections_from_sources[0][0],
        source=0,
        detections_from_sources=detections_from_sources,
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
    detections_from_sources = [
        [
            {
                "detection_id": "a",
                "x": 100,
                "y": 200,
                "height": 20,
                "width": 40,
                "class": "a",
            },
            {
                "detection_id": "b",
                "x": 100,
                "y": 200,
                "height": 20,
                "width": 40,
                "class": "a",
            },
        ],
        [
            {
                "detection_id": "c",
                "x": 100,
                "y": 200,
                "height": 20,
                "width": 40,
                "class": "b",
            }
        ],
        [
            {
                "detection_id": "d",
                "x": 100,
                "y": 200,
                "height": 20,
                "width": 40,
                "class": "b",
            }
        ],
    ]

    # when
    result = get_detections_from_different_sources_with_max_overlap(
        detection=detections_from_sources[0][0],
        source=0,
        detections_from_sources=detections_from_sources,
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
    detections_from_sources = [
        [
            {
                "detection_id": "a",
                "x": 100,
                "y": 200,
                "height": 20,
                "width": 40,
                "class": "a",
            },
            {
                "detection_id": "b",
                "x": 100,
                "y": 200,
                "height": 20,
                "width": 40,
                "class": "a",
            },
        ],
        [
            {
                "detection_id": "c",
                "x": 100,
                "y": 200,
                "height": 20,
                "width": 40,
                "class": "b",
            }
        ],
        [
            {
                "detection_id": "d",
                "x": 100,
                "y": 200,
                "height": 20,
                "width": 40,
                "class": "b",
            }
        ],
    ]

    # when
    result = get_detections_from_different_sources_with_max_overlap(
        detection=detections_from_sources[0][0],
        source=0,
        detections_from_sources=detections_from_sources,
        iou_threshold=0.5,
        class_aware=False,
        detections_already_considered=set(),
    )

    # then
    assert result == {
        1: (
            {
                "detection_id": "c",
                "x": 100,
                "y": 200,
                "height": 20,
                "width": 40,
                "class": "b",
            },
            1.0,
        ),
        2: (
            {
                "detection_id": "d",
                "x": 100,
                "y": 200,
                "height": 20,
                "width": 40,
                "class": "b",
            },
            1.0,
        ),
    }, "In both sources other than source 0 it is expected to find fully overlapping prediction, but differ in class"


def test_get_detections_from_different_sources_with_max_overlap_when_multiple_candidates_can_be_found() -> (
    None
):
    # given
    detections_from_sources = [
        [
            {
                "detection_id": "a",
                "x": 100,
                "y": 200,
                "height": 20,
                "width": 40,
                "class": "a",
            },
            {
                "detection_id": "b",
                "x": 100,
                "y": 200,
                "height": 20,
                "width": 40,
                "class": "a",
            },
        ],
        [
            {
                "detection_id": "to_small",
                "x": 120,
                "y": 210,
                "height": 20,
                "width": 40,
                "class": "a",
            },
            {
                "detection_id": "c",
                "x": 100,
                "y": 200,
                "height": 20,
                "width": 40,
                "class": "a",
            },
        ],
        [
            {
                "detection_id": "to_small",
                "x": 120,
                "y": 210,
                "height": 20,
                "width": 40,
                "class": "a",
            },
            {
                "detection_id": "d",
                "x": 100,
                "y": 200,
                "height": 20,
                "width": 40,
                "class": "a",
            },
        ],
    ]

    # when
    result = get_detections_from_different_sources_with_max_overlap(
        detection=detections_from_sources[0][0],
        source=0,
        detections_from_sources=detections_from_sources,
        iou_threshold=0.5,
        class_aware=True,
        detections_already_considered=set(),
    )

    # then
    assert result == {
        1: (
            {
                "detection_id": "c",
                "x": 100,
                "y": 200,
                "height": 20,
                "width": 40,
                "class": "a",
            },
            1.0,
        ),
        2: (
            {
                "detection_id": "d",
                "x": 100,
                "y": 200,
                "height": 20,
                "width": 40,
                "class": "a",
            },
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
    predictions = [[{"class": "a"}], [], [{"class": "b"}, {"class": "a"}]]

    # when
    result = filter_predictions(predictions=predictions, classes_to_consider=["a", "c"])

    # then
    assert result == [[{"class": "a"}], [], [{"class": "a"}]]


def test_get_parent_id_of_detections_from_sources_when_parent_id_matches() -> None:
    # given
    detections_from_sources = [
        [{"parent_id": "a"}],
        [],
        [{"parent_id": "a"}, {"parent_id": "a"}],
    ]

    # when
    result = get_parent_id_of_detections_from_sources(
        detections_from_sources=detections_from_sources,
    )

    # then
    assert result == "a"


def test_get_parent_id_of_detections_from_sources_parent_id_does_not_match() -> None:
    # given
    detections_from_sources = [
        [{"parent_id": "b"}],
        [],
        [{"parent_id": "a"}, {"parent_id": "a"}],
    ]

    # when
    with pytest.raises(ValueError):
        _ = get_parent_id_of_detections_from_sources(
            detections_from_sources=detections_from_sources,
        )


def test_does_not_detected_objects_in_any_source_when_all_sources_give_empty_prediction() -> (
    None
):
    # given
    detections_from_sources = [
        [],
        [],
    ]

    # when
    result = does_not_detected_objects_in_any_source(
        detections_from_sources=detections_from_sources,
    )

    # then
    assert result is True


def test_does_not_detected_objects_in_any_source_when_no_source_registered() -> None:
    # given
    detections_from_sources = []

    # when
    result = does_not_detected_objects_in_any_source(
        detections_from_sources=detections_from_sources,
    )

    # then
    assert result is True


def test_does_not_detected_objects_in_any_source_when_not_all_sources_give_empty_prediction() -> (
    None
):
    # given
    detections_from_sources = [
        [],
        [{"parent_id": "b"}],
    ]

    # when
    result = does_not_detected_objects_in_any_source(
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
    detection = {
        "detection_id": "c",
        "x": 100,
        "y": 200,
        "height": 20,
        "width": 40,
        "class": "a",
        "class_id": 0,
        "confidence": 0.9,
        "parent_id": "some_parent",
    }
    detections_from_sources = [
        [detection],
    ]
    detections_already_considered = set()

    # when
    (
        consensus_detections,
        detections_already_considered,
    ) = get_consensus_for_single_detection(
        detection=detection,
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
    assert consensus_detections == [
        {
            "detection_id": "xxx",
            "x": 100,
            "y": 200,
            "height": 20,
            "width": 40,
            "class": "a",
            "class_id": 0,
            "confidence": 0.9,
            "parent_id": "some_parent",
        }
    ]


def test_get_consensus_for_single_detection_when_only_single_source_and_single_source_is_not_enough() -> (
    None
):
    # given
    detection = {
        "detection_id": "c",
        "x": 100,
        "y": 200,
        "height": 20,
        "width": 40,
        "class": "a",
        "class_id": 0,
        "confidence": 0.9,
        "parent_id": "some_parent",
    }
    detections_from_sources = [
        [detection],
    ]
    detections_already_considered = set()

    # when
    (
        consensus_detections,
        detections_already_considered,
    ) = get_consensus_for_single_detection(
        detection=detection,
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
    detection = {
        "detection_id": "c",
        "x": 100,
        "y": 200,
        "height": 20,
        "width": 40,
        "class": "a",
        "class_id": 0,
        "confidence": 0.9,
        "parent_id": "some_parent",
    }
    detections_from_sources = [
        [detection],
        [
            {
                "detection_id": "d",
                "x": 100,
                "y": 200,
                "height": 30,
                "width": 40,
                "class": "a",
                "class_id": 0,
                "confidence": 0.9,
                "parent_id": "some_parent",
            }
        ],
    ]
    detections_already_considered = set()

    # when
    (
        consensus_detections,
        detections_already_considered,
    ) = get_consensus_for_single_detection(
        detection=detection,
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
        {
            "detection_id": "xxx",
            "x": 100,
            "y": 200,
            "height": 25,
            "width": 40,
            "class": "a",
            "class_id": 0,
            "confidence": 0.9,
            "parent_id": "some_parent",
        }
    ]


def test_get_consensus_for_single_detection_when_only_multiple_sources_matches_but_not_enough_votes_collected() -> (
    None
):
    # given
    detection = {
        "detection_id": "c",
        "x": 100,
        "y": 200,
        "height": 20,
        "width": 40,
        "class": "a",
        "class_id": 0,
        "confidence": 0.9,
        "parent_id": "some_parent",
    }
    detections_from_sources = [
        [detection],
        [
            {
                "detection_id": "d",
                "x": 100,
                "y": 200,
                "height": 30,
                "width": 40,
                "class": "a",
                "class_id": 0,
                "confidence": 0.9,
                "parent_id": "some_parent",
            }
        ],
        [],
    ]
    detections_already_considered = set()

    # when
    (
        consensus_detections,
        detections_already_considered,
    ) = get_consensus_for_single_detection(
        detection=detection,
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
    detection = {
        "detection_id": "c",
        "x": 100,
        "y": 200,
        "height": 20,
        "width": 40,
        "class": "a",
        "class_id": 0,
        "confidence": 0.7,
        "parent_id": "some_parent",
    }
    detections_from_sources = [
        [detection],
        [
            {
                "detection_id": "d",
                "x": 100,
                "y": 200,
                "height": 30,
                "width": 40,
                "class": "a",
                "class_id": 0,
                "confidence": 0.7,
                "parent_id": "some_parent",
            }
        ],
    ]
    detections_already_considered = set()

    # when
    (
        consensus_detections,
        detections_already_considered,
    ) = get_consensus_for_single_detection(
        detection=detection,
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
    detection = {
        "detection_id": "c",
        "x": 100,
        "y": 200,
        "height": 20,
        "width": 40,
        "class": "a",
        "class_id": 0,
        "confidence": 0.7,
        "parent_id": "some_parent",
    }
    detections_from_sources = [
        [detection],
        [
            {
                "detection_id": "d",
                "x": 100,
                "y": 200,
                "height": 30,
                "width": 40,
                "class": "b",
                "class_id": 0,
                "confidence": 0.7,
                "parent_id": "some_parent",
            }
        ],
    ]
    detections_already_considered = set()

    # when
    (
        consensus_detections,
        detections_already_considered,
    ) = get_consensus_for_single_detection(
        detection=detection,
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
    # when
    result = check_objects_presence_in_consensus_detections(
        consensus_detections=[],
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
    consensus_detections = [
        {
            "detection_id": "xxx",
            "x": 100,
            "y": 200,
            "height": 25,
            "width": 40,
            "class": "a",
            "class_id": 0,
            "confidence": 0.9,
            "parent_id": "some_parent",
        }
    ]

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
    consensus_detections = [
        {
            "detection_id": "xxx",
            "x": 100,
            "y": 200,
            "height": 25,
            "width": 40,
            "class": "a",
            "class_id": 0,
            "confidence": 0.9,
            "parent_id": "some_parent",
        }
    ]

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
    consensus_detections = [
        {
            "detection_id": "xxx",
            "x": 100,
            "y": 200,
            "height": 25,
            "width": 40,
            "class": "a",
            "class_id": 0,
            "confidence": 0.9,
            "parent_id": "some_parent",
        },
        {
            "detection_id": "yyy",
            "x": 100,
            "y": 200,
            "height": 25,
            "width": 40,
            "class": "c",
            "class_id": 2,
            "confidence": 0.9,
            "parent_id": "some_parent",
        },
    ]

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
    consensus_detections = [
        {
            "detection_id": "xxx",
            "x": 100,
            "y": 200,
            "height": 25,
            "width": 40,
            "class": "a",
            "class_id": 0,
            "confidence": 8 / 10,
            "parent_id": "some_parent",
        },
        {
            "detection_id": "yyy",
            "x": 100,
            "y": 200,
            "height": 25,
            "width": 40,
            "class": "c",
            "class_id": 2,
            "confidence": 1,
            "parent_id": "some_parent",
        },
    ]

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
    consensus_detections = [
        {
            "detection_id": "xxx",
            "x": 100,
            "y": 200,
            "height": 25,
            "width": 40,
            "class": "a",
            "class_id": 0,
            "confidence": 0.9,
            "parent_id": "some_parent",
        },
        {
            "detection_id": "yyy",
            "x": 100,
            "y": 200,
            "height": 25,
            "width": 40,
            "class": "a",
            "class_id": 2,
            "confidence": 0.9,
            "parent_id": "some_parent",
        },
    ]

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
    consensus_detections = [
        {
            "detection_id": "xxx",
            "x": 100,
            "y": 200,
            "height": 25,
            "width": 40,
            "class": "a",
            "class_id": 0,
            "confidence": 0.9,
            "parent_id": "some_parent",
        },
        {
            "detection_id": "yyy",
            "x": 100,
            "y": 200,
            "height": 25,
            "width": 40,
            "class": "a",
            "class_id": 2,
            "confidence": 0.9,
            "parent_id": "some_parent",
        },
    ]

    # when
    result = check_objects_presence_in_consensus_detections(
        consensus_detections=consensus_detections,
        class_aware=True,
        aggregation_mode=AggregationMode.AVERAGE,
        required_objects={"a": 2},
    )

    # then
    assert result == (True, {"a": 9 / 10})


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
    detections_from_sources = [
        [
            {
                "detection_id": "d",
                "x": 100,
                "y": 200,
                "height": 30,
                "width": 40,
                "class": "b",
                "class_id": 0,
                "confidence": 0.7,
                "parent_id": "some_parent",
            }
        ],
        [],
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
    assert result == ("some_parent", False, {}, [])


@mock.patch.object(detections_consensus, "uuid4")
def test_agree_on_consensus_for_all_detections_sources_when_predictions_from_single_source_given_but_thats_enough_for_consensus(
    uuid_mock: MagicMock,
) -> None:
    # given
    uuid_mock.return_value = "xxx"
    detections_from_sources = [
        [
            {
                "detection_id": "a",
                "x": 100,
                "y": 200,
                "height": 30,
                "width": 40,
                "class": "b",
                "class_id": 0,
                "confidence": 0.7,
                "parent_id": "some_parent",
            },
            {
                "detection_id": "b",
                "x": 110,
                "y": 200,
                "height": 30,
                "width": 40,
                "class": "b",
                "class_id": 0,
                "confidence": 0.9,
                "parent_id": "some_parent",
            },
        ],
    ]

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
    expected_consensus = [
        {
            "detection_id": "xxx",
            "x": 100,
            "y": 200,
            "height": 30,
            "width": 40,
            "class": "b",
            "class_id": 0,
            "confidence": 0.7,
            "parent_id": "some_parent",
        },
        {
            "detection_id": "xxx",
            "x": 110,
            "y": 200,
            "height": 30,
            "width": 40,
            "class": "b",
            "class_id": 0,
            "confidence": 0.9,
            "parent_id": "some_parent",
        },
    ]
    assert result == ("some_parent", True, {"b": 0.9}, expected_consensus)


@mock.patch.object(detections_consensus, "uuid4")
def test_agree_on_consensus_for_all_detections_sources_when_predictions_from_multiple_sources_given_enough_for_consensus(
    uuid_mock: MagicMock,
) -> None:
    # given
    uuid_mock.return_value = "xxx"
    detections_from_sources = [
        [
            {
                "detection_id": "a",
                "x": 100,
                "y": 200,
                "height": 30,
                "width": 40,
                "class": "b",
                "class_id": 0,
                "confidence": 0.7,
                "parent_id": "some_parent",
            },
            {
                "detection_id": "b",
                "x": 110,
                "y": 200,
                "height": 30,
                "width": 40,
                "class": "b",
                "class_id": 0,
                "confidence": 0.9,
                "parent_id": "some_parent",
            },
        ],
        [
            {
                "detection_id": "c",
                "x": 100,
                "y": 200,
                "height": 34,
                "width": 40,
                "class": "b",
                "class_id": 0,
                "confidence": 0.8,
                "parent_id": "some_parent",
            },
            {
                "detection_id": "d",
                "x": 110,
                "y": 200,
                "height": 36,
                "width": 40,
                "class": "b",
                "class_id": 0,
                "confidence": 1.0,
                "parent_id": "some_parent",
            },
        ],
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
    expected_consensus = [
        {
            "detection_id": "xxx",
            "x": 100,
            "y": 200,
            "height": 32,
            "width": 40,
            "class": "b",
            "class_id": 0,
            "confidence": 75 / 100,
            "parent_id": "some_parent",
        },
        {
            "detection_id": "xxx",
            "x": 110,
            "y": 200,
            "height": 33,
            "width": 40,
            "class": "b",
            "class_id": 0,
            "confidence": 95 / 100,
            "parent_id": "some_parent",
        },
    ]

    assert result == ("some_parent", True, {"b": 0.95}, expected_consensus)
