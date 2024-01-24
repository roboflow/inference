from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest

from inference.enterprise.deployments.complier.steps_executors import auxiliary
from inference.enterprise.deployments.complier.steps_executors.auxiliary import (
    crop_image,
    extract_origin_size_from_images,
    offset_detection,
    run_condition_step,
    run_detection_filter,
    take_static_crop, aggregate_field_values, get_detection_sizes, get_largest_bounding_box, get_smallest_bounding_box,
    get_average_bounding_box, get_majority_class, get_class_of_most_confident_detection,
    get_class_of_least_confident_detection, merge_detections, detection_to_xyxy, calculate_iou, enumerate_detections,
)
from inference.enterprise.deployments.entities.steps import (
    AbsoluteStaticCrop,
    Condition,
    DetectionFilter,
    Operator,
    RelativeStaticCrop, AggregationMode,
)


def test_crop_image() -> None:
    # given
    image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    origin_size = {"height": 1000, "width": 1000}
    detections = [
        {"x": 10, "y": 10, "width": 20, "height": 20, "detection_id": "one"},
        {"x": 100, "y": 100, "width": 40, "height": 40, "detection_id": "two"},
        {"x": 500, "y": 500, "width": 100, "height": 100, "detection_id": "three"},
    ]
    image[0:20, 0:20] = 39
    image[80:120, 80:120] = 49
    image[450:550, 450:550] = 59

    # when
    result = crop_image(image=image, detections=detections, origin_size=origin_size)

    # then
    assert len(result) == 3, "Expected 3 crops to be created"
    assert (
        result[0]["type"] == "numpy_object"
    ), "Type of image is expected to be numpy_object"
    assert (
        result[0]["value"] == (np.ones((20, 20, 3), dtype=np.uint8) * 39)
    ).all(), "Image must have expected size and color"
    assert (
        result[0]["parent_id"] == "one"
    ), "Appropriate parent id (from detection id) must be attached"
    assert result[0]["origin_coordinates"] == {
        "center_x": 10,
        "center_y": 10,
        "origin_image_size": {"height": 1000, "width": 1000},
    }, "Appropriate origin coordinates must be attached"
    assert (
        result[1]["type"] == "numpy_object"
    ), "Type of image is expected to be numpy_object"
    assert (
        result[1]["value"] == (np.ones((40, 40, 3), dtype=np.uint8) * 49)
    ).all(), "Image must have expected size and color"
    assert (
        result[1]["parent_id"] == "two"
    ), "Appropriate parent id (from detection id) must be attached"
    assert result[1]["origin_coordinates"] == {
        "center_x": 100,
        "center_y": 100,
        "origin_image_size": {"height": 1000, "width": 1000},
    }, "Appropriate origin coordinates must be attached"
    assert (
        result[2]["type"] == "numpy_object"
    ), "Type of image is expected to be numpy_object"
    assert (
        result[2]["value"] == (np.ones((100, 100, 3), dtype=np.uint8) * 59)
    ).all(), "Image must have expected size and color"
    assert (
        result[2]["parent_id"] == "three"
    ), "Appropriate parent id (from detection id) must be attached"
    assert result[2]["origin_coordinates"] == {
        "center_x": 500,
        "center_y": 500,
        "origin_image_size": {"height": 1000, "width": 1000},
    }, "Appropriate origin coordinates must be attached"


@pytest.mark.asyncio
async def test_run_condition_step() -> None:
    # given
    step = Condition(
        type="Condition",
        name="step_1",
        left="$inputs.some",
        operator=Operator.EQUAL,
        right="$steps.step_0.top",
        step_if_true="$steps.step_2",
        step_if_false="$steps.step_3",
    )

    # when
    next_step, outputs_lookup = await run_condition_step(
        step=step,
        runtime_parameters={"some": "cat"},
        outputs_lookup={"$steps.step_0": {"top": "cat"}},
        model_manager=MagicMock(),
        api_key=None,
    )

    # then
    assert outputs_lookup == {
        "$steps.step_0": {"top": "cat"}
    }, "Output lookup must not be modified"
    assert next_step == "$steps.step_2"


@pytest.mark.asyncio
async def test_run_detection_filter_step_when_single_image_detections_given() -> None:
    # given
    step = DetectionFilter.parse_obj(
        {
            "type": "DetectionFilter",
            "name": "step_2",
            "predictions": "$steps.step_1.predictions",
            "filter_definition": {
                "type": "CompoundDetectionFilterDefinition",
                "left": {
                    "type": "DetectionFilterDefinition",
                    "field_name": "class_name",
                    "operator": "equal",
                    "reference_value": "car",
                },
                "operator": "and",
                "right": {
                    "type": "DetectionFilterDefinition",
                    "field_name": "confidence",
                    "operator": "greater_or_equal_than",
                    "reference_value": 0.5,
                },
            },
        }
    )
    detections = [
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
    ]

    # when
    next_step, outputs_lookup = await run_detection_filter(
        step=step,
        runtime_parameters={},
        outputs_lookup={
            "$steps.step_1": {
                "predictions": detections,
                "image": {"height": 100, "width": 100},
            }
        },
        model_manager=MagicMock(),
        api_key=None,
    )

    # then
    assert next_step is None, "Next step should not be set here"
    assert outputs_lookup["$steps.step_2"]["predictions"] == [
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
    ], "Only second prediction should survive"
    assert outputs_lookup["$steps.step_2"]["parent_id"] == [
        "p2"
    ], "Only second prediction should mark parent_id"
    assert outputs_lookup["$steps.step_2"]["image"] == {
        "height": 100,
        "width": 100,
    }, "image metadata must be copied from input"


@pytest.mark.asyncio
async def test_run_detection_filter_step_when_batch_detections_given() -> None:
    # given
    step = DetectionFilter.parse_obj(
        {
            "type": "DetectionFilter",
            "name": "step_2",
            "predictions": "$steps.step_1.predictions",
            "filter_definition": {
                "type": "CompoundDetectionFilterDefinition",
                "left": {
                    "type": "DetectionFilterDefinition",
                    "field_name": "class_name",
                    "operator": "equal",
                    "reference_value": "car",
                },
                "operator": "and",
                "right": {
                    "type": "DetectionFilterDefinition",
                    "field_name": "confidence",
                    "operator": "greater_or_equal_than",
                    "reference_value": 0.5,
                },
            },
        }
    )
    detections = [
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

    # when
    next_step, outputs_lookup = await run_detection_filter(
        step=step,
        runtime_parameters={},
        outputs_lookup={
            "$steps.step_1": {
                "predictions": detections,
                "image": [{"height": 100, "width": 100}] * 2,
            }
        },
        model_manager=MagicMock(),
        api_key=None,
    )

    # then
    assert next_step is None, "Next step should not be set here"
    assert outputs_lookup["$steps.step_2"][0]["predictions"] == [
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
    assert outputs_lookup["$steps.step_2"][1]["predictions"] == [
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
    assert outputs_lookup["$steps.step_2"][0]["parent_id"] == [
        "p2"
    ], "Only second prediction in each batch should mark parent_id"
    assert outputs_lookup["$steps.step_2"][1]["parent_id"] == [
        "p4"
    ], "Only second prediction in each batch should mark parent_id"
    assert outputs_lookup["$steps.step_2"][0]["image"] == {
        "height": 100,
        "width": 100,
    }, "image metadata must be copied from input"
    assert outputs_lookup["$steps.step_2"][1]["image"] == {
        "height": 100,
        "width": 100,
    }, "image metadata must be copied from input"


def test_offset_detection() -> None:
    # given
    detection = {
        "x": 100,
        "y": 200,
        "width": 20,
        "height": 20,
        "parent_id": "p2",
        "detection_id": "two",
        "class_name": "car",
        "confidence": 0.5,
    }

    # when
    result = offset_detection(
        detection=detection,
        offset_x=50,
        offset_y=100,
    )

    # then
    assert result["x"] == 100, "OX center should not be changed"
    assert result["y"] == 200, "OY center should not be changed"
    assert result["width"] == 70, "Width should be offset by 50px"
    assert result["height"] == 120, "Height should be offset by 100px"
    assert (
        result["parent_id"] == "two"
    ), "Parent id should be set to origin detection id"
    assert (
        result["detection_id"] != detection["detection_id"]
    ), "New detection id (random) must be assigned"


def test_extract_origin_size_from_images() -> None:
    # given
    input_images = [
        {
            "type": "url",
            "value": "https://some/image.jpg",
            "origin_coordinates": {
                "origin_image_size": {"height": 1000, "width": 1000}
            },
        },
        {
            "type": "numpy_object",
            "value": np.zeros((192, 168, 3), dtype=np.uint8),
        },
    ]
    decoded_image = [
        np.zeros((200, 200, 3), dtype=np.uint8),
        np.zeros((192, 168, 3), dtype=np.uint8),
    ]

    # when
    result = extract_origin_size_from_images(
        input_images=input_images,
        decoded_images=decoded_image,
    )

    # then
    assert result == [{"height": 1000, "width": 1000}, {"height": 192, "width": 168}]


def test_take_relative_static_crop() -> None:
    # given
    crop = RelativeStaticCrop(
        type="RelativeStaticCrop",
        name="some",
        image="$inputs.image",
        x_center=0.5,
        y_center=0.6,
        width=0.1,
        height=0.2,
    )
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[50:70, 45:55] = 30  # painted the crop into (30, 30, 30)

    # when
    result = take_static_crop(
        image=image,
        crop=crop,
        runtime_parameters={},
        outputs_lookup={},
        origin_size={"height": 100, "width": 100},
    )

    # then
    assert (
        result["value"] == (np.ones((20, 10, 3), dtype=np.uint8) * 30)
    ).all(), "Crop must have the exact size and color"
    assert (
        result["parent_id"] == "$steps.some"
    ), "Parent must be set at crop step identifier"
    assert result["origin_coordinates"] == {
        "center_x": 50,
        "center_y": 60,
        "origin_image_size": {"height": 100, "width": 100},
    }, "Origin coordinates of crop and image size metadata must be preserved through the operation"


def test_take_absolute_static_crop() -> None:
    # given
    crop = AbsoluteStaticCrop(
        type="AbsoluteStaticCrop",
        name="some",
        image="$inputs.image",
        x_center=50,
        y_center=60,
        width=10,
        height=20,
    )
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[50:70, 45:55] = 30  # painted the crop into (30, 30, 30)

    # when
    result = take_static_crop(
        image=image,
        crop=crop,
        runtime_parameters={},
        outputs_lookup={},
        origin_size={"height": 100, "width": 100},
    )

    # then
    assert (
        result["value"] == (np.ones((20, 10, 3), dtype=np.uint8) * 30)
    ).all(), "Crop must have the exact size and color"
    assert (
        result["parent_id"] == "$steps.some"
    ), "Parent must be set at crop step identifier"
    assert result["origin_coordinates"] == {
        "center_x": 50,
        "center_y": 60,
        "origin_image_size": {"height": 100, "width": 100},
    }, "Origin coordinates of crop and image size metadata must be preserved through the operation"


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
    "mode",
    [AggregationMode.MIN, AggregationMode.MAX, AggregationMode.AVERAGE]
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


def test_get_detection_sizes_when_empty_input_provided() -> None:
    # when
    result = get_detection_sizes(detections=[])

    # then
    assert result == []


def test_get_detection_sizes_when_non_empty_input_provided() -> None:
    # given
    detections = [
        {"height": 30, "width": 40},
        {"height": 40, "width": 40}
    ]

    # when
    result = get_detection_sizes(detections=detections)

    # then
    assert result == [1200, 1600]


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


@mock.patch.object(auxiliary, "uuid4")
def test_merge_detections(uuid4_mock: MagicMock) -> None:
    # given
    uuid4_mock.return_value = "some_uuid"
    detections = [
        {"parent_id": "x", "class": "a", "class_id": 0, "confidence": 1/10, "x": 100, "y": 200, "height": 30, "width": 40},
        {"parent_id": "x", "class": "a", "class_id": 0, "confidence": 3/10, "x": 110, "y": 210, "height": 40, "width": 50},
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
        "confidence": 2/10,
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
    assert abs(result - 1/7) < 1e-5


def test_enumerate_detections_when_no_predictions_given() -> None:
    # when
    result = list(enumerate_detections(predictions=[]))

    # then
    assert result == []


def test_enumerate_detections_when_source_with_no_predictions_given() -> None:
    # given
    source_a = []
    source_b = [{"a": 1}, {"b": 1}]

    # when
    result = list(enumerate_detections(predictions=[source_a, source_b]))

    # then
    assert result == [(1, {"a": 1}), (1, {"b": 1})]


def test_enumerate_detections_when_sources_with_predictions_given() -> None:
    # given
    source_a = [{"a": 1}, {"b": 1}]
    source_b = [{"c": 1}, {"d": 1}]

    # when
    result = list(enumerate_detections(predictions=[source_a, source_b]))

    # then
    assert result == [(0, {"a": 1}), (0, {"b": 1}), (1, {"c": 1}), (1, {"d": 1})]


def test_enumerate_detections_when_sources_with_predictions_given_and_source_to_be_excluded() -> None:
    # given
    source_a = [{"a": 1}, {"b": 1}]
    source_b = [{"c": 1}, {"d": 1}]

    # when
    result = list(enumerate_detections(
        predictions=[source_a, source_b],
        excluded_source=1,
    ))

    # then
    assert result == [(0, {"a": 1}), (0, {"b": 1})]
