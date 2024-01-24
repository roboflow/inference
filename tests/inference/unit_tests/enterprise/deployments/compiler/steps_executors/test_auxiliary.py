from unittest.mock import MagicMock

import numpy as np
import pytest

from inference.enterprise.deployments.complier.steps_executors.auxiliary import (
    crop_image,
    run_condition_step,
    run_detection_filter,
    offset_detection,
    extract_origin_size_from_images,
    take_static_crop,
)
from inference.enterprise.deployments.entities.steps import (
    Condition,
    Operator,
    DetectionFilter,
    RelativeStaticCrop,
    AbsoluteStaticCrop,
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
