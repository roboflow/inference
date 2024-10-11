from datetime import datetime
from typing import Tuple, Union

import numpy as np
import pytest
import supervision as sv
from pydantic import ValidationError

from inference.core.workflows.core_steps.transformations.dynamic_crop.v1 import (
    BlockManifest,
    convert_color_to_bgr_tuple,
    crop_image,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    OriginCoordinatesSystem,
    VideoMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize(
    "type_alias", ["roboflow_core/dynamic_crop@v1", "DynamicCrop", "Crop"]
)
@pytest.mark.parametrize("images_field_alias", ["images", "image"])
@pytest.mark.parametrize(
    "background_color", ["$steps.some.color", "$inputs.color", (10, 20, 30), "#fff"]
)
def test_crop_validation_when_valid_manifest_is_given(
    type_alias: str,
    images_field_alias: str,
    background_color: Union[str, Tuple[int, int, int]],
) -> None:
    # given
    data = {
        "type": type_alias,
        "name": "some",
        images_field_alias: "$inputs.image",
        "predictions": "$steps.detection.predictions",
        "mask_opacity": 0.3,
        "background_color": background_color,
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result == BlockManifest(
        type=type_alias,
        name="some",
        images="$inputs.image",
        predictions="$steps.detection.predictions",
        mask_opacity=0.3,
        background_color=background_color,
    )


def test_crop_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "Crop",
        "name": "some",
        "images": "invalid",
        "predictions": "$steps.detection.predictions",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


@pytest.mark.parametrize("mask_opacity", [-0.1, 1.1])
def test_crop_validation_when_invalid_mask_opacity_is_given(
    mask_opacity: float,
) -> None:
    # given
    data = {
        "type": "Crop",
        "name": "some",
        "images": "$inputs.image",
        "predictions": "$steps.detection.predictions",
        "mask_opacity": mask_opacity,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_crop_image() -> None:
    # given
    np_image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    np_image[0:20, 0:20] = 39
    np_image[80:120, 80:120] = 49
    np_image[450:550, 450:550] = 59
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="origin_image"),
        numpy_image=np_image,
        video_metadata=VideoMetadata(
            video_identifier="some",
            frame_number=0,
            frame_timestamp=datetime.now(),
            fps=100,
        ),
    )
    detections = sv.Detections(
        xyxy=np.array(
            [[0, 0, 20, 20], [80, 80, 120, 120], [450, 450, 550, 550]], dtype=np.float64
        ),
        class_id=np.array([1, 1, 1]),
        confidence=np.array([0.5, 0.5, 0.5], dtype=np.float64),
        data={
            "detection_id": np.array(["one", "two", "three"]),
            "class_name": np.array(["cat", "cat", "cat"]),
        },
    )

    # when
    result = crop_image(
        image=image, detections=detections, mask_opacity=0.0, background_color=(0, 0, 0)
    )

    # then
    assert len(result) == 3, "Expected 3 crops to be created"
    assert (
        result[0]["crops"].numpy_image == (np.ones((20, 20, 3), dtype=np.uint8) * 39)
    ).all(), "Image must have expected size and color"
    assert (
        result[0]["crops"].parent_metadata.parent_id == "one"
    ), "Appropriate parent id (from detection id) must be attached"
    assert result[0][
        "crops"
    ].parent_metadata.origin_coordinates == OriginCoordinatesSystem(
        left_top_x=0,
        left_top_y=0,
        origin_width=1000,
        origin_height=1000,
    ), "Appropriate origin coordinates must be attached"
    assert (
        result[0]["crops"].video_metadata.fps == 30
    ), "Expected default video metadata to be generated"
    assert (
        result[1]["crops"].numpy_image == (np.ones((40, 40, 3), dtype=np.uint8) * 49)
    ).all(), "Image must have expected size and color"
    assert (
        result[1]["crops"].parent_metadata.parent_id == "two"
    ), "Appropriate parent id (from detection id) must be attached"
    assert result[1][
        "crops"
    ].parent_metadata.origin_coordinates == OriginCoordinatesSystem(
        left_top_x=80,
        left_top_y=80,
        origin_width=1000,
        origin_height=1000,
    ), "Appropriate origin coordinates must be attached"
    assert (
        result[1]["crops"].video_metadata.fps == 30
    ), "Expected default video metadata to be generated"
    assert (
        result[2]["crops"].numpy_image == (np.ones((100, 100, 3), dtype=np.uint8) * 59)
    ).all(), "Image must have expected size and color"
    assert (
        result[2]["crops"].parent_metadata.parent_id == "three"
    ), "Appropriate parent id (from detection id) must be attached"
    assert result[2][
        "crops"
    ].parent_metadata.origin_coordinates == OriginCoordinatesSystem(
        left_top_x=450,
        left_top_y=450,
        origin_width=1000,
        origin_height=1000,
    ), "Appropriate origin coordinates must be attached"
    assert (
        result[2]["crops"].video_metadata.fps == 30
    ), "Expected default video metadata to be generated"


def test_crop_image_on_empty_detections() -> None:
    # given
    np_image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="origin_image"),
        numpy_image=np_image,
    )
    detections = sv.Detections.empty()

    # when
    result = crop_image(
        image=image, detections=detections, mask_opacity=0.0, background_color=(0, 0, 0)
    )

    # then
    assert result == [], "Expected empty list"


def test_crop_image_on_zero_size_detections() -> None:
    # given
    np_image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="origin_image"),
        numpy_image=np_image,
    )
    detections = sv.Detections(
        xyxy=np.array(
            [[0, 0, 0, 0], [80, 80, 120, 120], [0, 0, 0, 0]], dtype=np.float64
        ),
        class_id=np.array([1, 1, 1]),
        confidence=np.array([0.5, 0.5, 0.5], dtype=np.float64),
        data={
            "detection_id": np.array(["one", "two", "three"]),
            "class_name": np.array(["cat", "cat", "cat"]),
        },
    )

    # when
    result = crop_image(
        image=image, detections=detections, mask_opacity=0.0, background_color=(0, 0, 0)
    )

    # then
    assert len(result) == 3, "Expected 3 outputs"
    assert result[0] == {"crops": None}, "Expected first element empty"
    assert (
        result[1]["crops"].parent_metadata.parent_id == "two"
    ), "Appropriate parent id (from detection id) must be attached"
    assert result[2] == {"crops": None}, "Expected last element empty"


def test_crop_image_when_detections_without_ids_provided() -> None:
    # given
    np_image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="origin_image"),
        numpy_image=np_image,
    )
    detections = sv.Detections(
        xyxy=np.array(
            [[0, 0, 20, 20], [80, 80, 120, 120], [450, 450, 550, 550]], dtype=np.float64
        ),
        class_id=np.array([1, 1, 1]),
        confidence=np.array([0.5, 0.5, 0.5], dtype=np.float64),
        data={
            "class_name": np.array(["cat", "cat", "cat"]),
        },
    )

    # when
    with pytest.raises(ValueError):
        _ = crop_image(
            image=image,
            detections=detections,
            mask_opacity=0.0,
            background_color=(0, 0, 0),
        )


def test_convert_color_to_bgr_tuple_when_valid_tuple_given() -> None:
    # when
    result = convert_color_to_bgr_tuple(color=(255, 0, 0))

    # then
    assert result == (0, 0, 255), "Expected RGB to be converted into BGR"


def test_convert_color_to_bgr_tuple_when_invalid_tuple_given() -> None:
    # when
    with pytest.raises(ValueError):
        _ = convert_color_to_bgr_tuple(color=(256, 0, 0, 0))


def test_convert_color_to_bgr_tuple_when_valid_hex_string_given() -> None:
    # when
    result = convert_color_to_bgr_tuple(color="#ff000A")

    # then
    assert result == (10, 0, 255), "Expected RGB to be converted into BGR"


def test_convert_color_to_bgr_tuple_when_valid_short_hex_string_given() -> None:
    # when
    result = convert_color_to_bgr_tuple(color="#f0A")

    # then
    assert result == (170, 0, 255), "Expected RGB to be converted into BGR"


def test_convert_color_to_bgr_tuple_when_invalid_hex_string_given() -> None:
    # when
    with pytest.raises(ValueError):
        _ = convert_color_to_bgr_tuple(color="#invalid")


def test_convert_color_to_bgr_tuple_when_tuple_string_given() -> None:
    # when
    result = convert_color_to_bgr_tuple(color="(255, 0, 128)")

    # then
    assert result == (128, 0, 255), "Expected RGB to be converted into BGR"


def test_convert_color_to_bgr_tuple_when_invalid_tuple_string_given() -> None:
    # when
    with pytest.raises(ValueError):
        _ = convert_color_to_bgr_tuple(color="(255, 0, a)")


def test_convert_color_to_bgr_tuple_when_invalid_value() -> None:
    # when
    with pytest.raises(ValueError):
        _ = convert_color_to_bgr_tuple(color="invalid")


def test_crop_image_when_background_removal_requested_and_mask_not_found() -> None:
    # given
    np_image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    np_image[0:20, 0:20] = 39
    np_image[80:120, 80:120] = 49
    np_image[450:550, 450:550] = 59
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="origin_image"),
        numpy_image=np_image,
    )
    detections = sv.Detections(
        xyxy=np.array(
            [[0, 0, 20, 20], [80, 80, 120, 120], [450, 450, 550, 550]], dtype=np.float64
        ),
        class_id=np.array([1, 1, 1]),
        confidence=np.array([0.5, 0.5, 0.5], dtype=np.float64),
        data={
            "detection_id": np.array(["one", "two", "three"]),
            "class_name": np.array(["cat", "cat", "cat"]),
        },
    )

    # when
    result = crop_image(
        image=image, detections=detections, mask_opacity=1.0, background_color=(0, 0, 0)
    )

    # then
    assert len(result) == 3, "Expected 3 crops to be created"
    assert (
        result[0]["crops"].numpy_image == (np.ones((20, 20, 3), dtype=np.uint8) * 39)
    ).all(), "Image must have expected size and color"
    assert (
        result[1]["crops"].numpy_image == (np.ones((40, 40, 3), dtype=np.uint8) * 49)
    ).all(), "Image must have expected size and color"
    assert (
        result[2]["crops"].numpy_image == (np.ones((100, 100, 3), dtype=np.uint8) * 59)
    ).all(), "Image must have expected size and color"


def test_crop_image_when_background_removal_requested_and_mask_found() -> None:
    # given
    np_image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    np_image[0:20, 0:20] = 39
    np_image[80:120, 80:120] = 49
    np_image[450:550, 450:550] = 59
    mask = np.zeros((3, 1000, 1000), dtype=np.bool_)
    mask[0, 0:15, 0:15] = 1
    mask[1, 80:90, 80:90] = 1
    mask[2, 450:460, 450:460] = 1
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="origin_image"),
        numpy_image=np_image,
    )
    detections = sv.Detections(
        xyxy=np.array(
            [[0, 0, 20, 20], [80, 80, 120, 120], [450, 450, 550, 550]], dtype=np.float64
        ),
        class_id=np.array([1, 1, 1]),
        mask=mask,
        confidence=np.array([0.5, 0.5, 0.5], dtype=np.float64),
        data={
            "detection_id": np.array(["one", "two", "three"]),
            "class_name": np.array(["cat", "cat", "cat"]),
        },
    )

    # when
    result = crop_image(
        image=image,
        detections=detections,
        mask_opacity=1.0,
        background_color=(127, 127, 127),
    )

    # then
    assert len(result) == 3, "Expected 3 crops to be created"
    expected_first_crop = np.ones((20, 20, 3), dtype=np.uint8) * 127
    expected_first_crop[0:15, 0:15, :] = 39
    assert (
        result[0]["crops"].numpy_image == expected_first_crop
    ).all(), "Image must have expected size and color"
    expected_second_crop = np.ones((40, 40, 3), dtype=np.uint8) * 127
    expected_second_crop[0:10, 0:10, :] = 49
    assert (
        result[1]["crops"].numpy_image == expected_second_crop
    ).all(), "Image must have expected size and color"
    expected_third_crop = np.ones((100, 100, 3), dtype=np.uint8) * 127
    expected_third_crop[0:10, 0:10, :] = 59
    assert (
        result[2]["crops"].numpy_image == expected_third_crop
    ).all(), "Image must have expected size and color"
