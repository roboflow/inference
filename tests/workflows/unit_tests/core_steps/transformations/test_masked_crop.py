from typing import Tuple, Union

import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.transformations.masked_crop.v1 import (
    BlockManifest,
    convert_color_to_bgr_tuple,
    mask_and_crop_image,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    OriginCoordinatesSystem,
    WorkflowImageData,
)


@pytest.mark.parametrize("image_field", ["image", "images"])
@pytest.mark.parametrize(
    "background_color", ["$steps.some.color", "$inputs.color", (10, 20, 30), "#fff"]
)
def test_manifest_parsing_when_input_data_valid(
    image_field: str,
    background_color: Union[str, Tuple[int, int, int]],
) -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/masked_crop@v1",
        "name": "crop",
        image_field: "$inputs.image",
        "predictions": "$steps.model.predictions",
        "background_color": background_color,
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result == BlockManifest(
        type="roboflow_core/masked_crop@v1",
        name="crop",
        images="$inputs.image",
        predictions="$steps.model.predictions",
        background_color=background_color,
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


def test_mask_and_crop_image_when_empty_detection_found() -> None:
    # given
    np_image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="origin_image"),
        numpy_image=np_image,
    )
    detections = sv.Detections.empty()

    # when
    result = mask_and_crop_image(
        image=image,
        detections=detections,
        background_color=(127, 127, 127),
    )

    # then
    assert len(result) == 0, "Expected 0 output crop"


def test_mask_and_crop_image_when_detection_without_mask_found() -> None:
    # given
    np_image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="origin_image"),
        numpy_image=np_image,
    )
    detections = sv.Detections(
        xyxy=np.array([[0, 0, 0, 0]], dtype=np.float64),
        class_id=np.array([1]),
        confidence=np.array([0.5], dtype=np.float64),
        data={
            "detection_id": np.array(["one"]),
            "class_name": np.array(["cat"]),
        },
    )

    # when
    with pytest.raises(ValueError):
        _ = mask_and_crop_image(
            image=image,
            detections=detections,
            background_color=(127, 127, 127),
        )


def test_mask_and_crop_image_when_detection_without_detection_id_found() -> None:
    # given
    np_image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="origin_image"),
        numpy_image=np_image,
    )
    detections = sv.Detections(
        xyxy=np.array([[0, 0, 0, 0]], dtype=np.float64),
        class_id=np.array([1]),
        mask=np.zeros((1, 1000, 1000), dtype=np.bool_),
        confidence=np.array([0.5], dtype=np.float64),
        data={
            "class_name": np.array(["cat"]),
        },
    )

    # when
    with pytest.raises(ValueError):
        _ = mask_and_crop_image(
            image=image,
            detections=detections,
            background_color=(127, 127, 127),
        )


def test_mask_and_crop_image_when_detection_with_zero_size_mask_found() -> None:
    # given
    np_image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="origin_image"),
        numpy_image=np_image,
    )
    detections = sv.Detections(
        xyxy=np.array([[0, 0, 0, 0]], dtype=np.float64),
        mask=np.zeros((1, 1000, 1000), dtype=np.bool_),
        class_id=np.array([1]),
        confidence=np.array([0.5], dtype=np.float64),
        data={
            "detection_id": np.array(["one"]),
            "class_name": np.array(["cat"]),
        },
    )

    # when
    result = mask_and_crop_image(
        image=image,
        detections=detections,
        background_color=(127, 127, 127),
    )

    # then
    assert len(result) == 1, "Expected 1 output crop"
    assert result[0] == {"crops": None}, "Expected first element empty"


def test_mask_and_crop_image_when_valid_detection_found() -> None:
    # given
    np_image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    np_image[0:20, 0:20] = 39
    np_image[80:120, 80:120] = 49
    np_image[450:550, 450:550] = 59
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="origin_image"),
        numpy_image=np_image,
    )
    mask = np.zeros((3, 1000, 1000), dtype=np.bool_)
    mask[0, 0:15, 0:15] = 1
    mask[1, 80:90, 80:90] = 1
    mask[2, 450:460, 450:460] = 1
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
    result = mask_and_crop_image(
        image=image,
        detections=detections,
        background_color=(127, 127, 127),
    )

    # then
    assert len(result) == 3, "Expected 3 crops to be created"
    expected_first_crop = np.ones((20, 20, 3), dtype=np.uint8) * 127
    expected_first_crop[0:15, 0:15, :] = 39
    assert (
        result[0]["crops"].numpy_image == expected_first_crop
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
    expected_second_crop = np.ones((40, 40, 3), dtype=np.uint8) * 127
    expected_second_crop[0:10, 0:10, :] = 49
    assert (
        result[1]["crops"].numpy_image == expected_second_crop
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
    expected_third_crop = np.ones((100, 100, 3), dtype=np.uint8) * 127
    expected_third_crop[0:10, 0:10, :] = 59
    assert (
        result[2]["crops"].numpy_image == expected_third_crop
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
