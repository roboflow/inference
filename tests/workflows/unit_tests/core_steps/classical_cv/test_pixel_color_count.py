import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.classical_cv.pixel_color_count.v1 import (
    ColorPixelCountManifest,
    PixelationCountBlockV1,
    convert_color_to_bgr_tuple,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_pixelation_validation_when_valid_manifest_is_given(
    images_field_alias: str,
) -> None:
    # given
    data = {
        "type": "roboflow_core/pixel_color_count@v1",  # Correct type
        "name": "pixelation1",
        images_field_alias: "$inputs.image",
        "target_color": (255, 0, 0),  # Add target_color field
    }

    # when
    result = ColorPixelCountManifest.model_validate(data)

    # then
    assert result == ColorPixelCountManifest(
        type="roboflow_core/pixel_color_count@v1",
        name="pixelation1",
        images="$inputs.image",
        target_color=(255, 0, 0),
    )


def test_pixelation_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "roboflow_core/pixel_color_count@v1",  # Correct type
        "name": "pixelation1",
        "images": "invalid",
        "target_color": (255, 0, 0),  # Add target_color field
    }

    # when
    with pytest.raises(ValidationError):
        _ = ColorPixelCountManifest.model_validate(data)


def test_pixelation_block() -> None:
    # given
    block = PixelationCountBlockV1()
    image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    image[0:100, 0:100] = (0, 0, 245)
    image[0:10, 0:10] = (0, 0, 255)

    # when
    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=image,
        ),
        target_color=(255, 0, 0),
        tolerance=10,
    )

    assert output is not None
    assert output["matching_pixels_count"] == 100 * 100, (
        "Expected 100*100 square to be matched, as 100 pixels match dominant color, "
        "and remaining are within margin"
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
