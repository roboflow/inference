from typing import Tuple, Union

import pytest

from inference.core.workflows.core_steps.transformations.masked_crop.v1 import (
    BlockManifest,
    convert_color_to_bgr_tuple,
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
