import pytest
import supervision as sv

from inference.core.workflows.core_steps.visualizations.common.utils import str_to_color


def test_str_to_color_with_hex_color() -> None:
    # given
    color = "#FF0000"

    # when
    result = str_to_color(color)

    # then
    assert result == sv.Color.from_hex(color)


def test_str_to_color_with_rgb_color() -> None:
    # given
    color = "rgb(255, 0, 0)"
    expected_color = sv.Color.from_rgb_tuple((255, 0, 0))

    # when
    result = str_to_color(color)

    # then
    assert result == expected_color


def test_str_to_color_with_bgr_color() -> None:
    # given
    color = "bgr(0, 0, 255)"
    expected_color = sv.Color.from_bgr_tuple((0, 0, 255))

    # when
    result = str_to_color(color)

    # then
    assert result == expected_color


def test_str_to_color_with_color_name() -> None:
    # given
    color = "WHITE"

    # when
    result = str_to_color(color)

    # then
    assert result == sv.Color.WHITE


def test_str_to_color_with_invalid_color() -> None:
    # given
    color = "invalid"

    # when
    with pytest.raises(ValueError):
        _ = str_to_color(color)
