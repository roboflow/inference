"""Colors of the annotation blocks: palettes, color strings and per-row lookup.

Oracle: the latest V1 classes the blocks port (``ColorableVisualizationBlock``
palettes, ``NegativeSafeColorPalette`` lookup, ``str_to_color``), all exact::

    palette index of row i  = class_id[i] | i | tracker_id[i]   (CLASS|INDEX|TRACK)
    color                   = GREY if index < 0 else palette[index % len(palette)]

The host lookup (``colors_of``) and the device lookup (``device_colors``) must
agree with V1 and with each other, including ids beyond float64 precision.
"""

from typing import List

import numpy as np
import pytest
import supervision as sv
import torch
from roboflow_workflows.core_steps.visualizations.common.base_colorable import (
    ColorableVisualizationBlock,
)
from roboflow_workflows.core_steps.visualizations.common.utils import (
    NegativeSafeColorPalette,
    str_to_color,
)
from roboflow_workflows.v2.blocks.visualization.annotations.blend import device_colors
from roboflow_workflows.v2.blocks.visualization.annotations.colors import (
    COLOR_AXES,
    PALETTE_NAMES,
    UNKNOWN_COLOR,
    colors_of,
    palette_indices,
    parse_color,
    resolve_palette,
)

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
DEVICES += [
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    )
]
GREY = (128, 128, 128)
# Ids around float64's exact-integer limit and the int64 extremes.
LARGE_IDS = [2**53 - 1, 2**53, 2**53 + 1, 2**62 + 3, 2**63 - 1, -(2**63)]


def v1_rgb_colors(
    name: str, *, palette_size: int, custom_colors: List[str]
) -> np.ndarray:
    palette = ColorableVisualizationBlock.getPalette(name, palette_size, custom_colors)
    colors = np.asarray([color.as_rgb() for color in palette.colors], dtype=np.uint8)

    return colors


@pytest.mark.parametrize(
    "text",
    [
        "#FF8000",
        "#f80",
        "#ff800080",
        "rgb(1, 2, 3)",
        "rgb(255,0,10)",
        "bgr(1, 2, 3)",
        "WHITE",
        "black",
        "Roboflow",
        "grey",
    ],
)
def test_parse_color_equals_v1_str_to_color(text: str) -> None:
    expected = str_to_color(text).as_rgb()

    assert parse_color(text, field="color") == expected


def test_parse_color_reads_bgr_in_reverse_order() -> None:
    # Hand-computed: bgr(B, G, R) names blue first.
    assert parse_color("bgr(10, 20, 30)", field="color") == (30, 20, 10)
    assert parse_color("rgb(10, 20, 30)", field="color") == (10, 20, 30)


@pytest.mark.parametrize(
    "text", ["", "#GG0000", "rgb(1, 2)", "bgr(a, b, c)", "PURPLISH", "from_hex"]
)
def test_parse_color_rejects_unknown_formats_naming_the_field(text: str) -> None:
    with pytest.raises(ValueError, match=r"text_color: invalid color") as error:
        parse_color(text, field="text_color")

    assert "#RRGGBB" in str(error.value)


@pytest.mark.parametrize("value", [None, 255, (255, 0, 0)])
def test_parse_color_rejects_non_strings(value: object) -> None:
    with pytest.raises(ValueError, match="color must be a color string"):
        parse_color(value, field="color")


@pytest.mark.parametrize("name", [name for name in PALETTE_NAMES if name != "CUSTOM"])
@pytest.mark.parametrize("palette_size", [1, 7])
def test_named_palettes_equal_v1_palettes(name: str, palette_size: int) -> None:
    expected = v1_rgb_colors(name, palette_size=palette_size, custom_colors=[])

    palette = resolve_palette(name, palette_size=palette_size, custom_colors=[])

    np.testing.assert_array_equal(palette, expected)
    assert palette.dtype == np.uint8 and palette.shape == (len(expected), 3)


def test_custom_palette_keeps_the_given_colors_in_order() -> None:
    custom = ["#FF0000", "rgb(0, 255, 0)", "bgr(255, 0, 0)", "YELLOW"]

    palette = resolve_palette("CUSTOM", palette_size=99, custom_colors=custom)

    expected = [[255, 0, 0], [0, 255, 0], [0, 0, 255], list(sv.Color.YELLOW.as_rgb())]
    np.testing.assert_array_equal(palette, np.asarray(expected, dtype=np.uint8))
    np.testing.assert_array_equal(
        palette, v1_rgb_colors("CUSTOM", palette_size=99, custom_colors=custom)
    )


def test_resolved_palettes_cannot_be_changed_through_lookups() -> None:
    palette = resolve_palette("DEFAULT", palette_size=1, custom_colors=[])
    colors = colors_of(palette, np.array([0, 1]))
    colors[:] = 0

    again = resolve_palette("DEFAULT", palette_size=1, custom_colors=[])

    assert not palette.flags.writeable
    np.testing.assert_array_equal(
        again[:2], v1_rgb_colors("DEFAULT", palette_size=1, custom_colors=[])[:2]
    )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"name": "SUNSET"}, "color_palette must be one of"),
        ({"name": "DEFAULT", "palette_size": 0}, "palette_size must be at least 1"),
        ({"name": "DEFAULT", "palette_size": True}, "palette_size must be an integer"),
        ({"name": "DEFAULT", "palette_size": 2.0}, "palette_size must be an integer"),
        ({"name": "CUSTOM", "custom_colors": []}, "CUSTOM needs custom_colors"),
        ({"name": "CUSTOM", "custom_colors": "#FF0000"}, "must be a list of colors"),
        (
            {"name": "CUSTOM", "custom_colors": ["#FF0000", "nope"]},
            r"custom_colors\[1\]: invalid color 'nope'",
        ),
    ],
)
def test_resolve_palette_rejects_invalid_parameters(kwargs: dict, message: str) -> None:
    arguments = {"palette_size": 3, "custom_colors": [], **kwargs}

    with pytest.raises(ValueError, match=message):
        resolve_palette(arguments.pop("name"), **arguments)


def test_row_colors_equal_v1_negative_safe_palette() -> None:
    palette = resolve_palette("DEFAULT", palette_size=1, custom_colors=[])
    v1 = NegativeSafeColorPalette(colors=list(sv.ColorPalette.DEFAULT.colors))
    indices = np.array(
        [0, 1, len(palette) - 1, len(palette), 3 * len(palette) + 2, -1, -7]
    )

    colors = colors_of(palette, indices)

    expected = [v1.by_idx(int(index)).as_rgb() for index in indices]
    np.testing.assert_array_equal(colors, np.asarray(expected, dtype=np.uint8))
    assert UNKNOWN_COLOR == GREY


@pytest.mark.parametrize("ids", [LARGE_IDS, [-1, -(2**40), 0, 5]])
@pytest.mark.parametrize("device", DEVICES)
def test_host_and_device_lookups_agree_with_exact_integer_modulo(
    ids: List[int],
    device: str,
) -> None:
    palette = np.asarray([[10, 0, 0], [0, 20, 0], [0, 0, 30]], dtype=np.uint8)
    expected = np.asarray(
        [GREY if index < 0 else palette[index % 3] for index in ids], dtype=np.uint8
    )

    host = colors_of(palette, np.asarray(ids, dtype=np.int64))
    colors = device_colors(palette, torch.tensor(ids, dtype=torch.int64, device=device))

    np.testing.assert_array_equal(host, expected)
    np.testing.assert_array_equal(colors.cpu().numpy(), expected)
    assert colors.device.type == device


@pytest.mark.parametrize("device", DEVICES)
def test_device_colors_keep_index_shape_device_and_handle_empty(device: str) -> None:
    palette = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    grid = torch.tensor([[0, 1, -1], [2, 3, -2]], device=device)

    colors = device_colors(palette, grid)
    empty = device_colors(palette, torch.zeros(0, dtype=torch.int64, device=device))

    assert colors.shape == (2, 3, 3) and colors.dtype == torch.uint8
    assert colors.device.type == device
    np.testing.assert_array_equal(
        colors.cpu().numpy(),
        [[[1, 2, 3], [4, 5, 6], GREY], [[1, 2, 3], [4, 5, 6], GREY]],
    )
    assert empty.shape == (0, 3)


def test_empty_rows_have_empty_indices_and_colors() -> None:
    palette = resolve_palette("ROBOFLOW", palette_size=1, custom_colors=[])

    for axis in ("CLASS", "INDEX"):
        indices = palette_indices(axis, class_id=np.zeros(0, np.int64), tracker_id=None)
        colors = colors_of(palette, indices)
        assert indices.shape == (0,) and colors.shape == (0, 3)


@pytest.mark.parametrize("axis", COLOR_AXES)
def test_palette_indices_follow_the_color_axis(axis: str) -> None:
    class_id = np.array([4, -1, 2**53 + 1], dtype=np.int64)
    tracker_id = np.array([7, 2**62 + 1, -3], dtype=np.int64)
    expected = {
        "CLASS": class_id,
        "INDEX": np.array([0, 1, 2]),
        "TRACK": tracker_id,
    }[axis]

    host = palette_indices(axis, class_id=class_id, tracker_id=tracker_id)
    device = palette_indices(
        axis, class_id=torch.from_numpy(class_id), tracker_id=tracker_id.tolist()
    )

    np.testing.assert_array_equal(host, expected)
    np.testing.assert_array_equal(device.numpy(), expected)
    assert host.dtype == np.int64 and device.dtype == torch.int64


def test_track_axis_without_tracker_ids_asks_for_a_tracker() -> None:
    class_id = np.array([0, 1])

    with pytest.raises(ValueError, match="add a tracker step"):
        palette_indices("TRACK", class_id=class_id, tracker_id=None)
    with pytest.raises(ValueError, match="add a tracker step"):
        palette_indices("TRACK", class_id=torch.from_numpy(class_id), tracker_id=None)


def test_unknown_color_axis_is_rejected_on_host_and_device() -> None:
    class_id = np.array([0])

    with pytest.raises(ValueError, match="color_axis must be one of"):
        palette_indices("PER_ROW", class_id=class_id, tracker_id=None)
    with pytest.raises(ValueError, match="color_axis must be one of"):
        palette_indices("PER_ROW", class_id=torch.from_numpy(class_id), tracker_id=None)
