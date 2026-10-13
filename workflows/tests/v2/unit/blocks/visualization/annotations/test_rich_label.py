"""``v2/rich_label_visualization@v1`` against the latest V1 block.

The V1 oracle (``support.run_v1``) gets exactly the validated V2 values;
labels, fonts, wrapping, anchors and edge clipping must be pixel-equal.
Fonts are the approved fonts installed with ``roboflow_workflows``.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

import numpy as np
import pytest
from PIL import ImageFont
from roboflow_workflows.core_steps.visualizations.common import fonts
from roboflow_workflows.core_steps.visualizations.common.fonts.registry import (
    FONTS_REGISTRY,
)
from roboflow_workflows.core_steps.visualizations.rich_label.v1 import (
    RichLabelVisualizationBlockV1,
)
from roboflow_workflows.v2.blocks.visualization.annotations import rich_label
from roboflow_workflows.v2.blocks.visualization.annotations.rich_label import (
    RichLabelVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.text import background_rect
from roboflow_workflows.v2.engine.declaration import spec_of
from roboflow_workflows.v2.engine.errors import ParamsValidationError

from tests.v2.unit.blocks.visualization.annotations.support import (
    detections,
    image_of,
    instance_detections,
    object_masks,
    pixels_of,
    run_v1,
    scene,
    v1_detections,
)

HEIGHT, WIDTH = 96, 128
TRACKERS = [{"tracker_id": 7}, {"tracker_id": 3}, {"tracker_id": 7}]


def validated(**parameters: Any) -> Dict[str, Any]:
    params = spec_of(RichLabelVisualizationBlock).validate_params(
        {"image": "$inputs.image", "predictions": "$inputs.predictions", **parameters}
    )

    return dict(params)


def label(pixels: np.ndarray, predictions: Any = None, **parameters: Any) -> np.ndarray:
    image = image_of(pixels)
    predictions = detections(image) if predictions is None else predictions
    arguments = {**validated(**parameters), "image": image, "predictions": predictions}
    result = RichLabelVisualizationBlock().run(**arguments)
    drawn = pixels_of(result["image"])

    return drawn


def v1_label(
    pixels: np.ndarray, predictions: Optional[Any] = None, **parameters: Any
) -> np.ndarray:
    arguments = {
        name: value
        for name, value in validated(**parameters).items()
        if name not in ("image", "predictions", "mask_quality")
    }
    predictions = v1_detections() if predictions is None else predictions
    expected = run_v1(
        RichLabelVisualizationBlockV1, pixels, predictions=predictions, **arguments
    )

    return expected


@pytest.mark.parametrize(
    "position",
    [
        "TOP_LEFT",
        "TOP_CENTER",
        "TOP_RIGHT",
        "CENTER_LEFT",
        "CENTER",
        "CENTER_RIGHT",
        "BOTTOM_LEFT",
        "BOTTOM_CENTER",
        "BOTTOM_RIGHT",
    ],
)
def test_every_box_position_equals_v1(position: str) -> None:
    # Labels of the top-right object cross the image edge and are clipped.
    pixels = scene(HEIGHT, WIDTH)
    parameters = dict(text="Class and Confidence", text_position=position, font_size=11)

    drawn = label(pixels, **parameters)

    np.testing.assert_array_equal(drawn, v1_label(pixels, **parameters))
    assert (drawn != pixels).any()


V1_CASES = {
    "wrapped, rounded, small padding": dict(
        text="Class and Confidence", max_line_length=3, border_radius=5, text_padding=3
    ),
    "automatic size": dict(text="Class", text_size_mode="Automatic", font_size=40),
    "Noto Sans by legacy id": dict(text="Class", font_family="noto_sans", font_size=13),
    "custom palette by index": dict(
        text="Index",
        color_palette="CUSTOM",
        custom_colors=["#FF0000", "rgb(0, 0, 255)"],
        color_axis="INDEX",
        text_color="BLACK",
    ),
    "dimensions": dict(text="Dimensions", font_size=9, text_position="BOTTOM_LEFT"),
    "zero padding": dict(text="Confidence", text_padding=0),
}


@pytest.mark.parametrize("case", list(V1_CASES))
def test_options_equal_v1(case: str) -> None:
    pixels = scene(HEIGHT, WIDTH)

    drawn = label(pixels, **V1_CASES[case])

    np.testing.assert_array_equal(drawn, v1_label(pixels, **V1_CASES[case]))


def test_tracker_ids_color_and_label_rows_like_v1() -> None:
    pixels = scene(HEIGHT, WIDTH)
    image = image_of(pixels)
    parameters = dict(text="Tracker Id", color_axis="TRACK", font_size=10)

    drawn = label(pixels, detections(image, rows=TRACKERS), **parameters)

    expected = v1_label(pixels, v1_detections(tracker_id=[7, 3, 7]), **parameters)
    np.testing.assert_array_equal(drawn, expected)


@pytest.mark.parametrize(
    "parameters",
    [
        dict(text="Class", text_position="CENTER_OF_MASS"),
        dict(text="Area", text_position="TOP_LEFT", font_size=9),
    ],
)
def test_mask_readers_equal_v1(parameters: Dict[str, Any]) -> None:
    pixels = scene(HEIGHT, WIDTH)
    masks = object_masks(HEIGHT, WIDTH)

    drawn = label(
        pixels, instance_detections(image_of(pixels), masks=masks), **parameters
    )

    expected = v1_label(pixels, v1_detections(masks=masks), **parameters)
    np.testing.assert_array_equal(drawn, expected)


def test_later_rows_draw_over_earlier_rows() -> None:
    # Given: two rows with the same box; only the second label's color shows
    pixels = scene(HEIGHT, WIDTH)
    image = image_of(pixels)
    same_box = [(30, 40, 90, 80, 0, 0.9), (30, 40, 90, 80, 1, 0.8)]
    parameters = dict(
        text="Class", color_palette="CUSTOM", custom_colors=["#FF0000", "#0000FF"]
    )

    drawn = label(pixels, detections(image, same_box), **parameters)

    np.testing.assert_array_equal(
        drawn, v1_label(pixels, v1_detections(same_box), **parameters)
    )
    label_area = drawn[20:40, 30:60].reshape(-1, 3)
    assert (label_area == (0, 0, 255)).all(axis=1).any()
    assert not (label_area == (255, 0, 0)).all(axis=1).any()


def test_no_rows_return_an_unchanged_copy() -> None:
    pixels = scene(HEIGHT, WIDTH)
    image = image_of(pixels, image_id="kept")
    params = validated()

    result = RichLabelVisualizationBlock().run(
        **{**params, "image": image, "predictions": detections(image, ())}
    )

    np.testing.assert_array_equal(pixels_of(result["image"]), pixels)
    assert result["image"].image_id == "kept"


def test_input_image_is_unchanged() -> None:
    pixels = scene(HEIGHT, WIDTH)
    image = image_of(pixels)

    RichLabelVisualizationBlock().run(
        **{**validated(), "image": image, "predictions": detections(image)}
    )

    np.testing.assert_array_equal(pixels_of(image), pixels)


def test_changed_custom_colors_take_effect_in_the_same_step() -> None:
    # V1 cached its annotator without custom_colors and kept the first ones.
    pixels = scene(HEIGHT, WIDTH)
    image = image_of(pixels)
    block = RichLabelVisualizationBlock()
    params = validated(
        color_palette="CUSTOM", custom_colors="$inputs.colors", color_axis="INDEX"
    )

    red = block.run(
        **{
            **params,
            "image": image,
            "predictions": detections(image),
            "custom_colors": ["#FF0000"],
        }
    )
    blue = block.run(
        **{
            **params,
            "image": image,
            "predictions": detections(image),
            "custom_colors": ["#0000FF"],
        }
    )

    np.testing.assert_array_equal(
        pixels_of(blue["image"]),
        label(
            pixels,
            color_palette="CUSTOM",
            custom_colors=["#0000FF"],
            color_axis="INDEX",
        ),
    )
    assert (pixels_of(red["image"]) != pixels_of(blue["image"])).any()


def test_concurrent_calls_draw_what_sequential_calls_draw() -> None:
    pixels = scene(HEIGHT, WIDTH)
    image = image_of(pixels)
    block = RichLabelVisualizationBlock()
    requests = [
        validated(text=text, font_family=font, font_size=size)
        for text in ("Class", "Confidence")
        for font in ("Geist Mono", "Noto Sans")
        for size in (10, 14)
    ]

    def draw(params: Dict[str, Any]) -> np.ndarray:
        result = block.run(
            **{**params, "image": image, "predictions": detections(image)}
        )

        return pixels_of(result["image"])

    sequential = [draw(params) for params in requests]
    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent = list(pool.map(draw, requests * 3))

    for index, drawn in enumerate(concurrent):
        np.testing.assert_array_equal(drawn, sequential[index % len(requests)])


def test_unknown_font_lists_the_approved_fonts() -> None:
    # A literal fails the font Literal, which lists the approved names.
    with pytest.raises(
        ParamsValidationError, match="font_family: Input should be .*'Geist Mono'"
    ):
        validated(font_family="Comic Sans")

    params = validated(font_family="$inputs.font")
    image = image_of(scene())
    with pytest.raises(ValueError, match="Unknown font family: 'Comic Sans'"):
        RichLabelVisualizationBlock().run(
            **{
                **params,
                "image": image,
                "predictions": detections(image),
                "font_family": "Comic Sans",
            }
        )


def test_missing_font_names_the_install_command(monkeypatch, tmp_path) -> None:
    # Given: no installed fonts and an empty V1 cache
    monkeypatch.setattr(fonts, "ASSETS_DIR", tmp_path / "assets")
    monkeypatch.setattr(fonts, "FONTS_CACHE_DIR", tmp_path / "cache")
    rich_label._local_font.cache_clear()

    try:
        with pytest.raises(
            ParamsValidationError, match="'Inter' is not installed.*make download_fonts"
        ):
            validated(font_family="Inter")
    finally:
        rich_label._local_font.cache_clear()


def test_v1_cached_font_is_used_only_with_the_registry_checksum(
    monkeypatch, tmp_path
) -> None:
    metadata = FONTS_REGISTRY["inter"]
    installed = fonts.ASSETS_DIR / metadata.identifier / metadata.file_name
    cached = tmp_path / "cache" / "inter" / installed.name
    cached.parent.mkdir(parents=True)
    cached.write_bytes(installed.read_bytes())
    monkeypatch.setattr(fonts, "ASSETS_DIR", tmp_path / "assets")
    monkeypatch.setattr(fonts, "FONTS_CACHE_DIR", tmp_path / "cache")
    rich_label._local_font.cache_clear()

    try:
        assert rich_label.font_path("Inter") == cached
        rich_label._local_font.cache_clear()
        cached.write_bytes(b"tampered")
        with pytest.raises(ValueError, match="not installed"):
            rich_label.font_path("Inter")
    finally:
        rich_label._local_font.cache_clear()


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("text_color", "nope", "text_color: invalid color"),
        ("font_size", 0, "font_size must be an integer >= 1"),
        ("text_padding", -1, "text_padding must be an integer >= 0"),
        ("border_radius", -1, "border_radius must be an integer >= 0"),
        ("max_line_length", 0, "max_line_length must be an integer >= 1"),
    ],
)
def test_literal_and_selected_values_fail_with_the_same_message(
    field: str, value: Any, message: str
) -> None:
    with pytest.raises(ParamsValidationError, match=message):
        validated(**{field: value})

    params = validated(**{field: "$inputs.value"})
    image = image_of(scene())
    with pytest.raises(ValueError, match=message):
        RichLabelVisualizationBlock().run(
            **{**params, "image": image, "predictions": detections(image), field: value}
        )


def test_center_of_mass_without_masks_names_the_requirement() -> None:
    with pytest.raises(ValueError, match="CENTER_OF_MASS needs instance masks"):
        label(scene(), text_position="CENTER_OF_MASS")


def test_copy_image_is_rejected() -> None:
    with pytest.raises(ParamsValidationError, match="copy_image is not a V2 parameter"):
        validated(copy_image=False)


# sv resolve_text_background_xyxy for a 9 x 5 label at (100, 50).
SV_BACKGROUNDS = {
    "TOP_LEFT": (100, 45, 109, 50),
    "TOP_RIGHT": (91, 45, 100, 50),
    "TOP_CENTER": (96, 45, 104, 50),
    "CENTER": (96, 48, 104, 52),
    "CENTER_OF_MASS": (96, 48, 104, 52),
    "BOTTOM_LEFT": (100, 50, 109, 55),
    "BOTTOM_RIGHT": (91, 50, 100, 55),
    "BOTTOM_CENTER": (96, 50, 104, 55),
    "CENTER_LEFT": (91, 48, 100, 52),
    "CENTER_RIGHT": (100, 48, 109, 52),
}


@pytest.mark.parametrize("position", list(SV_BACKGROUNDS))
def test_rich_labels_use_the_shared_sv_background_placement(position: str) -> None:
    # Odd sizes show the halving: centred sides lose a pixel, as in sv.
    font = ImageFont.truetype(str(rich_label.font_path(fonts.DEFAULT_FONT_FAMILY)), 12)
    (laid_out,) = rich_label.rich_labels(
        ["x"],
        np.array([[100, 50]]),
        position=position,
        font=font,
        padding=0,
        max_line_length=None,
        background_rgb=[(0, 0, 0)],
    )
    x1, y1, x2, y2 = laid_out.background

    assert background_rect(100, 50, width=9, height=5, position=position) == (
        SV_BACKGROUNDS[position]
    )
    assert laid_out.background == background_rect(
        100, 50, width=x2 - x1, height=y2 - y1, position=position
    )
