"""``roboflow_core/icon_visualization@v1``: icons at fixed points or on detections.

Icons are small arrays generated here; nothing is read from files or URLs.
Expected pixels come from hand-computed placements of solid icons and from
the latest V1 block on the same data. V2 resizes with float32 weights, so a
resized gradient icon may differ from V1 by 1 level; placement is exact.

Transparency (``icon_alpha``) is checked against the blend formula by hand
and against V1 given the same icon as an RGBA PNG (V1 decodes its base64
source again to recover alpha).
"""

import base64
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import pytest
import torch
from roboflow_workflows.core_steps.visualizations.icon.v1 import (
    IconVisualizationBlockV1,
)
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)
from roboflow_workflows.v2.blocks.visualization.annotations.icon_visualization.v1 import IconVisualizationBlock
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.engine.declaration import spec_of
from roboflow_workflows.v2.engine.errors import ParamsValidationError

from tests.v2.unit.blocks.visualization.annotations.support import (
    detections,
    grid_view,
    image_of,
    instance_detections,
    object_masks,
    pixels_of,
    run_v1,
    scene,
    v1_detections,
)

RED = (255, 0, 0)
BOX_POSITIONS = (
    "CENTER",
    "CENTER_LEFT",
    "CENTER_RIGHT",
    "TOP_CENTER",
    "TOP_LEFT",
    "TOP_RIGHT",
    "BOTTOM_LEFT",
    "BOTTOM_CENTER",
    "BOTTOM_RIGHT",
)


def solid_icon(height: int, width: int, color: Tuple[int, int, int] = RED) -> ImageData:
    """A one-color RGB icon."""
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    pixels[:] = color

    return ImageData.from_numpy_rgb(pixels, image_id="icon")


def gradient_icon(height: int = 5, width: int = 7) -> np.ndarray:
    """A smooth RGB icon whose resize depends on interpolation."""
    ys, xs = np.mgrid[0:height, 0:width]
    pixels = np.stack([xs * 36, ys * 60, 255 - xs * 20 - ys * 10], axis=-1)

    return pixels.astype(np.uint8)


def draw(
    image: ImageData,
    icon: ImageData,
    predictions: Any = None,
    *,
    icon_alpha: Optional[torch.Tensor] = None,
    **params: Any,
):
    """Validate ``params`` as the engine does, run the block, return the image."""
    bound = {"image": "$inputs.image", "icon": "$inputs.icon"}
    if predictions is not None:
        bound["predictions"] = "$inputs.predictions"
    if icon_alpha is not None:
        bound["icon_alpha"] = "$inputs.icon_alpha"
    validated = spec_of(IconVisualizationBlock).validate_params({**params, **bound})
    arguments = {
        **dict(validated),
        "image": image,
        "icon": icon,
        "icon_alpha": icon_alpha,
        "predictions": predictions,
    }
    result = IconVisualizationBlock().run(**arguments)

    return result["image"]


def black(height: int = 20, width: int = 24) -> ImageData:
    return image_of(np.zeros((height, width, 3), dtype=np.uint8))


def drawn_box(output: ImageData) -> Tuple[int, int, int, int]:
    """Return the half-open ``(x0, y0, x1, y1)`` extent of the drawn pixels."""
    ys, xs = np.nonzero(pixels_of(output).any(axis=2))

    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def test_a_static_icon_has_its_top_left_corner_at_the_position() -> None:
    output = draw(
        black(),
        solid_icon(4, 4),
        mode="static",
        icon_width=4,
        icon_height=4,
        x_position=3,
        y_position=5,
    )

    expected = np.zeros((20, 24, 3), dtype=np.uint8)
    expected[5:9, 3:7] = RED
    np.testing.assert_array_equal(pixels_of(output), expected)


def test_negative_static_positions_measure_from_the_right_and_bottom() -> None:
    output = draw(
        black(),
        solid_icon(4, 4),
        mode="static",
        icon_width=4,
        icon_height=4,
        x_position=-2,
        y_position=-1,
    )

    # Right edge 2 px from x = 24, bottom edge 1 px from y = 20.
    assert drawn_box(output) == (18, 15, 22, 19)


def test_an_odd_static_icon_size_shifts_left_by_one_like_v1() -> None:
    output = draw(
        black(),
        solid_icon(5, 5),
        mode="static",
        icon_width=5,
        icon_height=5,
        x_position=3,
        y_position=5,
    )

    # Anchor 3 + 5 // 2 = 5; corner int(5 - 2.5) = 2.
    assert drawn_box(output) == (2, 4, 7, 9)


def test_an_icon_keeps_its_aspect_ratio_centred_in_its_box() -> None:
    output = draw(
        black(),
        solid_icon(2, 4),
        mode="static",
        icon_width=8,
        icon_height=8,
        x_position=4,
        y_position=4,
    )

    # 2 x 4 fits 8 x 8 as 4 x 8: 2 transparent rows above and below.
    assert drawn_box(output) == (4, 6, 12, 10)


def test_dynamic_icons_centre_on_each_box_anchor() -> None:
    image = black()
    boxes = detections(
        image, [(2.0, 2.0, 10.0, 8.0, 0, 0.9), (14.0, 9.0, 20.0, 15.0, 1, 0.8)]
    )

    output = draw(
        image, solid_icon(4, 4), boxes, icon_width=4, icon_height=4, position="CENTER"
    )

    expected = np.zeros((20, 24, 3), dtype=np.uint8)
    expected[3:7, 4:8] = RED  # centre (6, 5)
    expected[10:14, 15:19] = RED  # centre (17, 12)
    np.testing.assert_array_equal(pixels_of(output), expected)


def test_an_icon_partly_outside_is_clipped_with_its_corner_truncated_to_zero() -> None:
    image = black()
    boxes = detections(image, [(0.0, 0.0, 6.0, 6.0, 0, 0.9)])

    output = draw(
        image, solid_icon(5, 5), boxes, icon_width=5, icon_height=5, position="TOP_LEFT"
    )

    # Corner int(0 - 2.5) = -2: columns and rows 0..2 remain.
    assert drawn_box(output) == (0, 0, 3, 3)


def test_an_icon_outside_the_image_draws_nothing() -> None:
    image = black()
    boxes = detections(image, [(40.0, 40.0, 50.0, 50.0, 0, 0.9)])

    output = draw(image, solid_icon(4, 4), boxes, icon_width=4, icon_height=4)

    assert not pixels_of(output).any()


def test_a_later_row_icon_covers_an_earlier_one() -> None:
    image = black()
    icon_pixels = np.arange(48, dtype=np.uint8).reshape(4, 4, 3) * 5
    # Centres (6, 6) and (8, 6): corners (4, 4) and (6, 4); they overlap at x 6..7.
    boxes = detections(
        image, [(4.0, 4.0, 8.0, 8.0, 0, 0.9), (6.0, 4.0, 10.0, 8.0, 1, 0.9)]
    )

    output = draw(
        image,
        ImageData.from_numpy_rgb(icon_pixels),
        boxes,
        icon_width=4,
        icon_height=4,
        position="CENTER",
    )

    np.testing.assert_array_equal(pixels_of(output)[4:8, 4:6], icon_pixels[:, 0:2])
    np.testing.assert_array_equal(pixels_of(output)[4:8, 6:10], icon_pixels)


def test_a_grey_icon_is_drawn_in_grey() -> None:
    gray = ImageData.from_tensor(
        ImageData.from_numpy_rgb(np.full((3, 3, 3), 77, dtype=np.uint8))
        .tensor_image[:1]
        .clone()
    )

    output = draw(
        black(),
        gray,
        mode="static",
        icon_width=3,
        icon_height=3,
        x_position=0,
        y_position=0,
    )

    assert (pixels_of(output)[0:3, 0:3] == 77).all()


@pytest.mark.parametrize("position", BOX_POSITIONS)
def test_dynamic_icons_equal_v1(position: str) -> None:
    pixels = scene()
    icon_pixels = gradient_icon()
    image = image_of(pixels)

    output = draw(
        image,
        ImageData.from_numpy_rgb(icon_pixels),
        detections(image),
        icon_width=10,
        icon_height=6,
        position=position,
    )
    expected = run_v1(
        IconVisualizationBlockV1,
        pixels,
        mode="dynamic",
        icon=v1_icon(icon_pixels),
        predictions=v1_detections(),
        icon_width=10,
        icon_height=6,
        position=position,
        x_position=10,
        y_position=10,
    )

    assert_within_one_level(pixels_of(output), expected)


def test_center_of_mass_icons_equal_v1() -> None:
    pixels = scene()
    icon_pixels = gradient_icon()
    image = image_of(pixels)

    output = draw(
        image,
        ImageData.from_numpy_rgb(icon_pixels),
        instance_detections(image),
        icon_width=6,
        icon_height=6,
        position="CENTER_OF_MASS",
    )
    expected = run_v1(
        IconVisualizationBlockV1,
        pixels,
        mode="dynamic",
        icon=v1_icon(icon_pixels),
        predictions=v1_detections(masks=object_masks()),
        icon_width=6,
        icon_height=6,
        position="CENTER_OF_MASS",
        x_position=10,
        y_position=10,
    )

    assert_within_one_level(pixels_of(output), expected)


@pytest.mark.parametrize(
    "x_position, y_position", [(10, 10), (-10, -3), (0, -1), (-40, 3)]
)
def test_static_icons_equal_v1(x_position: int, y_position: int) -> None:
    pixels = scene()
    icon_pixels = gradient_icon()

    output = draw(
        image_of(pixels),
        ImageData.from_numpy_rgb(icon_pixels),
        mode="static",
        icon_width=9,
        icon_height=7,
        x_position=x_position,
        y_position=y_position,
    )
    expected = run_v1(
        IconVisualizationBlockV1,
        pixels,
        mode="static",
        icon=v1_icon(icon_pixels),
        predictions=None,
        icon_width=9,
        icon_height=7,
        position="TOP_CENTER",
        x_position=x_position,
        y_position=y_position,
    )

    assert_within_one_level(pixels_of(output), expected)


def test_static_mode_ignores_predictions() -> None:
    image = black()
    params = dict(
        mode="static", icon_width=4, icon_height=4, x_position=1, y_position=1
    )

    with_rows = draw(
        image,
        solid_icon(4, 4),
        detections(image, [(10.0, 10.0, 14.0, 14.0, 0, 0.9)]),
        **params,
    )
    without = draw(image, solid_icon(4, 4), **params)

    np.testing.assert_array_equal(pixels_of(with_rows), pixels_of(without))


def test_no_rows_return_an_unchanged_copy() -> None:
    pixels = scene()
    image = image_of(pixels)

    output = draw(image, solid_icon(4, 4), detections(image, []))

    np.testing.assert_array_equal(pixels_of(output), pixels)


def test_box_positions_never_upscale_a_mask_view() -> None:
    image = image_of(scene())
    view = grid_view(image)

    from_view = draw(image, solid_icon(4, 4), view, position="CENTER")
    from_boxes = draw(image, solid_icon(4, 4), detections(image), position="CENTER")

    np.testing.assert_array_equal(pixels_of(from_view), pixels_of(from_boxes))
    assert not view.is_full_res_materialized()


def test_center_of_mass_without_masks_is_an_error() -> None:
    image = image_of(scene())

    with pytest.raises(ValueError, match="CENTER_OF_MASS needs instance masks"):
        draw(image, solid_icon(4, 4), detections(image), position="CENTER_OF_MASS")


def test_inputs_are_unchanged_and_the_output_keeps_the_image_identity() -> None:
    pixels, icon_pixels = scene(), gradient_icon()
    image, icon = image_of(pixels), ImageData.from_numpy_rgb(icon_pixels)

    output = draw(image, icon, detections(image), position="CENTER")

    np.testing.assert_array_equal(pixels_of(image), pixels)
    np.testing.assert_array_equal(pixels_of(icon), icon_pixels)
    assert output.image_id == image.image_id
    assert output.root == image.root


def test_dynamic_mode_without_predictions_is_a_compile_error() -> None:
    with pytest.raises(ParamsValidationError) as error:
        spec_of(IconVisualizationBlock).validate_params(
            {"image": "$inputs.image", "icon": "$inputs.icon", "mode": "dynamic"}
        )

    assert "needs predictions" in str(error.value) + str(error.value.__cause__)


def test_a_selected_dynamic_mode_without_predictions_is_a_run_error() -> None:
    with pytest.raises(ValueError, match="needs predictions"):
        draw_selected(black(), solid_icon(4, 4), {"mode": "dynamic"})


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("icon_width", 0, "icon_width must be an integer >= 1, got 0"),
        ("x_position", 1.5, "x_position must be an integer, got 1.5"),
        ("mode", "floating", "mode must be one of ['static', 'dynamic']"),
        ("position", "MIDDLE", "position must be one of"),
    ],
)
def test_an_invalid_selected_value_is_a_run_error(
    field: str, value: Any, message: str
) -> None:
    with pytest.raises(ValueError) as error:
        draw_selected(black(), solid_icon(4, 4), {"mode": "static", field: value})

    assert message in str(error.value)


def test_an_invalid_icon_size_literal_is_a_compile_error() -> None:
    with pytest.raises(ParamsValidationError) as error:
        spec_of(IconVisualizationBlock).validate_params(
            {
                "image": "$inputs.image",
                "icon": "$inputs.icon",
                "mode": "static",
                "icon_height": 0,
            }
        )

    assert "icon_height must be an integer >= 1, got 0" in str(error.value) + str(
        error.value.__cause__
    )


def draw_selected(image: ImageData, icon: ImageData, selected: dict) -> ImageData:
    """Validate ``selected`` fields as selectors, then run with their values."""
    validated = spec_of(IconVisualizationBlock).validate_params(
        {
            **{name: f"$inputs.{name}" for name in selected},
            "image": "$inputs.image",
            "icon": "$inputs.icon",
        }
    )
    arguments = {
        **dict(validated),
        **selected,
        "image": image,
        "icon": icon,
        "predictions": None,
    }
    result = IconVisualizationBlock().run(**arguments)

    return result["image"]


def v1_icon(pixels: np.ndarray) -> WorkflowImageData:
    """The V1 form of an RGB icon: a BGR image without a file or base64 source."""
    icon = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="icon"),
        numpy_image=np.ascontiguousarray(pixels[:, :, ::-1]),
    )

    return icon


def assert_within_one_level(actual: np.ndarray, expected: np.ndarray) -> None:
    """Same drawn pixels; resized icon values may differ by 1 (float32 weights)."""
    difference = np.abs(actual.astype(int) - expected.astype(int))
    assert (
        difference.max() <= 1
    ), f"{(difference > 1).sum()} pixels differ by more than 1"


# --- transparency: icon_alpha ---------------------------------------------------


def blend(below: int, icon: int, alpha: int) -> int:
    """The documented blend: exact integers, rounded down."""
    return (below * (255 - alpha) + icon * alpha) // 255


def grey(height: int, width: int, value: int) -> ImageData:
    return image_of(np.full((height, width, 3), value, dtype=np.uint8))


def static(**params: Any) -> dict:
    return dict(mode="static", **params)


def test_alpha_0_keeps_the_image_255_draws_the_icon_and_between_blends() -> None:
    alphas = [0, 1, 128, 200, 254, 255]
    alpha = torch.tensor([alphas, alphas], dtype=torch.uint8)

    output = pixels_of(
        draw(
            grey(6, 9, 100),
            solid_icon(2, 6, (200, 200, 200)),
            icon_alpha=alpha,
            **static(icon_width=6, icon_height=2, x_position=2, y_position=3),
        )
    )

    # (100 * (255 - a) + 200 * a) // 255; 1 and 254 show the rounding down:
    # 100.39 -> 100 and 199.6 -> 199.
    expected = [100, 100, 150, 178, 199, 200]
    assert [blend(100, 200, a) for a in alphas] == expected
    assert output[3, 2:8, 0].tolist() == expected
    assert output[4, 2:8, 0].tolist() == expected
    untouched = np.ones((6, 9), dtype=bool)
    untouched[3:5, 2:8] = False
    assert (output[untouched] == 100).all()


def test_float_alpha_is_rounded_to_the_same_255_steps() -> None:
    params = static(icon_width=4, icon_height=1, x_position=0, y_position=0)
    as_uint8 = torch.tensor([[0, 128, 51, 255]], dtype=torch.uint8)

    drawn = [
        pixels_of(draw(grey(2, 4, 30), solid_icon(1, 4), icon_alpha=alpha, **params))
        for alpha in (
            as_uint8,
            torch.tensor([[0.0, 0.5, 0.2, 1.0]]),
            torch.tensor([[0.0, 0.5, 0.2, 1.0]], dtype=torch.float64),
            torch.tensor([[0.0, 0.5, 0.2, 1.0]], dtype=torch.float16),
        )
    ]

    for other in drawn[1:]:
        np.testing.assert_array_equal(other, drawn[0])


def test_an_all_255_alpha_draws_like_no_alpha() -> None:
    pixels, icon_pixels = scene(), gradient_icon()
    params = dict(icon_width=10, icon_height=6, position="CENTER")
    image = image_of(pixels)

    opaque = draw(
        image, ImageData.from_numpy_rgb(icon_pixels), detections(image), **params
    )
    full = draw(
        image,
        ImageData.from_numpy_rgb(icon_pixels),
        detections(image),
        icon_alpha=torch.full((5, 7), 255, dtype=torch.uint8),
        **params,
    )

    np.testing.assert_array_equal(pixels_of(full), pixels_of(opaque))


def test_an_alpha_icon_partly_outside_blends_only_its_visible_corner() -> None:
    # A 4 x 2 icon on anchor (0, 0) has its corner at (-2, -1): only the
    # icon's bottom-right 2 x 1 pixels land in the image, at (0, 0) and (1, 0).
    image = grey(5, 6, 0)
    alpha = torch.tensor([[0, 50, 100, 255], [10, 60, 110, 200]], dtype=torch.uint8)

    output = pixels_of(
        draw(
            image,
            solid_icon(2, 4, (250, 250, 250)),
            detections(image, [(0.0, 0.0, 4.0, 4.0, 0, 0.9)]),
            icon_alpha=alpha,
            icon_width=4,
            icon_height=2,
            position="TOP_LEFT",
        )
    )

    assert output[0, :2, 0].tolist() == [blend(0, 250, 110), blend(0, 250, 200)]
    assert output[0, 2:].max() == 0 and output[1:].max() == 0


def test_an_alpha_icon_outside_the_image_draws_nothing() -> None:
    output = draw(
        grey(5, 6, 9),
        solid_icon(2, 2),
        icon_alpha=torch.full((2, 2), 200, dtype=torch.uint8),
        **static(icon_width=2, icon_height=2, x_position=40, y_position=0),
    )

    assert (pixels_of(output) == 9).all()


def test_a_resized_alpha_follows_its_icon_and_the_border_stays_transparent() -> None:
    # A 1 x 2 icon in a 4 x 4 box: letterboxed to 4 x 2, rows 1..2 of the box.
    # The left half is transparent, the right half opaque; INTER_LINEAR blends
    # the two middle columns with the same weights as the color.
    icon = ImageData.from_numpy_rgb(
        np.array([[[0, 0, 0], [240, 240, 240]]], dtype=np.uint8)
    )
    alpha = torch.tensor([[0, 255]], dtype=torch.uint8)

    output = pixels_of(
        draw(
            grey(6, 6, 60),
            icon,
            icon_alpha=alpha,
            **static(icon_width=4, icon_height=4, x_position=1, y_position=1),
        )
    )[:, :, 0]

    # resized color and alpha per column: 0, 60, 180, 240 and 0, 64, 191, 255
    row = [60, blend(60, 60, 64), blend(60, 180, 191), 240]
    assert output[2, 1:5].tolist() == row and output[3, 1:5].tolist() == row
    assert (output[[0, 1, 4, 5]] == 60).all() and (output[:, [0, 5]] == 60).all()


def test_a_grey_icon_blends_in_grey() -> None:
    grey_icon = ImageData.from_tensor(torch.full((1, 2, 2), 200, dtype=torch.uint8))

    output = pixels_of(
        draw(
            grey(3, 3, 0),
            grey_icon,
            icon_alpha=torch.full((2, 2), 0.5),
            **static(icon_width=2, icon_height=2, x_position=0, y_position=0),
        )
    )

    assert (output[:2, :2] == blend(0, 200, 128)).all()


@pytest.mark.parametrize(
    "alpha, message",
    [
        (
            torch.zeros((5,), dtype=torch.uint8),
            "(height, width) (4, 4), got shape (5,)",
        ),
        (torch.zeros((4, 4, 1), dtype=torch.uint8), "got shape (4, 4, 1)"),
        (torch.zeros((4, 3), dtype=torch.uint8), "got shape (4, 3)"),
        (torch.zeros((4, 4), dtype=torch.int64), "uint8 (0..255) or float (0..1)"),
        (torch.zeros((4, 4), dtype=torch.bool), "got torch.bool"),
        (torch.full((4, 4), 1.5), "finite and within 0..1"),
        (torch.full((4, 4), -0.1), "finite and within 0..1"),
        (torch.full((4, 4), float("nan")), "finite and within 0..1"),
        (torch.full((4, 4), float("inf")), "finite and within 0..1"),
        (np.zeros((4, 4), dtype=np.uint8), "must be a torch.Tensor, got ndarray"),
    ],
)
def test_an_invalid_icon_alpha_is_a_run_error(alpha: Any, message: str) -> None:
    with pytest.raises(ValueError) as error:
        draw(
            black(),
            solid_icon(4, 4),
            icon_alpha=alpha,
            **static(icon_width=4, icon_height=4),
        )

    assert message in str(error.value)


def test_alpha_is_checked_even_when_no_icon_is_drawn() -> None:
    image = black()

    with pytest.raises(ValueError, match="got shape"):
        draw(
            image,
            solid_icon(4, 4),
            detections(image, []),
            icon_alpha=torch.zeros((2, 2), dtype=torch.uint8),
        )


def test_alpha_inputs_are_unchanged_and_the_output_keeps_the_image_identity() -> None:
    pixels = scene()
    image = image_of(pixels)
    alpha = torch.linspace(0, 1, 35).reshape(5, 7)
    kept = alpha.clone()

    output = draw(
        image,
        ImageData.from_numpy_rgb(gradient_icon()),
        detections(image),
        icon_alpha=alpha,
        position="CENTER",
    )

    torch.testing.assert_close(alpha, kept, rtol=0, atol=0)
    np.testing.assert_array_equal(pixels_of(image), pixels)
    assert output.image_id == image.image_id
    assert output.root == image.root


def gradient_alpha(height: int = 5, width: int = 7) -> np.ndarray:
    """Transparent left edge, opaque right edge, partial values between."""
    ys, xs = np.mgrid[0:height, 0:width]

    return np.clip(xs * 45 - 10 + ys * 5, 0, 255).astype(np.uint8)


def v1_rgba_icon(pixels: np.ndarray, alpha: np.ndarray) -> WorkflowImageData:
    """The V1 form of an RGBA PNG icon: BGR pixels plus its base64 PNG source."""
    _, png = cv2.imencode(".png", np.dstack([pixels[:, :, ::-1], alpha]))
    icon = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="icon"),
        numpy_image=np.ascontiguousarray(pixels[:, :, ::-1]),
        base64_image=base64.b64encode(png.tobytes()).decode("ascii"),
    )

    return icon


@pytest.mark.parametrize("x_position, y_position", [(10, 10), (-3, -2), (-40, 3)])
def test_unresized_alpha_icons_equal_v1_or_are_one_level_brighter(
    x_position: int, y_position: int
) -> None:
    # Same size as the box: no resize, so only the blend rounding can differ.
    pixels, icon_pixels, alpha = scene(), gradient_icon(), gradient_alpha()
    params = dict(
        icon_width=7, icon_height=5, x_position=x_position, y_position=y_position
    )

    output = draw(
        image_of(pixels),
        ImageData.from_numpy_rgb(icon_pixels),
        icon_alpha=torch.from_numpy(alpha),
        mode="static",
        **params,
    )
    expected = run_v1(
        IconVisualizationBlockV1,
        pixels,
        mode="static",
        icon=v1_rgba_icon(icon_pixels, alpha),
        predictions=None,
        position="TOP_CENTER",
        **params,
    )

    difference = pixels_of(output).astype(int) - expected.astype(int)
    assert set(np.unique(difference).tolist()) <= {0, 1}
    assert (expected != pixels).any() and (expected == pixels).any()


@pytest.mark.parametrize("position", ["CENTER", "TOP_LEFT", "BOTTOM_RIGHT"])
def test_resized_alpha_icons_equal_v1_within_resize_and_blend_rounding(
    position: str,
) -> None:
    pixels, icon_pixels, alpha = scene(), gradient_icon(), gradient_alpha()
    image = image_of(pixels)

    output = draw(
        image,
        ImageData.from_numpy_rgb(icon_pixels),
        detections(image),
        icon_alpha=torch.from_numpy(alpha),
        icon_width=10,
        icon_height=6,
        position=position,
    )
    expected = run_v1(
        IconVisualizationBlockV1,
        pixels,
        mode="dynamic",
        icon=v1_rgba_icon(icon_pixels, alpha),
        predictions=v1_detections(),
        icon_width=10,
        icon_height=6,
        position=position,
        x_position=10,
        y_position=10,
    )

    # Color and alpha may each differ by 1 after the resize (float32 weights)
    # and the blend by 1 more, so 2 levels bound every pixel.
    difference = np.abs(pixels_of(output).astype(int) - expected.astype(int))
    assert difference.max() <= 2
