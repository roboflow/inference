"""``v2/color_visualization@v1``: solid box fills blended over the image.

Expected pixels come from small hand-computed oracles (which pixels a box
covers, the blend formula) and from the latest V1 block on the same data.
"""

from typing import Any

import numpy as np
import pytest
from roboflow_workflows.core_steps.visualizations.color.v1 import (
    ColorVisualizationBlockV1,
)
from roboflow_workflows.v2.blocks.visualization.annotations.color import (
    ColorVisualizationBlock,
)
from roboflow_workflows.v2.engine.declaration import spec_of
from roboflow_workflows.v2.engine.errors import ParamsValidationError

from tests.v2.unit.blocks.visualization.annotations.support import (
    V1_COLOR_DEFAULTS,
    detections,
    grid_view,
    image_of,
    pixels_of,
    run_v1,
    run_v2,
    scene,
    v1_detections,
)

RED, GREEN = (255, 0, 0), (0, 255, 0)
SOLID = dict(color_palette="CUSTOM", custom_colors=["#FF0000", "#00FF00"])


def paint(
    objects: Any, *, height: int = 6, width: int = 8, **params: Any
) -> np.ndarray:
    """Run the block on a black image; return its RGB pixels."""
    image = image_of(np.zeros((height, width, 3), dtype=np.uint8))
    output = run_v2(
        ColorVisualizationBlock,
        image=image,
        predictions=detections(image, objects),
        **params,
    )

    return pixels_of(output)


def test_a_box_fills_its_truncated_corners_inclusively() -> None:
    painted = paint([(1.2, 1.7, 3.9, 4.0, 0, 0.9)], opacity=1.0, **SOLID)

    expected = np.zeros((6, 8, 3), dtype=np.uint8)
    expected[1:5, 1:4] = RED
    np.testing.assert_array_equal(painted, expected)


def test_opacity_mixes_box_pixels_with_the_image() -> None:
    pixels = np.full((4, 4, 3), 100, dtype=np.uint8)
    image = image_of(pixels)
    output = run_v2(
        ColorVisualizationBlock,
        image=image,
        predictions=detections(image, [(0, 0, 1, 1, 0, 0.9)]),
        opacity=0.25,
        **SOLID,
    )

    painted = pixels_of(output)
    # round(255 * 0.25 + 100 * 0.75) = 139; 0 * 0.25 + 100 * 0.75 = 75.
    np.testing.assert_array_equal(painted[0, 0], (139, 75, 75))
    np.testing.assert_array_equal(painted[2:, :], pixels[2:, :])


def test_a_later_row_paints_over_an_earlier_one() -> None:
    painted = paint([(0, 0, 4, 4, 0, 0.9), (2, 2, 6, 5, 1, 0.9)], opacity=1.0, **SOLID)

    np.testing.assert_array_equal(painted[1, 1], RED)
    np.testing.assert_array_equal(painted[3, 3], GREEN)
    np.testing.assert_array_equal(painted[5, 0], (0, 0, 0))


@pytest.mark.parametrize(
    "box, filled",
    [
        ((-3.5, -2.0, 2.0, 2.0), (slice(0, 3), slice(0, 3))),  # past the top left
        ((5.0, 3.0, 20.0, 9.0), (slice(3, 6), slice(5, 8))),  # past the bottom right
        ((3.0, 4.0, 1.0, 2.0), (slice(2, 5), slice(1, 4))),  # corners swapped
    ],
)
def test_boxes_are_clipped_to_the_image(box: tuple, filled: tuple) -> None:
    painted = paint([(*box, 0, 0.9)], opacity=1.0, **SOLID)

    expected = np.zeros((6, 8, 3), dtype=np.uint8)
    expected[filled] = RED
    np.testing.assert_array_equal(painted, expected)


def test_a_box_outside_the_image_draws_nothing() -> None:
    painted = paint([(10.0, 1.0, 14.0, 3.0, 0, 0.9)], opacity=1.0, **SOLID)

    assert not painted.any()


@pytest.mark.parametrize("opacity", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("color_axis", ["CLASS", "INDEX"])
def test_fills_equal_v1(opacity: float, color_axis: str) -> None:
    pixels = scene()
    image = image_of(pixels)
    parameters = {**V1_COLOR_DEFAULTS, "color_axis": color_axis, "opacity": opacity}

    output = run_v2(
        ColorVisualizationBlock,
        image=image,
        predictions=detections(image),
        **parameters,
    )
    expected = run_v1(
        ColorVisualizationBlockV1, pixels, predictions=v1_detections(), **parameters
    )

    np.testing.assert_array_equal(pixels_of(output), expected)


def test_other_opacities_are_within_one_level_of_v1() -> None:
    pixels = scene()
    image = image_of(pixels)
    parameters = {**V1_COLOR_DEFAULTS, "opacity": 0.3}

    output = run_v2(
        ColorVisualizationBlock,
        image=image,
        predictions=detections(image),
        **parameters,
    )
    expected = run_v1(
        ColorVisualizationBlockV1, pixels, predictions=v1_detections(), **parameters
    )

    difference = np.abs(pixels_of(output).astype(int) - expected.astype(int))
    assert difference.max() <= 1


def test_track_axis_colors_rows_by_tracker_id_like_v1() -> None:
    pixels = scene()
    image = image_of(pixels)
    tracker_ids = [7, 3, 7]
    parameters = {**V1_COLOR_DEFAULTS, "color_axis": "TRACK", "opacity": 1.0}

    output = run_v2(
        ColorVisualizationBlock,
        image=image,
        predictions=detections(
            image, rows=[{"tracker_id": value} for value in tracker_ids]
        ),
        **parameters,
    )
    expected = run_v1(
        ColorVisualizationBlockV1,
        pixels,
        predictions=v1_detections(tracker_id=tracker_ids),
        **parameters,
    )

    np.testing.assert_array_equal(pixels_of(output), expected)


def test_no_rows_return_an_unchanged_copy_even_with_track_axis() -> None:
    pixels = scene()
    image = image_of(pixels)

    output = run_v2(
        ColorVisualizationBlock,
        image=image,
        predictions=detections(image, []),
        color_axis="TRACK",
        opacity=1.0,
    )

    np.testing.assert_array_equal(pixels_of(output), pixels)
    assert output.tensor_image.data_ptr() != image.tensor_image.data_ptr()


def test_the_input_image_is_unchanged_and_the_output_keeps_its_identity() -> None:
    pixels = scene()
    image = image_of(pixels)

    output = run_v2(
        ColorVisualizationBlock, image=image, predictions=detections(image), opacity=1.0
    )

    np.testing.assert_array_equal(pixels_of(image), pixels)
    assert output.image_id == image.image_id
    assert output.root == image.root


def test_a_mask_view_is_drawn_from_its_boxes_without_upscaling_masks() -> None:
    image = image_of(scene())
    view = grid_view(image)

    from_view = run_v2(ColorVisualizationBlock, image=image, predictions=view)
    from_boxes = run_v2(
        ColorVisualizationBlock, image=image, predictions=detections(image)
    )

    np.testing.assert_array_equal(pixels_of(from_view), pixels_of(from_boxes))
    assert not view.is_full_res_materialized()


def test_an_invalid_opacity_fails_alike_as_literal_and_as_selected_value() -> None:
    image = image_of(scene())
    message = "opacity must be a number in [0.0, 1.0], got 1.5"

    with pytest.raises(ParamsValidationError) as compile_error:
        spec_of(ColorVisualizationBlock).validate_params(
            {"image": "$inputs.image", "predictions": "$inputs.p", "opacity": 1.5}
        )
    with pytest.raises(ValueError) as run_error:
        run_selected(image, detections(image), {"opacity": 1.5})

    assert message in str(compile_error.value) + str(compile_error.value.__cause__)
    assert message in str(run_error.value)


def test_an_unknown_selected_color_axis_is_an_error_naming_the_choices() -> None:
    image = image_of(scene())

    with pytest.raises(ValueError, match="color_axis must be one of"):
        run_selected(image, detections(image), {"color_axis": "ROW"})


def run_selected(image: Any, predictions: Any, selected: dict) -> Any:
    """Validate ``selected`` fields as selectors, then run with their values."""
    params = spec_of(ColorVisualizationBlock).validate_params(
        {
            **{name: f"$inputs.{name}" for name in selected},
            "image": "$inputs.image",
            "predictions": "$inputs.predictions",
        }
    )
    arguments = {**dict(params), **selected, "image": image, "predictions": predictions}
    result = ColorVisualizationBlock().run(**arguments)

    return result["image"]
