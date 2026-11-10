"""``roboflow_core/ellipse_visualization@v1``: pixels equal V1 ``ellipse_visualization@v1``.

The oracle is the latest V1 block (``sv.EllipseAnnotator``) run on the same
rows; V1 test modules are not imported. One deliberate difference: a box
with ``x2 < x1`` draws nothing, where V1 fails in OpenCV.
"""

from typing import Tuple

import numpy as np
import pytest
import torch
from roboflow_workflows.core_steps.visualizations.ellipse.v1 import (
    EllipseVisualizationBlockV1,
)
from roboflow_workflows.v2.blocks.visualization.annotations.ellipse_visualization.v1 import EllipseVisualizationBlock
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.engine.declaration import spec_of
from roboflow_workflows.v2.engine.errors import ParamsValidationError

from tests.v2.unit.blocks.visualization.annotations.support import (
    OBJECTS,
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

# Arcs crossing every frame edge; the last row lies fully outside.
BEYOND_THE_FRAME = (
    (-12.5, 30.0, 14.0, 47.5, 0, 0.9),
    (40.0, 10.0, 75.0, 60.0, 1, 0.8),
    (70.0, -30.0, 90.0, -20.0, 2, 0.7),
)


def _input_image() -> Tuple[np.ndarray, ImageData]:
    pixels = scene()

    return pixels, image_of(pixels)


def _assert_same_pixels(actual: np.ndarray, expected: np.ndarray) -> None:
    differ = (actual != expected).any(axis=2)
    assert not differ.any(), (
        f"{int(differ.sum())} pixels differ from V1, first at (y, x) "
        f"{np.argwhere(differ)[:3].tolist()}"
    )


@pytest.mark.parametrize(
    ("thickness", "start_angle", "end_angle"),
    [(2, -45, 235), (1, 0, 360), (4, 90, 180), (-1, -45, 235), (3, 200, 20)],
)
def test_arcs_equal_v1(thickness: int, start_angle: int, end_angle: int) -> None:
    pixels, image = _input_image()
    style = dict(thickness=thickness, start_angle=start_angle, end_angle=end_angle)

    drawn = run_v2(
        EllipseVisualizationBlock, image=image, predictions=detections(image), **style
    )

    expected = run_v1(
        EllipseVisualizationBlockV1,
        pixels,
        predictions=v1_detections(),
        **style,
        **V1_COLOR_DEFAULTS,
    )
    _assert_same_pixels(pixels_of(drawn), expected)


def test_arcs_crossing_or_beyond_the_frame_are_clipped_like_v1() -> None:
    pixels, image = _input_image()
    style = dict(thickness=3, start_angle=0, end_angle=360)

    drawn = run_v2(
        EllipseVisualizationBlock,
        image=image,
        predictions=detections(image, BEYOND_THE_FRAME),
        **style,
    )

    expected = run_v1(
        EllipseVisualizationBlockV1,
        pixels,
        predictions=v1_detections(BEYOND_THE_FRAME),
        **style,
        **V1_COLOR_DEFAULTS,
    )
    _assert_same_pixels(pixels_of(drawn), expected)


def test_a_box_with_x2_left_of_x1_draws_nothing_and_the_others_still_draw() -> None:
    _, image = _input_image()
    inverted = (30.0, 10.0, 12.0, 30.0, 1, 0.5)

    drawn = run_v2(
        EllipseVisualizationBlock,
        image=image,
        predictions=detections(image, (*OBJECTS, inverted)),
    )

    expected = run_v2(
        EllipseVisualizationBlock, image=image, predictions=detections(image)
    )
    _assert_same_pixels(pixels_of(drawn), pixels_of(expected))


def test_a_mask_view_draws_its_boxes_without_upscaling() -> None:
    _, image = _input_image()
    view = grid_view(image)

    drawn = run_v2(EllipseVisualizationBlock, image=image, predictions=view)

    expected = run_v2(
        EllipseVisualizationBlock, image=image, predictions=detections(image)
    )
    _assert_same_pixels(pixels_of(drawn), pixels_of(expected))
    assert not view.is_full_res_materialized()


def test_empty_predictions_give_an_unchanged_new_image() -> None:
    pixels, image = _input_image()

    drawn = run_v2(
        EllipseVisualizationBlock, image=image, predictions=detections(image, ())
    )

    _assert_same_pixels(pixels_of(drawn), pixels)
    assert drawn.tensor_image.data_ptr() != image.tensor_image.data_ptr()


def test_input_image_is_not_changed() -> None:
    _, image = _input_image()
    before = image.tensor_image.clone()

    drawn = run_v2(
        EllipseVisualizationBlock, image=image, predictions=detections(image)
    )

    assert torch.equal(image.tensor_image, before)
    assert not torch.equal(drawn.tensor_image, before)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("thickness", 0, "thickness must be an integer >= 1, or -1"),
        ("start_angle", 1.5, "start_angle must be an integer number of degrees"),
        ("end_angle", True, "end_angle must be an integer number of degrees"),
    ],
)
def test_invalid_values_fail_as_literal_and_as_selector(
    field: str, value, message: str
) -> None:
    _, image = _input_image()
    spec = spec_of(EllipseVisualizationBlock)
    base = {"image": "$inputs.image", "predictions": "$inputs.predictions"}

    with pytest.raises(ParamsValidationError, match=field):
        spec.validate_params({**base, field: value})
    params = dict(spec.validate_params({**base, field: f"$inputs.{field}"}))
    params.update({"image": image, "predictions": detections(image), field: value})
    with pytest.raises(ValueError, match=message):
        EllipseVisualizationBlock().run(**params)
