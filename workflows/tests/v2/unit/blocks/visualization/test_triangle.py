"""``v2/triangle_visualization@v1``: pixels equal V1 ``triangle_visualization@v1``.

The oracle is the latest V1 block (``sv.TriangleAnnotator``) run on the same
rows; V1 test modules are not imported. ``CENTER_OF_MASS`` reads masks at
``mask_quality``; every other position must leave a mask view untouched.
"""

from typing import Tuple

import numpy as np
import pytest
from roboflow_workflows.core_steps.visualizations.triangle.v1 import (
    TriangleVisualizationBlockV1,
)
from roboflow_workflows.v2.blocks.visualization.annotations.params import (
    POSITIONS_WITH_MASS,
)
from roboflow_workflows.v2.blocks.visualization.annotations.triangle import (
    TriangleVisualizationBlock,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.engine.declaration import spec_of
from roboflow_workflows.v2.engine.errors import ParamsValidationError

from tests.v2.unit.blocks.visualization.annotations.support import (
    OBJECTS,
    V1_COLOR_DEFAULTS,
    detections,
    grid_view,
    image_of,
    instance_detections,
    key_points,
    object_masks,
    pixels_of,
    rle_detections,
    run_v1,
    run_v2,
    scene,
    v1_detections,
)

BOX_POSITIONS = POSITIONS_WITH_MASS[:-1]
# Tips on and beyond every frame edge; the last row is fully outside.
BEYOND_THE_FRAME = (
    (-8.0, 0.0, 6.0, 5.0, 0, 0.9),
    (55.0, 40.0, 70.0, 52.0, 1, 0.8),
    (90.0, 80.0, 99.0, 99.0, 2, 0.7),
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


@pytest.mark.parametrize("position", BOX_POSITIONS)
@pytest.mark.parametrize(
    ("base", "height", "outline_thickness"),
    [(10, 10, 0), (7, 12, 2), (12, -9, 1), (0, 5, 0)],
)
def test_triangles_on_box_anchors_equal_v1(
    position: str, base: int, height: int, outline_thickness: int
) -> None:
    pixels, image = _input_image()
    style = dict(
        position=position, base=base, height=height, outline_thickness=outline_thickness
    )

    drawn = run_v2(
        TriangleVisualizationBlock, image=image, predictions=detections(image), **style
    )

    expected = run_v1(
        TriangleVisualizationBlockV1,
        pixels,
        predictions=v1_detections(),
        **style,
        **V1_COLOR_DEFAULTS,
    )
    _assert_same_pixels(pixels_of(drawn), expected)


def test_triangles_crossing_or_beyond_the_frame_are_clipped_like_v1() -> None:
    pixels, image = _input_image()
    style = dict(position="TOP_CENTER", base=14, height=12, outline_thickness=2)

    drawn = run_v2(
        TriangleVisualizationBlock,
        image=image,
        predictions=detections(image, BEYOND_THE_FRAME),
        **style,
    )

    expected = run_v1(
        TriangleVisualizationBlockV1,
        pixels,
        predictions=v1_detections(BEYOND_THE_FRAME),
        **style,
        **V1_COLOR_DEFAULTS,
    )
    _assert_same_pixels(pixels_of(drawn), expected)


@pytest.mark.parametrize("carrier", ["instance_dense", "instance_rle"])
def test_triangles_at_the_centre_of_mass_equal_v1(carrier: str) -> None:
    pixels, image = _input_image()
    style = dict(position="CENTER_OF_MASS", base=8, height=6, outline_thickness=1)
    carriers = {"instance_dense": instance_detections, "instance_rle": rle_detections}

    drawn = run_v2(
        TriangleVisualizationBlock,
        image=image,
        predictions=carriers[carrier](image),
        **style,
    )

    expected = run_v1(
        TriangleVisualizationBlockV1,
        pixels,
        predictions=v1_detections(masks=object_masks()),
        **style,
        **V1_COLOR_DEFAULTS,
    )
    _assert_same_pixels(pixels_of(drawn), expected)


def test_box_positions_never_read_view_masks() -> None:
    _, image = _input_image()
    view = grid_view(image)

    drawn = run_v2(TriangleVisualizationBlock, image=image, predictions=view)

    expected = run_v2(
        TriangleVisualizationBlock, image=image, predictions=detections(image)
    )
    _assert_same_pixels(pixels_of(drawn), pixels_of(expected))
    assert not view.is_full_res_materialized()


def test_keypoints_with_boxes_draw_on_their_boxes() -> None:
    _, image = _input_image()

    drawn = run_v2(
        TriangleVisualizationBlock, image=image, predictions=key_points(image)
    )

    expected = run_v2(
        TriangleVisualizationBlock,
        image=image,
        predictions=detections(image, OBJECTS[:2]),
    )
    _assert_same_pixels(pixels_of(drawn), pixels_of(expected))


def test_centre_of_mass_without_masks_is_an_error_naming_the_remedy() -> None:
    _, image = _input_image()

    with pytest.raises(ValueError, match="CENTER_OF_MASS needs instance masks"):
        run_v2(
            TriangleVisualizationBlock,
            image=image,
            predictions=key_points(image),
            position="CENTER_OF_MASS",
        )


def test_empty_predictions_give_an_unchanged_new_image() -> None:
    pixels, image = _input_image()

    drawn = run_v2(
        TriangleVisualizationBlock, image=image, predictions=detections(image, ())
    )

    _assert_same_pixels(pixels_of(drawn), pixels)
    assert drawn.tensor_image.data_ptr() != image.tensor_image.data_ptr()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("base", -2, "base must be an integer >= 0"),
        ("height", 2.5, "height must be an integer"),
        ("outline_thickness", -1, "outline_thickness must be an integer >= 0"),
        ("position", "ABOVE", "position must be one of"),
    ],
)
def test_invalid_values_fail_as_literal_and_as_selector(
    field: str, value, message: str
) -> None:
    _, image = _input_image()
    spec = spec_of(TriangleVisualizationBlock)
    base = {"image": "$inputs.image", "predictions": "$inputs.predictions"}

    with pytest.raises(ParamsValidationError, match=field):
        spec.validate_params({**base, field: value})
    params = dict(spec.validate_params({**base, field: f"$inputs.{field}"}))
    params.update({"image": image, "predictions": detections(image), field: value})
    with pytest.raises(ValueError, match=message):
        TriangleVisualizationBlock().run(**params)
