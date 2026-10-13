"""``v2/corner_visualization@v1``: pixels equal V1 ``corner_visualization@v1``.

The oracle is the latest V1 block (``sv.BoxCornerAnnotator``) run on the
same rows; V1 test modules are not imported.
"""

from typing import Tuple

import numpy as np
import pytest
import torch
from roboflow_workflows.core_steps.visualizations.corner.v1 import (
    CornerVisualizationBlockV1,
)
from roboflow_workflows.v2.blocks.visualization.annotations.corner import (
    CornerVisualizationBlock,
    corner_segments,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.engine.declaration import spec_of
from roboflow_workflows.v2.engine.errors import ParamsValidationError

from inference_models.models.base.object_detection import Detections
from tests.v2.unit.blocks.visualization.annotations.support import (
    OBJECTS,
    V1_COLOR_DEFAULTS,
    detections,
    grid_view,
    image_of,
    pixels_of,
    rle_detections,
    run_v1,
    run_v2,
    scene,
    v1_detections,
)

# Corners crossing every frame edge, a zero-width box and one fully outside.
BEYOND_THE_FRAME = (
    (-10.0, -6.5, 20.0, 18.0, 0, 0.9),
    (44.0, 30.0, 80.0, 60.0, 1, 0.8),
    (30.0, 5.0, 30.0, 25.0, 2, 0.7),
    (-40.0, -40.0, -20.0, -20.0, 1, 0.6),
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
    ("thickness", "corner_length"), [(4, 15), (1, 5), (2, 40), (3, -6), (6, 0)]
)
def test_corner_marks_equal_v1(thickness: int, corner_length: int) -> None:
    pixels, image = _input_image()
    style = dict(thickness=thickness, corner_length=corner_length)

    drawn = run_v2(
        CornerVisualizationBlock, image=image, predictions=detections(image), **style
    )

    expected = run_v1(
        CornerVisualizationBlockV1,
        pixels,
        predictions=v1_detections(),
        **style,
        **V1_COLOR_DEFAULTS,
    )
    _assert_same_pixels(pixels_of(drawn), expected)


def test_marks_crossing_or_beyond_the_frame_are_clipped_like_v1() -> None:
    pixels, image = _input_image()
    style = dict(thickness=3, corner_length=12)

    drawn = run_v2(
        CornerVisualizationBlock,
        image=image,
        predictions=detections(image, BEYOND_THE_FRAME),
        **style,
    )

    expected = run_v1(
        CornerVisualizationBlockV1,
        pixels,
        predictions=v1_detections(BEYOND_THE_FRAME),
        **style,
        **V1_COLOR_DEFAULTS,
    )
    _assert_same_pixels(pixels_of(drawn), expected)


def test_corner_segments_point_along_the_box_edges() -> None:
    segments = corner_segments((10, 20, 50, 60), corner_length=5)

    assert segments == [
        ((10, 20), (15, 20)),
        ((10, 20), (10, 25)),
        ((50, 20), (45, 20)),
        ((50, 20), (50, 25)),
        ((10, 60), (15, 60)),
        ((10, 60), (10, 55)),
        ((50, 60), (45, 60)),
        ((50, 60), (50, 55)),
    ]


def test_marks_never_paint_the_middle_of_a_large_box() -> None:
    # Only the scanned corner regions may change; the box interior and the
    # middle of each edge keep their pixels.
    pixels = scene(96, 128)
    image = image_of(pixels)
    box = ((10.0, 10.0, 110.0, 80.0, 0, 0.9),)

    drawn = pixels_of(
        run_v2(
            CornerVisualizationBlock,
            image=image,
            predictions=detections(image, box),
            thickness=2,
            corner_length=10,
        )
    )

    changed = (drawn != pixels).any(axis=2)
    assert changed[10, 10] and changed[80, 110]
    assert not changed[10, 40:80].any() and not changed[30:60, 10].any()
    assert not changed[20:70, 20:100].any()


@pytest.mark.parametrize("carrier", ["instance_rle", "grid_view"])
def test_mask_carriers_draw_their_boxes_without_reading_masks(carrier: str) -> None:
    _, image = _input_image()
    view = grid_view(image)
    carriers = {"instance_rle": rle_detections(image), "grid_view": view}

    drawn = run_v2(CornerVisualizationBlock, image=image, predictions=carriers[carrier])

    expected = run_v2(
        CornerVisualizationBlock, image=image, predictions=detections(image)
    )
    _assert_same_pixels(pixels_of(drawn), pixels_of(expected))
    assert not view.is_full_res_materialized()


def test_noncontiguous_boxes_draw_like_contiguous_ones() -> None:
    _, image = _input_image()
    boxes = detections(image)
    columns = torch.zeros((len(OBJECTS), 6))
    columns[:, 1:5] = boxes.xyxy
    strided = Detections(
        xyxy=columns[:, 1:5],
        class_id=boxes.class_id,
        confidence=boxes.confidence,
        image_metadata=boxes.image_metadata,
    )
    assert not strided.xyxy.is_contiguous()

    drawn = run_v2(CornerVisualizationBlock, image=image, predictions=strided)

    expected = run_v2(CornerVisualizationBlock, image=image, predictions=boxes)
    _assert_same_pixels(pixels_of(drawn), pixels_of(expected))


def test_empty_predictions_give_an_unchanged_new_image() -> None:
    pixels, image = _input_image()

    drawn = run_v2(
        CornerVisualizationBlock, image=image, predictions=detections(image, ())
    )

    _assert_same_pixels(pixels_of(drawn), pixels)
    assert drawn.tensor_image.data_ptr() != image.tensor_image.data_ptr()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("thickness", 0, "thickness must be an integer >= 1"),
        ("corner_length", 2.5, "corner_length must be an integer"),
    ],
)
def test_invalid_values_fail_as_literal_and_as_selector(
    field: str, value, message: str
) -> None:
    _, image = _input_image()
    spec = spec_of(CornerVisualizationBlock)
    base = {"image": "$inputs.image", "predictions": "$inputs.predictions"}

    with pytest.raises(ParamsValidationError, match=field):
        spec.validate_params({**base, field: value})
    params = dict(spec.validate_params({**base, field: f"$inputs.{field}"}))
    params.update({"image": image, "predictions": detections(image), field: value})
    with pytest.raises(ValueError, match=message):
        CornerVisualizationBlock().run(**params)
