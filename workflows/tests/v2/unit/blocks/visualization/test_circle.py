"""``roboflow_core/circle_visualization@v1``: pixels equal V1 ``circle_visualization@v1``.

The oracle is the latest V1 block (``sv.CircleAnnotator``) run on the same
rows; V1 test modules are not imported. Fixtures are shared with the
annotation suites (``annotations/support.py``).
"""

from typing import Tuple

import numpy as np
import pytest
import torch
from roboflow_workflows.core_steps.visualizations.circle.v1 import (
    CircleVisualizationBlockV1,
)
from roboflow_workflows.v2.blocks.visualization.annotations.circle_visualization.v1 import CircleVisualizationBlock
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
    instance_detections,
    key_points,
    pixels_of,
    rle_detections,
    run_v1,
    run_v2,
    scene,
    v1_detections,
)

# Rows crossing every frame edge, one fully outside and one degenerate.
BEYOND_THE_FRAME = (
    (-20.5, -10.25, 15.75, 12.0, 0, 0.9),
    (50.0, 30.0, 90.0, 70.0, 1, 0.8),
    (100.0, 100.0, 140.0, 130.0, 2, 0.7),
    (10.0, 40.0, 10.0, 40.0, 0, 0.6),
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


@pytest.mark.parametrize("thickness", [1, 2, 5, -1])
def test_circles_equal_v1(thickness: int) -> None:
    pixels, image = _input_image()

    drawn = run_v2(
        CircleVisualizationBlock,
        image=image,
        predictions=detections(image),
        thickness=thickness,
    )

    expected = run_v1(
        CircleVisualizationBlockV1,
        pixels,
        predictions=v1_detections(),
        thickness=thickness,
        **V1_COLOR_DEFAULTS,
    )
    _assert_same_pixels(pixels_of(drawn), expected)


def test_circles_crossing_or_beyond_the_frame_are_clipped_like_v1() -> None:
    pixels, image = _input_image()

    drawn = run_v2(
        CircleVisualizationBlock,
        image=image,
        predictions=detections(image, BEYOND_THE_FRAME),
        thickness=3,
    )

    expected = run_v1(
        CircleVisualizationBlockV1,
        pixels,
        predictions=v1_detections(BEYOND_THE_FRAME),
        thickness=3,
        **V1_COLOR_DEFAULTS,
    )
    _assert_same_pixels(pixels_of(drawn), expected)


@pytest.mark.parametrize(
    ("axis", "rows", "tracker_id"),
    [
        ("INDEX", None, None),
        ("TRACK", [{"tracker_id": 7}, {"tracker_id": 1}, {"tracker_id": 7}], [7, 1, 7]),
    ],
)
def test_index_and_track_colors_equal_v1(axis, rows, tracker_id) -> None:
    pixels, image = _input_image()
    colors = {
        **V1_COLOR_DEFAULTS,
        "color_axis": axis,
        "color_palette": "CUSTOM",
        "custom_colors": ["#FF0000", "#00FF00"],
    }

    drawn = run_v2(
        CircleVisualizationBlock,
        image=image,
        predictions=detections(image, rows=rows),
        **colors,
    )

    expected = run_v1(
        CircleVisualizationBlockV1,
        pixels,
        predictions=v1_detections(tracker_id=tracker_id),
        thickness=2,
        **colors,
    )
    _assert_same_pixels(pixels_of(drawn), expected)


def test_track_colors_with_a_row_missing_its_tracker_id_are_an_error() -> None:
    _, image = _input_image()
    rows = [{"tracker_id": 7}, {}, {"tracker_id": 2}]

    with pytest.raises(ValueError, match="(?i)track"):
        run_v2(
            CircleVisualizationBlock,
            image=image,
            predictions=detections(image, rows=rows),
            color_axis="TRACK",
        )


@pytest.mark.parametrize(
    "carrier", ["instance_dense", "instance_rle", "grid_view", "keypoints_with_boxes"]
)
def test_every_box_carrier_draws_its_boxes_without_reading_masks(
    carrier: str,
) -> None:
    _, image = _input_image()
    view = grid_view(image)
    carriers = {
        "instance_dense": instance_detections(image),
        "instance_rle": rle_detections(image),
        "grid_view": view,
        "keypoints_with_boxes": key_points(image),
    }
    objects = OBJECTS[:2] if carrier == "keypoints_with_boxes" else OBJECTS

    drawn = run_v2(CircleVisualizationBlock, image=image, predictions=carriers[carrier])

    expected = run_v2(
        CircleVisualizationBlock, image=image, predictions=detections(image, objects)
    )
    _assert_same_pixels(pixels_of(drawn), pixels_of(expected))
    assert not view.is_full_res_materialized(), "a mask view must not be upscaled"


def test_keypoints_without_boxes_are_an_error_naming_the_remedy() -> None:
    _, image = _input_image()

    with pytest.raises(ValueError, match="no bounding boxes"):
        run_v2(
            CircleVisualizationBlock,
            image=image,
            predictions=key_points(image, with_boxes=False),
        )


def test_empty_predictions_give_an_unchanged_new_image() -> None:
    pixels, image = _input_image()

    drawn = run_v2(
        CircleVisualizationBlock, image=image, predictions=detections(image, ())
    )

    _assert_same_pixels(pixels_of(drawn), pixels)
    assert drawn.tensor_image.data_ptr() != image.tensor_image.data_ptr()


def test_noncontiguous_image_and_boxes_draw_like_contiguous_ones() -> None:
    pixels, image = _input_image()
    wide = torch.zeros((3, image.height, image.width + 5), dtype=torch.uint8)
    wide[:, :, : image.width] = image.tensor_image
    strided_image = ImageData.from_tensor(wide[:, :, : image.width], image_id="frame")
    boxes = detections(image)
    columns = torch.zeros((len(OBJECTS), 6))
    columns[:, 1:5] = boxes.xyxy
    strided_boxes = Detections(
        xyxy=columns[:, 1:5],
        class_id=torch.stack([boxes.class_id, boxes.class_id], dim=1)[:, 1],
        confidence=boxes.confidence,
        image_metadata=boxes.image_metadata,
    )
    assert not strided_image.tensor_image.is_contiguous()
    assert not strided_boxes.xyxy.is_contiguous()

    drawn = run_v2(
        CircleVisualizationBlock, image=strided_image, predictions=strided_boxes
    )

    expected = run_v2(CircleVisualizationBlock, image=image, predictions=boxes)
    _assert_same_pixels(pixels_of(drawn), pixels_of(expected))
    assert drawn.tensor_image.is_contiguous()


def test_input_image_and_boxes_are_not_changed() -> None:
    _, image = _input_image()
    boxes = detections(image)
    image_before, boxes_before = image.tensor_image.clone(), boxes.xyxy.clone()

    drawn = run_v2(CircleVisualizationBlock, image=image, predictions=boxes)

    assert torch.equal(image.tensor_image, image_before)
    assert torch.equal(boxes.xyxy, boxes_before)
    assert not torch.equal(drawn.tensor_image, image_before)
    assert drawn.image_id == image.image_id and drawn.root == image.root


@pytest.mark.parametrize("thickness", [0, -2, True, 1.5])
def test_invalid_thickness_fails_with_one_message_as_literal_and_selector(
    thickness,
) -> None:
    _, image = _input_image()
    spec = spec_of(CircleVisualizationBlock)
    base = {"image": "$inputs.image", "predictions": "$inputs.predictions"}

    with pytest.raises(ParamsValidationError, match="thickness"):
        spec.validate_params({**base, "thickness": thickness})
    params = dict(spec.validate_params({**base, "thickness": "$inputs.thickness"}))
    params.update(image=image, predictions=detections(image), thickness=thickness)
    with pytest.raises(ValueError, match="thickness must be an integer >= 1, or -1"):
        CircleVisualizationBlock().run(**params)
