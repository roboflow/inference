"""Canonical box implementations preserve the contract and expose raster fidelity."""

import cv2
import numpy as np
import pytest
import torch
from roboflow_workflows.v2.blocks.visualization.annotations.bounding_box_visualization.v1 import (
    AccurateBoundingBoxes,
    BoundingBoxVisualizationBlock,
    FastBoundingBoxes,
)
from roboflow_workflows.v2.engine.declaration import spec_of

from tests.v2.unit.blocks.visualization.annotations.support import (
    detections,
    image_of,
    pixels_of,
    scene,
)


def _run(implementation, image, predictions, **style):
    params = spec_of(BoundingBoxVisualizationBlock).validate_params(
        {"image": "$inputs.image", "predictions": "$inputs.predictions", **style}
    )
    result = implementation().run(
        **{**dict(params), "image": image, "predictions": predictions}
    )

    return result["image"]


@pytest.mark.parametrize("implementation", [AccurateBoundingBoxes, FastBoundingBoxes])
def test_square_one_pixel_borders_match_independent_opencv_oracle(implementation):
    pixels = scene()
    image = image_of(pixels)
    objects = (
        (-8, 5, 18, 22, 0, 0.9),
        (10, 12, 38, 35, 1, 0.8),
        (30, 1, 80, 30, 2, 0.7),
    )
    colors = [(241, 13, 29), (7, 229, 41), (19, 37, 233)]
    expected = pixels.copy()
    for (x1, y1, x2, y2, class_id, _), color in zip(objects, colors):
        cv2.rectangle(expected, (x1, y1), (x2, y2), color, thickness=1)

    output = _run(
        implementation,
        image,
        detections(image, objects),
        thickness=1,
        color_palette="CUSTOM",
        custom_colors=["#f10d1d", "#07e529", "#1325e9"],
    )

    np.testing.assert_array_equal(pixels_of(output), expected)
    np.testing.assert_array_equal(pixels_of(image), pixels)
    assert output.tensor_image.data_ptr() != image.tensor_image.data_ptr()
    assert output.image_id == image.image_id
    assert output.parent == image.parent
    assert output.root == image.root


@pytest.mark.parametrize("implementation", [AccurateBoundingBoxes, FastBoundingBoxes])
def test_track_colors_and_rounded_width_are_honored(implementation):
    image = image_of(np.zeros((64, 80, 3), dtype=np.uint8))
    found = detections(
        image,
        ((10, 10, 60, 50, 0, 0.9),),
        rows=[{"tracker_id": 1}],
    )
    style = dict(
        color_palette="CUSTOM", custom_colors=["red", "blue"], color_axis="TRACK"
    )
    thin = _run(implementation, image, found, thickness=1, **style)
    rounded = _run(implementation, image, found, thickness=5, roundness=0.8, **style)

    assert torch.equal(rounded.tensor_image[:, 10, 35], torch.tensor([0, 0, 255]))
    assert rounded.tensor_image[:, 10, 10].sum() == 0
    assert rounded.tensor_image.count_nonzero() > thin.tensor_image.count_nonzero()
    assert image.tensor_image.count_nonzero() == 0


@pytest.mark.parametrize("implementation", [AccurateBoundingBoxes, FastBoundingBoxes])
def test_empty_predictions_still_return_an_independent_image(implementation):
    image = image_of(scene())
    output = _run(implementation, image, detections(image, ()))

    assert torch.equal(output.tensor_image, image.tensor_image)
    output.tensor_image.zero_()
    assert image.tensor_image.count_nonzero() > 0


@pytest.mark.parametrize("implementation", [AccurateBoundingBoxes, FastBoundingBoxes])
def test_invalid_selected_width_fails_before_drawing(implementation):
    image = image_of(scene())
    original = image.tensor_image.clone()
    params = spec_of(BoundingBoxVisualizationBlock).validate_params(
        {
            "image": "$inputs.image",
            "predictions": "$inputs.predictions",
            "thickness": "$inputs.width",
        }
    )

    with pytest.raises(ValueError, match="thickness must be an integer >= 1"):
        implementation().run(
            **{
                **dict(params),
                "image": image,
                "predictions": detections(image),
                "thickness": 0,
            }
        )

    assert torch.equal(image.tensor_image, original)


def test_fast_rounded_box_preserves_visible_arc_when_straight_edges_are_off_screen():
    image = image_of(np.zeros((20, 20, 3), dtype=np.uint8))
    found = detections(image, ((-30, -30, 20, 20, 0, 0.9),))
    style = dict(thickness=1, roundness=1.0)

    expected = _run(AccurateBoundingBoxes, image, found, **style)
    actual = _run(FastBoundingBoxes, image, found, **style)

    assert expected.tensor_image.count_nonzero() > 0
    assert torch.equal(actual.tensor_image, expected.tensor_image)
    assert image.tensor_image.count_nonzero() == 0
