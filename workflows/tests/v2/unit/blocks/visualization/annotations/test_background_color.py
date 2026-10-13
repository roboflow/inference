"""``v2/background_color_visualization@v1``: cover everything outside detections.

Expected pixels come from small hand-computed oracles (which pixels a box or
mask keeps, the blend formula) and from the latest V1 block on the same
data. One deliberate difference is pinned: boxes starting left of or above
the image are clipped, where V1's negative slice keeps nothing.
"""

from typing import Any

import numpy as np
import pytest
import torch
from roboflow_workflows.core_steps.visualizations.background_color.v1 import (
    BackgroundColorVisualizationBlockV1,
)
from roboflow_workflows.v2.blocks.visualization.annotations.background_color import (
    BackgroundColorVisualizationBlock,
)
from roboflow_workflows.v2.engine.declaration import spec_of
from roboflow_workflows.v2.engine.errors import ParamsValidationError

from inference_models.models.base.instance_segmentation import InstanceDetections
from tests.v2.unit.blocks.visualization.annotations.support import (
    detections,
    grid_view,
    image_of,
    instance_detections,
    key_points,
    metadata_for,
    object_masks,
    pixels_of,
    rle_detections,
    run_v1,
    run_v2,
    scene,
    v1_detections,
)

GREY = 100


def cover(predictions_of: Any, *, height: int = 6, width: int = 8, **params: Any):
    """Run the block on a uniform grey image; return input and output pixels."""
    pixels = np.full((height, width, 3), GREY, dtype=np.uint8)
    image = image_of(pixels)
    output = run_v2(
        BackgroundColorVisualizationBlock,
        image=image,
        predictions=predictions_of(image),
        **params,
    )

    return pixels, pixels_of(output)


def test_a_box_keeps_its_pixels_with_exclusive_ends_and_covers_the_rest() -> None:
    _, covered = cover(
        lambda image: detections(image, [(1.6, 1.2, 4.0, 3.9, 0, 0.9)]),
        color="WHITE",
        opacity=1.0,
    )

    expected = np.full((6, 8, 3), 255, dtype=np.uint8)
    expected[1:3, 1:4] = GREY
    np.testing.assert_array_equal(covered, expected)


def test_opacity_mixes_the_color_into_the_background() -> None:
    _, covered = cover(
        lambda image: detections(image, []), color="rgb(255, 0, 40)", opacity=0.25
    )

    # round(255 * .25 + 100 * .75) = 139, round(0 + 75) = 75, round(10 + 75) = 85.
    np.testing.assert_array_equal(covered[0, 0], (139, 75, 85))
    assert (covered == covered[0, 0]).all()


def test_masks_keep_their_exact_shapes() -> None:
    pixels = scene()
    image = image_of(pixels)
    masks = object_masks()

    output = run_v2(
        BackgroundColorVisualizationBlock,
        image=image,
        predictions=instance_detections(image),
        opacity=1.0,
    )

    kept = masks.any(axis=0)
    np.testing.assert_array_equal(pixels_of(output)[kept], pixels[kept])
    assert not pixels_of(output)[~kept].any()


@pytest.mark.parametrize("opacity", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("color", ["BLACK", "#3366CC", "bgr(10, 200, 30)"])
@pytest.mark.parametrize("carrier", ["boxes", "dense", "rle"])
def test_covered_images_equal_v1(opacity: float, color: str, carrier: str) -> None:
    pixels = scene()
    image = image_of(pixels)
    predictions, v1_masks = {
        "boxes": (detections(image), None),
        "dense": (instance_detections(image), object_masks()),
        "rle": (rle_detections(image), object_masks()),
    }[carrier]

    output = run_v2(
        BackgroundColorVisualizationBlock,
        image=image,
        predictions=predictions,
        color=color,
        opacity=opacity,
    )
    expected = run_v1(
        BackgroundColorVisualizationBlockV1,
        pixels,
        predictions=v1_detections(masks=v1_masks),
        color=color,
        opacity=opacity,
    )

    np.testing.assert_array_equal(pixels_of(output), expected)


def test_other_opacities_are_within_one_level_of_v1() -> None:
    pixels = scene()
    image = image_of(pixels)

    output = run_v2(
        BackgroundColorVisualizationBlock,
        image=image,
        predictions=detections(image),
        color="#3366CC",
        opacity=0.3,
    )
    expected = run_v1(
        BackgroundColorVisualizationBlockV1,
        pixels,
        predictions=v1_detections(),
        color="#3366CC",
        opacity=0.3,
    )

    difference = np.abs(pixels_of(output).astype(int) - expected.astype(int))
    assert difference.max() <= 1


def test_a_box_past_the_top_left_corner_is_clipped_unlike_v1() -> None:
    box = (-3.5, -2.0, 3.0, 3.0, 0, 0.9)
    pixels, covered = cover(lambda image: detections(image, [box]), opacity=1.0)
    v1 = run_v1(
        BackgroundColorVisualizationBlockV1,
        pixels,
        predictions=v1_detections([box]),
        color="BLACK",
        opacity=1.0,
    )

    expected = np.zeros_like(pixels)
    expected[0:3, 0:3] = GREY
    np.testing.assert_array_equal(covered, expected)
    # V1 slices [-3:3] and [-2:3]: from the far edge, so nothing is kept.
    assert not v1.any()


def test_a_box_past_the_bottom_right_corner_keeps_its_inside_part() -> None:
    _, covered = cover(
        lambda image: detections(image, [(5.0, 4.0, 30.0, 20.0, 0, 0.9)]), opacity=1.0
    )

    expected = np.zeros((6, 8, 3), dtype=np.uint8)
    expected[4:6, 5:8] = GREY
    np.testing.assert_array_equal(covered, expected)


@pytest.mark.parametrize("carrier", ["boxes", "dense"])
def test_no_rows_cover_the_whole_image(carrier: str) -> None:
    def empty(image):
        if carrier == "boxes":
            return detections(image, [])
        return InstanceDetections(
            xyxy=torch.zeros((0, 4)),
            class_id=torch.zeros((0,), dtype=torch.int64),
            confidence=torch.zeros((0,)),
            mask=torch.zeros((0, 6, 8), dtype=torch.bool),
            image_metadata=metadata_for(image),
        )

    _, covered = cover(empty, opacity=1.0, color="WHITE")

    assert (covered == 255).all()


def test_keypoints_keep_their_boxes() -> None:
    pixels = scene()
    image = image_of(pixels)

    from_keypoints = run_v2(
        BackgroundColorVisualizationBlock,
        image=image,
        predictions=key_points(image),
    )
    from_boxes = run_v2(
        BackgroundColorVisualizationBlock,
        image=image,
        predictions=key_points(image)[1],
    )

    np.testing.assert_array_equal(pixels_of(from_keypoints), pixels_of(from_boxes))


def test_a_mask_view_keeps_its_masks_at_the_requested_quality() -> None:
    image = image_of(scene())
    coarse_view, accurate_view = grid_view(image), grid_view(image)

    coarse = run_v2(
        BackgroundColorVisualizationBlock,
        image=image,
        predictions=coarse_view,
        mask_quality="coarse",
        opacity=1.0,
    )
    accurate = run_v2(
        BackgroundColorVisualizationBlock,
        image=image,
        predictions=accurate_view,
        opacity=1.0,
    )

    assert not coarse_view.is_full_res_materialized()
    assert accurate_view.is_full_res_materialized()
    # Coarse masks are whole 4 x 4 cells.
    kept = pixels_of(coarse).any(axis=2)
    cells = kept.reshape(12, 4, 16, 4)
    assert (cells.all(axis=(1, 3)) | ~cells.any(axis=(1, 3))).all()
    assert pixels_of(accurate).any()


def test_the_input_image_is_unchanged_and_the_output_keeps_its_identity() -> None:
    pixels = scene()
    image = image_of(pixels)

    output = run_v2(
        BackgroundColorVisualizationBlock,
        image=image,
        predictions=instance_detections(image),
        opacity=1.0,
    )

    np.testing.assert_array_equal(pixels_of(image), pixels)
    assert output.image_id == image.image_id
    assert output.root == image.root


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("color", "NOT_A_COLOR", "color: invalid color 'NOT_A_COLOR'"),
        ("opacity", -0.1, "opacity must be a number in [0.0, 1.0]"),
    ],
)
def test_an_invalid_value_fails_alike_as_literal_and_as_selected_value(
    field: str, value: Any, message: str
) -> None:
    image = image_of(scene())

    with pytest.raises(ParamsValidationError) as compile_error:
        spec_of(BackgroundColorVisualizationBlock).validate_params(
            {"image": "$inputs.image", "predictions": "$inputs.p", field: value}
        )
    with pytest.raises(ValueError) as run_error:
        run_selected(image, detections(image), {field: value})

    assert message in str(compile_error.value) + str(compile_error.value.__cause__)
    assert message in str(run_error.value)


def test_an_unknown_selected_mask_quality_fails_even_for_boxes() -> None:
    image = image_of(scene())

    with pytest.raises(ValueError, match="mask_quality must be one of"):
        run_selected(image, detections(image), {"mask_quality": "fast"})


def run_selected(image: Any, predictions: Any, selected: dict) -> Any:
    """Validate ``selected`` fields as selectors, then run with their values."""
    params = spec_of(BackgroundColorVisualizationBlock).validate_params(
        {
            **{name: f"$inputs.{name}" for name in selected},
            "image": "$inputs.image",
            "predictions": "$inputs.predictions",
        }
    )
    arguments = {**dict(params), **selected, "image": image, "predictions": predictions}
    result = BackgroundColorVisualizationBlock().run(**arguments)

    return result["image"]
