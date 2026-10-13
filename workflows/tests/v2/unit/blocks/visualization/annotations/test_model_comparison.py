"""``v2/model_comparison_visualization@v1`` against V1 and analytic pixels.

The V1 oracle is the latest V1 block run on the same rows (``support``).
Opacity 0.5 is an exact blend weight; V1's default 0.7 may differ by one
level at exact .5 ties (``blend``). Box clipping is checked analytically,
because V1 mishandles boxes with negative corners.
"""

from typing import Any

import numpy as np
import pytest
import torch
from roboflow_workflows.core_steps.visualizations.model_comparison.v1 import (
    ModelComparisonVisualizationBlockV1,
)
from roboflow_workflows.v2.blocks.visualization.annotations.model_comparison import (
    ModelComparisonVisualizationBlock,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.engine.declaration import spec_of
from roboflow_workflows.v2.engine.errors import ParamsValidationError

from inference_models.models.base.object_detection import Detections
from tests.v2.unit.blocks.visualization.annotations.support import (
    OBJECTS,
    detections,
    grid_view,
    image_of,
    instance_detections,
    object_masks,
    pixels_of,
    rle_detections,
    run_v1,
    scene,
    v1_detections,
)

GREEN, RED, BLACK = (0, 255, 0), (255, 0, 0), (0, 0, 0)


def compare(image: ImageData, a: Any, b: Any, **parameters: Any) -> np.ndarray:
    params = spec_of(ModelComparisonVisualizationBlock).validate_params(
        {
            **parameters,
            "image": "$inputs.image",
            "predictions_a": "$inputs.a",
            "predictions_b": "$inputs.b",
        }
    )
    arguments = {**dict(params), "image": image, "predictions_a": a, "predictions_b": b}
    result = ModelComparisonVisualizationBlock().run(**arguments)
    pixels = pixels_of(result["image"])

    return pixels


def v1_compare(pixels: np.ndarray, a: Any, b: Any, *, opacity: float) -> np.ndarray:
    expected = run_v1(
        ModelComparisonVisualizationBlockV1,
        pixels,
        predictions_a=a,
        predictions_b=b,
        color_a="GREEN",
        color_b="RED",
        background_color="BLACK",
        opacity=opacity,
    )

    return expected


def tinted(pixels: np.ndarray, color, *, opacity: float) -> np.ndarray:
    blended = np.round(pixels * (1 - opacity) + np.array(color) * opacity)

    return blended.astype(np.uint8)


def test_boxes_equal_v1_at_an_exact_opacity() -> None:
    # Given: model A found objects 0 and 1, model B found objects 1 and 2
    pixels = scene()
    image = image_of(pixels)

    # When
    drawn = compare(
        image,
        detections(image, OBJECTS[:2]),
        detections(image, OBJECTS[1:]),
        opacity=0.5,
    )

    # Then
    expected = v1_compare(
        pixels, v1_detections(OBJECTS[:2]), v1_detections(OBJECTS[1:]), opacity=0.5
    )
    np.testing.assert_array_equal(drawn, expected)


def test_default_opacity_is_within_one_level_of_v1() -> None:
    pixels = scene()
    image = image_of(pixels)

    drawn = compare(
        image, detections(image, OBJECTS[:2]), detections(image, OBJECTS[1:])
    )

    expected = v1_compare(
        pixels, v1_detections(OBJECTS[:2]), v1_detections(OBJECTS[1:]), opacity=0.7
    )
    assert np.abs(drawn.astype(int) - expected).max() <= 1


@pytest.mark.parametrize("carrier", ["dense", "rle"])
def test_mask_carriers_compare_masks_like_v1(carrier: str) -> None:
    # Given: A has masks (compared as masks), B has only boxes
    pixels = scene()
    image = image_of(pixels)
    masked = instance_detections(image) if carrier == "dense" else rle_detections(image)

    drawn = compare(image, masked, detections(image, OBJECTS[2:]), opacity=0.5)

    expected = v1_compare(
        pixels,
        v1_detections(masks=object_masks()),
        v1_detections(OBJECTS[2:]),
        opacity=0.5,
    )
    np.testing.assert_array_equal(drawn, expected)


def test_regions_get_their_colors_and_shared_pixels_stay() -> None:
    pixels = scene()
    image = image_of(pixels)
    a = [(0, 0, 20, 20, 0, 0.9)]
    b = [(10, 10, 30, 30, 1, 0.9)]

    drawn = compare(image, detections(image, a), detections(image, b), opacity=0.5)

    np.testing.assert_array_equal(drawn[10:20, 10:20], pixels[10:20, 10:20])
    np.testing.assert_array_equal(
        drawn[0:10, 0:10], tinted(pixels[0:10, 0:10], GREEN, opacity=0.5)
    )
    np.testing.assert_array_equal(
        drawn[20:30, 20:30], tinted(pixels[20:30, 20:30], RED, opacity=0.5)
    )
    np.testing.assert_array_equal(
        drawn[40:, 40:], tinted(pixels[40:, 40:], BLACK, opacity=0.5)
    )


def test_two_empty_inputs_tint_the_whole_image_like_v1() -> None:
    pixels = scene()
    image = image_of(pixels)

    drawn = compare(image, detections(image, ()), detections(image, ()), opacity=0.5)

    expected = v1_compare(pixels, v1_detections(()), v1_detections(()), opacity=0.5)
    np.testing.assert_array_equal(drawn, expected)
    np.testing.assert_array_equal(drawn, tinted(pixels, BLACK, opacity=0.5))


def test_box_partly_outside_the_image_is_clipped() -> None:
    # V1 slices with the negative corner and colors nothing; V2 clips.
    pixels = scene()
    image = image_of(pixels)
    a = [(-6.5, -4.0, 12.9, 9.0, 0, 0.9)]

    drawn = compare(image, detections(image, a), detections(image, ()), opacity=0.5)

    # Corners truncate toward zero, as V1: x in [0, 12), y in [0, 9).
    np.testing.assert_array_equal(
        drawn[0:9, 0:12], tinted(pixels[0:9, 0:12], GREEN, opacity=0.5)
    )
    np.testing.assert_array_equal(
        drawn[9:, :], tinted(pixels[9:, :], BLACK, opacity=0.5)
    )
    np.testing.assert_array_equal(
        drawn[:9, 12:], tinted(pixels[:9, 12:], BLACK, opacity=0.5)
    )


@pytest.mark.parametrize("quality", ["accurate", "coarse"])
def test_mask_view_is_read_at_the_requested_quality(quality: str) -> None:
    # Given: the same objects as a stride-4 mask view and as dense masks
    pixels = scene()
    image = image_of(pixels)
    view = grid_view(image)
    dense = view.full_res() if quality == "accurate" else None

    drawn = compare(
        image, view, detections(image, ()), opacity=0.5, mask_quality=quality
    )

    # Then: green exactly where the view's masks are, at that quality
    green = (drawn == tinted(pixels, GREEN, opacity=0.5)).all(axis=2)
    if dense is not None:
        expected = dense.mask.any(dim=0).numpy()
        np.testing.assert_array_equal(green, expected)
    assert 0 < green.sum() < green.size


def test_input_image_is_unchanged_and_output_keeps_identity() -> None:
    pixels = scene()
    image = image_of(pixels, image_id="kept")
    params = spec_of(ModelComparisonVisualizationBlock).validate_params(
        {
            "image": "$inputs.image",
            "predictions_a": "$inputs.a",
            "predictions_b": "$inputs.b",
        }
    )

    result = ModelComparisonVisualizationBlock().run(
        **{
            **dict(params),
            "image": image,
            "predictions_a": detections(image, OBJECTS[:1]),
            "predictions_b": detections(image, OBJECTS[1:]),
        }
    )

    np.testing.assert_array_equal(pixels_of(image), pixels)
    assert result["image"].image_id == "kept"
    assert result["image"].tensor_image.is_contiguous()


def test_noncontiguous_image_and_predictions_give_the_same_pixels() -> None:
    pixels = scene()
    image = image_of(pixels)
    strided = ImageData.from_tensor(
        torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 1, 0))).transpose(
            1, 2
        ),
        image_id="frame",
    )
    boxes = detections(image, OBJECTS)
    # Column-major (N, 4) boxes: a transposed view of a (4, N) tensor.
    strided_boxes = Detections(
        xyxy=boxes.xyxy.T.contiguous().T,
        class_id=boxes.class_id,
        confidence=boxes.confidence,
        image_metadata=detections(strided, OBJECTS).image_metadata,
    )

    expected = compare(image, boxes, detections(image, ()), opacity=0.5)
    drawn = compare(strided, strided_boxes, detections(strided, ()), opacity=0.5)

    assert not strided.tensor_image.is_contiguous()
    assert not strided_boxes.xyxy.is_contiguous()
    np.testing.assert_array_equal(drawn, expected)


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("color_a", "not-a-color", "color_a: invalid color"),
        ("background_color", "transparent", "background_color: invalid color"),
        ("opacity", 1.5, r"opacity must be a number in \[0.0, 1.0\]"),
        # A literal fails the shared Literal type; the wording differs.
        ("mask_quality", "fine", "mask_quality.*'accurate'"),
    ],
)
def test_literal_and_selected_values_fail_with_the_same_message(
    field: str, value: Any, message: str
) -> None:
    # Given: a literal at compile time, then the same value from a selector
    image = image_of(scene())
    with pytest.raises(ParamsValidationError, match=message):
        spec_of(ModelComparisonVisualizationBlock).validate_params(
            {
                "image": "$inputs.image",
                "predictions_a": "$inputs.a",
                "predictions_b": "$inputs.b",
                field: value,
            }
        )

    params = spec_of(ModelComparisonVisualizationBlock).validate_params(
        {
            "image": "$inputs.image",
            "predictions_a": "$inputs.a",
            "predictions_b": "$inputs.b",
            field: "$inputs.value",
        }
    )
    arguments = {
        **dict(params),
        "image": image,
        "predictions_a": detections(image),
        "predictions_b": detections(image),
        field: value,
    }
    with pytest.raises(ValueError, match=message):
        ModelComparisonVisualizationBlock().run(**arguments)


def test_copy_image_is_rejected_with_the_chaining_remedy() -> None:
    with pytest.raises(ParamsValidationError, match=r"\$steps.<previous>.image"):
        spec_of(ModelComparisonVisualizationBlock).validate_params(
            {
                "image": "$inputs.image",
                "predictions_a": "$inputs.a",
                "predictions_b": "$inputs.b",
                "copy_image": False,
            }
        )
