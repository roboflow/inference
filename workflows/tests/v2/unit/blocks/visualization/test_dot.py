"""``v2/dot_visualization@v1``: pixels equal V1 ``dot_visualization@v1``.

The oracle is the latest V1 block (``sv.DotAnnotator``) run on the same rows;
V1 test modules are not imported. ``CENTER_OF_MASS`` reads masks at
``mask_quality``; every other position must leave a mask view untouched.
"""

from typing import Tuple

import numpy as np
import pytest
import torch
from roboflow_workflows.core_steps.visualizations.dot.v1 import DotVisualizationBlockV1
from roboflow_workflows.v2.blocks.visualization.annotations.dot import (
    DotVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.params import (
    POSITIONS_WITH_MASS,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.engine.declaration import spec_of
from roboflow_workflows.v2.engine.errors import ParamsValidationError

from inference_models.models.base.instance_segmentation import InstanceDetections
from tests.v2.unit.blocks.visualization.annotations.support import (
    V1_COLOR_DEFAULTS,
    detections,
    grid_view,
    image_of,
    instance_detections,
    object_masks,
    pixels_of,
    run_v1,
    run_v2,
    scene,
    v1_detections,
)

BOX_POSITIONS = POSITIONS_WITH_MASS[:-1]
# Anchors on and beyond every frame edge; the last row is fully outside.
BEYOND_THE_FRAME = (
    (-6.0, -4.0, 6.0, 4.0, 0, 0.9),
    (58.5, 40.25, 70.0, 52.0, 1, 0.8),
    (80.0, 60.0, 90.0, 70.0, 2, 0.7),
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
    ("radius", "outline_thickness"), [(4, 0), (6, 2), (0, 0), (3, 5)]
)
def test_dots_on_box_anchors_equal_v1(
    position: str, radius: int, outline_thickness: int
) -> None:
    pixels, image = _input_image()
    style = dict(position=position, radius=radius, outline_thickness=outline_thickness)

    drawn = run_v2(
        DotVisualizationBlock, image=image, predictions=detections(image), **style
    )

    expected = run_v1(
        DotVisualizationBlockV1,
        pixels,
        predictions=v1_detections(),
        **style,
        **V1_COLOR_DEFAULTS,
    )
    _assert_same_pixels(pixels_of(drawn), expected)


def test_dots_crossing_or_beyond_the_frame_are_clipped_like_v1() -> None:
    pixels, image = _input_image()
    style = dict(position="BOTTOM_RIGHT", radius=5, outline_thickness=2)

    drawn = run_v2(
        DotVisualizationBlock,
        image=image,
        predictions=detections(image, BEYOND_THE_FRAME),
        **style,
    )

    expected = run_v1(
        DotVisualizationBlockV1,
        pixels,
        predictions=v1_detections(BEYOND_THE_FRAME),
        **style,
        **V1_COLOR_DEFAULTS,
    )
    _assert_same_pixels(pixels_of(drawn), expected)


def test_dots_at_the_centre_of_mass_of_dense_masks_equal_v1() -> None:
    pixels, image = _input_image()
    style = dict(position="CENTER_OF_MASS", radius=3, outline_thickness=1)

    drawn = run_v2(
        DotVisualizationBlock,
        image=image,
        predictions=instance_detections(image),
        **style,
    )

    expected = run_v1(
        DotVisualizationBlockV1,
        pixels,
        predictions=v1_detections(masks=object_masks()),
        **style,
        **V1_COLOR_DEFAULTS,
    )
    _assert_same_pixels(pixels_of(drawn), expected)


def test_centre_of_mass_reads_noncontiguous_masks_like_contiguous_ones() -> None:
    _, image = _input_image()
    dense = instance_detections(image)
    wide = torch.zeros((3, image.height, image.width + 7), dtype=torch.bool)
    wide[:, :, : image.width] = dense.mask
    strided = InstanceDetections(
        xyxy=dense.xyxy,
        class_id=dense.class_id,
        confidence=dense.confidence,
        mask=wide[:, :, : image.width],
        image_metadata=dense.image_metadata,
    )
    assert not strided.mask.is_contiguous()

    drawn = run_v2(
        DotVisualizationBlock,
        image=image,
        predictions=strided,
        position="CENTER_OF_MASS",
    )

    expected = run_v2(
        DotVisualizationBlock,
        image=image,
        predictions=dense,
        position="CENTER_OF_MASS",
    )
    _assert_same_pixels(pixels_of(drawn), pixels_of(expected))


@pytest.mark.parametrize("position", ["CENTER", "TOP_LEFT"])
def test_box_positions_never_read_view_masks(position: str) -> None:
    _, image = _input_image()
    view = grid_view(image)

    drawn = run_v2(
        DotVisualizationBlock, image=image, predictions=view, position=position
    )

    expected = run_v2(
        DotVisualizationBlock,
        image=image,
        predictions=detections(image),
        position=position,
    )
    _assert_same_pixels(pixels_of(drawn), pixels_of(expected))
    assert not view.is_full_res_materialized()


def test_centre_of_mass_of_a_view_follows_mask_quality() -> None:
    _, image = _input_image()
    accurate_view, coarse_view = grid_view(image), grid_view(image)

    accurate = run_v2(
        DotVisualizationBlock,
        image=image,
        predictions=accurate_view,
        position="CENTER_OF_MASS",
    )
    coarse = run_v2(
        DotVisualizationBlock,
        image=image,
        predictions=coarse_view,
        position="CENTER_OF_MASS",
        mask_quality="coarse",
    )

    reference = InstanceDetections(
        xyxy=accurate_view.detections.xyxy,
        class_id=accurate_view.detections.class_id,
        confidence=accurate_view.detections.confidence,
        mask=accurate_view.full_res().mask,
        image_metadata=accurate_view.detections.image_metadata,
    )
    expected = run_v2(
        DotVisualizationBlock,
        image=image,
        predictions=reference,
        position="CENTER_OF_MASS",
    )
    _assert_same_pixels(pixels_of(accurate), pixels_of(expected))
    assert not coarse_view.is_full_res_materialized()
    assert (pixels_of(coarse) != pixels_of(image)).any()


def test_centre_of_mass_without_masks_is_an_error_naming_the_remedy() -> None:
    _, image = _input_image()

    with pytest.raises(ValueError, match="CENTER_OF_MASS needs instance masks"):
        run_v2(
            DotVisualizationBlock,
            image=image,
            predictions=detections(image),
            position="CENTER_OF_MASS",
        )


def test_empty_predictions_give_an_unchanged_new_image() -> None:
    pixels, image = _input_image()
    dense = instance_detections(image)
    empty = InstanceDetections(
        xyxy=dense.xyxy[:0],
        class_id=dense.class_id[:0],
        confidence=dense.confidence[:0],
        mask=dense.mask[:0],
        image_metadata=dense.image_metadata,
    )

    drawn = run_v2(
        DotVisualizationBlock,
        image=image,
        predictions=empty,
        position="CENTER_OF_MASS",
    )

    _assert_same_pixels(pixels_of(drawn), pixels)
    assert drawn.tensor_image.data_ptr() != image.tensor_image.data_ptr()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("radius", -1, "radius must be an integer >= 0"),
        ("outline_thickness", -1, "outline_thickness must be an integer >= 0"),
        ("position", "MIDDLE", "position must be one of"),
        ("mask_quality", "best", "mask_quality must be one of"),
    ],
)
def test_invalid_values_fail_as_literal_and_as_selector(
    field: str, value, message: str
) -> None:
    _, image = _input_image()
    spec = spec_of(DotVisualizationBlock)
    base = {"image": "$inputs.image", "predictions": "$inputs.predictions"}

    with pytest.raises(ParamsValidationError, match=field):
        spec.validate_params({**base, field: value})
    params = dict(spec.validate_params({**base, field: f"$inputs.{field}"}))
    params.update({"image": image, "predictions": detections(image), field: value})
    with pytest.raises(ValueError, match=message):
        DotVisualizationBlock().run(**params)
