"""``v2/halo_visualization@v2``: pixels equal V1 ``halo_visualization@v2``.

The oracle is the latest V1 block (``sv.HaloAnnotator``) run on the same
rows; V1 test modules are not imported.

Every ``kernel_size`` is exact, odd and even: ``box_blur`` rounds as
``cv2.blur`` does. Dim colors are checked separately: there a 1-level glow
pixel sets ``max(gray)``, so any rounding difference would change the image
by up to ~160 levels.

One deliberate difference: sums above 255 saturate, where V1's ``uint8``
cast wraps them. That case is checked against ``_saturating_v1_halo``, V1's
formula recomputed here with OpenCV and NumPy and a clip before the cast.
"""

from typing import Tuple

import cv2
import numpy as np
import pytest
import torch
from roboflow_workflows.core_steps.visualizations.halo.v2 import (
    HaloVisualizationBlockV2,
)
from roboflow_workflows.v2.blocks.visualization.annotations.halo import (
    HaloVisualizationBlock,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.engine.declaration import spec_of
from roboflow_workflows.v2.engine.errors import ParamsValidationError

from inference_models.models.base.instance_segmentation import InstanceDetections
from tests.v2.unit.blocks.visualization.annotations.support import (
    HEIGHT,
    V1_COLOR_DEFAULTS,
    WIDTH,
    detections,
    grid_view,
    image_of,
    instance_detections,
    metadata_for,
    object_masks,
    pixels_of,
    rle_detections,
    run_v1,
    run_v2,
    scene,
    v1_detections,
)

# Two disjoint squares: a blue glow (low gray) beside a white one (high gray)
# on a bright image drives V1's sum above 255.
SQUARES = ((6.0, 10.0, 22.0, 26.0, 0, 0.9), (38.0, 10.0, 54.0, 26.0, 1, 0.8))
BLUE_AND_WHITE = {
    "color_palette": "CUSTOM",
    "custom_colors": ["#0000FF", "#FFFFFF"],
    "color_axis": "INDEX",
}


def _input_image() -> Tuple[np.ndarray, ImageData]:
    pixels = scene()

    return pixels, image_of(pixels)


def _assert_same_pixels(actual: np.ndarray, expected: np.ndarray) -> None:
    differ = (actual != expected).any(axis=2)
    assert not differ.any(), (
        f"{int(differ.sum())} pixels differ, first at (y, x) "
        f"{np.argwhere(differ)[:3].tolist()}"
    )


def _square_masks() -> np.ndarray:
    masks = np.zeros((2, HEIGHT, WIDTH), dtype=bool)
    for mask, (x1, y1, x2, y2, _, _) in zip(masks, SQUARES):
        mask[int(y1) : int(y2), int(x1) : int(x2)] = True

    return masks


def _squares(image: ImageData) -> InstanceDetections:
    masks = _square_masks()
    predictions = InstanceDetections(
        xyxy=torch.tensor([square[:4] for square in SQUARES]),
        class_id=torch.tensor([square[4] for square in SQUARES]),
        confidence=torch.tensor([square[5] for square in SQUARES]),
        mask=torch.from_numpy(masks),
        image_metadata=metadata_for(image),
    )

    return predictions


def _saturating_v1_halo(
    pixels: np.ndarray, masks: np.ndarray, colors_rgb, *, opacity: float, kernel: int
) -> Tuple[np.ndarray, np.ndarray]:
    # sv.HaloAnnotator on disjoint masks (paint order is irrelevant), in BGR.
    # Returns the clipped result and the unclipped float sums.
    scene_bgr = pixels[:, :, ::-1].astype(np.float64)
    colored = np.zeros(pixels.shape, dtype=np.uint8)
    for mask, (red, green, blue) in zip(masks, colors_rgb):
        colored[mask] = (blue, green, red)
    colored = cv2.blur(colored, (kernel, kernel))
    colored[masks.any(axis=0)] = 0
    gray = cv2.cvtColor(colored, cv2.COLOR_BGR2GRAY)
    alpha = (opacity * gray / gray.max())[:, :, np.newaxis]
    sums = scene_bgr * (1 - alpha) + colored * opacity
    clipped = np.clip(sums, 0, 255).astype(np.uint8)[:, :, ::-1]

    return clipped, sums[:, :, ::-1]


def _halo_and_v1(opacity: float, kernel_size: int) -> Tuple[np.ndarray, np.ndarray]:
    pixels, image = _input_image()
    style = dict(opacity=opacity, kernel_size=kernel_size)

    drawn = run_v2(
        HaloVisualizationBlock,
        image=image,
        predictions=instance_detections(image),
        **style,
    )
    expected = run_v1(
        HaloVisualizationBlockV2,
        pixels,
        predictions=v1_detections(masks=object_masks()),
        **style,
        **V1_COLOR_DEFAULTS,
    )
    if kernel_size > 1:
        assert (expected != pixels).any(), "the case must draw a glow"

    return pixels_of(drawn), expected


@pytest.mark.parametrize(
    ("opacity", "kernel_size"),
    [(0.8, 41), (0.5, 5), (1.0, 9), (0.3, 101), (0.65, 3), (1.0, 1)],
)
def test_halos_with_odd_kernels_equal_v1(opacity: float, kernel_size: int) -> None:
    # kernel_size 101 is larger than the image: the blur reflects repeatedly.
    drawn, expected = _halo_and_v1(opacity, kernel_size)

    _assert_same_pixels(drawn, expected)


@pytest.mark.parametrize(
    "palette", ["ROBOFLOW", "Matplotlib Viridis", "Matplotlib Pastel1"]
)
@pytest.mark.parametrize("seed", [1, 2])
def test_palettes_equal_v1(palette: str, seed: int) -> None:
    # Many glow colors: the gray weights must be OpenCV's for every value.
    pixels = scene(seed=seed)
    image = image_of(pixels)
    colors = {**V1_COLOR_DEFAULTS, "color_palette": palette}

    drawn = run_v2(
        HaloVisualizationBlock,
        image=image,
        predictions=instance_detections(image),
        kernel_size=21,
        **colors,
    )

    expected = run_v1(
        HaloVisualizationBlockV2,
        pixels,
        predictions=v1_detections(masks=object_masks()),
        opacity=0.8,
        kernel_size=21,
        **colors,
    )
    _assert_same_pixels(pixels_of(drawn), expected)


@pytest.mark.parametrize("kernel_size", [2, 4, 6, 8, 10, 40, 64])
@pytest.mark.parametrize("opacity", [0.8, 0.5])
def test_halos_with_even_kernels_equal_v1(opacity: float, kernel_size: int) -> None:
    drawn, expected = _halo_and_v1(opacity, kernel_size)

    _assert_same_pixels(drawn, expected)


def _one_mask_halo_and_v1(
    pixels: np.ndarray, mask: np.ndarray, color: str, *, kernel_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    # One CUSTOM-colored mask at opacity 0.8; its box spans the mask's pixels.
    image = image_of(pixels)
    rows, columns = np.nonzero(mask)
    box = (
        (float(columns.min()), float(rows.min()), columns.max() + 1.0, rows.max() + 1.0)
        if len(rows)
        else (0.0, 0.0, 1.0, 1.0)
    )
    predictions = InstanceDetections(
        xyxy=torch.tensor([box]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([0.9]),
        mask=torch.from_numpy(mask[None]),
        image_metadata=metadata_for(image),
    )
    style = dict(
        opacity=0.8,
        kernel_size=kernel_size,
        color_palette="CUSTOM",
        custom_colors=[color],
        color_axis="CLASS",
    )

    drawn = run_v2(
        HaloVisualizationBlock, image=image, predictions=predictions, **style
    )

    expected = run_v1(
        HaloVisualizationBlockV2,
        pixels,
        predictions=v1_detections([(*box, 0, 0.9)], masks=mask[None]),
        **{**V1_COLOR_DEFAULTS, **style},
    )

    return pixels_of(drawn), expected


def _gray_image_with_mask(height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    pixels = np.full((height, width, 3), 200, dtype=np.uint8)
    mask = np.zeros((height, width), dtype=bool)
    mask[4:12, 5:14] = True

    return pixels, mask


def test_a_dim_glow_darkens_like_v1() -> None:
    # The glow of #010203 with kernel 4 is 0 or 1 level. Where gray is 1 it
    # is max(gray), so alpha is the full 0.8: 200 becomes about 40.
    pixels, mask = _gray_image_with_mask(17, 19)

    drawn, expected = _one_mask_halo_and_v1(pixels, mask, "#010203", kernel_size=4)

    _assert_same_pixels(drawn, expected)
    assert drawn[3, 7].tolist() == [39, 40, 40]
    assert drawn[0, 0].tolist() == [200, 200, 200]


DIM_COLORS = ["#010203", "#000003", "#020202", "#040000", "#000100", "#010101"]


@pytest.mark.parametrize("kernel_size", [2, 3, 4, 6, 8, 16, 40])
@pytest.mark.parametrize("color", DIM_COLORS)
def test_dim_colors_equal_v1(color: str, kernel_size: int) -> None:
    pixels, mask = _gray_image_with_mask(17, 19)

    drawn, expected = _one_mask_halo_and_v1(
        pixels, mask, color, kernel_size=kernel_size
    )

    _assert_same_pixels(drawn, expected)


def _mask_at(height: int, width: int, *spans: Tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for y1, y2, x1, x2 in spans:
        mask[y1:y2, x1:x2] = True

    return mask


@pytest.mark.parametrize(
    "mask",
    [
        _mask_at(1, 1, (0, 1, 0, 1)),
        _mask_at(1, 19, (0, 1, 3, 6)),
        _mask_at(23, 1, (10, 13, 0, 1)),
        _mask_at(5, 3, (0, 2, 0, 1)),
        _mask_at(17, 19, (0, 6, 12, 19)),  # touches the top and right edges
        _mask_at(17, 19, (2, 5, 2, 5), (11, 15, 13, 17)),  # two blobs
        _mask_at(17, 19),  # empty
    ],
    ids=["1x1", "1x19", "23x1", "5x3", "edges", "two-blobs", "empty"],
)
@pytest.mark.parametrize("kernel_size", [2, 4, 40])
def test_dim_glows_at_edges_and_of_odd_masks_equal_v1(
    mask: np.ndarray, kernel_size: int
) -> None:
    pixels = np.full((*mask.shape, 3), 180, dtype=np.uint8)
    pixels[::2] = 90

    drawn, expected = _one_mask_halo_and_v1(
        pixels, mask, "#060606", kernel_size=kernel_size
    )

    _assert_same_pixels(drawn, expected)
    if mask.any() and not mask.all():
        assert (expected != pixels).any(), "the case must draw a glow"


def test_track_colors_equal_v1() -> None:
    pixels, image = _input_image()
    rows = [{"tracker_id": 4}, {"tracker_id": 9}, {"tracker_id": 4}]
    dense = instance_detections(image)
    tracked = InstanceDetections(
        xyxy=dense.xyxy,
        class_id=dense.class_id,
        confidence=dense.confidence,
        mask=dense.mask,
        image_metadata=dense.image_metadata,
        bboxes_metadata=rows,
    )
    colors = {**V1_COLOR_DEFAULTS, "color_axis": "TRACK"}

    drawn = run_v2(
        HaloVisualizationBlock,
        image=image,
        predictions=tracked,
        kernel_size=41,
        **colors,
    )

    expected = run_v1(
        HaloVisualizationBlockV2,
        pixels,
        predictions=v1_detections(masks=object_masks(), tracker_id=[4, 9, 4]),
        opacity=0.8,
        kernel_size=41,
        **colors,
    )
    _assert_same_pixels(pixels_of(drawn), expected)


def test_sums_above_255_saturate_where_v1_wraps() -> None:
    pixels = np.full((HEIGHT, WIDTH, 3), 240, dtype=np.uint8)
    image = image_of(pixels)
    style = dict(opacity=0.8, kernel_size=9)

    drawn = pixels_of(
        run_v2(
            HaloVisualizationBlock,
            image=image,
            predictions=_squares(image),
            **style,
            **BLUE_AND_WHITE,
        )
    )

    v1 = run_v1(
        HaloVisualizationBlockV2,
        pixels,
        predictions=v1_detections(SQUARES, masks=_square_masks()),
        **style,
        **{**V1_COLOR_DEFAULTS, **BLUE_AND_WHITE},
    )
    saturated, sums = _saturating_v1_halo(
        pixels,
        _square_masks(),
        [(0, 0, 255), (255, 255, 255)],
        opacity=0.8,
        kernel=9,
    )
    overflow = sums >= 256
    assert overflow.any(), "the case must overflow"
    _assert_same_pixels(drawn, saturated)
    assert (drawn[overflow] == 255).all()
    assert (drawn[~overflow] == v1[~overflow]).all(), "V1 pixels below 256 kept"
    assert (v1[overflow] < 255).all(), "V1 wraps these sums"


def test_pixels_inside_the_masks_never_change() -> None:
    pixels, image = _input_image()

    drawn = pixels_of(
        run_v2(
            HaloVisualizationBlock,
            image=image,
            predictions=instance_detections(image),
            opacity=1.0,
            kernel_size=15,
        )
    )

    inside = object_masks().any(axis=0)
    assert (drawn[inside] == pixels[inside]).all()
    assert (drawn[~inside] != pixels[~inside]).any()


@pytest.mark.parametrize(
    "style",
    [
        {"opacity": 0.0},
        # Gray of this dim red is 0 everywhere: V1 returns the image unchanged.
        {"color_palette": "CUSTOM", "custom_colors": ["#010000"]},
    ],
)
def test_an_invisible_glow_leaves_the_image_unchanged_like_v1(style) -> None:
    pixels, image = _input_image()

    drawn = run_v2(
        HaloVisualizationBlock,
        image=image,
        predictions=instance_detections(image),
        **style,
    )

    expected = run_v1(
        HaloVisualizationBlockV2,
        pixels,
        predictions=v1_detections(masks=object_masks()),
        **{**V1_COLOR_DEFAULTS, "opacity": 0.8, "kernel_size": 40, **style},
    )
    _assert_same_pixels(pixels_of(drawn), expected)
    _assert_same_pixels(pixels_of(drawn), pixels)


def test_rle_masks_and_an_accurate_view_equal_their_dense_masks() -> None:
    _, image = _input_image()
    view = grid_view(image)
    dense_view = InstanceDetections(
        xyxy=view.detections.xyxy,
        class_id=view.detections.class_id,
        confidence=view.detections.confidence,
        mask=view.full_res().mask,
        image_metadata=view.detections.image_metadata,
    )

    from_rle = run_v2(
        HaloVisualizationBlock, image=image, predictions=rle_detections(image)
    )
    from_view = run_v2(HaloVisualizationBlock, image=image, predictions=view)

    from_dense = run_v2(
        HaloVisualizationBlock, image=image, predictions=instance_detections(image)
    )
    from_dense_view = run_v2(
        HaloVisualizationBlock, image=image, predictions=dense_view
    )
    _assert_same_pixels(pixels_of(from_rle), pixels_of(from_dense))
    _assert_same_pixels(pixels_of(from_view), pixels_of(from_dense_view))


def test_a_coarse_view_glows_without_the_accurate_upscaling() -> None:
    pixels, image = _input_image()
    view = grid_view(image)

    drawn = run_v2(
        HaloVisualizationBlock, image=image, predictions=view, mask_quality="coarse"
    )

    assert not view.is_full_res_materialized()
    assert (pixels_of(drawn) != pixels).any()


def test_noncontiguous_image_and_masks_draw_like_contiguous_ones() -> None:
    pixels, image = _input_image()
    dense = instance_detections(image)
    wide_image = torch.zeros((3, HEIGHT, WIDTH + 3), dtype=torch.uint8)
    wide_image[:, :, :WIDTH] = image.tensor_image
    wide_masks = torch.zeros((3, HEIGHT, WIDTH + 5), dtype=torch.bool)
    wide_masks[:, :, :WIDTH] = dense.mask
    strided = InstanceDetections(
        xyxy=dense.xyxy,
        class_id=dense.class_id,
        confidence=dense.confidence,
        mask=wide_masks[:, :, :WIDTH],
        image_metadata=dense.image_metadata,
    )
    strided_image = ImageData.from_tensor(wide_image[:, :, :WIDTH], image_id="frame")

    drawn = run_v2(HaloVisualizationBlock, image=strided_image, predictions=strided)

    expected = run_v2(HaloVisualizationBlock, image=image, predictions=dense)
    _assert_same_pixels(pixels_of(drawn), pixels_of(expected))
    assert drawn.tensor_image.is_contiguous()


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

    drawn = run_v2(HaloVisualizationBlock, image=image, predictions=empty)

    _assert_same_pixels(pixels_of(drawn), pixels)
    assert drawn.tensor_image.data_ptr() != image.tensor_image.data_ptr()


def test_input_image_and_masks_are_not_changed() -> None:
    _, image = _input_image()
    dense = instance_detections(image)
    image_before, masks_before = image.tensor_image.clone(), dense.mask.clone()

    drawn = run_v2(HaloVisualizationBlock, image=image, predictions=dense)

    assert torch.equal(image.tensor_image, image_before)
    assert torch.equal(dense.mask, masks_before)
    assert not torch.equal(drawn.tensor_image, image_before)


def test_boxes_without_masks_are_an_error() -> None:
    _, image = _input_image()

    with pytest.raises(ValueError, match="needs instance masks"):
        run_v2(HaloVisualizationBlock, image=image, predictions=detections(image))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("opacity", 1.5, "opacity must be a number in"),
        ("kernel_size", 0, "kernel_size must be an integer >= 1"),
        ("mask_quality", "fast", "mask_quality must be one of"),
    ],
)
def test_invalid_values_fail_as_literal_and_as_selector(
    field: str, value, message: str
) -> None:
    _, image = _input_image()
    spec = spec_of(HaloVisualizationBlock)
    base = {"image": "$inputs.image", "predictions": "$inputs.predictions"}

    with pytest.raises(ParamsValidationError, match=field):
        spec.validate_params({**base, field: value})
    params = dict(spec.validate_params({**base, field: f"$inputs.{field}"}))
    params.update(
        {"image": image, "predictions": instance_detections(image), field: value}
    )
    with pytest.raises(ValueError, match=message):
        HaloVisualizationBlock().run(**params)
