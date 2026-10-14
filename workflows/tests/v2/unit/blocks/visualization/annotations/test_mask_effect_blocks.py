"""Correctness of the mask and region-effect annotation blocks on the CPU.

Families: mask, polygon, blur, semantic_segmentation, crop and pixelate.

Oracles::

    existing V1 block   the latest V1 class run on the same rows (support.run_v1)
                        exact where the port promises V1 pixels; <= 1 level only
                        where its docstring documents resampling (crop, pixelate
                        cells) or OpenCV's 8-bit even-kernel blur rounding
    mask view           the same block on dense masks: view.full_res() for
                        mask_quality accurate, nearest_full_res(view) for coarse
    semantic grid       the mask block over one dense mask per painted class

Views count only their accurate full-resolution materializations
(``materialization_count``); a zero count is not a claim of zero allocation.
"""

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import pytest
import torch
from roboflow_workflows.core_steps.visualizations.blur.v1 import (
    BlurVisualizationBlockV1,
)
from roboflow_workflows.core_steps.visualizations.crop.v1 import (
    CropVisualizationBlockV1,
)
from roboflow_workflows.core_steps.visualizations.mask.v1 import (
    MaskVisualizationBlockV1,
)
from roboflow_workflows.core_steps.visualizations.pixelate.v1 import (
    PixelateVisualizationBlockV1,
)
from roboflow_workflows.core_steps.visualizations.polygon.v2 import (
    PolygonVisualizationBlockV2,
)
from roboflow_workflows.v2.blocks.visualization.annotations import (
    BlurVisualizationBlock,
    CropVisualizationBlock,
    MaskVisualizationBlock,
    PixelateVisualizationBlock,
    PolygonVisualizationBlock,
    SemanticSegmentationVisualizationBlock,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.masks import SegmentationView
from roboflow_workflows.v2.data.masks.operations import nearest_full_res
from roboflow_workflows.v2.data.row_identity import RowRef
from roboflow_workflows.v2.engine.declaration import spec_of

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.semantic_segmentation import (
    SemanticSegmentationResult,
)

from .support import (
    OBJECTS,
    V1_COLOR_DEFAULTS,
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
    semantic,
    v1_detections,
)

ACCURATE, COARSE = "accurate", "coarse"
MASK_QUALITY_BLOCKS = (
    MaskVisualizationBlock,
    PolygonVisualizationBlock,
    BlurVisualizationBlock,
)
# support.object_masks cuts this hole into row 1 (the person).
HOLE = (slice(28, 32), slice(30, 36))
# Three boxes that do not overlap, for per-box pixelate cell checks.
SEPARATE_OBJECTS = (
    (2.3, 3.7, 30.8, 21.2, 0, 0.9),
    (34.0, 5.0, 61.6, 40.9, 1, 0.8),
    (3.0, 26.0, 29.0, 46.0, 2, 0.7),
)
CROP_POSITIONS = ("TOP_CENTER", "CENTER", "BOTTOM_LEFT", "TOP_RIGHT", "CENTER_LEFT")


@pytest.fixture
def pixels() -> np.ndarray:
    return scene()


@pytest.fixture
def image(pixels: np.ndarray) -> ImageData:
    return image_of(pixels)


# ---------------------------------------------------------------- helpers


def run_selected(
    block_class: type,
    *,
    image: ImageData,
    predictions: Any,
    values: Dict[str, Any],
    **literals: Any,
) -> ImageData:
    """Run with ``values`` bound to ``$inputs`` selectors, as the engine does."""
    selectors = {name: f"$inputs.{name}" for name in values}
    params = spec_of(block_class).validate_params(
        {
            **literals,
            **selectors,
            "image": "$inputs.image",
            "predictions": "$inputs.predictions",
        }
    )
    arguments = {**dict(params), **values, "image": image, "predictions": predictions}
    result = (getattr(block_class, "implementations", ()) or (block_class,))[0]().run(
        **arguments
    )

    return result["image"]


def validate_literals(block_class: type, **literals: Any) -> None:
    spec_of(block_class).validate_params(
        {**literals, "image": "$inputs.image", "predictions": "$inputs.predictions"}
    )


def with_masks(rows: Any, masks: torch.Tensor) -> InstanceDetections:
    """Dense predictions with the boxes and metadata of ``rows``."""
    boxes = rows.detections if isinstance(rows, SegmentationView) else rows
    predictions = InstanceDetections(
        xyxy=boxes.xyxy,
        class_id=boxes.class_id,
        confidence=boxes.confidence,
        mask=masks,
        image_metadata=boxes.image_metadata,
        bboxes_metadata=boxes.bboxes_metadata,
    )

    return predictions


def masked_rows(
    image: ImageData,
    objects: Sequence[Tuple],
    masks: np.ndarray,
    *,
    rows: Optional[Sequence[dict]] = None,
) -> InstanceDetections:
    """Dense predictions of local ``objects`` and ``masks``."""
    boxes = detections(image, objects, rows=rows)
    predictions = with_masks(boxes, torch.from_numpy(masks))

    return predictions


def empty_rows(image: ImageData, *, with_mask: bool) -> Any:
    boxes = detections(image, ())
    if not with_mask:
        return boxes

    predictions = with_masks(boxes, torch.zeros((0, *image.size_hw), dtype=torch.bool))

    return predictions


def level_difference(first: np.ndarray, second: np.ndarray) -> int:
    difference = int(np.abs(first.astype(np.int16) - second.astype(np.int16)).max())

    return difference


def changed(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """``(H, W)`` bool: pixels where any channel differs."""
    pixels_changed = (first != second).any(axis=-1)

    return pixels_changed


def padded_rectangles(
    xyxy: np.ndarray, *, padding: int, size_hw: Tuple[int, int]
) -> np.ndarray:
    """``(H, W)`` bool union of boxes grown by ``padding``, clipped to the frame."""
    height, width = size_hw
    region = np.zeros((height, width), dtype=bool)
    for x1, y1, x2, y2 in xyxy:
        left, top = max(int(x1) - padding, 0), max(int(y1) - padding, 0)
        right, bottom = min(int(x2) + padding + 1, width), min(
            int(y2) + padding + 1, height
        )
        region[top:bottom, left:right] = True

    return region


def mask_rectangles(masks: np.ndarray, *, padding: int) -> np.ndarray:
    """``(H, W)`` bool union of each mask's pixel extent grown by ``padding``."""
    boxes = []
    for mask in masks:
        ys, xs = np.nonzero(mask)
        boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
    region = padded_rectangles(
        np.array(boxes), padding=padding, size_hw=masks.shape[1:]
    )

    return region


def nearest_cells(height: int, width: int, *, pixel_size: int) -> np.ndarray:
    """``(H, W)`` cell labels of V1's pixelation of one region, from OpenCV."""
    if min(height, width) < pixel_size:
        cells = np.zeros((height, width), dtype=np.int64)
        return cells

    small_height, small_width = cv2.resize(
        np.zeros((height, width), dtype=np.uint8),
        None,
        fx=1 / pixel_size,
        fy=1 / pixel_size,
        interpolation=cv2.INTER_LINEAR,
    ).shape
    labels = np.arange(small_height * small_width, dtype=np.float32)
    cells = cv2.resize(
        labels.reshape(small_height, small_width),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int64)

    return cells


def assert_constant_per_cell(region: np.ndarray, cells: np.ndarray) -> None:
    for cell in np.unique(cells):
        colors = region[cells == cell]
        assert (colors == colors[0]).all(), f"cell {cell} is not one flat color"


# ------------------------------------------------------------------- mask


@pytest.mark.parametrize("opacity", [0.5, 0.3, 0.0, 1.0])
@pytest.mark.parametrize("color_axis", ["CLASS", "INDEX"])
def test_mask_matches_v1_on_dense_masks(
    pixels: np.ndarray, image: ImageData, opacity: float, color_axis: str
) -> None:
    colors = {**V1_COLOR_DEFAULTS, "color_axis": color_axis}

    output = run_v2(
        MaskVisualizationBlock,
        image=image,
        predictions=instance_detections(image),
        opacity=opacity,
        **colors,
    )

    expected = run_v1(
        MaskVisualizationBlockV1,
        pixels,
        predictions=v1_detections(masks=object_masks()),
        opacity=opacity,
        **colors,
    )
    np.testing.assert_array_equal(pixels_of(output), expected)


def test_mask_on_rle_masks_equals_dense_masks_and_v1(
    pixels: np.ndarray, image: ImageData
) -> None:
    from_rle = run_v2(
        MaskVisualizationBlock, image=image, predictions=rle_detections(image)
    )

    from_dense = run_v2(
        MaskVisualizationBlock, image=image, predictions=instance_detections(image)
    )
    expected = run_v1(
        MaskVisualizationBlockV1,
        pixels,
        predictions=v1_detections(masks=object_masks()),
        opacity=0.5,
        **V1_COLOR_DEFAULTS,
    )
    np.testing.assert_array_equal(pixels_of(from_rle), pixels_of(from_dense))
    np.testing.assert_array_equal(pixels_of(from_rle), expected)


def test_mask_paints_equal_areas_in_v1_order(
    pixels: np.ndarray, image: ImageData
) -> None:
    # Four 12 x 12 squares of different classes, each overlapping the next.
    objects = tuple(
        (4 + 9 * i, 6 + 7 * i, 16 + 9 * i, 18 + 7 * i, i % 3, 0.9) for i in range(4)
    )
    masks = np.zeros((4, *image.size_hw), dtype=bool)
    for index, (x1, y1, x2, y2, _, _) in enumerate(objects):
        masks[index, y1:y2, x1:x2] = True

    output = run_v2(
        MaskVisualizationBlock,
        image=image,
        predictions=masked_rows(image, objects, masks),
        opacity=1.0,
        color_axis="INDEX",
    )

    expected = run_v1(
        MaskVisualizationBlockV1,
        pixels,
        predictions=v1_detections(objects, masks=masks),
        opacity=1.0,
        **{**V1_COLOR_DEFAULTS, "color_axis": "INDEX"},
    )
    np.testing.assert_array_equal(pixels_of(output), expected)
    # Equal areas: the earlier row ends on top where two squares overlap.
    overlap = (slice(13, 18), slice(13, 16))
    first_only = (slice(6, 13), slice(4, 13))
    assert (pixels_of(output)[overlap] == pixels_of(output)[first_only][0, 0]).all()


def test_mask_track_axis_matches_v1(pixels: np.ndarray, image: ImageData) -> None:
    tracker_ids = [7, 3, 12]
    predictions = masked_rows(
        image,
        OBJECTS,
        object_masks(),
        rows=[{"tracker_id": tracker_id} for tracker_id in tracker_ids],
    )

    output = run_v2(
        MaskVisualizationBlock, image=image, predictions=predictions, color_axis="TRACK"
    )

    expected = run_v1(
        MaskVisualizationBlockV1,
        pixels,
        predictions=v1_detections(masks=object_masks(), tracker_id=tracker_ids),
        opacity=0.5,
        **{**V1_COLOR_DEFAULTS, "color_axis": "TRACK"},
    )
    np.testing.assert_array_equal(pixels_of(output), expected)


def test_mask_leaves_inner_hole_unpainted(pixels: np.ndarray, image: ImageData) -> None:
    output = run_v2(
        MaskVisualizationBlock, image=image, predictions=instance_detections(image)
    )

    np.testing.assert_array_equal(pixels_of(output)[HOLE], pixels[HOLE])
    assert changed(pixels_of(output), pixels)[27, 30:36].all()


# ---------------------------------------------------------------- polygon


@pytest.mark.parametrize("thickness", [1, 2, 4])
@pytest.mark.parametrize("color_axis", ["CLASS", "INDEX"])
def test_polygon_matches_v1_on_dense_masks(
    pixels: np.ndarray, image: ImageData, thickness: int, color_axis: str
) -> None:
    colors = {**V1_COLOR_DEFAULTS, "color_axis": color_axis}

    output = run_v2(
        PolygonVisualizationBlock,
        image=image,
        predictions=instance_detections(image),
        thickness=thickness,
        **colors,
    )

    expected = run_v1(
        PolygonVisualizationBlockV2,
        pixels,
        predictions=v1_detections(masks=object_masks()),
        thickness=thickness,
        **colors,
    )
    np.testing.assert_array_equal(pixels_of(output), expected)


def test_polygon_on_rle_masks_equals_dense_masks(image: ImageData) -> None:
    from_rle = run_v2(
        PolygonVisualizationBlock, image=image, predictions=rle_detections(image)
    )

    from_dense = run_v2(
        PolygonVisualizationBlock, image=image, predictions=instance_detections(image)
    )
    np.testing.assert_array_equal(pixels_of(from_rle), pixels_of(from_dense))


def test_polygon_outlines_the_inner_hole(pixels: np.ndarray, image: ImageData) -> None:
    filled = object_masks()
    filled[1][HOLE] = True

    with_hole = pixels_of(
        run_v2(
            PolygonVisualizationBlock,
            image=image,
            predictions=instance_detections(image),
            thickness=1,
        )
    )

    without_hole = pixels_of(
        run_v2(
            PolygonVisualizationBlock,
            image=image,
            predictions=instance_detections(image, masks=filled),
            thickness=1,
        )
    )
    hole_outline = changed(with_hole, without_hole)
    assert hole_outline[27:33, 29:37].sum() >= 16
    assert not hole_outline[:25].any() and not hole_outline[35:].any()
    np.testing.assert_array_equal(with_hole[29:31, 32:34], pixels[29:31, 32:34])


def test_polygon_skips_rows_with_empty_masks(image: ImageData) -> None:
    masks = np.concatenate([object_masks(), np.zeros((1, *image.size_hw), dtype=bool)])
    objects = (*OBJECTS, (10.0, 10.0, 20.0, 20.0, 2, 0.4))

    output = run_v2(
        PolygonVisualizationBlock,
        image=image,
        predictions=masked_rows(image, objects, masks),
        color_axis="INDEX",
    )

    expected = run_v2(
        PolygonVisualizationBlock,
        image=image,
        predictions=instance_detections(image),
        color_axis="INDEX",
    )
    np.testing.assert_array_equal(pixels_of(output), pixels_of(expected))


@pytest.mark.parametrize(
    "block_class", [MaskVisualizationBlock, PolygonVisualizationBlock]
)
def test_mask_readers_refuse_box_only_predictions(
    image: ImageData, block_class: type
) -> None:
    with pytest.raises(ValueError, match="instance masks"):
        run_v2(block_class, image=image, predictions=detections(image))


# ------------------------------------------------------------------- blur


@pytest.mark.parametrize(
    "kernel_size, padding", [(15, 0), (5, 3), (3, 2), (1, 0), (31, 0)]
)
@pytest.mark.parametrize("with_mask", [False, True])
def test_blur_matches_v1_for_odd_kernels(
    pixels: np.ndarray,
    image: ImageData,
    kernel_size: int,
    padding: int,
    with_mask: bool,
) -> None:
    predictions = instance_detections(image) if with_mask else detections(image)

    output = run_v2(
        BlurVisualizationBlock,
        image=image,
        predictions=predictions,
        kernel_size=kernel_size,
        padding=padding,
    )

    expected = run_v1(
        BlurVisualizationBlockV1,
        pixels,
        predictions=v1_detections(masks=object_masks() if with_mask else None),
        kernel_size=kernel_size,
        padding=padding,
    )
    np.testing.assert_array_equal(pixels_of(output), expected)


@pytest.mark.parametrize("kernel_size, padding", [(4, 0), (6, 2), (16, 1)])
@pytest.mark.parametrize("with_mask", [False, True])
def test_blur_even_kernels_are_within_one_level_of_v1(
    pixels: np.ndarray,
    image: ImageData,
    kernel_size: int,
    padding: int,
    with_mask: bool,
) -> None:
    predictions = instance_detections(image) if with_mask else detections(image)

    output = pixels_of(
        run_v2(
            BlurVisualizationBlock,
            image=image,
            predictions=predictions,
            kernel_size=kernel_size,
            padding=padding,
        )
    )

    expected = run_v1(
        BlurVisualizationBlockV1,
        pixels,
        predictions=v1_detections(masks=object_masks() if with_mask else None),
        kernel_size=kernel_size,
        padding=padding,
    )
    assert level_difference(output, expected) <= 1
    boxes = np.array([obj[:4] for obj in OBJECTS])
    region = padded_rectangles(boxes, padding=padding, size_hw=image.size_hw)
    if with_mask:
        region |= mask_rectangles(object_masks(), padding=padding)
    assert not changed(output, pixels)[~region].any()
    assert not changed(expected, pixels)[~region].any()


def test_blur_of_masks_changes_only_mask_pixels(
    pixels: np.ndarray, image: ImageData
) -> None:
    output = pixels_of(
        run_v2(
            BlurVisualizationBlock, image=image, predictions=instance_detections(image)
        )
    )

    outside = ~object_masks().any(axis=0)
    np.testing.assert_array_equal(output[outside], pixels[outside])
    np.testing.assert_array_equal(output[HOLE], pixels[HOLE])
    assert changed(output, pixels)[object_masks().any(axis=0)].mean() > 0.5


def test_blur_on_rle_masks_equals_dense_masks(image: ImageData) -> None:
    from_rle = run_v2(
        BlurVisualizationBlock, image=image, predictions=rle_detections(image)
    )

    from_dense = run_v2(
        BlurVisualizationBlock, image=image, predictions=instance_detections(image)
    )
    np.testing.assert_array_equal(pixels_of(from_rle), pixels_of(from_dense))


@pytest.mark.parametrize(
    "block_class", [BlurVisualizationBlock, PixelateVisualizationBlock]
)
def test_region_blocks_use_keypoint_boxes(image: ImageData, block_class: type) -> None:
    points, boxes = key_points(image)

    output = run_v2(block_class, image=image, predictions=(points, boxes))

    expected = run_v2(block_class, image=image, predictions=boxes)
    np.testing.assert_array_equal(pixels_of(output), pixels_of(expected))
    with pytest.raises(ValueError, match="keypoint_visualization"):
        run_v2(
            block_class, image=image, predictions=key_points(image, with_boxes=False)
        )


# --------------------------------------------------------------- pixelate


@pytest.mark.parametrize("pixel_size", [1, 3, 40])
def test_pixelate_matches_v1_for_exact_cases(
    pixels: np.ndarray, image: ImageData, pixel_size: int
) -> None:
    # 1: identity; 3: cell colors exact here; 40: every box is a mean fill.
    output = run_v2(
        PixelateVisualizationBlock,
        image=image,
        predictions=detections(image),
        pixel_size=pixel_size,
    )

    expected = run_v1(
        PixelateVisualizationBlockV1,
        pixels,
        predictions=v1_detections(),
        pixel_size=pixel_size,
    )
    np.testing.assert_array_equal(pixels_of(output), expected)


@pytest.mark.parametrize("pixel_size", [6, 7, 20])
def test_pixelate_cells_follow_v1_layout_within_one_level(
    pixels: np.ndarray, image: ImageData, pixel_size: int
) -> None:
    output = pixels_of(
        run_v2(
            PixelateVisualizationBlock,
            image=image,
            predictions=detections(image, SEPARATE_OBJECTS),
            pixel_size=pixel_size,
        )
    )

    expected = run_v1(
        PixelateVisualizationBlockV1,
        pixels,
        predictions=v1_detections(SEPARATE_OBJECTS),
        pixel_size=pixel_size,
    )
    assert level_difference(output, expected) <= 1
    boxes = np.array([obj[:4] for obj in SEPARATE_OBJECTS]).astype(int)
    inside = np.zeros(image.size_hw, dtype=bool)
    for x1, y1, x2, y2 in boxes:
        inside[y1:y2, x1:x2] = True
        cells = nearest_cells(y2 - y1, x2 - x1, pixel_size=pixel_size)
        assert_constant_per_cell(output[y1:y2, x1:x2], cells)
        assert_constant_per_cell(expected[y1:y2, x1:x2], cells)
    np.testing.assert_array_equal(output[~inside], pixels[~inside])


@pytest.mark.parametrize("pixel_size", [6, 20])
def test_pixelate_overlapping_rows_are_within_one_level_of_v1(
    pixels: np.ndarray, image: ImageData, pixel_size: int
) -> None:
    output = run_v2(
        PixelateVisualizationBlock,
        image=image,
        predictions=detections(image),
        pixel_size=pixel_size,
    )

    expected = run_v1(
        PixelateVisualizationBlockV1,
        pixels,
        predictions=v1_detections(),
        pixel_size=pixel_size,
    )
    assert level_difference(pixels_of(output), expected) <= 1


def test_pixelate_ignores_masks(image: ImageData) -> None:
    from_masks = run_v2(
        PixelateVisualizationBlock, image=image, predictions=instance_detections(image)
    )

    from_boxes = run_v2(
        PixelateVisualizationBlock, image=image, predictions=detections(image)
    )
    np.testing.assert_array_equal(pixels_of(from_masks), pixels_of(from_boxes))


# ------------------------------------------------------------------- crop


@pytest.mark.parametrize("position", CROP_POSITIONS)
def test_crop_without_scaling_matches_v1(
    pixels: np.ndarray, image: ImageData, position: str
) -> None:
    # Scale 1 copies pixels: placement, clipping and borders are exact.
    output = run_v2(
        CropVisualizationBlock,
        image=image,
        predictions=detections(image),
        position=position,
        scale_factor=1.0,
        **V1_COLOR_DEFAULTS,
    )

    expected = run_v1(
        CropVisualizationBlockV1,
        pixels,
        predictions=v1_detections(),
        position=position,
        scale_factor=1.0,
        border_thickness=2,
        **V1_COLOR_DEFAULTS,
    )
    np.testing.assert_array_equal(pixels_of(output), expected)


@pytest.mark.parametrize("position", CROP_POSITIONS)
@pytest.mark.parametrize("scale_factor", [2.0, 0.5, 1.3])
def test_crop_scaled_copies_are_within_one_level_of_v1(
    pixels: np.ndarray, image: ImageData, position: str, scale_factor: float
) -> None:
    output = pixels_of(
        run_v2(
            CropVisualizationBlock,
            image=image,
            predictions=detections(image),
            position=position,
            scale_factor=scale_factor,
            **V1_COLOR_DEFAULTS,
        )
    )

    expected = run_v1(
        CropVisualizationBlockV1,
        pixels,
        predictions=v1_detections(),
        position=position,
        scale_factor=scale_factor,
        border_thickness=2,
        **V1_COLOR_DEFAULTS,
    )
    assert level_difference(output, expected) <= 1
    # Only resampled copies may differ; borders and untouched pixels are exact.
    flat = np.full_like(pixels, (17, 99, 201))
    borders = changed(
        run_v1(
            CropVisualizationBlockV1,
            flat,
            predictions=v1_detections(),
            position=position,
            scale_factor=scale_factor,
            border_thickness=2,
            **V1_COLOR_DEFAULTS,
        ),
        flat,
    )
    assert borders.sum() > 0
    np.testing.assert_array_equal(output[borders], expected[borders])


def test_crop_at_center_of_mass_is_within_one_level_of_v1(
    pixels: np.ndarray, image: ImageData
) -> None:
    output = run_v2(
        CropVisualizationBlock,
        image=image,
        predictions=instance_detections(image),
        position="CENTER_OF_MASS",
        **V1_COLOR_DEFAULTS,
    )

    expected = run_v1(
        CropVisualizationBlockV1,
        pixels,
        predictions=v1_detections(masks=object_masks()),
        position="CENTER_OF_MASS",
        scale_factor=2.0,
        border_thickness=2,
        **V1_COLOR_DEFAULTS,
    )
    assert level_difference(pixels_of(output), expected) <= 1


def test_crop_at_center_of_mass_needs_masks(image: ImageData) -> None:
    with pytest.raises(ValueError, match="CENTER_OF_MASS needs instance masks"):
        run_v2(
            CropVisualizationBlock,
            image=image,
            predictions=detections(image),
            position="CENTER_OF_MASS",
        )


def test_crop_skips_rows_without_pixels(image: ImageData) -> None:
    # V1 raised in cv2.resize for an empty crop; V2 skips the row.
    objects = (*OBJECTS, (30.0, 10.0, 30.4, 20.0, 1, 0.5))

    output = run_v2(
        CropVisualizationBlock, image=image, predictions=detections(image, objects)
    )

    expected = run_v2(
        CropVisualizationBlock, image=image, predictions=detections(image)
    )
    np.testing.assert_array_equal(pixels_of(output), pixels_of(expected))


# --------------------------------------------------- mask views and quality


def view_runs() -> Dict[str, Tuple[type, Dict[str, Any]]]:
    runs = {
        "mask": (MaskVisualizationBlock, {}),
        "polygon": (PolygonVisualizationBlock, {}),
        "blur": (BlurVisualizationBlock, {}),
        "crop": (CropVisualizationBlock, {"position": "CENTER_OF_MASS"}),
    }

    return runs


@pytest.mark.parametrize("name", list(view_runs()))
def test_accurate_view_equals_dense_full_res_with_one_materialization(
    image: ImageData, name: str
) -> None:
    block_class, parameters = view_runs()[name]
    view = grid_view(image)

    output = run_v2(
        block_class, image=image, predictions=view, mask_quality=ACCURATE, **parameters
    )

    assert view.materialization_count == 1
    expected = run_v2(
        block_class, image=image, predictions=view.full_res(), **parameters
    )
    assert view.materialization_count == 1
    np.testing.assert_array_equal(pixels_of(output), pixels_of(expected))


@pytest.mark.parametrize("name", list(view_runs()))
def test_coarse_view_draws_nearest_cell_masks_without_materializing(
    image: ImageData, name: str
) -> None:
    block_class, parameters = view_runs()[name]
    view = grid_view(image)

    coarse = run_v2(
        block_class, image=image, predictions=view, mask_quality=COARSE, **parameters
    )

    assert view.materialization_count == 0
    cell_masks = with_masks(view, nearest_full_res(view))
    expected = run_v2(block_class, image=image, predictions=cell_masks, **parameters)
    np.testing.assert_array_equal(pixels_of(coarse), pixels_of(expected))


@pytest.mark.parametrize("block_class", MASK_QUALITY_BLOCKS)
def test_coarse_is_an_approximation_of_accurate(
    image: ImageData, block_class: type
) -> None:
    view = grid_view(image)

    coarse = pixels_of(
        run_v2(block_class, image=image, predictions=view, mask_quality=COARSE)
    )

    accurate = pixels_of(
        run_v2(block_class, image=image, predictions=view, mask_quality=ACCURATE)
    )
    # The fixture's cells disagree with the interpolated masks at edges.
    assert changed(coarse, accurate).sum() > 0
    assert changed(coarse, accurate).mean() < 0.3


def test_coarse_mask_overlay_differs_only_where_cell_masks_differ(
    image: ImageData,
) -> None:
    view = grid_view(image)
    disagree = (nearest_full_res(view) != view.full_res().mask).any(dim=0).numpy()

    coarse = pixels_of(
        run_v2(
            MaskVisualizationBlock, image=image, predictions=view, mask_quality=COARSE
        )
    )

    accurate = pixels_of(
        run_v2(
            MaskVisualizationBlock, image=image, predictions=view, mask_quality=ACCURATE
        )
    )
    differs = changed(coarse, accurate)
    assert differs.any()
    assert not (differs & ~disagree).any()


def carriers() -> Dict[str, Callable[[ImageData], Any]]:
    by_name = {
        "dense": instance_detections,
        "rle": rle_detections,
        "dense_view": lambda image: SegmentationView.from_dense(
            instance_detections(image)
        ),
    }

    return by_name


@pytest.mark.parametrize("carrier", list(carriers()))
@pytest.mark.parametrize("name", list(view_runs()))
def test_mask_quality_is_a_no_op_for_dense_and_rle_masks(
    image: ImageData, carrier: str, name: str
) -> None:
    block_class, parameters = view_runs()[name]
    predictions = carriers()[carrier](image)

    coarse = run_v2(
        block_class,
        image=image,
        predictions=predictions,
        mask_quality=COARSE,
        **parameters,
    )

    accurate = run_v2(
        block_class,
        image=image,
        predictions=predictions,
        mask_quality=ACCURATE,
        **parameters,
    )
    reference = run_v2(
        block_class, image=image, predictions=instance_detections(image), **parameters
    )
    np.testing.assert_array_equal(pixels_of(coarse), pixels_of(accurate))
    np.testing.assert_array_equal(pixels_of(accurate), pixels_of(reference))


def test_dense_fallback_view_never_materializes(image: ImageData) -> None:
    view = SegmentationView.from_dense(instance_detections(image))

    for block_class, parameters in view_runs().values():
        run_v2(
            block_class,
            image=image,
            predictions=view,
            mask_quality=ACCURATE,
            **parameters,
        )

    assert view.materialization_count == 0


def test_mask_readers_share_one_accurate_materialization(image: ImageData) -> None:
    view = grid_view(image)

    for block_class, parameters in view_runs().values():
        run_v2(
            block_class,
            image=image,
            predictions=view,
            mask_quality=COARSE,
            **parameters,
        )
    assert view.materialization_count == 0
    for block_class, parameters in view_runs().values():
        run_v2(
            block_class,
            image=image,
            predictions=view,
            mask_quality=ACCURATE,
            **parameters,
        )

    assert view.materialization_count == 1


@pytest.mark.parametrize(
    "block_class, parameters",
    [
        (PixelateVisualizationBlock, {}),
        (CropVisualizationBlock, {"position": "TOP_CENTER", "mask_quality": ACCURATE}),
        (CropVisualizationBlock, {"position": "CENTER", "mask_quality": COARSE}),
    ],
)
def test_box_only_paths_never_materialize_a_view(
    image: ImageData, block_class: type, parameters: Dict[str, Any]
) -> None:
    view = grid_view(image)

    output = run_v2(block_class, image=image, predictions=view, **parameters)

    assert view.materialization_count == 0
    expected = run_v2(
        block_class, image=image, predictions=view.detections, **parameters
    )
    np.testing.assert_array_equal(pixels_of(output), pixels_of(expected))


def test_blur_of_box_only_predictions_reads_no_masks(image: ImageData) -> None:
    view = grid_view(image)

    output = run_v2(BlurVisualizationBlock, image=image, predictions=view.detections)

    assert view.materialization_count == 0
    expected = run_v2(
        BlurVisualizationBlock, image=image, predictions=detections(image)
    )
    np.testing.assert_array_equal(pixels_of(output), pixels_of(expected))


@pytest.mark.parametrize("name", list(view_runs()))
def test_unknown_mask_quality_is_rejected(image: ImageData, name: str) -> None:
    block_class, parameters = view_runs()[name]
    # The provider's internal names are not public values.
    with pytest.raises(ValueError):
        validate_literals(block_class, mask_quality="reference", **parameters)

    with pytest.raises(ValueError, match="mask_quality"):
        run_selected(
            block_class,
            image=image,
            predictions=grid_view(image),
            values={"mask_quality": "fast"},
            **parameters,
        )


# --------------------------------------------------- semantic segmentation


def class_masks(
    result: SemanticSegmentationResult, *, painted: Sequence[int]
) -> InstanceDetections:
    """One dense mask per painted class: the mask block's view of the grid."""
    grid = result.segmentation_map
    masks = torch.stack([grid == class_id for class_id in painted])
    predictions = InstanceDetections(
        xyxy=torch.zeros((len(painted), 4)),
        class_id=torch.tensor(list(painted), dtype=torch.int64),
        confidence=torch.ones(len(painted)),
        mask=masks,
        image_metadata=result.image_metadata,
    )

    return predictions


def semantic_with(
    image: ImageData,
    grid: torch.Tensor,
    *,
    names: Optional[Any],
    frame: bool = True,
) -> SemanticSegmentationResult:
    metadata = metadata_for(image) if frame else {}
    metadata.pop("class_names", None)
    if names is not None:
        metadata["class_names"] = names
    result = SemanticSegmentationResult(
        segmentation_map=grid,
        confidence=torch.full(grid.shape, 0.9),
        image_metadata=metadata,
    )

    return result


@pytest.mark.parametrize(
    "skip_background, painted", [(True, [1, 2]), (False, [0, 1, 2])]
)
@pytest.mark.parametrize("opacity", [0.5, 0.8])
def test_semantic_equals_mask_block_over_class_masks(
    image: ImageData, skip_background: bool, painted: Sequence[int], opacity: float
) -> None:
    result = semantic(image)

    output = run_v2(
        SemanticSegmentationVisualizationBlock,
        image=image,
        predictions=result,
        skip_background=skip_background,
        opacity=opacity,
    )

    expected = run_v2(
        MaskVisualizationBlock,
        image=image,
        predictions=class_masks(result, painted=painted),
        opacity=opacity,
    )
    np.testing.assert_array_equal(pixels_of(output), pixels_of(expected))


def test_semantic_colors_ids_beyond_the_palette_like_the_mask_block(
    image: ImageData,
) -> None:
    grid = semantic(image).segmentation_map.clone()
    grid[2:8, 2:12] = 13
    names = {0: "background", 1: "road", 2: "car", 13: "sign"}
    result = semantic_with(image, grid, names=names)

    output = run_v2(
        SemanticSegmentationVisualizationBlock,
        image=image,
        predictions=result,
        palette_size=4,
        color_palette="Matplotlib Viridis",
    )

    expected = run_v2(
        MaskVisualizationBlock,
        image=image,
        predictions=class_masks(result, painted=[1, 2, 13]),
        palette_size=4,
        color_palette="Matplotlib Viridis",
    )
    np.testing.assert_array_equal(pixels_of(output), pixels_of(expected))


@pytest.mark.parametrize(
    "names, skipped",
    [
        ({0: "Background", 1: "road", 2: "car"}, {0}),
        ({0: "BACKGROUND", 1: "road", 2: "car"}, {0}),
        ({0: "background_area", 1: "road", 2: "car"}, set()),
        ({0: "sky", 1: "road", 2: "background"}, {2}),
        (["background", "road", "car"], {0}),
        ({0: "sky", 1: "road", 2: "car"}, set()),
    ],
)
def test_semantic_skips_classes_named_background(
    pixels: np.ndarray, image: ImageData, names: Any, skipped: set
) -> None:
    result = semantic_with(image, semantic(image).segmentation_map, names=names)

    output = pixels_of(
        run_v2(SemanticSegmentationVisualizationBlock, image=image, predictions=result)
    )

    grid = result.segmentation_map.numpy()
    unpainted = np.isin(grid, list(skipped))
    np.testing.assert_array_equal(output[unpainted], pixels[unpainted])
    assert changed(output, pixels)[~unpainted].mean() > 0.9


def test_semantic_skip_classes_by_name(pixels: np.ndarray, image: ImageData) -> None:
    result = semantic(image)

    output = pixels_of(
        run_v2(
            SemanticSegmentationVisualizationBlock,
            image=image,
            predictions=result,
            skip_classes=["road"],
        )
    )

    expected = run_v2(
        MaskVisualizationBlock,
        image=image,
        predictions=class_masks(result, painted=[2]),
    )
    np.testing.assert_array_equal(output, pixels_of(expected))


def test_semantic_rejects_unknown_skip_classes(image: ImageData) -> None:
    with pytest.raises(
        ValueError, match=r"\['sky'\].*classes: \['background', 'car', 'road'\]"
    ):
        run_v2(
            SemanticSegmentationVisualizationBlock,
            image=image,
            predictions=semantic(image),
            skip_classes=["sky"],
        )


def test_semantic_without_class_names(pixels: np.ndarray, image: ImageData) -> None:
    result = semantic_with(image, semantic(image).segmentation_map, names=None)

    output = run_v2(
        SemanticSegmentationVisualizationBlock, image=image, predictions=result
    )

    # No names: nothing is known to be background, every class is painted.
    expected = run_v2(
        MaskVisualizationBlock,
        image=image,
        predictions=class_masks(result, painted=[0, 1, 2]),
    )
    np.testing.assert_array_equal(pixels_of(output), pixels_of(expected))
    with pytest.raises(ValueError, match="class_names"):
        run_v2(
            SemanticSegmentationVisualizationBlock,
            image=image,
            predictions=result,
            skip_classes=["road"],
        )


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int32])
def test_semantic_grid_dtype_does_not_change_pixels(
    image: ImageData, dtype: torch.dtype
) -> None:
    result = semantic(image)
    narrow = semantic_with(
        image,
        result.segmentation_map.to(dtype),
        names=result.image_metadata["class_names"],
    )

    output = run_v2(
        SemanticSegmentationVisualizationBlock, image=image, predictions=narrow
    )

    expected = run_v2(
        SemanticSegmentationVisualizationBlock, image=image, predictions=result
    )
    np.testing.assert_array_equal(pixels_of(output), pixels_of(expected))


def test_semantic_refuses_grids_of_other_images(
    pixels: np.ndarray, image: ImageData
) -> None:
    other = image_of(pixels, image_id="other")
    smaller = image_of(pixels[:40, :60], image_id="frame")

    with pytest.raises(ValueError, match="'other'.*'frame'"):
        run_v2(
            SemanticSegmentationVisualizationBlock,
            image=image,
            predictions=semantic(other),
        )
    with pytest.raises(ValueError, match=r"\(40, 60\).*\(48, 64\)"):
        run_v2(
            SemanticSegmentationVisualizationBlock,
            image=image,
            predictions=semantic(smaller),
        )


def test_semantic_draws_a_frame_free_grid_of_the_image_size(image: ImageData) -> None:
    result = semantic(image)
    local = semantic_with(
        image,
        result.segmentation_map,
        names=result.image_metadata["class_names"],
        frame=False,
    )

    output = run_v2(
        SemanticSegmentationVisualizationBlock, image=image, predictions=local
    )

    expected = run_v2(
        SemanticSegmentationVisualizationBlock, image=image, predictions=result
    )
    np.testing.assert_array_equal(pixels_of(output), pixels_of(expected))


@pytest.mark.parametrize(
    "values, message",
    [
        ({"skip_background": "yes"}, "skip_background must be true or false"),
        ({"skip_classes": "road"}, "skip_classes must be a list"),
        ({"skip_classes": ["sky"]}, "not classes of this model"),
    ],
)
def test_semantic_checks_selector_values(
    image: ImageData, values: Dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        run_selected(
            SemanticSegmentationVisualizationBlock,
            image=image,
            predictions=semantic(image),
            values=values,
        )


# ------------------------------------------- shared contract, all six blocks


def all_blocks() -> Dict[str, Tuple[type, Callable[[ImageData], Any]]]:
    blocks = {
        "mask": (MaskVisualizationBlock, instance_detections),
        "polygon": (PolygonVisualizationBlock, instance_detections),
        "blur": (BlurVisualizationBlock, instance_detections),
        "pixelate": (PixelateVisualizationBlock, detections),
        "crop": (CropVisualizationBlock, detections),
        "semantic": (SemanticSegmentationVisualizationBlock, semantic),
    }

    return blocks


def predictions_for(block_class: type, image: ImageData) -> Any:
    predictions_of = {block: factory for block, factory in all_blocks().values()}

    return predictions_of[block_class](image)


@pytest.mark.parametrize(
    "block_class, predictions",
    [
        (MaskVisualizationBlock, lambda image: empty_rows(image, with_mask=True)),
        (
            MaskVisualizationBlock,
            lambda image: grid_view(image).select(torch.zeros(0, dtype=torch.int64)),
        ),
        (PolygonVisualizationBlock, lambda image: empty_rows(image, with_mask=True)),
        (BlurVisualizationBlock, lambda image: empty_rows(image, with_mask=True)),
        (BlurVisualizationBlock, lambda image: empty_rows(image, with_mask=False)),
        (PixelateVisualizationBlock, lambda image: empty_rows(image, with_mask=False)),
        (CropVisualizationBlock, lambda image: empty_rows(image, with_mask=False)),
        (
            CropVisualizationBlock,
            lambda image: grid_view(image).select(torch.zeros(0, dtype=torch.int64)),
        ),
        (
            SemanticSegmentationVisualizationBlock,
            lambda image: semantic_with(
                image,
                torch.zeros(image.size_hw, dtype=torch.int64),
                names={0: "background"},
            ),
        ),
    ],
)
def test_empty_predictions_return_an_unchanged_copy(
    pixels: np.ndarray,
    image: ImageData,
    block_class: type,
    predictions: Callable[[ImageData], Any],
) -> None:
    output = run_v2(block_class, image=image, predictions=predictions(image))

    np.testing.assert_array_equal(pixels_of(output), pixels)
    assert output.tensor_image.data_ptr() != image.tensor_image.data_ptr()


@pytest.mark.parametrize("name", list(all_blocks()))
def test_input_is_untouched_and_identity_is_kept(pixels: np.ndarray, name: str) -> None:
    block_class, predictions_of = all_blocks()[name]
    crop = image_of(pixels).crop((0, 0, 64, 48), image_id="crop")
    crop = crop.with_parent_row(RowRef(namespace="detections", id=2))
    before = crop.tensor_image.clone()

    output = run_v2(block_class, image=crop, predictions=predictions_of(crop))

    assert torch.equal(crop.tensor_image, before)
    assert changed(pixels_of(output), pixels).any()
    assert output.tensor_image.data_ptr() != crop.tensor_image.data_ptr()
    assert output.tensor_image.is_contiguous()
    assert output.image_id == "crop" and output.size_hw == crop.size_hw
    assert output.parent == crop.parent and output.root == crop.root
    assert output.parent_row == crop.parent_row
    assert output.video_metadata is crop.video_metadata
    assert output.composite_sources == crop.composite_sources


@pytest.mark.parametrize("name", ["mask", "polygon", "blur", "pixelate", "crop"])
def test_predictions_of_another_input_are_refused(
    pixels: np.ndarray, image: ImageData, name: str
) -> None:
    block_class, predictions_of = all_blocks()[name]
    other = image_of(pixels, image_id="other")

    with pytest.raises(ValueError, match="'other'.*'frame'"):
        run_v2(block_class, image=image, predictions=predictions_of(other))


SELECTOR_VALUES = [
    (MaskVisualizationBlock, {"opacity": 0.3}),
    (MaskVisualizationBlock, {"mask_quality": COARSE}),
    (PolygonVisualizationBlock, {"thickness": 3}),
    (BlurVisualizationBlock, {"kernel_size": 5, "padding": 2}),
    (PixelateVisualizationBlock, {"pixel_size": 6}),
    (
        CropVisualizationBlock,
        {"position": "CENTER", "scale_factor": 1.5, "border_thickness": 3},
    ),
    (
        SemanticSegmentationVisualizationBlock,
        {"opacity": 0.7, "skip_background": False},
    ),
    (SemanticSegmentationVisualizationBlock, {"skip_classes": ["road"]}),
]


@pytest.mark.parametrize("block_class, values", SELECTOR_VALUES)
def test_selector_values_draw_like_literals(
    image: ImageData, block_class: type, values: Dict[str, Any]
) -> None:
    if "mask_quality" in values:
        view_or_rows = grid_view(image)
    else:
        view_or_rows = predictions_for(block_class, image)

    selected = run_selected(
        block_class, image=image, predictions=view_or_rows, values=values
    )

    literal = run_v2(block_class, image=image, predictions=view_or_rows, **values)
    np.testing.assert_array_equal(pixels_of(selected), pixels_of(literal))


OUT_OF_RANGE_VALUES = [
    (MaskVisualizationBlock, "opacity", 1.5),
    (MaskVisualizationBlock, "opacity", -0.1),
    (PolygonVisualizationBlock, "thickness", 0),
    (BlurVisualizationBlock, "kernel_size", 0),
    (BlurVisualizationBlock, "padding", -1),
    (PixelateVisualizationBlock, "pixel_size", 0),
    (CropVisualizationBlock, "scale_factor", 0.0),
    (CropVisualizationBlock, "border_thickness", 0),
    (SemanticSegmentationVisualizationBlock, "opacity", 2.0),
]
UNKNOWN_CHOICES = [
    (CropVisualizationBlock, "position", "MIDDLE"),
    (CropVisualizationBlock, "color_axis", "DIAGONAL"),
    (MaskVisualizationBlock, "color_axis", "DIAGONAL"),
    (SemanticSegmentationVisualizationBlock, "color_palette", "NOT A PALETTE"),
]


@pytest.mark.parametrize("block_class, name, value", OUT_OF_RANGE_VALUES)
def test_out_of_range_values_fail_with_one_message_as_literals_and_selectors(
    image: ImageData, block_class: type, name: str, value: Any
) -> None:
    with pytest.raises(ValueError) as at_run:
        run_selected(
            block_class,
            image=image,
            predictions=predictions_for(block_class, image),
            values={name: value},
        )

    with pytest.raises(ValueError) as at_compile:
        validate_literals(block_class, **{name: value})

    assert name in str(at_run.value)
    assert str(at_run.value) in str(at_compile.value)


@pytest.mark.parametrize("block_class, name, value", UNKNOWN_CHOICES)
def test_unknown_choices_fail_as_literals_and_selectors(
    image: ImageData, block_class: type, name: str, value: Any
) -> None:
    # Literals fail on the declared choices at compile time; selectors at run.
    with pytest.raises(ValueError, match=name) as at_run:
        run_selected(
            block_class,
            image=image,
            predictions=predictions_for(block_class, image),
            values={name: value},
        )

    with pytest.raises(ValueError, match=name):
        validate_literals(block_class, **{name: value})

    assert repr(value) in str(at_run.value)
