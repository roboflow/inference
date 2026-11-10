"""Mask-reading annotation blocks on predictions of another frame.

Masks of a crop are drawn where they lie in the parent; masks of the parent
are drawn on a crop and cut at its border::

    parent 64x48 ─ crop (10, 8, 50, 40) ─▶ crop 40x32
    crop -> parent:  masks pasted at (10, 8), boxes shifted by (+10, +8)
    parent -> crop:  masks sliced to the window, boxes shifted by (-10, -8)

Oracle: the latest V1 block drawing the same rows given to it already in the
drawn image's frame (shifted boxes, pasted or sliced masks). Mask views use
``mask_quality`` accurate, so their masks are materialized in their own frame
(``view.full_res()``) and then mapped.
"""

from typing import Any, Sequence, Tuple

import numpy as np
import pycocotools.mask as mask_utils
import pytest
import torch
import torch.nn.functional as F
from roboflow_workflows.core_steps.visualizations.blur.v1 import (
    BlurVisualizationBlockV1,
)
from roboflow_workflows.core_steps.visualizations.crop.v1 import (
    CropVisualizationBlockV1,
)
from roboflow_workflows.core_steps.visualizations.label.v2 import (
    LabelVisualizationBlockV2,
)
from roboflow_workflows.core_steps.visualizations.mask.v1 import (
    MaskVisualizationBlockV1,
)
from roboflow_workflows.core_steps.visualizations.polygon.v2 import (
    PolygonVisualizationBlockV2,
)
from roboflow_workflows.v2.blocks.visualization.annotations import (
    BlurVisualizationBlock,
    CropVisualizationBlock,
    LabelVisualizationBlock,
    MaskVisualizationBlock,
    PolygonVisualizationBlock,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.masks import MaskGridGeometry, SegmentationView

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks

from .support import (
    OBJECTS,
    V1_COLOR_DEFAULTS,
    image_of,
    metadata_for,
    object_masks,
    pixels_of,
    run_v1,
    run_v2,
    scene,
    v1_detections,
)

CROP_XYXY = (10, 8, 50, 40)
# Two crop-local objects (x1, y1, x2, y2, class id, confidence); masks are the
# ellipses inside the boxes.
LOCAL = (
    (2.5, 3.25, 20.75, 18.5, 0, 0.9),
    (15.0, 10.5, 38.25, 31.0, 2, 0.4),
)
CARRIERS = ("dense", "rle", "view")
STRIDE = 4
SMALL_TEXT = {
    "text_color": "WHITE",
    "text_size_mode": "Manual",
    "text_scale": 0.4,
    "text_thickness": 1,
    "text_padding": 3,
    "border_radius": 2,
}
# name -> (V2 block, latest V1 block, parameters, V1 colour parameters)
CROP_TO_PARENT = {
    "mask": (MaskVisualizationBlock, MaskVisualizationBlockV1, {"opacity": 0.5}, True),
    "polygon": (
        PolygonVisualizationBlock,
        PolygonVisualizationBlockV2,
        {"thickness": 2},
        True,
    ),
    "blur": (
        BlurVisualizationBlock,
        BlurVisualizationBlockV1,
        {"kernel_size": 5, "padding": 0},
        False,
    ),
    "label-class-centre-of-mass": (
        LabelVisualizationBlock,
        LabelVisualizationBlockV2,
        {**SMALL_TEXT, "text": "Class", "text_position": "CENTER_OF_MASS"},
        True,
    ),
    "label-area-top-left": (
        LabelVisualizationBlock,
        LabelVisualizationBlockV2,
        {**SMALL_TEXT, "text": "Area", "text_position": "TOP_LEFT"},
        True,
    ),
    "label-area-centre-of-mass": (
        LabelVisualizationBlock,
        LabelVisualizationBlockV2,
        {**SMALL_TEXT, "text": "Area", "text_position": "CENTER_OF_MASS"},
        True,
    ),
    "crop-centre-of-mass": (
        CropVisualizationBlock,
        CropVisualizationBlockV1,
        {"position": "CENTER_OF_MASS", "scale_factor": 1.0, "border_thickness": 2},
        True,
    ),
}
PARENT_TO_CROP = {
    "mask": CROP_TO_PARENT["mask"],
    "polygon": CROP_TO_PARENT["polygon"],
}


def ellipse_masks(size_hw: Tuple[int, int], objects: Sequence[Tuple]) -> np.ndarray:
    """Return ``(N, H, W)`` bool masks: the ellipse inside each box."""
    height, width = size_hw
    ys, xs = np.mgrid[0:height, 0:width]
    masks = []
    for x1, y1, x2, y2, *_ in objects:
        centre_x, centre_y = (x1 + x2) / 2, (y1 + y2) / 2
        radius_x, radius_y = (x2 - x1) / 2, (y2 - y1) / 2
        masks.append(
            ((xs - centre_x) / radius_x) ** 2 + ((ys - centre_y) / radius_y) ** 2 <= 1
        )
    stacked = np.stack(masks)

    return stacked


def rows_in(
    image: ImageData, objects: Sequence[Tuple], masks: np.ndarray, *, carrier: str
) -> Any:
    """Predictions of ``image`` in one mask carrier: dense, RLE or a mask view."""
    xyxy = torch.tensor([list(obj[:4]) for obj in objects], dtype=torch.float32)
    class_id = torch.tensor([obj[4] for obj in objects], dtype=torch.int64)
    confidence = torch.tensor([obj[5] for obj in objects], dtype=torch.float32)
    if carrier == "view":
        boxes = Detections(
            xyxy=xyxy,
            class_id=class_id,
            confidence=confidence,
            image_metadata=metadata_for(image),
        )
        view = stride_view(boxes, masks, size_hw=image.size_hw)
        return view

    if carrier == "dense":
        mask = torch.from_numpy(masks)
    else:
        encoded = [
            mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
            for mask in masks
        ]
        mask = InstancesRLEMasks.from_coco_rle_masks(image.size_hw, encoded)
    predictions = InstanceDetections(
        xyxy=xyxy,
        class_id=class_id,
        confidence=confidence,
        mask=mask,
        image_metadata=metadata_for(image),
    )

    return predictions


def stride_view(
    boxes: Detections, masks: np.ndarray, *, size_hw: Tuple[int, int]
) -> SegmentationView:
    """A stride-4 logit grid of ``masks``: ``16 * (cell coverage) - 8``."""
    height, width = size_hw
    coverage = F.avg_pool2d(torch.from_numpy(masks).float().unsqueeze(1), STRIDE)
    view = SegmentationView(
        detections=boxes,
        scores=coverage.squeeze(1) * 16.0 - 8.0,
        score_type="logits",
        threshold=0.0,
        geometry=MaskGridGeometry(
            grid_size_hw=(height // STRIDE, width // STRIDE),
            inference_size_hw=(height, width),
            padding_ltrb=(0, 0, 0, 0),
            scale_xy=(1.0, 1.0),
            crop_size_hw=(height, width),
            original_size_hw=(height, width),
            static_crop_xywh=(0, 0, width, height),
        ),
        take_ownership=True,
    )

    return view


def own_frame_masks(predictions: Any, masks: np.ndarray) -> np.ndarray:
    """The masks the block reads in the predictions' own frame.

    Dense and RLE rows carry ``masks`` exactly; a view's accurate masks are
    its full-resolution materialization.
    """
    if not isinstance(predictions, SegmentationView):
        return masks

    materialized = predictions.full_res().mask.cpu().numpy()

    return materialized


def shifted(objects: Sequence[Tuple], *, dx: float, dy: float) -> list:
    moved = [
        (x1 + dx, y1 + dy, x2 + dx, y2 + dy, *rest) for x1, y1, x2, y2, *rest in objects
    ]

    return moved


def assert_same_pixels(actual: np.ndarray, expected: np.ndarray) -> None:
    differing = (actual != expected).any(axis=-1)
    assert not differing.any(), (
        f"{int(differing.sum())} pixels differ from V1; first at (y, x)="
        f"{tuple(int(v) for v in np.argwhere(differing)[0])}"
    )


@pytest.mark.parametrize("carrier", CARRIERS)
@pytest.mark.parametrize("case", list(CROP_TO_PARENT))
def test_crop_masks_are_drawn_where_they_lie_in_the_parent(
    case: str, carrier: str
) -> None:
    # given: rows segmented on a crop, drawn on its parent
    block_class, v1_class, parameters, colored = CROP_TO_PARENT[case]
    pixels = scene()
    image = image_of(pixels)
    crop = image.crop(CROP_XYXY, image_id="crop")
    local_masks = ellipse_masks(crop.size_hw, LOCAL)
    predictions = rows_in(crop, LOCAL, local_masks, carrier=carrier)

    # when
    drawn = run_v2(
        block_class,
        image=image,
        predictions=predictions,
        mask_quality="accurate",
        **parameters,
    )

    # then: V1 given the rows already in the parent draws the same pixels
    x0, y0 = CROP_XYXY[:2]
    crop_height, crop_width = crop.size_hw
    parent_masks = np.zeros((len(LOCAL), *image.size_hw), dtype=bool)
    parent_masks[:, y0 : y0 + crop_height, x0 : x0 + crop_width] = own_frame_masks(
        predictions, local_masks
    )
    expected = run_v1(
        v1_class,
        pixels,
        predictions=v1_detections(shifted(LOCAL, dx=x0, dy=y0), masks=parent_masks),
        **parameters,
        **(V1_COLOR_DEFAULTS if colored else {}),
    )
    assert drawn.size_hw == image.size_hw
    assert_same_pixels(pixels_of(drawn), expected)


@pytest.mark.parametrize("carrier", CARRIERS)
@pytest.mark.parametrize("case", list(PARENT_TO_CROP))
def test_parent_masks_drawn_on_a_crop_are_clipped_to_it(
    case: str, carrier: str
) -> None:
    # given: all three parent objects cross the crop border
    block_class, v1_class, parameters, colored = PARENT_TO_CROP[case]
    pixels = scene()
    image = image_of(pixels)
    crop = image.crop(CROP_XYXY, image_id="crop")
    parent_masks = object_masks()
    predictions = rows_in(image, OBJECTS, parent_masks, carrier=carrier)

    # when
    drawn = run_v2(
        block_class,
        image=crop,
        predictions=predictions,
        mask_quality="accurate",
        **parameters,
    )

    # then: V1 given the window of each mask draws the same crop pixels
    x0, y0, x1, y1 = CROP_XYXY
    whole = own_frame_masks(predictions, parent_masks)
    window = whole[:, y0:y1, x0:x1]
    visible, total = window.sum(axis=(1, 2)), whole.sum(axis=(1, 2))
    assert ((visible > 0) & (visible < total)).all()  # every mask is cut
    expected = run_v1(
        v1_class,
        pixels_of(crop),
        predictions=v1_detections(shifted(OBJECTS, dx=-x0, dy=-y0), masks=window),
        **parameters,
        **(V1_COLOR_DEFAULTS if colored else {}),
    )
    assert drawn.size_hw == crop.size_hw
    assert_same_pixels(pixels_of(drawn), expected)
