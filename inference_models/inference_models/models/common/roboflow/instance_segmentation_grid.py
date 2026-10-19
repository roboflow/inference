"""Construct compact mask snapshots without materializing output rasters."""

from math import isfinite
from typing import Literal, Tuple

import torch

from inference_models.models.base.types import InstanceSegmentationGrid
from inference_models.models.common.roboflow._grid_geometry_cache import (
    geometry_constants,
)
from inference_models.models.common.roboflow.model_packages import PreProcessingMetadata


def build_instance_segmentation_grid(
    *,
    image_bboxes: torch.Tensor,
    masks: torch.Tensor,
    metadata: PreProcessingMetadata,
    score_type: Literal["logits", "probabilities"],
    binarization_threshold: float = 0.0,
) -> Tuple[torch.Tensor, InstanceSegmentationGrid]:
    """Snapshot prepared scores and continuous original-image boxes.

    Perform no mask resize, threshold, canvas allocation or tensor-value reads.
    The caller publishes returned tensors using its existing stream/lifetime
    protocol, and sets all enclosing detection size fields to original_size.

    Args:
        image_bboxes: Selected ``(N, 4)`` source boxes before alignment/finalization.
        masks: Selected floating ``(N, mh, mw)`` scores after model-specific steps.
        metadata: This image's preprocessing geometry; RF-DETR's non-square
            intermediate dimensions take precedence for resolving grid padding.
        score_type: Logit or probability meaning of the prepared scores.
        binarization_threshold: Strict foreground threshold for later materialization.

    Returns:
        Independent float32 image-space boxes and a selected-score snapshot.
        Boxes are intentionally neither clipped nor rounded: those operations
        lose information needed by consumers requesting another mask raster.
        Reprojection may differ numerically from direct source-to-raster mapping.

    Raises:
        ValueError: For incompatible tensor shapes/devices or invalid geometry.
    """
    if (
        not isinstance(image_bboxes, torch.Tensor)
        or image_bboxes.ndim != 2
        or image_bboxes.shape[1] != 4
        or not isinstance(masks, torch.Tensor)
        or masks.ndim != 3
        or image_bboxes.shape[0] != masks.shape[0]
        or image_bboxes.device != masks.device
    ):
        raise ValueError("boxes and scores must be row-aligned tensors on one device")

    inference_size = metadata.nonsquare_intermediate_size or metadata.inference_size
    if min(inference_size.height, inference_size.width) <= 0:
        raise ValueError("effective inference dimensions must be positive")
    if any(
        not isfinite(scale) or scale <= 0
        for scale in (metadata.scale_width, metadata.scale_height)
    ):
        raise ValueError("preprocessing scales must be finite and positive")

    crop = metadata.static_crop_offset
    if (crop.crop_height, crop.crop_width) != tuple(metadata.size_after_pre_processing):
        raise ValueError("static crop size must match size_after_pre_processing")
    if (
        crop.offset_x + crop.crop_width > metadata.original_size.width
        or crop.offset_y + crop.crop_height > metadata.original_size.height
    ):
        raise ValueError("static crop must fit within the original image")

    mask_height, mask_width = masks.shape[1:]
    grid_padding = (
        round(mask_width / inference_size.width * metadata.pad_left),
        round(mask_height / inference_size.height * metadata.pad_top),
        round(mask_width / inference_size.width * metadata.pad_right),
        round(mask_height / inference_size.height * metadata.pad_bottom),
    )
    grid = InstanceSegmentationGrid(
        scores=masks.detach().clone(memory_format=torch.contiguous_format),
        score_type=score_type,
        threshold=binarization_threshold,
        grid_padding_ltrb=grid_padding,
        image_crop_xywh=(
            crop.offset_x,
            crop.offset_y,
            crop.crop_width,
            crop.crop_height,
        ),
    )
    boxes = image_bboxes.detach().to(dtype=torch.float32).clone()
    padding, scale, offset = geometry_constants(
        boxes,
        padding=[metadata.pad_left, metadata.pad_top] * 2,
        scale=[metadata.scale_width, metadata.scale_height] * 2,
        offset=[crop.offset_x, crop.offset_y] * 2,
    )
    boxes.sub_(padding).div_(scale).add_(offset)

    return boxes, grid
