from typing import Dict, Generator, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torchvision
from torchvision.transforms import functional

from inference_models.configuration import (
    INFERENCE_MODELS_DEFAULT_CONFIDENCE,
    INFERENCE_MODELS_INSTANCE_SEG_MASK_PROCESSING_CHUNK_SIZE,
)
from inference_models.entities import Confidence, ImageDimensions
from inference_models.logger import LOGGER
from inference_models.models.base.semantic_segmentation import (
    SemanticSegmentationResult,
)
from inference_models.models.common.rle_utils import (
    torch_mask_to_coco_rle,
    torch_masks_to_coco_rle_batch,
)
from inference_models.models.common.roboflow.model_packages import (
    PreProcessingMetadata,
    StaticCropOffset,
)
from inference_models.models.common.roboflow.semantic_segmentation import (
    insert_background_class,
)
from inference_models.weights_providers.entities import RecommendedParameters


def run_nms_for_object_detection(
    output: torch.Tensor,
    conf_thresh: Union[float, torch.Tensor] = 0.25,
    iou_thresh: float = 0.45,
    max_detections: int = 100,
    class_agnostic: bool = False,
    box_format: Literal["xywh", "xyxy"] = "xywh",
) -> List[torch.Tensor]:
    """
    `conf_thresh`: scalar applies to all classes; 1-D tensor of shape
    (num_classes,) indexed by class_id for per-class thresholds.
    """
    bs = output.shape[0]
    boxes = output[:, :4, :]
    scores = output[:, 4:, :]
    results = []
    for b in range(bs):
        class_scores = scores[b]
        class_conf, class_ids = class_scores.max(0)
        if isinstance(conf_thresh, torch.Tensor):
            mask = class_conf > conf_thresh.to(output.device)[class_ids]
        else:
            mask = class_conf > conf_thresh
        if not torch.any(mask):
            results.append(torch.zeros((0, 6), device=output.device))
            continue
        bboxes = boxes[b][:, mask].T  # (num, 4) -- selects and then transposes
        class_conf = class_conf[mask]
        class_ids = class_ids[mask]
        if box_format == "xywh":
            # Vectorized [x, y, w, h] -> [x1, y1, x2, y2]
            xy = bboxes[:, :2]
            wh = bboxes[:, 2:]
            half_wh = wh / 2
            xyxy = torch.cat((xy - half_wh, xy + half_wh), 1)
        else:
            xyxy = bboxes
        # Class-agnostic NMS -> use dummy class ids
        nms_class_ids = torch.zeros_like(class_ids) if class_agnostic else class_ids
        # NMS and limiting max detections
        keep = torchvision.ops.batched_nms(xyxy, class_conf, nms_class_ids, iou_thresh)
        if keep.numel() > max_detections:
            keep = keep[:max_detections]
        detections = torch.cat(
            (
                xyxy[keep],
                class_conf[keep, None],  # unsqueeze(1) is replaced with None
                class_ids[keep, None].float(),
            ),
            1,
        )  # [x1, y1, x2, y2, conf, cls]

        results.append(detections)
    return results


def post_process_nms_fused_model_output(
    output: torch.Tensor,
    conf_thresh: Union[float, torch.Tensor] = 0.25,
) -> List[torch.Tensor]:
    """
    `conf_thresh`: scalar applies to all classes; 1-D tensor of shape
    (num_classes,) indexed by class_id (col 5 of `output`).
    """
    bs = output.shape[0]
    nms_results = []
    for batch_element_id in range(bs):
        batch_element_result = output[batch_element_id]
        if isinstance(conf_thresh, torch.Tensor):
            class_ids = batch_element_result[:, 5].long()
            batch_element_result = batch_element_result[
                batch_element_result[:, 4] >= conf_thresh.to(output.device)[class_ids]
            ]
        else:
            batch_element_result = batch_element_result[
                batch_element_result[:, 4] >= conf_thresh
            ]
        nms_results.append(batch_element_result)
    return nms_results


def run_nms_for_instance_segmentation(
    output: torch.Tensor,
    conf_thresh: Union[float, torch.Tensor] = 0.25,
    iou_thresh: float = 0.45,
    max_detections: int = 100,
    class_agnostic: bool = False,
    box_format: Literal["xywh", "xyxy"] = "xywh",
) -> List[torch.Tensor]:
    """
    `conf_thresh`: scalar applies to all classes; 1-D tensor of shape
    (num_classes,) indexed by class_id for per-class thresholds.
    """
    bs = output.shape[0]
    boxes = output[:, :4, :]  # (N, 4, 8400)
    scores = output[:, 4:-32, :]  # (N, 80, 8400)
    masks = output[:, -32:, :]
    results = []

    for b in range(bs):
        bboxes = boxes[b].T  # (8400, 4)
        class_scores = scores[b].T  # (8400, 80)
        box_masks = masks[b].T
        class_conf, class_ids = class_scores.max(1)  # (8400,), (8400,)
        if isinstance(conf_thresh, torch.Tensor):
            mask = class_conf > conf_thresh.to(output.device)[class_ids]
        else:
            mask = class_conf > conf_thresh
        if mask.sum() == 0:
            results.append(torch.zeros((0, 38), device=output.device))
            continue
        bboxes = bboxes[mask]
        class_conf = class_conf[mask]
        class_ids = class_ids[mask]
        box_masks = box_masks[mask]
        if box_format == "xywh":
            # Vectorized [x, y, w, h] -> [x1, y1, x2, y2]
            xy = bboxes[:, :2]
            wh = bboxes[:, 2:]
            half_wh = wh / 2
            xyxy = torch.cat((xy - half_wh, xy + half_wh), 1)
        else:
            xyxy = bboxes
        # Class-agnostic NMS -> use dummy class ids
        nms_class_ids = torch.zeros_like(class_ids) if class_agnostic else class_ids
        keep = torchvision.ops.batched_nms(xyxy, class_conf, nms_class_ids, iou_thresh)
        keep = keep[:max_detections]
        detections = torch.cat(
            [
                xyxy[keep],
                class_conf[keep].unsqueeze(1),
                class_ids[keep].unsqueeze(1).float(),
                box_masks[keep],
            ],
            dim=1,
        )  # [x1, y1, x2, y2, conf, cls]
        results.append(detections)
    return results


def run_nms_for_key_points_detection(
    output: torch.Tensor,
    num_classes: int,
    key_points_slots_in_prediction: int,
    conf_thresh: Union[float, torch.Tensor] = 0.25,
    iou_thresh: float = 0.45,
    max_detections: int = 100,
    class_agnostic: bool = False,
) -> List[torch.Tensor]:
    """
    `conf_thresh`: scalar applies to all classes; 1-D tensor of shape
    (num_classes,) indexed by class_id for per-class thresholds.
    """
    bs = output.shape[0]
    boxes = output[:, :4, :]
    scores = output[:, 4 : 4 + num_classes, :]
    key_points = output[:, 4 + num_classes :, :]
    results = []
    for b in range(bs):
        class_scores = scores[b]
        class_conf, class_ids = class_scores.max(0)
        if isinstance(conf_thresh, torch.Tensor):
            mask = class_conf > conf_thresh.to(output.device)[class_ids]
        else:
            mask = class_conf > conf_thresh
        if not torch.any(mask):
            results.append(
                torch.zeros(
                    (0, 6 + key_points_slots_in_prediction * 3), device=output.device
                )
            )
            continue
        bboxes = boxes[b][:, mask].T
        image_key_points = key_points[b, :, mask].T
        class_conf = class_conf[mask]
        class_ids = class_ids[mask]
        xy = bboxes[:, :2]
        wh = bboxes[:, 2:]
        half_wh = wh / 2
        xyxy = torch.cat((xy - half_wh, xy + half_wh), 1)
        # Class-agnostic NMS -> use dummy class ids
        nms_class_ids = torch.zeros_like(class_ids) if class_agnostic else class_ids
        # NMS and limiting max detections
        keep = torchvision.ops.batched_nms(xyxy, class_conf, nms_class_ids, iou_thresh)
        if keep.numel() > max_detections:
            keep = keep[:max_detections]
        detections = torch.cat(
            (
                xyxy[keep],
                class_conf[keep, None],  # unsqueeze(1) is replaced with None
                class_ids[keep, None].float(),
                image_key_points[keep],
            ),
            1,
        )  # [x1, y1, x2, y2, conf, cls, keypoints....]
        results.append(detections)
    return results


def rescale_detections(
    detections: List[torch.Tensor], images_metadata: List[PreProcessingMetadata]
) -> List[torch.Tensor]:
    for image_detections, metadata in zip(detections, images_metadata):
        _ = rescale_image_detections(
            image_detections=image_detections, image_metadata=metadata
        )
    return detections


def rescale_image_detections(
    image_detections: torch.Tensor,
    image_metadata: PreProcessingMetadata,
) -> torch.Tensor:
    # in-place processing
    offsets = torch.as_tensor(
        [
            image_metadata.pad_left,
            image_metadata.pad_top,
            image_metadata.pad_left,
            image_metadata.pad_top,
        ],
        dtype=image_detections.dtype,
        device=image_detections.device,
    )
    image_detections[:, :4].sub_(offsets)  # in-place subtraction for speed/memory
    scale = torch.as_tensor(
        [
            image_metadata.scale_width,
            image_metadata.scale_height,
            image_metadata.scale_width,
            image_metadata.scale_height,
        ],
        dtype=image_detections.dtype,
        device=image_detections.device,
    )
    image_detections[:, :4].div_(scale)
    if (
        image_metadata.static_crop_offset.offset_x != 0
        or image_metadata.static_crop_offset.offset_y != 0
    ):
        static_crop_offsets = torch.as_tensor(
            [
                image_metadata.static_crop_offset.offset_x,
                image_metadata.static_crop_offset.offset_y,
                image_metadata.static_crop_offset.offset_x,
                image_metadata.static_crop_offset.offset_y,
            ],
            dtype=image_detections.dtype,
            device=image_detections.device,
        )
        image_detections[:, :4].add_(static_crop_offsets)
    xyxy_max = torch.as_tensor(
        [
            image_metadata.original_size.width,
            image_metadata.original_size.height,
            image_metadata.original_size.width,
            image_metadata.original_size.height,
        ],
        dtype=image_detections.dtype,
        device=image_detections.device,
    )
    image_detections[:, :4].clamp_(min=torch.zeros_like(xyxy_max), max=xyxy_max)
    return image_detections


def rescale_key_points_detections(
    detections: List[torch.Tensor],
    images_metadata: List[PreProcessingMetadata],
    num_classes: int,
    key_points_slots_in_prediction: int,
) -> List[torch.Tensor]:
    for image_detections, metadata in zip(detections, images_metadata):
        offsets = torch.as_tensor(
            [metadata.pad_left, metadata.pad_top, metadata.pad_left, metadata.pad_top],
            dtype=image_detections.dtype,
            device=image_detections.device,
        )
        image_detections[:, :4].sub_(offsets)  # in-place subtraction for speed/memory
        scale = torch.as_tensor(
            [
                metadata.scale_width,
                metadata.scale_height,
                metadata.scale_width,
                metadata.scale_height,
            ],
            dtype=image_detections.dtype,
            device=image_detections.device,
        )
        image_detections[:, :4].div_(scale)
        key_points_offsets = torch.as_tensor(
            [metadata.pad_left, metadata.pad_top, 0],
            dtype=image_detections.dtype,
            device=image_detections.device,
        ).repeat(key_points_slots_in_prediction)
        image_detections[:, 6:].sub_(key_points_offsets)
        key_points_scale = torch.as_tensor(
            [metadata.scale_width, metadata.scale_height, 1.0],
            dtype=image_detections.dtype,
            device=image_detections.device,
        ).repeat(key_points_slots_in_prediction)
        image_detections[:, 6:].div_(key_points_scale)
        if (
            metadata.static_crop_offset.offset_x != 0
            or metadata.static_crop_offset.offset_y != 0
        ):
            static_crop_offset_length = (image_detections.shape[1] - 6) // 3
            static_crop_offsets = torch.as_tensor(
                [
                    metadata.static_crop_offset.offset_x,
                    metadata.static_crop_offset.offset_y,
                    0,
                ]
                * static_crop_offset_length,
                dtype=image_detections.dtype,
                device=image_detections.device,
            )
            image_detections[:, 6:].add_(static_crop_offsets)
            static_crop_offsets = torch.as_tensor(
                [
                    metadata.static_crop_offset.offset_x,
                    metadata.static_crop_offset.offset_y,
                    metadata.static_crop_offset.offset_x,
                    metadata.static_crop_offset.offset_y,
                ],
                dtype=image_detections.dtype,
                device=image_detections.device,
            )
            image_detections[:, :4].add_(static_crop_offsets)
        xyxy_max = torch.as_tensor(
            [
                metadata.original_size.width,
                metadata.original_size.height,
                metadata.original_size.width,
                metadata.original_size.height,
            ],
            dtype=image_detections.dtype,
            device=image_detections.device,
        )
        image_detections[:, :4].clamp_(min=torch.zeros_like(xyxy_max), max=xyxy_max)
    return detections


def preprocess_segmentation_masks(
    protos: torch.Tensor,
    masks_in: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum("chw,nc->nhw", protos, masks_in)


def crop_masks_to_boxes(
    boxes: torch.Tensor,
    masks: torch.Tensor,
    scaling: float = 0.25,
) -> torch.Tensor:
    n, h, w = masks.shape
    scaled_boxes = torch.round(boxes * scaling)
    x1, y1, x2, y2 = (
        scaled_boxes[:, 0][:, None, None],
        scaled_boxes[:, 1][:, None, None],
        scaled_boxes[:, 2][:, None, None],
        scaled_boxes[:, 3][:, None, None],
    )
    rows = torch.arange(w, device=masks.device)[None, None, :]  # shape: [1, 1, w]
    cols = torch.arange(h, device=masks.device)[None, :, None]  # shape: [1, h, 1]
    crop_mask = (rows >= x1) & (rows < x2) & (cols >= y1) & (cols < y2)
    return masks * crop_mask


def scale_polygons_to_image(
    polygons: List[np.ndarray],
    *,
    mask_size: ImageDimensions,
    image_size: ImageDimensions,
) -> List[np.ndarray]:
    """Lift polygon coordinates from mask space into image space.

    Mask contours are extracted in the coordinate system of the mask they came
    from. When that mask is smaller than the image it describes, the contour
    must be rescaled before it can be reported as a prediction. Each axis is
    scaled independently, since the mask and the image need not share an aspect
    ratio.

    Coordinates are multiplied directly, without a pixel-centre correction, so
    a contour touching the far edge of the mask lands up to `(scale - 1)` pixels
    short of the image edge - 11 px at a 12x scale. This matches
    `inference.core.utils.postprocess.scale_polygons`, which the legacy backend
    uses, and keeping the two in agreement is worth more than halving the bias
    on one of them. Changing the convention would make the same input produce
    different polygons depending on which backend served it.

    Args:
        polygons: Contours in mask coordinates, each of shape `(k, 2)` as `(x, y)`.
        mask_size: Dimensions of the mask the contours were extracted from.
        image_size: Dimensions of the image the contours should be reported in.

    Returns:
        Contours in image coordinates. The input is returned unchanged when the
        two sizes already match.
    """
    if not polygons:
        return []

    x_scale = image_size.width / mask_size.width
    y_scale = image_size.height / mask_size.height
    if x_scale == 1.0 and y_scale == 1.0:
        return polygons

    scale = np.array([x_scale, y_scale], dtype=np.float32)
    scaled_polygons = [polygon * scale for polygon in polygons]

    return scaled_polygons


def resolve_mask_frame_size(metadata: PreProcessingMetadata) -> ImageDimensions:
    """Resolve the image-space extent covered by aligned masks.

    Args:
        metadata: Preprocessing transforms used to align the masks.

    Returns:
        Original image dimensions, including for crops anchored at the origin.
    """
    return metadata.original_size


def resolve_mask_target_size(
    mask_height: int,
    mask_width: int,
    *,
    size_after_pre_processing: ImageDimensions,
    masks_resolution_factor: float,
) -> Tuple[int, int]:
    """Interpolate the mask resize target between the mask grid and the image.

    The endpoints are explicit: `1.0` selects `size_after_pre_processing`,
    reproducing the behaviour from before this parameter existed, and `0.0`
    selects the grid the model produced masks on. Values in between interpolate
    linearly between those two sizes.

    **A lower factor does not guarantee smaller masks or lower latency.** When
    the image is smaller than the model's mask grid the interpolation runs the
    other way: with a 160x160 grid and a 100x100 image, `0.0` yields 25,600
    mask pixels against 10,000 at `1.0`. The `fast` and `accurate` names on
    `mask_decode_mode` describe the common case where the image is larger than
    the grid, and are misleading below it. Callers sensitive to this should
    compare `size_after_pre_processing` against the mask grid rather than
    assume a direction.

    Args:
        mask_height: Height of the mask grid after letterbox padding is removed.
        mask_width: Width of that grid.
        size_after_pre_processing: Dimensions masks are reported against at a
            factor of `1.0`.
        masks_resolution_factor: Interpolation factor in `[0.0, 1.0]`.

    Returns:
        The `(height, width)` to resize masks to.

    Raises:
        ValueError: If the resolution factor is outside the finite range [0, 1].
    """
    if not 0.0 <= masks_resolution_factor <= 1.0:
        raise ValueError("masks_resolution_factor must be finite and in [0.0, 1.0]")

    if masks_resolution_factor >= 1.0:
        return size_after_pre_processing.height, size_after_pre_processing.width

    factor = masks_resolution_factor
    height = max(
        1, round(mask_height * (1 - factor) + size_after_pre_processing.height * factor)
    )
    width = max(
        1, round(mask_width * (1 - factor) + size_after_pre_processing.width * factor)
    )

    return height, width


def resolve_unpadded_mask_grid(
    mask_height: int,
    mask_width: int,
    *,
    padding: Tuple[int, int, int, int],
    inference_size: ImageDimensions,
) -> Tuple[int, int]:
    """Size of the mask grid once letterbox padding is removed.

    The populated path slices the padding away before resizing, so the resize
    target is interpolated from the unpadded grid. The empty path has no masks
    to slice, and must compute the same number or it reports a shape the
    populated path would never produce.

    Args:
        mask_height: Height of the mask grid as the model produced it.
        mask_width: Width of that grid.
        padding: Letterbox padding as `(left, top, right, bottom)` in
            network-input pixels.
        inference_size: Network input dimensions the padding refers to.

    Returns:
        The unpadded `(height, width)` of the mask grid.
    """
    pad_left, pad_top, pad_right, pad_bottom = padding
    height_scale = mask_height / inference_size.height
    width_scale = mask_width / inference_size.width
    unpadded_height = (
        mask_height - round(height_scale * pad_top) - round(height_scale * pad_bottom)
    )
    unpadded_width = (
        mask_width - round(width_scale * pad_left) - round(width_scale * pad_right)
    )

    return max(1, unpadded_height), max(1, unpadded_width)


def resolve_mask_output_size(
    mask_height: int,
    mask_width: int,
    *,
    padding: Tuple[int, int, int, int],
    inference_size: ImageDimensions,
    original_size: ImageDimensions,
    size_after_pre_processing: ImageDimensions,
    static_crop_offset: StaticCropOffset,
    masks_resolution_factor: float = 1.0,
) -> Tuple[int, int]:
    """Resolve the final mask canvas dimensions without materializing a mask.

    Args:
        mask_height: Native mask height before removing network padding.
        mask_width: Native mask width before removing network padding.
        padding: Network padding as ``(left, top, right, bottom)``.
        inference_size: Network dimensions corresponding to the native masks.
        original_size: Original image dimensions before static cropping.
        size_after_pre_processing: Image dimensions after static cropping.
        static_crop_offset: Crop location and dimensions in the original image.
        masks_resolution_factor: Mask-grid interpolation factor in ``[0, 1]``.

    Returns:
        Final ``(height, width)``, including rounded static-crop canvas placement.

    Raises:
        ValueError: If the resolution factor is outside the finite range [0, 1].
    """
    unpadded_height, unpadded_width = resolve_unpadded_mask_grid(
        mask_height,
        mask_width,
        padding=padding,
        inference_size=inference_size,
    )
    target_height, target_width = resolve_mask_target_size(
        mask_height=unpadded_height,
        mask_width=unpadded_width,
        size_after_pre_processing=size_after_pre_processing,
        masks_resolution_factor=masks_resolution_factor,
    )
    output_height, output_width = target_height, target_width
    if (
        static_crop_offset.offset_x > 0
        or static_crop_offset.offset_y > 0
        or size_after_pre_processing != original_size
    ):
        height_scale = target_height / size_after_pre_processing.height
        width_scale = target_width / size_after_pre_processing.width
        output_height = max(
            1,
            round(original_size.height * height_scale),
            round(static_crop_offset.offset_y * height_scale) + target_height,
        )
        output_width = max(
            1,
            round(original_size.width * width_scale),
            round(static_crop_offset.offset_x * width_scale) + target_width,
        )

    return output_height, output_width


def _align_boxes_to_mask_grid(
    boxes: torch.Tensor,
    *,
    padding: Tuple[int, int, int, int],
    scale_width: float,
    scale_height: float,
    size_after_pre_processing: ImageDimensions,
    target_size: Tuple[int, int],
    canvas_size: Tuple[int, int],
    canvas_offset: Tuple[int, int],
) -> torch.Tensor:
    """Map network boxes directly to the resized mask and its canvas placement."""
    target_height, target_width = target_size
    if target_size != tuple(size_after_pre_processing):
        boxes = boxes.float()
        scale_width *= size_after_pre_processing.width / target_width
        scale_height *= size_after_pre_processing.height / target_height

    pad_left, pad_top, _, _ = padding
    offset_x, offset_y = canvas_offset
    canvas_height, canvas_width = canvas_size
    pad = boxes.new_tensor([pad_left, pad_top, pad_left, pad_top])
    scale = boxes.new_tensor([scale_width, scale_height, scale_width, scale_height])
    maximum = boxes.new_tensor(
        [canvas_width, canvas_height, canvas_width, canvas_height]
    )
    boxes[:, :4].sub_(pad).div_(scale)
    if offset_x or offset_y:
        offset = boxes.new_tensor([offset_x, offset_y, offset_x, offset_y])
        boxes[:, :4].add_(offset)

    boxes[:, :4].clamp_(min=torch.zeros_like(maximum), max=maximum)

    return boxes


def finalize_instance_segmentation_boxes(
    boxes: torch.Tensor,
    *,
    mask_size: Tuple[int, int],
    image_size: ImageDimensions,
) -> torch.Tensor:
    """Preserve fractional mask-grid boxes and legacy image-grid rounding.

    Args:
        boxes: Aligned ``xyxy`` boxes already on the final mask grid.
        mask_size: Encoded mask dimensions as ``(height, width)``.
        image_size: Original image dimensions.

    Returns:
        Integer boxes when the mask and image grids match, rounded once in
        that final grid. Otherwise, float32 boxes without quantization.
    """
    if tuple(mask_size) == tuple(image_size):
        final_boxes = boxes.round().int()
    else:
        final_boxes = boxes.float()

    return final_boxes


def align_instance_segmentation_results(
    image_bboxes: torch.Tensor,
    masks: torch.Tensor,
    padding: Tuple[int, int, int, int],
    scale_width: float,
    scale_height: float,
    original_size: ImageDimensions,
    size_after_pre_processing: ImageDimensions,
    inference_size: ImageDimensions,
    static_crop_offset: StaticCropOffset,
    binarization_threshold: float = 0.0,
    mask_chunk_size: int = INFERENCE_MODELS_INSTANCE_SEG_MASK_PROCESSING_CHUNK_SIZE,
    masks_resolution_factor: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Align network boxes and dense masks onto the same output grid.

    Args:
        image_bboxes: Network-space boxes in the first four columns. Other
            columns are retained; the input may be modified in place.
        masks: Per-instance mask logits on the model grid.
        padding: Network-input padding as ``(left, top, right, bottom)``.
        scale_width: Horizontal image-to-network resize ratio.
        scale_height: Vertical image-to-network resize ratio.
        original_size: Original image dimensions, before any static crop.
        size_after_pre_processing: Image dimensions after static cropping.
        inference_size: Network input dimensions corresponding to the masks.
        static_crop_offset: Location and dimensions of the static crop.
        binarization_threshold: Threshold applied after resizing the masks.
        mask_chunk_size: Maximum masks resized together.
        masks_resolution_factor: Interpolation between model and image grids.

    Returns:
        Boxes and boolean masks on the final mask canvas. Boxes map directly
        from network coordinates without intermediate image-space rounding,
        using the same rounded crop placement as the masks.
    """
    if image_bboxes.shape[0] == 0:
        empty_height, empty_width = resolve_mask_output_size(
            masks.shape[1],
            masks.shape[2],
            padding=padding,
            inference_size=inference_size,
            original_size=original_size,
            size_after_pre_processing=size_after_pre_processing,
            static_crop_offset=static_crop_offset,
            masks_resolution_factor=masks_resolution_factor,
        )
        empty_masks = torch.empty(
            size=(0, empty_height, empty_width),
            dtype=torch.bool,
            device=image_bboxes.device,
        )
        return image_bboxes, empty_masks

    pad_left, pad_top, pad_right, pad_bottom = padding
    n, mh, mw = masks.shape
    mask_h_scale = mh / inference_size.height
    mask_w_scale = mw / inference_size.width
    mask_pad_top, mask_pad_bottom, mask_pad_left, mask_pad_right = (
        round(mask_h_scale * pad_top),
        round(mask_h_scale * pad_bottom),
        round(mask_w_scale * pad_left),
        round(mask_w_scale * pad_right),
    )
    if (
        mask_pad_top < 0
        or mask_pad_bottom < 0
        or mask_pad_left < 0
        or mask_pad_right < 0
    ):
        masks = torch.nn.functional.pad(
            masks,
            (
                abs(min(mask_pad_left, 0)),
                abs(min(mask_pad_right, 0)),
                abs(min(mask_pad_top, 0)),
                abs(min(mask_pad_bottom, 0)),
            ),
            "constant",
            0,
        )
        padded_mask_offset_top = max(mask_pad_top, 0)
        padded_mask_offset_bottom = max(mask_pad_bottom, 0)
        padded_mask_offset_left = max(mask_pad_left, 0)
        padded_mask_offset_right = max(mask_pad_right, 0)
        masks = masks[
            :,
            padded_mask_offset_top : masks.shape[1] - padded_mask_offset_bottom,
            padded_mask_offset_left : masks.shape[2] - padded_mask_offset_right,
        ]
    else:
        masks = masks[
            :, mask_pad_top : mh - mask_pad_bottom, mask_pad_left : mw - mask_pad_right
        ]
    # Resize to full resolution in chunks: the float32 working set is
    # chunk x H x W instead of n x H x W (17+ GiB at 300 masks on a 12MP
    # image); only the n x H x W bool output is ever materialized whole.
    mask_chunk_size = max(1, int(mask_chunk_size))
    target_height, target_width = resolve_mask_target_size(
        mask_height=masks.shape[1],
        mask_width=masks.shape[2],
        size_after_pre_processing=size_after_pre_processing,
        masks_resolution_factor=masks_resolution_factor,
    )
    binarized_masks = torch.empty(
        (masks.shape[0], target_height, target_width),
        dtype=torch.bool,
        device=masks.device,
    )
    for start in range(0, masks.shape[0], mask_chunk_size):
        binarized_masks[start : start + mask_chunk_size] = functional.resize(
            masks[start : start + mask_chunk_size],
            [target_height, target_width],
            interpolation=functional.InterpolationMode.BILINEAR,
        ).gt_(binarization_threshold)
    masks = binarized_masks
    canvas_offset_x = canvas_offset_y = 0
    if (
        static_crop_offset.offset_x > 0
        or static_crop_offset.offset_y > 0
        or size_after_pre_processing != original_size
    ):
        # the canvas keeps the mask grid's ratio to the image, so a reduced
        # factor shrinks the whole output rather than planting a small mask on
        # a full-size canvas; crop offsets move into the same space
        canvas_h_scale = target_height / size_after_pre_processing.height
        canvas_w_scale = target_width / size_after_pre_processing.width
        canvas_offset_y = round(static_crop_offset.offset_y * canvas_h_scale)
        canvas_offset_x = round(static_crop_offset.offset_x * canvas_w_scale)
        # The canvas must contain the pasted extent. Rounding the canvas and
        # the offset independently can leave the canvas one pixel short, so
        # take whichever is larger rather than trusting the rounded product.
        mask_canvas = torch.zeros(
            (
                masks.shape[0],
                max(
                    1,
                    round(original_size.height * canvas_h_scale),
                    canvas_offset_y + masks.shape[1],
                ),
                max(
                    1,
                    round(original_size.width * canvas_w_scale),
                    canvas_offset_x + masks.shape[2],
                ),
            ),
            dtype=torch.bool,
            device=masks.device,
        )
        mask_canvas[
            :,
            canvas_offset_y : canvas_offset_y + masks.shape[1],
            canvas_offset_x : canvas_offset_x + masks.shape[2],
        ] = masks
        masks = mask_canvas
    image_bboxes = _align_boxes_to_mask_grid(
        image_bboxes,
        padding=padding,
        scale_width=scale_width,
        scale_height=scale_height,
        size_after_pre_processing=size_after_pre_processing,
        target_size=(target_height, target_width),
        canvas_size=tuple(masks.shape[1:]),
        canvas_offset=(canvas_offset_x, canvas_offset_y),
    )
    return image_bboxes, masks


def align_instance_segmentation_results_to_rle_masks_batched(
    image_bboxes: torch.Tensor,
    masks: torch.Tensor,
    padding: Tuple[int, int, int, int],
    scale_width: float,
    scale_height: float,
    original_size: ImageDimensions,
    size_after_pre_processing: ImageDimensions,
    inference_size: ImageDimensions,
    static_crop_offset: StaticCropOffset,
    binarization_threshold: float = 0.0,
    mask_chunk_size: int = INFERENCE_MODELS_INSTANCE_SEG_MASK_PROCESSING_CHUNK_SIZE,
    masks_resolution_factor: float = 1.0,
) -> Tuple[torch.Tensor, List[dict]]:
    """Chunked batch variant of align_instance_segmentation_results_to_rle_masks.

    Aligns and RLE-encodes detections ``mask_chunk_size`` at a time: each chunk
    goes through align_instance_segmentation_results and a single
    device->host transfer in torch_masks_to_coco_rle_batch. Compared to the
    per-detection generator this cuts host syncs from ~2*N to ceil(N / chunk),
    while peak memory stays bounded by ``chunk x H x W`` full-resolution bool
    masks instead of materializing all ``N x H x W`` at once. Output (boxes and
    RLE) is identical to the generator.

    NOTE: image_bboxes is modified in-place (same behaviour as the other
    variants). Pass a .clone() if that's not acceptable.
    """
    if image_bboxes.shape[0] == 0:
        return image_bboxes, []
    mask_chunk_size = max(1, int(mask_chunk_size))
    aligned_boxes_chunks, rle_masks = [], []
    for start in range(0, image_bboxes.shape[0], mask_chunk_size):
        end = start + mask_chunk_size
        chunk_boxes, chunk_masks = align_instance_segmentation_results(
            image_bboxes=image_bboxes[start:end],
            masks=masks[start:end],
            padding=padding,
            scale_width=scale_width,
            scale_height=scale_height,
            original_size=original_size,
            size_after_pre_processing=size_after_pre_processing,
            inference_size=inference_size,
            static_crop_offset=static_crop_offset,
            binarization_threshold=binarization_threshold,
            mask_chunk_size=mask_chunk_size,
            masks_resolution_factor=masks_resolution_factor,
        )
        aligned_boxes_chunks.append(chunk_boxes)
        rle_masks.extend(torch_masks_to_coco_rle_batch(chunk_masks))
        del chunk_masks
    return torch.cat(aligned_boxes_chunks, dim=0), rle_masks


def align_instance_segmentation_results_to_rle_masks(
    image_bboxes: torch.Tensor,
    masks: torch.Tensor,
    padding: Tuple[int, int, int, int],
    scale_width: float,
    scale_height: float,
    original_size: ImageDimensions,
    size_after_pre_processing: ImageDimensions,
    inference_size: ImageDimensions,
    static_crop_offset: StaticCropOffset,
    binarization_threshold: float = 0.0,
    masks_resolution_factor: float = 1.0,
) -> Generator[Tuple[torch.Tensor, dict], None, None]:
    """Align boxes and encode one mask at a time on their shared output grid.

    Args:
        image_bboxes: Network-space boxes in the first four columns. Other
            columns are retained; the input may be modified in place.
        masks: Per-instance mask logits on the model grid.
        padding: Network-input padding as ``(left, top, right, bottom)``.
        scale_width: Horizontal image-to-network resize ratio.
        scale_height: Vertical image-to-network resize ratio.
        original_size: Original image dimensions, before any static crop.
        size_after_pre_processing: Image dimensions after static cropping.
        inference_size: Network input dimensions corresponding to the masks.
        static_crop_offset: Location and dimensions of the static crop.
        binarization_threshold: Threshold applied after resizing the masks.
        masks_resolution_factor: Interpolation between model and image grids.

    Yields:
        A box and COCO RLE mask sharing the final mask canvas. Boxes map
        directly from network coordinates with no intermediate rounding.
        Only one resized dense mask is materialized at a time.
    """
    if image_bboxes.shape[0] == 0:
        return None

    pad_left, pad_top, pad_right, pad_bottom = padding
    needs_canvas = (
        static_crop_offset.offset_x > 0
        or static_crop_offset.offset_y > 0
        or size_after_pre_processing != original_size
    )
    n, mh, mw = masks.shape
    mask_h_scale = mh / inference_size.height
    mask_w_scale = mw / inference_size.width
    mask_pad_top, mask_pad_bottom, mask_pad_left, mask_pad_right = (
        round(mask_h_scale * pad_top),
        round(mask_h_scale * pad_bottom),
        round(mask_w_scale * pad_left),
        round(mask_w_scale * pad_right),
    )
    if (
        mask_pad_top < 0
        or mask_pad_bottom < 0
        or mask_pad_left < 0
        or mask_pad_right < 0
    ):
        masks = torch.nn.functional.pad(
            masks,
            (
                abs(min(mask_pad_left, 0)),
                abs(min(mask_pad_right, 0)),
                abs(min(mask_pad_top, 0)),
                abs(min(mask_pad_bottom, 0)),
            ),
            "constant",
            0,
        )
        padded_mask_offset_top = max(mask_pad_top, 0)
        padded_mask_offset_bottom = max(mask_pad_bottom, 0)
        padded_mask_offset_left = max(mask_pad_left, 0)
        padded_mask_offset_right = max(mask_pad_right, 0)
        masks = masks[
            :,
            padded_mask_offset_top : masks.shape[1] - padded_mask_offset_bottom,
            padded_mask_offset_left : masks.shape[2] - padded_mask_offset_right,
        ]
    else:
        masks = masks[
            :, mask_pad_top : mh - mask_pad_bottom, mask_pad_left : mw - mask_pad_right
        ]

    target_h, target_w = resolve_mask_target_size(
        mask_height=masks.shape[1],
        mask_width=masks.shape[2],
        size_after_pre_processing=size_after_pre_processing,
        masks_resolution_factor=masks_resolution_factor,
    )
    # the canvas keeps the mask grid's ratio to the image, matching the dense
    # path, so crop offsets move into the same space
    canvas_h_scale = target_h / size_after_pre_processing.height
    canvas_w_scale = target_w / size_after_pre_processing.width
    offset_y = round(static_crop_offset.offset_y * canvas_h_scale)
    offset_x = round(static_crop_offset.offset_x * canvas_w_scale)
    # as above: the canvas must contain offset + mask, not merely the rounded
    # product of the original size and the scale
    canvas_height = max(
        1, round(original_size.height * canvas_h_scale), offset_y + target_h
    )
    canvas_width = max(
        1, round(original_size.width * canvas_w_scale), offset_x + target_w
    )
    image_bboxes = _align_boxes_to_mask_grid(
        image_bboxes,
        padding=padding,
        scale_width=scale_width,
        scale_height=scale_height,
        size_after_pre_processing=size_after_pre_processing,
        target_size=(target_h, target_w),
        canvas_size=(
            (canvas_height, canvas_width) if needs_canvas else (target_h, target_w)
        ),
        canvas_offset=(offset_x, offset_y) if needs_canvas else (0, 0),
    )
    num_instances = image_bboxes.shape[0]
    for i in range(num_instances):
        # keep a batch dim so functional.resize is unambiguous
        single = masks[i : i + 1]
        resized = (
            functional.resize(
                single,
                [target_h, target_w],
                interpolation=functional.InterpolationMode.BILINEAR,
            )
            .gt_(binarization_threshold)
            .to(dtype=torch.bool)
        )
        if needs_canvas:
            mask_canvas = torch.zeros(
                (canvas_height, canvas_width),
                dtype=torch.bool,
                device=resized.device,
            )
            mask_canvas[
                offset_y : offset_y + resized.shape[1],
                offset_x : offset_x + resized.shape[2],
            ] = resized[0]
            converted = torch_mask_to_coco_rle(mask_canvas)
            del mask_canvas
        else:
            converted = torch_mask_to_coco_rle(resized[0])
        del resized
        yield image_bboxes[i], converted
    return None


class ConfidenceFilter:
    """Resolves per-class confidence thresholds.

    ``confidence`` selects the mode:

      - ``"best"`` — per-class → global → model default.
      - ``"default"`` — skip recommended_parameters, use model default.
      - ``float`` — uniform user override for all classes.
    """

    def __init__(
        self,
        *,
        confidence: Confidence = "default",
        recommended_parameters: Optional[RecommendedParameters] = None,
        default_confidence: float = INFERENCE_MODELS_DEFAULT_CONFIDENCE,
    ):
        self._class_to_threshold_map = self._resolve_class_to_threshold_map(
            confidence, recommended_parameters
        )
        self._fallback_threshold = self._resolve_fallback_threshold(
            confidence, recommended_parameters, default_confidence
        )
        LOGGER.debug(
            "ConfidenceFilter: confidence=%s, recommended_parameters=%s, "
            "default_confidence=%.4f -> class_to_threshold_map=%s, "
            "fallback_threshold=%.4f",
            confidence,
            recommended_parameters,
            default_confidence,
            self._class_to_threshold_map,
            self._fallback_threshold,
        )

    def get_threshold(self, class_names: List[str]) -> Union[float, torch.Tensor]:
        """Return the confidence threshold to apply.

        Returns a scalar float when the same threshold applies to all
        classes (fast path). Returns a 1-D CPU tensor of shape
        `(len(class_names),)` indexed by class_id when per-class
        thresholds are in effect.
        """
        if self._class_to_threshold_map is None:
            return self._fallback_threshold
        return torch.tensor(
            [
                self._class_to_threshold_map.get(name, self._fallback_threshold)
                for name in class_names
            ]
        )

    @staticmethod
    def _resolve_class_to_threshold_map(
        confidence: Confidence,
        recommended_parameters: Optional[RecommendedParameters],
    ) -> Optional[Dict[str, float]]:
        if confidence != "best":
            return None
        if (
            recommended_parameters is not None
            and recommended_parameters.confidence is not None
            and recommended_parameters.per_class_confidence
        ):
            return recommended_parameters.per_class_confidence
        return None

    @staticmethod
    def _resolve_fallback_threshold(
        confidence: Confidence,
        recommended_parameters: Optional[RecommendedParameters],
        default_confidence: float,
    ) -> float:
        if isinstance(confidence, float):
            return confidence
        if confidence == "default":
            return default_confidence
        if (
            recommended_parameters is not None
            and recommended_parameters.confidence is not None
        ):
            return recommended_parameters.confidence
        return default_confidence


def post_process_semantic_segmentation_logits(
    model_results: torch.Tensor,
    pre_processing_meta: List[PreProcessingMetadata],
    class_names: List[str],
    background_class_id: int,
    device: torch.device,
    confidence: Confidence,
    recommended_parameters: Optional[RecommendedParameters],
    default_confidence: float,
    class_activation: Literal["softmax", "sigmoid"] = "softmax",
) -> List[SemanticSegmentationResult]:
    """Shared post-processing for semantic-segmentation models that emit
    (B, K, H, W) float logits. Used by DeepLabV3+, YOLO26-sem and RF-DETR-sem.

    Steps: crop out letterbox padding → resize back to pre-letterbox size →
    softmax over classes → argmax → place into original-image canvas if
    static_crop was applied → apply per-class confidence threshold
    (sub-threshold pixels collapse to background_class_id).

    Single-channel (K==1) outputs are the Ultralytics binary (``nc==1``) head:
    instead of softmax+argmax, the sigmoid foreground probability is used and the
    lone foreground class is read from ``class_names`` (``[background, <fg>]``).
    Sub-threshold pixels collapse to background via the same threshold step.

    ``class_activation="sigmoid"`` takes the top per-class sigmoid as the pixel
    confidence instead of the softmax maximum, for models trained with per-class
    BCE. The class map is the same argmax either way.
    """
    confidence_filter = ConfidenceFilter(
        confidence=confidence,
        recommended_parameters=recommended_parameters,
        default_confidence=default_confidence,
    )
    results: List[SemanticSegmentationResult] = []
    for image_results, image_metadata in zip(model_results, pre_processing_meta):
        inference_size = image_metadata.inference_size
        mask_h_scale = model_results.shape[2] / inference_size.height
        mask_w_scale = model_results.shape[3] / inference_size.width
        mask_pad_top, mask_pad_bottom, mask_pad_left, mask_pad_right = (
            round(mask_h_scale * image_metadata.pad_top),
            round(mask_h_scale * image_metadata.pad_bottom),
            round(mask_w_scale * image_metadata.pad_left),
            round(mask_w_scale * image_metadata.pad_right),
        )
        _, mh, mw = image_results.shape
        if (
            mask_pad_top < 0
            or mask_pad_bottom < 0
            or mask_pad_left < 0
            or mask_pad_right < 0
        ):
            image_results = torch.nn.functional.pad(
                image_results,
                (
                    abs(min(mask_pad_left, 0)),
                    abs(min(mask_pad_right, 0)),
                    abs(min(mask_pad_top, 0)),
                    abs(min(mask_pad_bottom, 0)),
                ),
                "constant",
                background_class_id,
            )
            padded_mask_offset_top = max(mask_pad_top, 0)
            padded_mask_offset_bottom = max(mask_pad_bottom, 0)
            padded_mask_offset_left = max(mask_pad_left, 0)
            padded_mask_offset_right = max(mask_pad_right, 0)
            image_results = image_results[
                :,
                padded_mask_offset_top : image_results.shape[1]
                - padded_mask_offset_bottom,
                padded_mask_offset_left : image_results.shape[2]
                - padded_mask_offset_right,
            ]
        else:
            image_results = image_results[
                :,
                mask_pad_top : mh - mask_pad_bottom,
                mask_pad_left : mw - mask_pad_right,
            ]
        if (
            image_results.shape[1] != image_metadata.size_after_pre_processing.height
            or image_results.shape[2] != image_metadata.size_after_pre_processing.width
        ):
            image_results = functional.resize(
                image_results,
                [
                    image_metadata.size_after_pre_processing.height,
                    image_metadata.size_after_pre_processing.width,
                ],
                interpolation=functional.InterpolationMode.BILINEAR,
            )
        if image_results.shape[0] == 1:
            image_confidence = image_results[0].sigmoid()
            image_class_ids = insert_background_class(
                torch.zeros_like(image_confidence, dtype=torch.long),
                background_class_id=background_class_id,
                num_classes=len(class_names),
            )
        else:
            if class_activation == "sigmoid":
                # Pick the class on the raw logits: sigmoid saturates, and ties at 1.0 would favor the lowest index.
                max_logits, image_class_ids = torch.max(image_results, dim=0)
                image_confidence = max_logits.sigmoid()
            else:
                image_results = torch.nn.functional.softmax(image_results, dim=0)
                image_confidence, image_class_ids = torch.max(image_results, dim=0)
            if len(class_names) == image_results.shape[0] + 1:
                image_class_ids = insert_background_class(
                    image_class_ids,
                    background_class_id=background_class_id,
                    num_classes=len(class_names),
                )
        if (
            image_metadata.static_crop_offset.offset_x > 0
            or image_metadata.static_crop_offset.offset_y > 0
        ):
            original_size_confidence_canvas = torch.zeros(
                (
                    image_metadata.original_size.height,
                    image_metadata.original_size.width,
                ),
                device=device,
                dtype=image_confidence.dtype,
            )
            original_size_confidence_canvas[
                image_metadata.static_crop_offset.offset_y : image_metadata.static_crop_offset.offset_y
                + image_confidence.shape[0],
                image_metadata.static_crop_offset.offset_x : image_metadata.static_crop_offset.offset_x
                + image_confidence.shape[1],
            ] = image_confidence
            original_size_confidence_class_id_canvas = (
                torch.ones(
                    (
                        image_metadata.original_size.height,
                        image_metadata.original_size.width,
                    ),
                    device=device,
                    dtype=image_class_ids.dtype,
                )
                * background_class_id
            )
            original_size_confidence_class_id_canvas[
                image_metadata.static_crop_offset.offset_y : image_metadata.static_crop_offset.offset_y
                + image_class_ids.shape[0],
                image_metadata.static_crop_offset.offset_x : image_metadata.static_crop_offset.offset_x
                + image_class_ids.shape[1],
            ] = image_class_ids
            image_class_ids = original_size_confidence_class_id_canvas
            image_confidence = original_size_confidence_canvas
        threshold = confidence_filter.get_threshold(class_names)
        if isinstance(threshold, torch.Tensor):
            threshold = threshold.to(
                dtype=image_confidence.dtype, device=image_confidence.device
            )
        below = image_confidence < (
            threshold[image_class_ids.long()]
            if isinstance(threshold, torch.Tensor)
            else threshold
        )
        image_class_ids = image_class_ids.clone()
        image_confidence = image_confidence.clone()
        image_class_ids[below] = background_class_id
        image_confidence[below] = 0.0
        results.append(
            SemanticSegmentationResult(
                segmentation_map=image_class_ids,
                confidence=image_confidence,
            )
        )
    return results


def post_process_depth_estimation_map(
    model_results: torch.Tensor,
    pre_processing_meta: List[PreProcessingMetadata],
    device: torch.device,
) -> List[torch.Tensor]:
    """Shared post-processing for depth-estimation models that emit dense
    (B, 1, H, W) or (B, H, W) depth maps. Used by YOLO26-depth.

    Values are preserved as-is (e.g. metric meters). Steps: crop out letterbox
    padding (offsets scaled to the map resolution, which may differ from the
    network input size) → resize back to pre-letterbox size → place into a
    zero-filled original-image canvas if static_crop was applied (depth outside
    the crop region is unknown and reported as 0.0).
    """
    if model_results.ndim == 3:
        model_results = model_results.unsqueeze(1)
    results: List[torch.Tensor] = []
    for image_results, image_metadata in zip(model_results, pre_processing_meta):
        inference_size = image_metadata.inference_size
        map_h_scale = model_results.shape[2] / inference_size.height
        map_w_scale = model_results.shape[3] / inference_size.width
        map_pad_top, map_pad_bottom, map_pad_left, map_pad_right = (
            round(map_h_scale * image_metadata.pad_top),
            round(map_h_scale * image_metadata.pad_bottom),
            round(map_w_scale * image_metadata.pad_left),
            round(map_w_scale * image_metadata.pad_right),
        )
        _, mh, mw = image_results.shape
        if (
            map_pad_top < 0
            or map_pad_bottom < 0
            or map_pad_left < 0
            or map_pad_right < 0
        ):
            image_results = torch.nn.functional.pad(
                image_results,
                (
                    abs(min(map_pad_left, 0)),
                    abs(min(map_pad_right, 0)),
                    abs(min(map_pad_top, 0)),
                    abs(min(map_pad_bottom, 0)),
                ),
                "constant",
                0.0,
            )
            padded_map_offset_top = max(map_pad_top, 0)
            padded_map_offset_bottom = max(map_pad_bottom, 0)
            padded_map_offset_left = max(map_pad_left, 0)
            padded_map_offset_right = max(map_pad_right, 0)
            image_results = image_results[
                :,
                padded_map_offset_top : image_results.shape[1]
                - padded_map_offset_bottom,
                padded_map_offset_left : image_results.shape[2]
                - padded_map_offset_right,
            ]
        else:
            image_results = image_results[
                :,
                map_pad_top : mh - map_pad_bottom,
                map_pad_left : mw - map_pad_right,
            ]
        if (
            image_results.shape[1] != image_metadata.size_after_pre_processing.height
            or image_results.shape[2] != image_metadata.size_after_pre_processing.width
        ):
            image_results = functional.resize(
                image_results,
                [
                    image_metadata.size_after_pre_processing.height,
                    image_metadata.size_after_pre_processing.width,
                ],
                interpolation=functional.InterpolationMode.BILINEAR,
            )
        depth_map = image_results[0].float()
        if (
            image_metadata.static_crop_offset.offset_x > 0
            or image_metadata.static_crop_offset.offset_y > 0
        ):
            original_size_canvas = torch.zeros(
                (
                    image_metadata.original_size.height,
                    image_metadata.original_size.width,
                ),
                device=device,
                dtype=depth_map.dtype,
            )
            original_size_canvas[
                image_metadata.static_crop_offset.offset_y : image_metadata.static_crop_offset.offset_y
                + depth_map.shape[0],
                image_metadata.static_crop_offset.offset_x : image_metadata.static_crop_offset.offset_x
                + depth_map.shape[1],
            ] = depth_map
            depth_map = original_size_canvas
        results.append(depth_map)
    return results
