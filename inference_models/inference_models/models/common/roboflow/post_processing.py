from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torchvision
from torchvision.transforms import functional

from inference_models.entities import ImageDimensions
from inference_models.logger import LOGGER
from inference_models.models.common.roboflow.model_packages import (
    PreProcessingMetadata,
    StaticCropOffset,
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
    bs = output.shape[0]
    boxes = output[:, :4, :]
    scores = output[:, 4:, :]
    results = []
    per_class_thresh = (
        conf_thresh.to(output.device) if isinstance(conf_thresh, torch.Tensor) else None
    )
    for b in range(bs):
        class_scores = scores[b]
        class_conf, class_ids = class_scores.max(0)
        if per_class_thresh is not None:
            mask = class_conf > per_class_thresh[class_ids]
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
    bs = output.shape[0]
    nms_results = []
    per_class_thresh = (
        conf_thresh.to(output.device) if isinstance(conf_thresh, torch.Tensor) else None
    )
    for batch_element_id in range(bs):
        batch_element_result = output[batch_element_id]
        if per_class_thresh is not None:
            class_ids = batch_element_result[:, 5].long()
            batch_element_result = batch_element_result[
                batch_element_result[:, 4] >= per_class_thresh[class_ids]
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
    bs = output.shape[0]
    boxes = output[:, :4, :]  # (N, 4, 8400)
    scores = output[:, 4:-32, :]  # (N, 80, 8400)
    masks = output[:, -32:, :]
    results = []
    per_class_thresh = (
        conf_thresh.to(output.device) if isinstance(conf_thresh, torch.Tensor) else None
    )

    for b in range(bs):
        bboxes = boxes[b].T  # (8400, 4)
        class_scores = scores[b].T  # (8400, 80)
        box_masks = masks[b].T
        class_conf, class_ids = class_scores.max(1)  # (8400,), (8400,)
        if per_class_thresh is not None:
            mask = class_conf > per_class_thresh[class_ids]
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
    bs = output.shape[0]
    boxes = output[:, :4, :]
    scores = output[:, 4 : 4 + num_classes, :]
    key_points = output[:, 4 + num_classes :, :]
    results = []
    per_class_thresh = (
        conf_thresh.to(output.device) if isinstance(conf_thresh, torch.Tensor) else None
    )
    for b in range(bs):
        class_scores = scores[b]
        class_conf, class_ids = class_scores.max(0)
        if per_class_thresh is not None:
            mask = class_conf > per_class_thresh[class_ids]
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    if image_bboxes.shape[0] == 0:
        empty_masks = torch.empty(
            size=(0, size_after_pre_processing.height, size_after_pre_processing.width),
            dtype=torch.bool,
            device=image_bboxes.device,
        )
        return image_bboxes, empty_masks
    pad_left, pad_top, pad_right, pad_bottom = padding
    offsets = torch.tensor(
        [pad_left, pad_top, pad_left, pad_top],
        device=image_bboxes.device,
    )
    image_bboxes[:, :4].sub_(offsets)
    scale = torch.as_tensor(
        [scale_width, scale_height, scale_width, scale_height],
        dtype=image_bboxes.dtype,
        device=image_bboxes.device,
    )
    image_bboxes[:, :4].div_(scale)
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
    masks = (
        functional.resize(
            masks,
            [size_after_pre_processing.height, size_after_pre_processing.width],
            interpolation=functional.InterpolationMode.BILINEAR,
        )
        .gt_(binarization_threshold)
        .to(dtype=torch.bool)
    )
    if static_crop_offset.offset_x > 0 or static_crop_offset.offset_y > 0:
        mask_canvas = torch.zeros(
            (
                masks.shape[0],
                original_size.height,
                original_size.width,
            ),
            dtype=torch.bool,
            device=masks.device,
        )
        mask_canvas[
            :,
            static_crop_offset.offset_y : static_crop_offset.offset_y + masks.shape[1],
            static_crop_offset.offset_x : static_crop_offset.offset_x + masks.shape[2],
        ] = masks
        static_crop_offsets = torch.as_tensor(
            [
                static_crop_offset.offset_x,
                static_crop_offset.offset_y,
                static_crop_offset.offset_x,
                static_crop_offset.offset_y,
            ],
            dtype=image_bboxes.dtype,
            device=image_bboxes.device,
        )
        image_bboxes[:, :4].add_(static_crop_offsets)
        masks = mask_canvas
    return image_bboxes, masks


class ConfidenceFilter:
    """
    Resolves the 4-tier priority chain (highest to lowest) for confidence
    thresholding, honoring per-model defaults and model-eval-derived
    `recommendedParameters`:

      1. Explicit user value — single global threshold for everything
      2. Per-class optimal — per-class filter with the global optimal (or
         model's hardcoded default) as fallback for classes not in the map
      3. Global optimal — single threshold for everything
      4. Model's hardcoded default — single threshold for everything

    Exposes:
      - `floor`: the lowest threshold across all classes.
      - `per_class_thresholds(class_names)`: thresholds aligned to
        `class_names` (for tiers without per-class data, every entry is the
        floor).

    Application of thresholds is the caller's responsibility — the filter
    intentionally owns no knowledge of domain output types.
    """

    def __init__(
        self,
        *,
        user_confidence: Optional[float],
        recommended_parameters: Optional[RecommendedParameters],
        default_confidence: float,
    ):
        self._per_class_map = self._resolve_per_class_map(
            user_confidence, recommended_parameters
        )
        self._fallback = self._resolve_fallback(
            user_confidence, recommended_parameters, default_confidence
        )
        LOGGER.debug(
            "ConfidenceFilter: fallback=%.4f, per_class=%s",
            self._fallback, self._per_class_map,
        )

    def per_class_thresholds(self, class_names: List[str]) -> torch.Tensor:
        if self._per_class_map is None:
            return torch.full((len(class_names),), self._fallback)
        return torch.tensor(
            [self._per_class_map.get(name, self._fallback) for name in class_names]
        )

    @staticmethod
    def _resolve_per_class_map(
        user_confidence: Optional[float],
        recommended_parameters: Optional[RecommendedParameters],
    ) -> Optional[Dict[str, float]]:
        if user_confidence is not None:
            return None
        if recommended_parameters is None:
            return None
        return recommended_parameters.per_class_confidence or None

    @staticmethod
    def _resolve_fallback(
        user_confidence: Optional[float],
        recommended_parameters: Optional[RecommendedParameters],
        default_confidence: float,
    ) -> float:
        if user_confidence is not None:
            return user_confidence
        if (
            recommended_parameters is not None
            and recommended_parameters.confidence is not None
        ):
            return recommended_parameters.confidence
        return default_confidence

