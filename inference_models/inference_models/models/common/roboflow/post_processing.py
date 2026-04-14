from typing import Dict, List, Literal, Optional, Tuple

import torch
import torchvision
from torchvision.transforms import functional

from inference_models.entities import ImageDimensions
from inference_models.logger import LOGGER
from inference_models.models.base.classification import (
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.semantic_segmentation import (
    SemanticSegmentationResult,
)
from inference_models.models.common.roboflow.model_packages import (
    PreProcessingMetadata,
    StaticCropOffset,
)
from inference_models.weights_providers.entities import RecommendedParameters


def run_nms_for_object_detection(
    output: torch.Tensor,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    max_detections: int = 100,
    class_agnostic: bool = False,
    box_format: Literal["xywh", "xyxy"] = "xywh",
) -> List[torch.Tensor]:
    bs = output.shape[0]
    boxes = output[:, :4, :]
    scores = output[:, 4:, :]
    results = []
    for b in range(bs):
        # Combine transpose & max for efficiency
        class_scores = scores[b]  # (80, 8400)
        class_conf, class_ids = class_scores.max(0)  # (8400,), (8400,)
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
    conf_thresh: float = 0.25,
) -> List[torch.Tensor]:
    bs = output.shape[0]
    nms_results = []
    for batch_element_id in range(bs):
        batch_element_result = output[batch_element_id]
        batch_element_result = batch_element_result[
            batch_element_result[:, 4] >= conf_thresh
        ]
        nms_results.append(batch_element_result)
    return nms_results


def run_nms_for_instance_segmentation(
    output: torch.Tensor,
    conf_thresh: float = 0.25,
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

    for b in range(bs):
        bboxes = boxes[b].T  # (8400, 4)
        class_scores = scores[b].T  # (8400, 80)
        box_masks = masks[b].T
        class_conf, class_ids = class_scores.max(1)  # (8400,), (8400,)
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
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    max_detections: int = 100,
    class_agnostic: bool = False,
) -> List[torch.Tensor]:
    bs = output.shape[0]
    boxes = output[:, :4, :]
    scores = output[:, 4 : 4 + num_classes, :]
    key_points = output[:, 4 + num_classes :, :]
    results = []
    for b in range(bs):
        class_scores = scores[b]
        class_conf, class_ids = class_scores.max(0)
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
      - `floor`: the lowest threshold to give the underlying model so its NMS
        doesn't drop boxes we still want to consider for per-class refinement.
      - `passes(class_name, confidence)`: per-detection refinement check.
      - `build_keep_mask(class_ids, confidences, class_names)`: vectorized
        mask for parallel-array detection shapes (OD/IS/KP).
      - `per_class_thresholds(class_names)`: lookup table for shapes that
        index by class_id (e.g. semantic segmentation per-pixel).
      - `has_per_class_refinement`: short-circuit hint when there's nothing
        to refine on top of the floor.
      - `refine_*(...)`: per-image refinement methods to call inline inside
        each concrete model's existing per-image loop.
    """

    def __init__(
        self,
        user_confidence: Optional[float],
        recommended_parameters: Optional[RecommendedParameters],
        default_confidence: float,
    ):
        # Tier 1: explicit user value wins outright. No per-class refinement
        # needed because the floor IS the final threshold.
        if user_confidence is not None:
            self._floor = user_confidence
            self._per_class: Optional[Dict[str, float]] = None
            self._fallback = user_confidence
            LOGGER.debug(
                "ConfidenceFilter: tier 1 (user override), floor=%.4f, fallback=%.4f, per_class=%s",
                self._floor, self._fallback, self._per_class,
            )
            return

        global_optimal = (
            recommended_parameters.confidence
            if recommended_parameters is not None
            else None
        )
        per_class = (
            recommended_parameters.per_class_confidence
            if recommended_parameters is not None
            else None
        )

        # Tier 2: per-class data present.
        if per_class:
            # Classes outside the per-class map fall back to the global
            # optimal, or the model's default if no global was set.
            self._fallback = (
                global_optimal
                if global_optimal is not None
                else default_confidence
            )
            self._per_class = dict(per_class)
            # Floor must be ≤ every threshold any class might use, so the
            # model doesn't NMS-drop boxes we'd accept after refinement.
            self._floor = min(min(per_class.values()), self._fallback)
            LOGGER.debug(
                "ConfidenceFilter: tier 2 (per-class), floor=%.4f, fallback=%.4f, per_class=%s",
                self._floor, self._fallback, self._per_class,
            )
            return

        # Tier 3: only global optimal.
        if global_optimal is not None:
            self._floor = global_optimal
            self._per_class = None
            self._fallback = global_optimal
            LOGGER.debug(
                "ConfidenceFilter: tier 3 (global optimal), floor=%.4f, fallback=%.4f, per_class=%s",
                self._floor, self._fallback, self._per_class,
            )
            return

        # Tier 4: model's default.
        self._floor = default_confidence
        self._per_class = None
        self._fallback = default_confidence
        LOGGER.debug(
            "ConfidenceFilter: tier 4 (model default), floor=%.4f, fallback=%.4f, per_class=%s",
            self._floor, self._fallback, self._per_class,
        )

    @property
    def floor(self) -> float:
        return self._floor

    @property
    def has_per_class_refinement(self) -> bool:
        """True iff `passes` / `build_keep_mask` may return a non-trivial
        result. Concrete models should short-circuit the per-image refine
        call when this is False."""
        return self._per_class is not None

    def passes(self, class_name: str, confidence: float) -> bool:
        """Per-detection refinement check. Returns True for tiers without
        per-class data because the model already filtered at the floor."""
        if not self.has_per_class_refinement:
            return True
        return confidence >= self._per_class.get(class_name, self._fallback)

    def build_keep_mask(
        self,
        class_ids: torch.Tensor,
        confidences: torch.Tensor,
        class_names: List[str],
    ) -> torch.Tensor:
        """Per-detection mask for the OD/IS/KP shape where `class_ids` and
        `confidences` are parallel arrays. Caller is responsible for short-
        circuiting via `has_per_class_refinement` when applicable; this method
        also handles the no-refinement case (returns all-True)."""
        n = len(class_ids)
        if not self.has_per_class_refinement:
            return torch.ones(n, dtype=torch.bool)

        # Vectorized: look up each detection's per-class threshold via
        # class_id indexing, then compare against confidence in one shot.
        # Out-of-range class_ids fall back to `_fallback` via the same
        # pattern per_class_thresholds uses.
        thresholds_per_class = torch.tensor(
            self.per_class_thresholds(class_names),
            dtype=confidences.dtype,
            device=confidences.device,
        )
        # Guard against class_ids outside [0, len(class_names)) — clamp to
        # a valid index and apply fallback for those positions separately.
        class_ids_long = class_ids.long()
        in_range = (class_ids_long >= 0) & (class_ids_long < len(class_names))
        safe_idx = class_ids_long.clamp(0, max(len(class_names) - 1, 0))
        per_detection_thresholds = torch.where(
            in_range,
            thresholds_per_class[safe_idx] if len(class_names) > 0 else torch.full_like(confidences, self._fallback),
            torch.full_like(confidences, self._fallback),
        )
        return confidences >= per_detection_thresholds

    def per_class_thresholds(self, class_names: List[str]) -> List[float]:
        """Per-class thresholds aligned to `class_names`, for shapes that
        index by class_id (e.g. semantic segmentation per-pixel). For tiers
        without per-class data, every entry is the floor."""
        if not self.has_per_class_refinement:
            return [self._floor] * len(class_names)
        return [
            self._per_class.get(name, self._fallback) for name in class_names
        ]

    # ------------------------------------------------------------------
    # Per-image refinement. Concrete models call these inside their
    # existing per-image loop when has_per_class_refinement is True.
    # ------------------------------------------------------------------

    def refine_detections(
        self, detections: Detections, class_names: List[str]
    ) -> Detections:
        keep = self.build_keep_mask(
            detections.class_id, detections.confidence, class_names
        )
        if bool(keep.all()):
            return detections
        bboxes_metadata = detections.bboxes_metadata
        if bboxes_metadata is not None:
            keep_indices = keep.nonzero(as_tuple=True)[0].tolist()
            bboxes_metadata = [bboxes_metadata[i] for i in keep_indices]
        return Detections(
            xyxy=detections.xyxy[keep],
            class_id=detections.class_id[keep],
            confidence=detections.confidence[keep],
            image_metadata=detections.image_metadata,
            bboxes_metadata=bboxes_metadata,
        )

    def refine_instance_detections(
        self, detections: InstanceDetections, class_names: List[str]
    ) -> InstanceDetections:
        keep = self.build_keep_mask(
            detections.class_id, detections.confidence, class_names
        )
        if bool(keep.all()):
            return detections
        bboxes_metadata = detections.bboxes_metadata
        if bboxes_metadata is not None:
            keep_indices = keep.nonzero(as_tuple=True)[0].tolist()
            bboxes_metadata = [bboxes_metadata[i] for i in keep_indices]
        return InstanceDetections(
            xyxy=detections.xyxy[keep],
            class_id=detections.class_id[keep],
            confidence=detections.confidence[keep],
            mask=detections.mask[keep],
            image_metadata=detections.image_metadata,
            bboxes_metadata=bboxes_metadata,
        )

    def refine_keypoints_and_detections(
        self,
        keypoints: KeyPoints,
        detections: Detections,
        class_names: List[str],
    ) -> Tuple[KeyPoints, Detections]:
        keep = self.build_keep_mask(
            detections.class_id, detections.confidence, class_names
        )
        if bool(keep.all()):
            return keypoints, detections
        refined_detections = Detections(
            xyxy=detections.xyxy[keep],
            class_id=detections.class_id[keep],
            confidence=detections.confidence[keep],
            image_metadata=detections.image_metadata,
            bboxes_metadata=detections.bboxes_metadata,
        )
        kp_metadata = keypoints.key_points_metadata
        if kp_metadata is not None:
            keep_indices = keep.nonzero(as_tuple=True)[0].tolist()
            kp_metadata = [kp_metadata[i] for i in keep_indices]
        refined_keypoints = KeyPoints(
            xy=keypoints.xy[keep],
            class_id=keypoints.class_id[keep],
            confidence=keypoints.confidence[keep],
            image_metadata=keypoints.image_metadata,
            key_points_metadata=kp_metadata,
        )
        return refined_keypoints, refined_detections

    def refine_multilabel_prediction(
        self,
        prediction: MultiLabelClassificationPrediction,
        class_names: List[str],
    ) -> MultiLabelClassificationPrediction:
        if prediction.class_ids.numel() == 0:
            return prediction
        class_ids_list = prediction.class_ids.tolist()
        kept_indices = [
            cid
            for cid in class_ids_list
            if self.passes(
                class_names[cid] if 0 <= cid < len(class_names) else str(cid),
                float(prediction.confidence[cid]),
            )
        ]
        if len(kept_indices) == len(class_ids_list):
            return prediction
        return MultiLabelClassificationPrediction(
            class_ids=torch.tensor(
                kept_indices, dtype=prediction.class_ids.dtype
            ),
            confidence=prediction.confidence,
            image_metadata=prediction.image_metadata,
        )

    def refine_segmentation_result(
        self,
        result: SemanticSegmentationResult,
        class_names: List[str],
        background_class_id: int,
    ) -> SemanticSegmentationResult:
        thresholds = self.per_class_thresholds(class_names)
        threshold_tensor = torch.tensor(
            thresholds,
            dtype=result.confidence.dtype,
            device=result.confidence.device,
        )
        per_pixel_thresholds = threshold_tensor[
            result.segmentation_map.long()
        ]
        keep = result.confidence >= per_pixel_thresholds
        if bool(keep.all()):
            return result
        new_segmentation_map = result.segmentation_map.clone()
        new_confidence = result.confidence.clone()
        new_segmentation_map[~keep] = background_class_id
        new_confidence[~keep] = 0.0
        return SemanticSegmentationResult(
            segmentation_map=new_segmentation_map,
            confidence=new_confidence,
            image_metadata=result.image_metadata,
        )
