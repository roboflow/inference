from typing import List, Optional, Tuple

import torch
from inference_exp import InstanceDetections
from inference_exp.entities import ImageDimensions
from inference_exp.errors import CorruptedModelPackageError
from inference_exp.models.common.roboflow.model_packages import (
    PreProcessingMetadata,
    StaticCropOffset,
)
from inference_exp.models.rfdetr.class_remapping import ClassesReMapping
from inference_exp.utils.file_system import read_json
from torchvision.transforms import functional


def parse_model_type(config_path: str) -> str:
    try:
        parsed_config = read_json(path=config_path)
        if not isinstance(parsed_config, dict):
            raise ValueError(
                f"decoded value is {type(parsed_config)}, but dictionary expected"
            )
        if "model_type" not in parsed_config or not isinstance(
            parsed_config["model_type"], str
        ):
            raise ValueError(
                "could not find required entries in config - either "
                "'model_type' field is missing or not a string"
            )
        return parsed_config["model_type"]
    except (IOError, OSError, ValueError) as error:
        raise CorruptedModelPackageError(
            message=f"Model type config file is malformed: "
            f"{error}. In case that the package is "
            f"hosted on the Roboflow platform - contact support. If you created model package manually, please "
            f"verify its consistency in docs.",
            help_url="https://todo",
        ) from error


def post_process_instance_segmentation_results(
    bboxes: torch.Tensor,
    logits: torch.Tensor,
    masks: torch.Tensor,
    pre_processing_meta: List[PreProcessingMetadata],
    threshold: float,
    classes_re_mapping: Optional[ClassesReMapping],
) -> List[InstanceDetections]:
    logits_sigmoid = torch.nn.functional.sigmoid(logits)
    results = []
    device = bboxes.device
    for image_bboxes, image_logits, image_masks, image_meta in zip(
        bboxes, logits_sigmoid, masks, pre_processing_meta
    ):
        confidence, top_classes = image_logits.max(dim=1)
        confidence_mask = confidence > threshold
        confidence = confidence[confidence_mask]
        top_classes = top_classes[confidence_mask]
        selected_boxes = image_bboxes[confidence_mask]
        selected_masks = image_masks[confidence_mask]
        confidence, sorted_indices = torch.sort(confidence, descending=True)
        top_classes = top_classes[sorted_indices]
        selected_boxes = selected_boxes[sorted_indices]
        selected_masks = selected_masks[sorted_indices]
        if classes_re_mapping is not None:
            remapping_mask = torch.isin(
                top_classes, classes_re_mapping.remaining_class_ids
            )
            top_classes = classes_re_mapping.class_mapping[top_classes[remapping_mask]]
            selected_boxes = selected_boxes[remapping_mask]
            confidence = confidence[remapping_mask]
        cxcy = selected_boxes[:, :2]
        wh = selected_boxes[:, 2:]
        xy_min = cxcy - 0.5 * wh
        xy_max = cxcy + 0.5 * wh
        selected_boxes_xyxy_pct = torch.cat([xy_min, xy_max], dim=-1)
        inference_size_hwhw = torch.tensor(
            [
                image_meta.inference_size.height,
                image_meta.inference_size.width,
                image_meta.inference_size.height,
                image_meta.inference_size.width,
            ],
            device=device,
        )
        padding = (
            image_meta.pad_left,
            image_meta.pad_top,
            image_meta.pad_right,
            image_meta.pad_bottom,
        )
        selected_boxes_xyxy = selected_boxes_xyxy_pct * inference_size_hwhw
        aligned_boxes, aligned_masks = align_instance_segmentation_results(
            image_bboxes=selected_boxes_xyxy,
            masks=selected_masks,
            padding=padding,
            scale_height=image_meta.scale_height,
            scale_width=image_meta.scale_width,
            original_size=image_meta.original_size,
            size_after_pre_processing=image_meta.size_after_pre_processing,
            inference_size=image_meta.inference_size,
            static_crop_offset=image_meta.static_crop_offset,
        )
        detections = InstanceDetections(
            xyxy=aligned_boxes.round().int(),
            confidence=confidence,
            class_id=top_classes.int(),
            mask=aligned_masks,
        )
        results.append(detections)
    return results


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
            padded_mask_offset_left : masks.shape[1] - padded_mask_offset_right,
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
