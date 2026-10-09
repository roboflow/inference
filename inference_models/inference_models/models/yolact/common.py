from typing import List

import torch

from inference_models import InstanceDetections, InstancesRLEMasks
from inference_models.models.common.roboflow.model_packages import PreProcessingMetadata
from inference_models.models.common.roboflow.post_processing import (
    align_instance_segmentation_results,
    align_instance_segmentation_results_to_rle_masks,
    crop_masks_to_boxes,
    finalize_instance_segmentation_boxes,
    resolve_mask_frame_size,
)


def prepare_dense_masks(
    nms_results: List[torch.Tensor],
    all_proto_data: torch.Tensor,
    pre_processing_meta: List[PreProcessingMetadata],
    masks_resolution_factor: float = 1.0,
) -> List[InstanceDetections]:
    final_results = []
    for image_bboxes, image_protos, image_meta in zip(
        nms_results, all_proto_data, pre_processing_meta
    ):
        pre_processed_masks = image_protos @ image_bboxes[:, 6:].T
        pre_processed_masks = 1 / (1 + torch.exp(-pre_processed_masks))
        pre_processed_masks = torch.permute(pre_processed_masks, (2, 0, 1))
        cropped_masks = crop_masks_to_boxes(image_bboxes[:, :4], pre_processed_masks)
        padding = (
            image_meta.pad_left,
            image_meta.pad_top,
            image_meta.pad_right,
            image_meta.pad_bottom,
        )
        aligned_boxes, aligned_masks = align_instance_segmentation_results(
            image_bboxes=image_bboxes,
            masks=cropped_masks,
            padding=padding,
            scale_height=image_meta.scale_height,
            scale_width=image_meta.scale_width,
            original_size=image_meta.original_size,
            size_after_pre_processing=image_meta.size_after_pre_processing,
            inference_size=image_meta.inference_size,
            static_crop_offset=image_meta.static_crop_offset,
            binarization_threshold=0.5,
            masks_resolution_factor=masks_resolution_factor,
        )
        final_results.append(
            InstanceDetections(
                xyxy=finalize_instance_segmentation_boxes(
                    aligned_boxes[:, :4],
                    mask_size=tuple(aligned_masks.shape[1:]),
                    image_size=image_meta.original_size,
                ),
                class_id=aligned_boxes[:, 5].int(),
                confidence=aligned_boxes[:, 4],
                mask=aligned_masks,
                image_size=tuple(image_meta.original_size),
                mask_frame_size=tuple(resolve_mask_frame_size(image_meta)),
            )
        )
    return final_results


def prepare_rle_masks(
    nms_results: List[torch.Tensor],
    all_proto_data: torch.Tensor,
    pre_processing_meta: List[PreProcessingMetadata],
    masks_resolution_factor: float = 1.0,
) -> List[InstanceDetections]:
    final_results = []
    for image_bboxes, image_protos, image_meta in zip(
        nms_results, all_proto_data, pre_processing_meta
    ):
        pre_processed_masks = image_protos @ image_bboxes[:, 6:].T
        pre_processed_masks = 1 / (1 + torch.exp(-pre_processed_masks))
        pre_processed_masks = torch.permute(pre_processed_masks, (2, 0, 1))
        cropped_masks = crop_masks_to_boxes(image_bboxes[:, :4], pre_processed_masks)
        padding = (
            image_meta.pad_left,
            image_meta.pad_top,
            image_meta.pad_right,
            image_meta.pad_bottom,
        )
        aligned_boxes, rle_masks = [], []
        for bbox, mask in align_instance_segmentation_results_to_rle_masks(
            image_bboxes=image_bboxes,
            masks=cropped_masks,
            padding=padding,
            scale_height=image_meta.scale_height,
            scale_width=image_meta.scale_width,
            original_size=image_meta.original_size,
            size_after_pre_processing=image_meta.size_after_pre_processing,
            inference_size=image_meta.inference_size,
            static_crop_offset=image_meta.static_crop_offset,
            binarization_threshold=0.5,
            masks_resolution_factor=masks_resolution_factor,
        ):
            aligned_boxes.append(bbox)
            rle_masks.append(mask)
        instances_masks = InstancesRLEMasks.from_coco_rle_masks(
            image_size=(
                image_meta.original_size.height,
                image_meta.original_size.width,
            ),
            masks=rle_masks,
            mask_size=tuple(rle_masks[0]["size"]) if rle_masks else None,
        )
        if len(aligned_boxes) > 0:
            aligned_boxes_tensor = torch.stack(aligned_boxes, dim=0)
            final_results.append(
                InstanceDetections(
                    xyxy=finalize_instance_segmentation_boxes(
                        aligned_boxes_tensor[:, :4],
                        mask_size=instances_masks.mask_size,
                        image_size=image_meta.original_size,
                    ),
                    class_id=aligned_boxes_tensor[:, 5].int(),
                    confidence=aligned_boxes_tensor[:, 4],
                    mask=instances_masks,
                    image_size=tuple(image_meta.original_size),
                    mask_frame_size=tuple(resolve_mask_frame_size(image_meta)),
                )
            )
        else:
            final_results.append(
                InstanceDetections(
                    xyxy=torch.empty(
                        (0, 4), dtype=torch.int32, device=image_bboxes.device
                    ),
                    class_id=torch.empty(
                        (0,), dtype=torch.int32, device=image_bboxes.device
                    ),
                    confidence=torch.empty((0,), device=image_bboxes.device),
                    mask=instances_masks,
                    image_size=tuple(image_meta.original_size),
                    mask_frame_size=tuple(resolve_mask_frame_size(image_meta)),
                )
            )
    return final_results
