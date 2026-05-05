from typing import List

import torch

from inference_models import InstanceDetections, InstancesRLEMasks
from inference_models.models.common.roboflow.model_packages import PreProcessingMetadata
from inference_models.models.common.roboflow.post_processing import (
    align_instance_segmentation_results,
    align_instance_segmentation_results_to_rle_masks,
    crop_masks_to_boxes,
    preprocess_segmentation_masks,
)


def prepare_dense_masks(
    nms_results: List[torch.Tensor],
    protos: torch.Tensor,
    pre_processing_meta: List[PreProcessingMetadata],
) -> List[InstanceDetections]:
    final_results = []
    for image_bboxes, image_protos, image_meta in zip(
        nms_results, protos, pre_processing_meta
    ):
        pre_processed_masks = preprocess_segmentation_masks(
            protos=image_protos,
            masks_in=image_bboxes[:, 6:],
        )
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
        )
        final_results.append(
            InstanceDetections(
                xyxy=aligned_boxes[:, :4].round().int(),
                class_id=aligned_boxes[:, 5].int(),
                confidence=aligned_boxes[:, 4],
                mask=aligned_masks,
            )
        )
    return final_results


def prepare_rle_masks(
    nms_results: List[torch.Tensor],
    protos: torch.Tensor,
    pre_processing_meta: List[PreProcessingMetadata],
) -> List[InstanceDetections]:
    final_results = []
    for image_bboxes, image_protos, image_meta in zip(
        nms_results, protos, pre_processing_meta
    ):
        pre_processed_masks = preprocess_segmentation_masks(
            protos=image_protos,
            masks_in=image_bboxes[:, 6:],
        )
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
        ):
            aligned_boxes.append(bbox)
            rle_masks.append(mask)
        instances_masks = InstancesRLEMasks.from_coco_rle_masks(
            image_size=(
                image_meta.original_size.height,
                image_meta.original_size.width,
            ),
            masks=rle_masks,
        )
        if len(aligned_boxes) > 0:
            aligned_boxes_tensor = torch.stack(aligned_boxes, dim=0)
            final_results.append(
                InstanceDetections(
                    xyxy=aligned_boxes_tensor[:, :4].round().int(),
                    class_id=aligned_boxes_tensor[:, 5].int(),
                    confidence=aligned_boxes_tensor[:, 4],
                    mask=instances_masks,
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
                )
            )
    return final_results
