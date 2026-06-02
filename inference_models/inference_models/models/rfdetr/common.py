from typing import List, Optional, Tuple, Union

import torch
from torchvision.transforms import functional

from inference_models import Detections, InstanceDetections, InstancesRLEMasks
from inference_models.configuration import (
    INFERENCE_MODELS_RFDETR_TRITON_POSTPROC_ENABLED,
)
from inference_models.errors import CorruptedModelPackageError
from inference_models.models.common.roboflow.model_packages import (
    PreProcessingMetadata,
)
from inference_models.models.common.roboflow.post_processing import (
    align_instance_segmentation_results,
    align_instance_segmentation_results_to_rle_masks,
    align_instance_segmentation_results_to_rle_masks_batch,
    rescale_image_detections,
)
from inference_models.models.rfdetr.class_remapping import ClassesReMapping
from inference_models.models.rfdetr.post_processor import select_topk_predictions
from inference_models.models.rfdetr.triton_postprocess import (
    post_process_single_instance_segmentation_result_to_rle_masks_triton,
)
from inference_models.utils.file_system import read_json

_TRITON_POSTPROC_ENABLED = INFERENCE_MODELS_RFDETR_TRITON_POSTPROC_ENABLED


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
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        ) from error


def _pre_processing_metadata_key(image_meta: PreProcessingMetadata) -> tuple:
    denorm_size = image_meta.nonsquare_intermediate_size or image_meta.inference_size
    static_crop_offset = image_meta.static_crop_offset
    return (
        image_meta.pad_left,
        image_meta.pad_top,
        image_meta.pad_right,
        image_meta.pad_bottom,
        image_meta.scale_width,
        image_meta.scale_height,
        image_meta.original_size.height,
        image_meta.original_size.width,
        image_meta.size_after_pre_processing.height,
        image_meta.size_after_pre_processing.width,
        denorm_size.height,
        denorm_size.width,
        static_crop_offset.offset_x,
        static_crop_offset.offset_y,
        static_crop_offset.crop_width,
        static_crop_offset.crop_height,
    )


def post_process_object_detection_results(
    bboxes: torch.Tensor,
    logits: torch.Tensor,
    pre_processing_meta: List[PreProcessingMetadata],
    threshold: Union[float, torch.Tensor],
    num_classes: int,
    classes_re_mapping: Optional[ClassesReMapping],
    device: torch.device,
) -> List[Detections]:
    logits_sigmoid = torch.nn.functional.sigmoid(logits)
    if isinstance(threshold, torch.Tensor):
        threshold = threshold.to(device=device, dtype=logits_sigmoid.dtype)
    results = []
    for image_bboxes, image_logits, image_meta in zip(
        bboxes, logits_sigmoid, pre_processing_meta
    ):
        predicted_confidence, top_classes, image_bboxes, _ = select_topk_predictions(
            logits_sigmoid=image_logits,
            bboxes_cxcywh=image_bboxes,
        )
        if classes_re_mapping is not None:
            remapping_mask = torch.isin(
                top_classes, classes_re_mapping.remaining_class_ids
            )
            top_classes = classes_re_mapping.class_mapping[top_classes[remapping_mask]]
            predicted_confidence = predicted_confidence[remapping_mask]
            image_bboxes = image_bboxes[remapping_mask]
        else:
            # drop DETR no-object rows
            named = top_classes < num_classes
            predicted_confidence = predicted_confidence[named]
            top_classes = top_classes[named]
            image_bboxes = image_bboxes[named]
        confidence_mask = predicted_confidence > (
            threshold[top_classes.long()]
            if isinstance(threshold, torch.Tensor)
            else threshold
        )
        predicted_confidence = predicted_confidence[confidence_mask]
        top_classes = top_classes[confidence_mask]
        selected_boxes = image_bboxes[confidence_mask]
        predicted_confidence, sorted_indices = torch.sort(
            predicted_confidence, descending=True
        )
        top_classes = top_classes[sorted_indices]
        selected_boxes = selected_boxes[sorted_indices]
        cxcy = selected_boxes[:, :2]
        wh = selected_boxes[:, 2:]
        xy_min = cxcy - 0.5 * wh
        xy_max = cxcy + 0.5 * wh
        selected_boxes_xyxy_pct = torch.cat([xy_min, xy_max], dim=-1)
        denorm_size = (
            image_meta.nonsquare_intermediate_size or image_meta.inference_size
        )
        inference_size_whwh = torch.tensor(
            [
                denorm_size.width,
                denorm_size.height,
                denorm_size.width,
                denorm_size.height,
            ],
            device=device,
        )
        selected_boxes_xyxy = selected_boxes_xyxy_pct * inference_size_whwh
        selected_boxes_xyxy = rescale_image_detections(
            image_detections=selected_boxes_xyxy,
            image_metadata=image_meta,
        )
        results.append(
            Detections(
                xyxy=selected_boxes_xyxy.round().int(),
                confidence=predicted_confidence,
                class_id=top_classes.int(),
            )
        )
    return results


def post_process_instance_segmentation_results(
    bboxes: torch.Tensor,
    logits: torch.Tensor,
    masks: torch.Tensor,
    pre_processing_meta: List[PreProcessingMetadata],
    threshold: Union[float, torch.Tensor],
    num_classes: int,
    classes_re_mapping: Optional[ClassesReMapping],
) -> List[InstanceDetections]:
    logits_sigmoid = torch.nn.functional.sigmoid(logits)
    results = []
    device = bboxes.device
    if isinstance(threshold, torch.Tensor):
        threshold = threshold.to(device=device, dtype=logits_sigmoid.dtype)
    for image_bboxes, image_logits, image_masks, image_meta in zip(
        bboxes, logits_sigmoid, masks, pre_processing_meta
    ):
        confidence, top_classes, image_bboxes, query_indices = select_topk_predictions(
            logits_sigmoid=image_logits,
            bboxes_cxcywh=image_bboxes,
        )
        image_masks = image_masks[query_indices]
        if classes_re_mapping is not None:
            remapping_mask = torch.isin(
                top_classes, classes_re_mapping.remaining_class_ids
            )
            top_classes = classes_re_mapping.class_mapping[top_classes[remapping_mask]]
            confidence = confidence[remapping_mask]
            image_bboxes = image_bboxes[remapping_mask]
            image_masks = image_masks[remapping_mask]
        else:
            # drop DETR no-object rows
            named = top_classes < num_classes
            confidence = confidence[named]
            top_classes = top_classes[named]
            image_bboxes = image_bboxes[named]
            image_masks = image_masks[named]
        confidence_mask = confidence > (
            threshold[top_classes.long()]
            if isinstance(threshold, torch.Tensor)
            else threshold
        )
        confidence = confidence[confidence_mask]
        top_classes = top_classes[confidence_mask]
        selected_boxes = image_bboxes[confidence_mask]
        selected_masks = image_masks[confidence_mask]
        confidence, sorted_indices = torch.sort(confidence, descending=True)
        top_classes = top_classes[sorted_indices]
        selected_boxes = selected_boxes[sorted_indices]
        selected_masks = selected_masks[sorted_indices]
        cxcy = selected_boxes[:, :2]
        wh = selected_boxes[:, 2:]
        xy_min = cxcy - 0.5 * wh
        xy_max = cxcy + 0.5 * wh
        selected_boxes_xyxy_pct = torch.cat([xy_min, xy_max], dim=-1)
        denorm_size = (
            image_meta.nonsquare_intermediate_size or image_meta.inference_size
        )
        denorm_size_whwh = torch.tensor(
            [
                denorm_size.width,
                denorm_size.height,
                denorm_size.width,
                denorm_size.height,
            ],
            device=device,
        )
        padding = (
            image_meta.pad_left,
            image_meta.pad_top,
            image_meta.pad_right,
            image_meta.pad_bottom,
        )
        selected_boxes_xyxy = selected_boxes_xyxy_pct * denorm_size_whwh
        aligned_boxes, aligned_masks = align_instance_segmentation_results(
            image_bboxes=selected_boxes_xyxy,
            masks=selected_masks,
            padding=padding,
            scale_height=image_meta.scale_height,
            scale_width=image_meta.scale_width,
            original_size=image_meta.original_size,
            size_after_pre_processing=image_meta.size_after_pre_processing,
            inference_size=denorm_size,
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


def _post_process_single_instance_segmentation_result_to_rle_masks(
    image_bboxes: torch.Tensor,
    image_logits: torch.Tensor,
    image_masks: torch.Tensor,
    image_meta: PreProcessingMetadata,
    threshold: Union[float, torch.Tensor],
    num_classes: int,
    classes_re_mapping: Optional[ClassesReMapping],
) -> InstanceDetections:
    if not _TRITON_POSTPROC_ENABLED:
        return _post_process_single_instance_segmentation_result_to_rle_masks_classic(
            image_bboxes=image_bboxes,
            image_logits=image_logits,
            image_masks=image_masks,
            image_meta=image_meta,
            threshold=threshold,
            num_classes=num_classes,
            classes_re_mapping=classes_re_mapping,
        )

    triton_result = (
        post_process_single_instance_segmentation_result_to_rle_masks_triton(
            image_bboxes=image_bboxes,
            image_scores=image_logits,
            image_masks=image_masks,
            image_meta=image_meta,
            threshold=threshold,
            classes_re_mapping=classes_re_mapping,
        )
    )
    if triton_result is not None:
        return triton_result

    return _post_process_single_instance_segmentation_result_to_rle_masks_classic(
        image_bboxes=image_bboxes,
        image_logits=image_logits,
        image_masks=image_masks,
        image_meta=image_meta,
        threshold=threshold,
        num_classes=num_classes,
        classes_re_mapping=classes_re_mapping,
    )


def _post_process_single_instance_segmentation_result_to_rle_masks_classic(
    image_bboxes: torch.Tensor,
    image_logits: torch.Tensor,
    image_masks: torch.Tensor,
    image_meta: PreProcessingMetadata,
    threshold: Union[float, torch.Tensor],
    num_classes: int,
    classes_re_mapping: Optional[ClassesReMapping],
) -> InstanceDetections:
    num_queries, num_logits_classes = image_logits.shape
    flat_scores = image_logits.reshape(-1)
    confidence, topk_indexes = torch.topk(flat_scores, num_queries)
    query_indices = topk_indexes // num_logits_classes
    top_classes = topk_indexes % num_logits_classes
    if classes_re_mapping is not None:
        if classes_re_mapping.class_mapping.shape[0] >= num_logits_classes:
            top_classes = classes_re_mapping.class_mapping[top_classes]
        else:
            mapped_classes = torch.full_like(top_classes, -1)
            mappable_classes = top_classes < classes_re_mapping.class_mapping.shape[0]
            mapped_classes[mappable_classes] = classes_re_mapping.class_mapping[
                top_classes[mappable_classes]
            ]
            top_classes = mapped_classes
        remapping_mask = top_classes >= 0
    else:
        named = top_classes < num_classes
        remapping_mask = named
    confidence_mask = confidence > (
        threshold[top_classes.clamp(min=0, max=threshold.shape[0] - 1).long()]
        if isinstance(threshold, torch.Tensor)
        else threshold
    )
    keep_mask = remapping_mask & confidence_mask
    confidence = confidence[keep_mask]
    top_classes = top_classes[keep_mask]
    query_indices = query_indices[keep_mask]
    confidence, sorted_indices = torch.sort(confidence, descending=True)
    top_classes = top_classes[sorted_indices]
    query_indices = query_indices[sorted_indices]
    selected_boxes = image_bboxes[query_indices]
    selected_masks = image_masks[query_indices]
    cxcy = selected_boxes[:, :2]
    wh = selected_boxes[:, 2:]
    xy_min = cxcy - 0.5 * wh
    xy_max = cxcy + 0.5 * wh
    selected_boxes_xyxy_pct = torch.cat([xy_min, xy_max], dim=-1)
    denorm_size = image_meta.nonsquare_intermediate_size or image_meta.inference_size
    denorm_size_whwh = torch.tensor(
        [
            denorm_size.width,
            denorm_size.height,
            denorm_size.width,
            denorm_size.height,
        ],
        device=image_bboxes.device,
    )
    padding = (
        image_meta.pad_left,
        image_meta.pad_top,
        image_meta.pad_right,
        image_meta.pad_bottom,
    )
    selected_boxes_xyxy = selected_boxes_xyxy_pct * denorm_size_whwh
    aligned_boxes, rle_masks = [], []
    for bbox, mask in align_instance_segmentation_results_to_rle_masks(
        image_bboxes=selected_boxes_xyxy,
        masks=selected_masks,
        padding=padding,
        scale_height=image_meta.scale_height,
        scale_width=image_meta.scale_width,
        original_size=image_meta.original_size,
        size_after_pre_processing=image_meta.size_after_pre_processing,
        inference_size=denorm_size,
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
    else:
        aligned_boxes_tensor = torch.empty(
            (0, 4), dtype=torch.int32, device=image_bboxes.device
        )
    return InstanceDetections(
        xyxy=aligned_boxes_tensor.round().int(),
        confidence=confidence,
        class_id=top_classes.int(),
        mask=instances_masks,
    )


def _post_process_instance_segmentation_results_to_rle_masks_batched_dense(
    bboxes: torch.Tensor,
    logits_sigmoid: torch.Tensor,
    masks: torch.Tensor,
    pre_processing_meta: List[PreProcessingMetadata],
    threshold: Union[float, torch.Tensor],
    num_classes: int,
    classes_re_mapping: Optional[ClassesReMapping],
) -> List[InstanceDetections]:
    batch_size, num_queries, num_logits_classes = logits_sigmoid.shape
    final_results: List[Optional[InstanceDetections]] = [None] * batch_size
    device = bboxes.device

    flat_scores = logits_sigmoid.reshape(batch_size, -1)
    confidence, topk_indexes = torch.topk(flat_scores, num_queries, dim=1)
    query_indices = topk_indexes // num_logits_classes
    top_classes = topk_indexes % num_logits_classes
    image_bboxes = torch.gather(
        bboxes,
        dim=1,
        index=query_indices[:, :, None].expand(-1, -1, bboxes.shape[-1]),
    )
    image_masks = torch.gather(
        masks,
        dim=1,
        index=query_indices[:, :, None, None].expand(
            -1,
            -1,
            masks.shape[-2],
            masks.shape[-1],
        ),
    )

    if classes_re_mapping is not None:
        mapped_classes = torch.full_like(top_classes, -1)
        mappable_classes = top_classes < classes_re_mapping.class_mapping.shape[0]
        mapped_classes[mappable_classes] = classes_re_mapping.class_mapping[
            top_classes[mappable_classes]
        ]
        class_mask = mapped_classes >= 0
        top_classes = mapped_classes
    else:
        # drop DETR no-object rows
        class_mask = top_classes < num_classes

    if isinstance(threshold, torch.Tensor):
        threshold_indices = top_classes.clamp(
            min=0,
            max=threshold.shape[0] - 1,
        ).long()
        threshold_values = threshold[threshold_indices]
        confidence_mask = class_mask & (confidence > threshold_values)
    else:
        confidence_mask = class_mask & (confidence > threshold)
    valid_counts = confidence_mask.sum(dim=1)
    valid_sorted = torch.zeros_like(confidence_mask)
    sorted_confidence = torch.zeros_like(confidence)
    sorted_classes = torch.zeros_like(top_classes)
    sorted_boxes = torch.zeros_like(image_bboxes)
    sorted_masks = torch.empty_like(image_masks)
    for valid_count in torch.unique(valid_counts).tolist():
        if valid_count == 0:
            continue
        image_indices = (valid_counts == valid_count).nonzero(as_tuple=True)[0]
        group_mask = confidence_mask[image_indices]
        group_size = image_indices.shape[0]
        group_confidence = confidence[image_indices][group_mask].reshape(
            group_size,
            valid_count,
        )
        group_classes = top_classes[image_indices][group_mask].reshape(
            group_size,
            valid_count,
        )
        group_boxes = image_bboxes[image_indices][group_mask].reshape(
            group_size,
            valid_count,
            image_bboxes.shape[-1],
        )
        group_masks = image_masks[image_indices][group_mask].reshape(
            group_size,
            valid_count,
            image_masks.shape[-2],
            image_masks.shape[-1],
        )
        group_confidence, sorted_indices = torch.sort(
            group_confidence,
            dim=1,
            descending=True,
        )
        sorted_confidence[image_indices, :valid_count] = group_confidence
        sorted_classes[image_indices, :valid_count] = torch.gather(
            group_classes,
            dim=1,
            index=sorted_indices,
        )
        sorted_boxes[image_indices, :valid_count] = torch.gather(
            group_boxes,
            dim=1,
            index=sorted_indices[:, :, None].expand(-1, -1, group_boxes.shape[-1]),
        )
        sorted_masks[image_indices, :valid_count] = torch.gather(
            group_masks,
            dim=1,
            index=sorted_indices[:, :, None, None].expand(
                -1,
                -1,
                group_masks.shape[-2],
                group_masks.shape[-1],
            ),
        )
        valid_sorted[image_indices, :valid_count] = True
    confidence = sorted_confidence
    top_classes = sorted_classes
    selected_boxes = sorted_boxes
    selected_masks = sorted_masks

    cxcy = selected_boxes[..., :2]
    wh = selected_boxes[..., 2:]
    xy_min = cxcy - 0.5 * wh
    xy_max = cxcy + 0.5 * wh
    selected_boxes_xyxy_pct = torch.cat([xy_min, xy_max], dim=-1)
    denorm_sizes = [
        image_meta.nonsquare_intermediate_size or image_meta.inference_size
        for image_meta in pre_processing_meta
    ]
    denorm_size_whwh = torch.tensor(
        [
            [
                denorm_size.width,
                denorm_size.height,
                denorm_size.width,
                denorm_size.height,
            ]
            for denorm_size in denorm_sizes
        ],
        device=device,
    )
    selected_boxes_xyxy = selected_boxes_xyxy_pct * denorm_size_whwh[:, None, :]

    metadata_groups = {}
    for image_index, image_meta in enumerate(pre_processing_meta):
        metadata_groups.setdefault(
            _pre_processing_metadata_key(image_meta),
            [],
        ).append(image_index)

    for image_indices in metadata_groups.values():
        image_meta = pre_processing_meta[image_indices[0]]
        denorm_size = (
            image_meta.nonsquare_intermediate_size or image_meta.inference_size
        )
        padding = (
            image_meta.pad_left,
            image_meta.pad_top,
            image_meta.pad_right,
            image_meta.pad_bottom,
        )
        group_valid = valid_sorted[image_indices]
        group_counts = group_valid.sum(dim=1).tolist()
        group_boxes = selected_boxes_xyxy[image_indices][group_valid]
        group_masks = selected_masks[image_indices][group_valid]
        group_confidence = confidence[image_indices][group_valid]
        group_classes = top_classes[image_indices][group_valid]
        aligned_boxes_tensor, rle_masks = (
            align_instance_segmentation_results_to_rle_masks_batch(
                image_bboxes=group_boxes,
                masks=group_masks,
                padding=padding,
                scale_height=image_meta.scale_height,
                scale_width=image_meta.scale_width,
                original_size=image_meta.original_size,
                size_after_pre_processing=image_meta.size_after_pre_processing,
                inference_size=denorm_size,
                static_crop_offset=image_meta.static_crop_offset,
            )
        )
        offset = 0
        for image_index, count in zip(image_indices, group_counts):
            next_offset = offset + count
            instances_masks = InstancesRLEMasks.from_coco_rle_masks(
                image_size=(
                    image_meta.original_size.height,
                    image_meta.original_size.width,
                ),
                masks=rle_masks[offset:next_offset],
            )
            final_results[image_index] = InstanceDetections(
                xyxy=aligned_boxes_tensor[offset:next_offset].round().int(),
                confidence=group_confidence[offset:next_offset],
                class_id=group_classes[offset:next_offset].int(),
                mask=instances_masks,
            )
            offset = next_offset
    return final_results


def _post_process_instance_segmentation_results_to_rle_masks_classic(
    bboxes: torch.Tensor,
    logits: torch.Tensor,
    masks: torch.Tensor,
    pre_processing_meta: List[PreProcessingMetadata],
    threshold: Union[float, torch.Tensor],
    num_classes: int,
    classes_re_mapping: Optional[ClassesReMapping],
) -> List[InstanceDetections]:
    logits_sigmoid = torch.nn.functional.sigmoid(logits)
    device = bboxes.device
    if isinstance(threshold, torch.Tensor):
        threshold = threshold.to(device=device, dtype=logits_sigmoid.dtype)
    return [
        _post_process_single_instance_segmentation_result_to_rle_masks_classic(
            image_bboxes=image_bboxes,
            image_logits=image_logits,
            image_masks=image_masks,
            image_meta=image_meta,
            threshold=threshold,
            num_classes=num_classes,
            classes_re_mapping=classes_re_mapping,
        )
        for image_bboxes, image_logits, image_masks, image_meta in zip(
            bboxes,
            logits_sigmoid,
            masks,
            pre_processing_meta,
        )
    ]


def post_process_instance_segmentation_results_to_rle_masks(
    bboxes: torch.Tensor,
    logits: torch.Tensor,
    masks: torch.Tensor,
    pre_processing_meta: List[PreProcessingMetadata],
    threshold: Union[float, torch.Tensor],
    num_classes: int,
    classes_re_mapping: Optional[ClassesReMapping],
) -> List[InstanceDetections]:
    if not _TRITON_POSTPROC_ENABLED:
        return _post_process_instance_segmentation_results_to_rle_masks_classic(
            bboxes=bboxes,
            logits=logits,
            masks=masks,
            pre_processing_meta=pre_processing_meta,
            threshold=threshold,
            num_classes=num_classes,
            classes_re_mapping=classes_re_mapping,
        )

    logits_sigmoid = torch.nn.functional.sigmoid(logits)
    batch_size = logits_sigmoid.shape[0]
    device = bboxes.device
    if isinstance(threshold, torch.Tensor):
        threshold = threshold.to(device=device, dtype=logits_sigmoid.dtype)
    if batch_size == 1:
        return [
            _post_process_single_instance_segmentation_result_to_rle_masks(
                image_bboxes=bboxes[0],
                image_logits=logits_sigmoid[0],
                image_masks=masks[0],
                image_meta=pre_processing_meta[0],
                threshold=threshold,
                num_classes=num_classes,
                classes_re_mapping=classes_re_mapping,
            )
        ]

    return _post_process_instance_segmentation_results_to_rle_masks_batched_dense(
        bboxes=bboxes,
        logits_sigmoid=logits_sigmoid,
        masks=masks,
        pre_processing_meta=pre_processing_meta,
        threshold=threshold,
        num_classes=num_classes,
        classes_re_mapping=classes_re_mapping,
    )
