import os
from typing import List, Optional, Tuple, Union

import torch
from torchvision.transforms import functional

from inference_models import Detections, InstanceDetections, InstancesRLEMasks
from inference_models.entities import ImageDimensions
from inference_models.errors import CorruptedModelPackageError
from inference_models.models.common.roboflow.model_packages import (
    PreProcessingMetadata,
    StaticCropOffset,
)
from inference_models.models.common.rle_utils import torch_mask_to_coco_rle
from inference_models.models.common.roboflow.post_processing import (
    align_instance_segmentation_results,
    align_instance_segmentation_results_to_rle_masks,
    rescale_image_detections,
)
from inference_models.models.rfdetr.class_remapping import ClassesReMapping
from inference_models.models.rfdetr.post_processor import select_topk_predictions
from inference_models.utils.file_system import read_json

_RFDETR_TRITON_FULLPOSTPROC = os.getenv("RFDETR_TRITON_FULLPOSTPROC", "false").lower() in (
    "true",
    "1",
)
if _RFDETR_TRITON_FULLPOSTPROC:
    try:
        from inference_models.models.rfdetr.triton_fullpostproc import (
            TRITON_AVAILABLE as _TRITON_FULLPOST_AVAILABLE,
            triton_rfdetr_fullpost,
        )
        _TRITON_FULLPOST_READY = _TRITON_FULLPOST_AVAILABLE and torch.cuda.is_available()
    except Exception:
        _TRITON_FULLPOST_READY = False
        triton_rfdetr_fullpost = None
else:
    _TRITON_FULLPOST_READY = False
    triton_rfdetr_fullpost = None


def _fullpost_upsample_masks(
    masks: torch.Tensor,
    survivor_q: torch.Tensor,
    inference_size_wh: Tuple[int, int],
    pad_ltrb: Tuple[int, int, int, int],
    orig_size_hw: Tuple[int, int],
) -> torch.Tensor:
    """Replicate ``align_instance_segmentation_results``'s mask path
    bit-for-bit (when ``size_after_pre_processing == inference_size`` and no
    static crop applies — both guaranteed by ``_fullpost_eligible``).

    Gathers surviving query rows from ``masks`` (shape ``(1, Q, mh, mw)``),
    crops the letterbox padding in mask coordinates, bilinear-resizes to
    ``(orig_h, orig_w)`` with ``antialias=True`` (matching torchvision's
    ``functional.resize`` default), and thresholds at 0.
    """
    selected = masks[0].index_select(0, survivor_q.long())
    if selected.shape[0] == 0:
        orig_h, orig_w = orig_size_hw
        return torch.empty(
            (0, orig_h, orig_w), dtype=torch.bool, device=masks.device
        )
    _, mh, mw = selected.shape
    inf_w, inf_h = inference_size_wh
    pad_l, pad_t, pad_r, pad_b = pad_ltrb
    mh_scale = mh / inf_h
    mw_scale = mw / inf_w
    mpt = round(mh_scale * pad_t)
    mpb = round(mh_scale * pad_b)
    mpl = round(mw_scale * pad_l)
    mpr = round(mw_scale * pad_r)
    selected = selected[:, mpt: mh - mpb, mpl: mw - mpr]
    orig_h, orig_w = orig_size_hw
    upsampled = torch.nn.functional.interpolate(
        selected.unsqueeze(1),
        size=(orig_h, orig_w),
        mode="bilinear",
        antialias=True,
        align_corners=False,
    ).squeeze(1)
    return upsampled > 0.0


def _fullpost_eligible(
    bboxes: torch.Tensor,
    pre_processing_meta: List[PreProcessingMetadata],
    classes_re_mapping: Optional[ClassesReMapping],
) -> bool:
    if not _TRITON_FULLPOST_READY or not bboxes.is_cuda:
        return False
    if bboxes.shape[0] != 1 or len(pre_processing_meta) != 1:
        return False
    meta = pre_processing_meta[0]
    if meta.nonsquare_intermediate_size is not None:
        return False
    if meta.static_crop_offset.offset_x != 0 or meta.static_crop_offset.offset_y != 0:
        return False
    if classes_re_mapping is None:
        return False
    return True


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
    if _fullpost_eligible(bboxes, pre_processing_meta, classes_re_mapping):
        meta = pre_processing_meta[0]
        thr_arg = threshold if isinstance(threshold, torch.Tensor) else float(threshold)
        combined, survivor_idx, counter, done_event = triton_rfdetr_fullpost(
            bboxes=bboxes,
            logits=logits,
            threshold=thr_arg,
            num_classes=num_classes,
            class_mapping=classes_re_mapping.class_mapping,
            inference_size_wh=(meta.inference_size.width, meta.inference_size.height),
            pad_ltrb=(meta.pad_left, meta.pad_top, meta.pad_right, meta.pad_bottom),
            scale_wh=(meta.scale_width, meta.scale_height),
            orig_size_wh=(meta.original_size.width, meta.original_size.height),
        )
        done_event.wait(torch.cuda.current_stream(bboxes.device))
        # Counter is incremented unconditionally before the slot-cap guard, so
        # cap by combined's row count (num_queries).
        n_survivors = min(int(counter.item()), combined.shape[0])
        combined_slice = combined[:n_survivors]
        mask_bin = _fullpost_upsample_masks(
            masks=masks,
            survivor_q=survivor_idx[:n_survivors],
            inference_size_wh=(meta.inference_size.width, meta.inference_size.height),
            pad_ltrb=(meta.pad_left, meta.pad_top, meta.pad_right, meta.pad_bottom),
            orig_size_hw=(meta.original_size.height, meta.original_size.width),
        )
        detections = InstanceDetections(
            xyxy=combined_slice[:, :4],
            confidence=combined_slice[:, 4].view(torch.float32),
            class_id=combined_slice[:, 5],
            mask=mask_bin,
        )
        detections.__dict__["_combined_gpu"] = combined
        detections.__dict__["_counter_gpu"] = counter
        detections.__dict__["_postproc_done_event"] = done_event
        return [detections]
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


def post_process_instance_segmentation_results_to_rle_masks(
    bboxes: torch.Tensor,
    logits: torch.Tensor,
    masks: torch.Tensor,
    pre_processing_meta: List[PreProcessingMetadata],
    threshold: Union[float, torch.Tensor],
    num_classes: int,
    classes_re_mapping: Optional[ClassesReMapping],
) -> List[InstanceDetections]:
    if _fullpost_eligible(bboxes, pre_processing_meta, classes_re_mapping):
        meta = pre_processing_meta[0]
        thr_arg = threshold if isinstance(threshold, torch.Tensor) else float(threshold)
        combined, survivor_idx, counter, done_event = triton_rfdetr_fullpost(
            bboxes=bboxes,
            logits=logits,
            threshold=thr_arg,
            num_classes=num_classes,
            class_mapping=classes_re_mapping.class_mapping,
            inference_size_wh=(meta.inference_size.width, meta.inference_size.height),
            pad_ltrb=(meta.pad_left, meta.pad_top, meta.pad_right, meta.pad_bottom),
            scale_wh=(meta.scale_width, meta.scale_height),
            orig_size_wh=(meta.original_size.width, meta.original_size.height),
        )
        done_event.wait(torch.cuda.current_stream(bboxes.device))
        # Counter is incremented unconditionally before the slot-cap guard, so
        # cap by combined's row count (num_queries).
        n_survivors = min(int(counter.item()), combined.shape[0])
        orig_h = meta.original_size.height
        orig_w = meta.original_size.width
        if n_survivors == 0:
            empty_xyxy = torch.empty(
                (0, 4), dtype=torch.int32, device=bboxes.device
            )
            empty_conf = torch.empty((0,), dtype=torch.float32, device=bboxes.device)
            empty_cls = torch.empty((0,), dtype=torch.int32, device=bboxes.device)
            return [
                InstanceDetections(
                    xyxy=empty_xyxy,
                    confidence=empty_conf,
                    class_id=empty_cls,
                    mask=InstancesRLEMasks.from_coco_rle_masks(
                        image_size=(orig_h, orig_w), masks=[]
                    ),
                )
            ]
        combined_slice = combined[:n_survivors]
        mask_slice = _fullpost_upsample_masks(
            masks=masks,
            survivor_q=survivor_idx[:n_survivors],
            inference_size_wh=(meta.inference_size.width, meta.inference_size.height),
            pad_ltrb=(meta.pad_left, meta.pad_top, meta.pad_right, meta.pad_bottom),
            orig_size_hw=(orig_h, orig_w),
        )
        rle_masks = [
            torch_mask_to_coco_rle(mask=mask_slice[i]) for i in range(n_survivors)
        ]
        instances_masks = InstancesRLEMasks.from_coco_rle_masks(
            image_size=(orig_h, orig_w), masks=rle_masks
        )
        xyxy = combined_slice[:, :4]
        confidence = combined_slice[:, 4].view(torch.float32)
        class_id = combined_slice[:, 5]
        return [
            InstanceDetections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id,
                mask=instances_masks,
            )
        ]
    logits_sigmoid = torch.nn.functional.sigmoid(logits)
    final_results = []
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
            final_results.append(
                InstanceDetections(
                    xyxy=aligned_boxes_tensor.round().int(),
                    confidence=confidence,
                    class_id=top_classes.int(),
                    mask=instances_masks,
                )
            )
        else:
            final_results.append(
                InstanceDetections(
                    xyxy=torch.empty(
                        (0, 4), dtype=torch.int32, device=image_bboxes.device
                    ),
                    class_id=top_classes.int(),
                    confidence=confidence,
                    mask=instances_masks,
                )
            )
    return final_results
