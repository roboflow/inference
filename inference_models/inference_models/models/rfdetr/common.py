from typing import List, Optional, Tuple, Union

import torch
from torchvision.transforms import functional

from inference_models import Detections, InstanceDetections, InstancesRLEMasks, KeyPoints
from inference_models.configuration import (
    INFERENCE_MODELS_RFDETR_TRITON_POSTPROC_ENABLED,
)
from inference_models.entities import ImageDimensions
from inference_models.errors import CorruptedModelPackageError
from inference_models.models.common.roboflow.model_packages import (
    PreProcessingMetadata,
)
from inference_models.models.common.roboflow.post_processing import (
    align_instance_segmentation_results,
    align_instance_segmentation_results_to_rle_masks,
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


def _post_process_single_instance_segmentation_result_to_rle_masks_with_triton(
    image_bboxes: torch.Tensor,
    image_logits: torch.Tensor,
    image_masks: torch.Tensor,
    image_meta: PreProcessingMetadata,
    threshold: Union[float, torch.Tensor],
    num_classes: int,
    classes_re_mapping: Optional[ClassesReMapping],
    defer_postprocess_sync: bool = False,
) -> InstanceDetections:
    triton_result = (
        post_process_single_instance_segmentation_result_to_rle_masks_triton(
            image_bboxes=image_bboxes,
            image_scores=image_logits,
            image_masks=image_masks,
            image_meta=image_meta,
            threshold=threshold,
            classes_re_mapping=classes_re_mapping,
            defer_postprocess_sync=defer_postprocess_sync,
        )
    )
    if triton_result is not None:
        return triton_result

    return _post_process_single_instance_segmentation_result_to_rle_masks(
        image_bboxes=image_bboxes,
        image_logits=image_logits,
        image_masks=image_masks,
        image_meta=image_meta,
        threshold=threshold,
        num_classes=num_classes,
        classes_re_mapping=classes_re_mapping,
    )


def _post_process_single_instance_segmentation_result_to_rle_masks(
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


def post_process_instance_segmentation_results_to_rle_masks(
    bboxes: torch.Tensor,
    logits: torch.Tensor,
    masks: torch.Tensor,
    pre_processing_meta: List[PreProcessingMetadata],
    threshold: Union[float, torch.Tensor],
    num_classes: int,
    classes_re_mapping: Optional[ClassesReMapping],
    defer_postprocess_sync: bool = False,
) -> List[InstanceDetections]:
    logits_sigmoid = torch.nn.functional.sigmoid(logits)
    device = bboxes.device
    if isinstance(threshold, torch.Tensor):
        threshold = threshold.to(device=device, dtype=logits_sigmoid.dtype)
    if _TRITON_POSTPROC_ENABLED:
        return [
            _post_process_single_instance_segmentation_result_to_rle_masks_with_triton(
                image_bboxes=image_bboxes,
                image_logits=image_logits,
                image_masks=image_masks,
                image_meta=image_meta,
                threshold=threshold,
                num_classes=num_classes,
                classes_re_mapping=classes_re_mapping,
                defer_postprocess_sync=defer_postprocess_sync,
            )
            for image_bboxes, image_logits, image_masks, image_meta in zip(
                bboxes,
                logits_sigmoid,
                masks,
                pre_processing_meta,
            )
        ]
    return [
        _post_process_single_instance_segmentation_result_to_rle_masks(
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


def cxcywh_to_xyxy(boxes):
    boxes = boxes.clone()
    boxes[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    boxes[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    boxes[..., 2] = boxes[..., 0] + boxes[..., 2]
    boxes[..., 3] = boxes[..., 1] + boxes[..., 3]
    return boxes


def post_process_keypoint_detection_results(
    bboxes: torch.Tensor,     # [B, N_q, 4] cxcywh, normalized [0, 1]
    out_logits: torch.Tensor,     # [B, N_q, C]
    out_keypoints: torch.Tensor,  # [B, N_q, K_padded, D]
    pre_processing_meta: List[PreProcessingMetadata],
    threshold: Union[float, torch.Tensor],
    key_points_threshold: float,
    num_classes: int,
    classes_re_mapping: Optional[ClassesReMapping],
    key_points_classes_for_instances,
    key_points_slots_in_prediction,
    device: torch.device,
) -> Tuple[List[KeyPoints], Optional[List[Detections]]]:
    # RF-DETR keypoint heads emit one slot per (class, max_keypoint), padded with zeros for
    # classes that have fewer keypoints than the max. The preview model is trained with
    # number of keypoints per class including background [0, 17], so K_padded = 17 * 2 = 34.
    B, N_q, C = out_logits.shape
    K_padded = out_keypoints.shape[2]
    D = out_keypoints.shape[3]
    assert K_padded % C == 0, f"K_padded={K_padded} not divisible by num_classes={C}"
    K_per_class = K_padded // C

    scores = out_logits.sigmoid()
    flat_scores = scores.view(B, -1)
    num_select = flat_scores.shape[1]

    topk_values, topk_indexes = torch.topk(flat_scores, num_select, dim=1)
    scores = topk_values
    topk_boxes = topk_indexes // C  # [B, num_select] query indices
    labels = topk_indexes % C       # [B, num_select] class indices

    bboxes = torch.gather(bboxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
    bboxes = cxcywh_to_xyxy(bboxes)

    # Gather keypoints per top-k query, then select the per-class slot.
    # Keep all D=8 dims: [x, y, findable_logit, visible_logit, log_l11, l21, log_l22, class_logit]
    # The model trains a 2D Gaussian per keypoint with precision matrix P = L L^T;
    # log(sqrt(det P)) = log_l11 + log_l22 — the model's own predicted localization sharpness.
    kp_gather_idx = topk_boxes.unsqueeze(-1).unsqueeze(-1).expand(B, num_select, K_padded, D)
    keypoints_g = torch.gather(out_keypoints, 1, kp_gather_idx)  # [B, num_select, K_padded, D]
    keypoints_g = keypoints_g.view(B, num_select, C, K_per_class, D)

    batch_idx = torch.arange(B, device=labels.device).unsqueeze(-1).expand_as(labels)
    query_idx = torch.arange(num_select, device=labels.device).unsqueeze(0).expand_as(labels)
    keypoints_sel = keypoints_g[batch_idx, query_idx, labels]  # [B, num_select, K_per_class, D=8]

    keypoints_xy = keypoints_sel[..., :2]
    keypoints_conf = keypoints_sel[..., 2:3].sigmoid()  # findable [0,1] per kp

    # Score fusion: object confidence × inverse mean expected squared error of findable kps.
    #   score = cs · (Σ_k w_k · trace(Σ_k) / Σ_k w_k)^(-α)
    # where Σ_k = (L_k L_k^T)^{-1} (model's predicted per-keypoint covariance) and
    # trace(Σ_k) = 1/l11² + 1/l22² + l21²/(l11·l22)² = E[(x-μ_x)² + (y-μ_y)²].
    # Note: σ_k (COCO bandwidths) NOT used — model's L already encodes per-kp difficulty
    # implicitly. Sigma-free form transfers to any keypoint domain. α=0.20 seems to work well.
    log_l11 = keypoints_sel[..., 4]
    l21     = keypoints_sel[..., 5]
    log_l22 = keypoints_sel[..., 6]
    # log(trace) per kp via logsumexp over the three log-terms (numerical stability)
    log_t1 = -2.0 * log_l11                                                   # log(1/l11²)
    log_t2 = -2.0 * log_l22                                                   # log(1/l22²)
    log_t3 = 2.0 * torch.log(l21.abs().clamp(min=1e-12)) + log_t1 + log_t2    # log(l21²/(l11·l22)²)
    log_trace = torch.logsumexp(torch.stack([log_t1, log_t2, log_t3], dim=-1), dim=-1)
    # Findable-weighted arithmetic mean of trace, in log space
    w_find = keypoints_conf.squeeze(-1)
    log_w = torch.log(w_find.clamp(min=1e-12))
    log_mean_trace = torch.logsumexp(log_trace + log_w, dim=-1) - torch.logsumexp(log_w, dim=-1)
    scores = scores * torch.exp(-0.20 * log_mean_trace)

    # normalize
    scores = scores / (1 + scores)

    keypoints_final = torch.cat([keypoints_xy, keypoints_conf], dim=-1)  # [B, num_select, K_per_class, 3]

    # iterate over batch and collect detections above thresholds
    all_key_points, detections = [], []

    if isinstance(threshold, torch.Tensor):
        threshold = threshold.to(device=device, dtype=keypoints_final.dtype)

    for bidx in range(len(keypoints_final)):
        predicted_confidence = scores[bidx]
        top_classes = labels[bidx]
        image_bboxes = bboxes[bidx]
        image_keypoints = keypoints_final[bidx]
        image_meta = pre_processing_meta[bidx]

        if classes_re_mapping is not None:
            remapping_mask = torch.isin(
                top_classes, classes_re_mapping.remaining_class_ids
            )
            top_classes = classes_re_mapping.class_mapping[top_classes[remapping_mask]]
            predicted_confidence = predicted_confidence[remapping_mask]
            image_bboxes = image_bboxes[remapping_mask]
            image_keypoints = image_keypoints[remapping_mask]
        else:
            # similar 'else' block for object detection is not correct
            raise ValueError("Not implemented")

        confidence_mask = predicted_confidence > (
            threshold[top_classes.long()]
            if isinstance(threshold, torch.Tensor)
            else threshold
        )

        predicted_confidence = predicted_confidence[confidence_mask]
        top_classes = top_classes[confidence_mask]
        selected_boxes = image_bboxes[confidence_mask]
        selected_keypoints = image_keypoints[confidence_mask] 
        predicted_confidence, sorted_indices = torch.sort(
            predicted_confidence, descending=True
        )
        top_classes = top_classes[sorted_indices]
        selected_boxes_xyxy_pct = selected_boxes[sorted_indices]
        selected_keypoints_xy_pct_conf = selected_keypoints[sorted_indices]
        selected_keypoints_xy_pct = selected_keypoints_xy_pct_conf[:, :, :2] 
        selected_keypoints_conf = selected_keypoints_xy_pct_conf[:, :, 2]

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
        selected_keypoints_xy = selected_keypoints_xy_pct * inference_size_whwh[:2]

        selected_boxes_xyxy = rescale_image_detections(
            image_detections=selected_boxes_xyxy,
            image_metadata=image_meta,
        )
        detections.append(
            Detections(
                xyxy=selected_boxes_xyxy.round().int(),
                confidence=predicted_confidence,
                class_id=top_classes.int(),
            )
        )

        # Similar to rescale_image_detections function, for keypoints. 
        offsets = torch.as_tensor([image_meta.pad_left, image_meta.pad_top],
            dtype=selected_keypoints_xy.dtype,
            device=selected_keypoints_xy.device,
        )
        selected_keypoints_xy.sub_(offsets)
        scale = torch.as_tensor([image_meta.scale_width, image_meta.scale_height],
            dtype=selected_keypoints_xy.dtype,
            device=selected_keypoints_xy.device,
        )
        selected_keypoints_xy.div_(scale)

        if (
            image_meta.static_crop_offset.offset_x != 0
            or image_meta.static_crop_offset.offset_y != 0
        ):
            static_crop_offsets = torch.as_tensor(
                [
                    image_meta.static_crop_offset.offset_x,
                    image_meta.static_crop_offset.offset_y,
                ],
                dtype=selected_keypoints_xy.dtype,
                device=selected_keypoints_xy.device,
            )
            selected_keypoints_xy.add_(static_crop_offsets)

        xy_max = torch.as_tensor(
            [image_meta.original_size.width, image_meta.original_size.height],
            dtype=selected_keypoints_xy.dtype,
            device=selected_keypoints_xy.device,
        )
        selected_keypoints_xy.clamp_(min=torch.zeros_like(xy_max), max=xy_max)

        # this is similar to the end of yolo26 keypoint postprocessing
        key_points_classes_for_instance_class = (
            (key_points_classes_for_instances[top_classes])
            .unsqueeze(1)
            .to(device=selected_keypoints_xy.device)
        )
        invalid_slot_keypoints = (
            torch.arange(key_points_slots_in_prediction, device=selected_keypoints_xy.device)
            .unsqueeze(0)
            .repeat(selected_keypoints_xy.shape[0], 1)
            >= key_points_classes_for_instance_class
        )
        keypoints_below_threshold = selected_keypoints_conf < key_points_threshold
        mask = invalid_slot_keypoints | keypoints_below_threshold
        selected_keypoints_xy[mask] = 0.0
        selected_keypoints_conf[mask] = 0.0
        all_key_points.append(
            KeyPoints(
                xy=selected_keypoints_xy.round().int(), 
                class_id=top_classes.int(),
                confidence=selected_keypoints_conf,
            )
        )

    return all_key_points, detections
