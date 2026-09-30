"""SAM3 native RLE output: filter, suppress and pack without decoding pixels."""

import uuid
from typing import Dict, List, Optional

import numpy as np
from pycocotools import mask as mask_utils
from roboflow_workflows.core_steps.models.foundation.segment_anything3.v1_tensor import (
    _assemble_detections,
)
from roboflow_workflows.core_steps.models.foundation.segment_anything3.v2_tensor import (
    _nms_greedy_pycocotools,
)
from roboflow_workflows.core_steps.models.foundation.segment_anything_common.prompts import (
    Sam3Prompt,
)
from roboflow_workflows.execution_engine.constants import (
    CLASS_NAME_KEY,
    DETECTION_ID_KEY,
)
from roboflow_workflows.execution_engine.entities.base import WorkflowImageData

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.types import InstancesRLEMasks


def build_native_rle_detections(
    per_prompt_results: List[dict],
    class_names: List[Optional[str]],
    class_mapping: Optional[Dict[str, str]],
    prompts: List[Sam3Prompt],
    global_confidence: float,
    apply_nms: bool,
    nms_iou_threshold: Optional[float],
    image: WorkflowImageData,
) -> InstanceDetections:
    """Preserve prompt ordering, per-class thresholds and cross-prompt mask NMS.

    Tight mask boxes use the existing inclusive maximum-coordinate convention;
    model-predicted boxes are deliberately not substituted for mask bounds.

    Args:
        per_prompt_results (List[dict]): Model RLE masks and scores by prompt.
        class_names (List[Optional[str]]): Names in prompt order.
        class_mapping (Optional[Dict[str, str]]): Output class-name overrides.
        prompts (List[Sam3Prompt]): Prompts with optional confidence thresholds.
        global_confidence (float): Threshold for prompts without an override.
        apply_nms (bool): Whether to suppress overlapping masks across prompts.
        nms_iou_threshold (Optional[float]): Mask IoU threshold for suppression.
        image (WorkflowImageData): Image providing shape and lineage metadata.

    Returns:
        InstanceDetections: Predictions retaining compressed masks and class metadata.
    """
    height, width = image._read_shape_without_materialization()
    by_prompt = {r.get("prompt_index", 0): r for r in per_prompt_results}
    rles, scores, class_ids = [], [], []
    for prompt_index, prompt in enumerate(prompts):
        threshold = prompt.output_prob_thresh
        if threshold is None:
            threshold = global_confidence
        result = by_prompt.get(prompt_index, {})
        for rle, score in zip(result.get("masks", []), result.get("scores", [])):
            if score >= threshold:
                # SAM3's GPU encoder returns JSON strings; native mask carriers
                # use bytes. Normalize without decoding or mutating model output.
                if isinstance(rle["counts"], str):
                    rle = {**rle, "counts": rle["counts"].encode("ascii")}
                rles.append(rle)
                scores.append(float(score))
                class_ids.append(prompt_index)
    if apply_nms and nms_iou_threshold is not None and rles:
        keep = _nms_greedy_pycocotools(rles, np.asarray(scores), nms_iou_threshold)
        rles = [r for r, kept in zip(rles, keep) if kept]
        scores = [s for s, kept in zip(scores, keep) if kept]
        class_ids = [c for c, kept in zip(class_ids, keep) if kept]

    boxes = mask_utils.toBbox(rles) if rles else np.empty((0, 4))
    xyxy, kept_scores, kept_ids, kept_rles, metadata = [], [], [], [], []
    names = {}
    for rle, score, class_id, (x, y, w, h) in zip(rles, scores, class_ids, boxes):
        # Thresholded SAM3 queries can have no foreground pixels.
        if w == 0 or h == 0:
            continue
        class_name = (
            class_names[class_id] if class_id < len(class_names) else None
        ) or "foreground"
        if class_mapping:
            class_name = class_mapping.get(class_name, class_name)
        names[class_id] = class_name
        xyxy.append([float(x), float(y), float(x + w - 1), float(y + h - 1)])
        kept_scores.append(score)
        kept_ids.append(class_id)
        kept_rles.append(rle)
        metadata.append(
            {DETECTION_ID_KEY: str(uuid.uuid4()), CLASS_NAME_KEY: class_name}
        )
    detections = _assemble_detections(
        image=image,
        xyxy=xyxy,
        confidences=kept_scores,
        class_ids=kept_ids,
        class_names_map=names,
        bboxes_metadata=metadata,
        mask=InstancesRLEMasks.from_coco_rle_masks(
            image_size=(height, width), masks=kept_rles
        ),
    )

    return detections
