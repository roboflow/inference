# Copyright 2026 Technology Innovation Institute (TII), Abu Dhabi.
# Licensed under the Apache License, Version 2.0.
# Adapted from https://github.com/tiiuae/Falcon-Perception for integration
# with the inference-models package.
#
# Post-processing: convert engine outputs to Detections/InstanceDetections.

from typing import List, Optional

import torch
import torch.nn.functional as F

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from inference_models.models.falcon_perception.config import FalconPerceptionConfig
from inference_models.models.falcon_perception.engine import (
    BatchInferenceResult,
    InstancePrediction,
    QueryResult,
    boxes_from_instances,
    decode_coord_bins,
    decode_size_bins,
    instance_confidence,
)
from inference_models.models.falcon_perception.model import FalconPerceptionModel
from inference_models.models.falcon_perception.preprocessing import ImageMetadata


def compute_masks_for_instances(
    model: FalconPerceptionModel,
    image_features: torch.Tensor,
    h_patches: int,
    w_patches: int,
    instances: List[InstancePrediction],
    target_height: int,
    target_width: int,
    config: FalconPerceptionConfig,
) -> Optional[torch.Tensor]:
    """Compute segmentation masks for all instances in a query.

    Args:
        model: The Falcon Perception model (for AnyUp and seg projector).
        image_features: (1, N_patches, D) image token features from transformer.
        h_patches: Number of patches in height.
        w_patches: Number of patches in width.
        instances: List of instance predictions with seg_hidden tensors.
        target_height: Output mask height (original image height).
        target_width: Output mask width (original image width).
        config: Model config.

    Returns:
        (N_instances, target_height, target_width) binary mask tensor, or None if
        no instances have seg_hidden.
    """
    seg_instances = [inst for inst in instances if inst.seg_hidden is not None]
    if not seg_instances:
        return None

    masks = []
    with torch.inference_mode():
        # Compute upsampled features once for all instances
        upsampled = model.anyup(image_features, h_patches, w_patches)  # (1, C, H_up, W_up)

        for inst in seg_instances:
            seg_proj = model.get_seg_projection(inst.seg_hidden)  # (1, C_anyup)
            mask_logits = model.compute_mask(seg_proj, upsampled)  # (1, H_up, W_up)

            # Resize mask to target image dimensions
            mask_resized = F.interpolate(
                mask_logits.unsqueeze(1),  # (1, 1, H_up, W_up)
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1).squeeze(0)  # (target_height, target_width)

            binary_mask = (mask_resized.sigmoid() > config.mask_threshold).bool()
            masks.append(binary_mask)

    if not masks:
        return None
    return torch.stack(masks, dim=0)  # (N, H, W)


def result_to_detections(
    result: BatchInferenceResult,
    image_metadata: ImageMetadata,
    config: FalconPerceptionConfig,
    prompts: List[str],
) -> Detections:
    """Convert batch inference result to Detections (detection-only mode).

    Flattens all instances across all queries into a single Detections object
    with class IDs mapping to prompt indices.

    Args:
        result: BatchInferenceResult from the engine.
        image_metadata: Original image dimensions.
        config: Model config.
        prompts: List of prompt strings.

    Returns:
        Detections with xyxy boxes, class IDs (prompt index), and confidence scores.
    """
    all_boxes = []
    all_class_ids = []
    all_confidences = []

    for query_idx, query_result in enumerate(result.query_results):
        if not query_result.present or not query_result.instances:
            continue

        boxes = boxes_from_instances(
            query_result.instances,
            image_metadata.original_width,
            image_metadata.original_height,
            config,
        )

        for inst, box in zip(query_result.instances, boxes):
            all_boxes.append(box)
            all_class_ids.append(query_idx)
            all_confidences.append(instance_confidence(inst))

    if not all_boxes:
        return Detections(
            xyxy=torch.zeros(0, 4, dtype=torch.int32),
            class_id=torch.zeros(0, dtype=torch.int32),
            confidence=torch.zeros(0, dtype=torch.float32),
            image_metadata={"class_names": prompts},
        )

    xyxy = torch.tensor(all_boxes, dtype=torch.float32).round().int()
    class_id = torch.tensor(all_class_ids, dtype=torch.int32)
    confidence = torch.tensor(all_confidences, dtype=torch.float32)

    return Detections(
        xyxy=xyxy,
        class_id=class_id,
        confidence=confidence,
        image_metadata={"class_names": prompts},
    )


def result_to_instance_detections(
    result: BatchInferenceResult,
    image_metadata: ImageMetadata,
    model: FalconPerceptionModel,
    config: FalconPerceptionConfig,
    prompts: List[str],
) -> InstanceDetections:
    """Convert batch inference result to InstanceDetections (segmentation mode).

    Flattens all instances across all queries into a single InstanceDetections
    object with class IDs mapping to prompt indices, including masks.

    Args:
        result: BatchInferenceResult from the engine (must have image_features).
        image_metadata: Original image dimensions.
        model: The Falcon Perception model (for mask computation).
        config: Model config.
        prompts: List of prompt strings.

    Returns:
        InstanceDetections with xyxy boxes, class IDs, confidence, and masks.
    """
    all_boxes = []
    all_class_ids = []
    all_confidences = []
    all_instances_flat: List[InstancePrediction] = []
    instance_to_query: List[int] = []

    for query_idx, query_result in enumerate(result.query_results):
        if not query_result.present or not query_result.instances:
            continue

        boxes = boxes_from_instances(
            query_result.instances,
            image_metadata.original_width,
            image_metadata.original_height,
            config,
        )

        for inst, box in zip(query_result.instances, boxes):
            all_boxes.append(box)
            all_class_ids.append(query_idx)
            all_confidences.append(instance_confidence(inst))
            all_instances_flat.append(inst)
            instance_to_query.append(query_idx)

    if not all_boxes:
        return InstanceDetections(
            xyxy=torch.zeros(0, 4, dtype=torch.int32),
            class_id=torch.zeros(0, dtype=torch.int32),
            confidence=torch.zeros(0, dtype=torch.float32),
            mask=torch.zeros(
                0,
                image_metadata.original_height,
                image_metadata.original_width,
                dtype=torch.bool,
            ),
            image_metadata={"class_names": prompts},
        )

    xyxy = torch.tensor(all_boxes, dtype=torch.float32).round().int()
    class_id = torch.tensor(all_class_ids, dtype=torch.int32)
    confidence = torch.tensor(all_confidences, dtype=torch.float32)

    # Compute masks if image features are available
    masks = None
    if result.image_features is not None:
        masks = compute_masks_for_instances(
            model=model,
            image_features=result.image_features,
            h_patches=result.h_patches,
            w_patches=result.w_patches,
            instances=all_instances_flat,
            target_height=image_metadata.original_height,
            target_width=image_metadata.original_width,
            config=config,
        )

    if masks is None:
        masks = torch.zeros(
            len(all_boxes),
            image_metadata.original_height,
            image_metadata.original_width,
            dtype=torch.bool,
        )

    return InstanceDetections(
        xyxy=xyxy,
        class_id=class_id,
        confidence=confidence,
        mask=masks,
        image_metadata={"class_names": prompts},
    )
