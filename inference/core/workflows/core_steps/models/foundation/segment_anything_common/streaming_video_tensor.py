"""Tensor-native adapters shared by SAM2/SAM3 streaming video blocks."""

import uuid
from typing import Dict, List, Tuple

import numpy as np
import torch

from inference.core.env import WORKFLOWS_IMAGE_TENSOR_DEVICE
from inference.core.workflows.core_steps.common.tensor_native import (
    build_native_image_metadata,
    split_key_point_prediction,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything_common.streaming_video import (
    BoxPromptMetadata,
)
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAME_KEY,
    CLASS_NAMES_KEY,
    DETECTION_ID_KEY,
    TRACKER_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.rle_utils import torch_mask_to_coco_rle

PREDICTION_TYPE = "instance-segmentation"


def _resolve_prompt_class_name(detections, index: int, bbox_metadata: dict) -> str:
    """Resolve a per-box class name before the image-level class-name map."""
    override = bbox_metadata.get(CLASS_NAME_KEY)
    if override is not None:
        return str(override)
    class_names = (detections.image_metadata or {}).get(CLASS_NAMES_KEY) or {}
    class_id = int(detections.class_id[index])
    return str(class_names.get(class_id, "foreground"))


def extract_box_prompts_tensor(
    boxes_for_image,
) -> Tuple[List[Tuple[float, float, float, float]], List[BoxPromptMetadata]]:
    """Convert tensor-native detections into prompts and metadata."""
    if boxes_for_image is None:
        return [], []
    _key_points, detections = split_key_point_prediction(boxes_for_image)
    if len(detections) == 0:
        return [], []

    boxes_xyxy: List[Tuple[float, float, float, float]] = []
    metadata: List[BoxPromptMetadata] = []
    for index in range(len(detections)):
        x1, y1, x2, y2 = detections.xyxy[index].tolist()
        boxes_xyxy.append((float(x1), float(y1), float(x2), float(y2)))
        bbox_metadata = (
            detections.bboxes_metadata[index]
            if detections.bboxes_metadata is not None
            and index < len(detections.bboxes_metadata)
            else {}
        )
        parent_id = bbox_metadata.get(DETECTION_ID_KEY)
        metadata.append(
            BoxPromptMetadata(
                class_id=int(detections.class_id[index]),
                class_name=_resolve_prompt_class_name(
                    detections=detections,
                    index=index,
                    bbox_metadata=bbox_metadata,
                ),
                confidence=(
                    float(detections.confidence[index])
                    if detections.confidence is not None
                    else 1.0
                ),
                parent_id=str(parent_id) if parent_id is not None else None,
            )
        )
    return boxes_xyxy, metadata


def masks_to_instance_detections(
    masks: np.ndarray,
    obj_ids: np.ndarray,
    image: WorkflowImageData,
    obj_id_metadata: Dict[int, BoxPromptMetadata],
    threshold: float,
    mask_representation: str,
    fallback_class_id: int = 0,
    fallback_class_name: str = "foreground",
) -> InstanceDetections:
    """Convert tracked masks into tensor-native instance detections."""
    height, width = image._read_shape_without_materialization()
    xyxy: List[List[float]] = []
    confidences: List[float] = []
    class_ids: List[int] = []
    class_names_map: Dict[int, str] = {}
    bboxes_metadata: List[dict] = []
    kept_masks: List[np.ndarray] = []

    for mask, obj_id in zip(masks, obj_ids.tolist()):
        metadata = obj_id_metadata.get(int(obj_id))
        confidence = metadata.confidence if metadata is not None else 1.0
        if confidence < threshold:
            continue
        ys, xs = np.where(mask)
        if xs.size == 0:
            continue
        class_id = int(metadata.class_id) if metadata is not None else fallback_class_id
        class_name = (
            metadata.class_name if metadata is not None else fallback_class_name
        )
        xyxy.append(
            [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
        )
        confidences.append(float(confidence))
        class_ids.append(class_id)
        class_names_map[class_id] = class_name
        bboxes_metadata.append(
            {
                DETECTION_ID_KEY: str(uuid.uuid4()),
                CLASS_NAME_KEY: class_name,
                TRACKER_ID_KEY: int(obj_id),
            }
        )
        kept_masks.append(mask.astype(bool))

    detection_count = len(kept_masks)
    if detection_count == 0:
        xyxy_tensor = torch.zeros((0, 4), dtype=torch.float32)
        class_id_tensor = torch.zeros((0,), dtype=torch.int64)
        confidence_tensor = torch.zeros((0,), dtype=torch.float32)
        output_masks = (
            InstancesRLEMasks(image_size=(height, width), masks=[])
            if mask_representation == "rle"
            else torch.zeros((0, height, width), dtype=torch.bool)
        )
    else:
        xyxy_tensor = torch.tensor(xyxy, dtype=torch.float32)
        class_id_tensor = torch.tensor(class_ids, dtype=torch.int64)
        confidence_tensor = torch.tensor(confidences, dtype=torch.float32)
        if mask_representation == "rle":
            rle_masks = [
                torch_mask_to_coco_rle(torch.from_numpy(mask)) for mask in kept_masks
            ]
            output_masks = InstancesRLEMasks.from_coco_rle_masks(
                image_size=(height, width), masks=rle_masks
            )
        else:
            output_masks = torch.from_numpy(np.stack(kept_masks, axis=0))

    detections = InstanceDetections(
        xyxy=xyxy_tensor.to(WORKFLOWS_IMAGE_TENSOR_DEVICE),
        class_id=class_id_tensor.to(WORKFLOWS_IMAGE_TENSOR_DEVICE),
        confidence=confidence_tensor.to(WORKFLOWS_IMAGE_TENSOR_DEVICE),
        mask=(
            output_masks
            if isinstance(output_masks, InstancesRLEMasks)
            else output_masks.to(WORKFLOWS_IMAGE_TENSOR_DEVICE)
        ),
    )
    detections.image_metadata = build_native_image_metadata(
        image=image,
        class_names=class_names_map,
        prediction_type=PREDICTION_TYPE,
        inference_id=str(uuid.uuid4()),
    )
    detections.bboxes_metadata = bboxes_metadata if detection_count else None
    return detections
