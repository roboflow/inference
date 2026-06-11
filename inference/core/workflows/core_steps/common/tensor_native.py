"""Shared helpers for tensor-native predictions (the inference_models dataclasses
used under ENABLE_TENSOR_DATA_REPRESENTATION).

These consolidate logic that otherwise gets copy-pasted into every tensor-native
block sibling: selecting a subset of detections by boolean mask or index list,
and normalising the keypoint-detection input shape. Kept here (rather than on the
inference_models dataclasses) to avoid bloating those types with workflow-specific
concerns.

Supported prediction shapes:
- ``inference_models.Detections``                (object detection)
- ``inference_models.InstanceDetections``        (instance segmentation; dense or RLE masks)
- ``inference_models.KeyPoints``                 (keypoints, standalone)
- ``Tuple[KeyPoints, Optional[Detections]]``     (keypoint-detection workflow kind)
"""

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.rle_utils import coco_rle_masks_to_numpy_mask

TensorNativeDetections = Union[Detections, InstanceDetections]
KeyPointPrediction = Tuple[KeyPoints, Optional[Detections]]
TensorNativePrediction = Union[Detections, InstanceDetections, KeyPoints, KeyPointPrediction]


def mask_to_indices(mask: Union[np.ndarray, Sequence[bool]]) -> List[int]:
    """Convert a boolean mask (numpy array, list, or torch tensor) into the list
    of surviving row indices, in ascending order."""
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().to("cpu").numpy()
    return np.nonzero(np.asarray(mask))[0].tolist()


def take_detections_by_indices(
    detections: TensorNativeDetections,
    indices: Sequence[int],
) -> TensorNativeDetections:
    """Select rows of a ``Detections`` / ``InstanceDetections`` by index list.

    Per-detection state (``bboxes_metadata``) and masks (dense torch or RLE) are
    carried over for the surviving rows; ``image_metadata`` is shared as-is.
    """
    indices = list(indices)
    index_tensor = torch.as_tensor(
        indices, dtype=torch.long, device=detections.xyxy.device
    )
    bboxes_metadata = None
    if detections.bboxes_metadata is not None:
        bboxes_metadata = [detections.bboxes_metadata[i] for i in indices]
    if isinstance(detections, InstanceDetections):
        mask_field = detections.mask
        if isinstance(mask_field, InstancesRLEMasks):
            new_mask: Union[torch.Tensor, InstancesRLEMasks] = InstancesRLEMasks(
                image_size=mask_field.image_size,
                masks=[mask_field.masks[i] for i in indices],
            )
        else:
            new_mask = mask_field[index_tensor]
        return InstanceDetections(
            xyxy=detections.xyxy[index_tensor],
            class_id=detections.class_id[index_tensor],
            confidence=detections.confidence[index_tensor],
            mask=new_mask,
            image_metadata=detections.image_metadata,
            bboxes_metadata=bboxes_metadata,
        )
    return Detections(
        xyxy=detections.xyxy[index_tensor],
        class_id=detections.class_id[index_tensor],
        confidence=detections.confidence[index_tensor],
        image_metadata=detections.image_metadata,
        bboxes_metadata=bboxes_metadata,
    )


def take_key_points_by_indices(
    key_points: KeyPoints,
    indices: Sequence[int],
) -> KeyPoints:
    """Select instances of a ``KeyPoints`` by index list (slices along the
    instance dimension; per-instance ``key_points_metadata`` carried over)."""
    indices = list(indices)
    index_tensor = torch.as_tensor(
        indices, dtype=torch.long, device=key_points.xy.device
    )
    key_points_metadata = None
    if key_points.key_points_metadata is not None:
        key_points_metadata = [key_points.key_points_metadata[i] for i in indices]
    return KeyPoints(
        xy=key_points.xy[index_tensor],
        class_id=key_points.class_id[index_tensor],
        confidence=key_points.confidence[index_tensor],
        image_metadata=key_points.image_metadata,
        key_points_metadata=key_points_metadata,
    )


def take_prediction_by_indices(
    prediction: TensorNativePrediction,
    indices: Sequence[int],
) -> TensorNativePrediction:
    """Select a subset of any tensor-native prediction by index list, returning
    the same shape. For the keypoint-detection tuple, both the ``KeyPoints`` and
    the bbox ``Detections`` components are sliced consistently."""
    if isinstance(prediction, tuple):
        key_points, detections = prediction
        sliced_key_points = take_key_points_by_indices(key_points, indices)
        sliced_detections = (
            take_detections_by_indices(detections, indices)
            if detections is not None
            else None
        )
        return sliced_key_points, sliced_detections
    if isinstance(prediction, KeyPoints):
        return take_key_points_by_indices(prediction, indices)
    return take_detections_by_indices(prediction, indices)


def take_prediction_by_mask(
    prediction: TensorNativePrediction,
    mask: Union[np.ndarray, Sequence[bool]],
) -> TensorNativePrediction:
    """Select a subset of any tensor-native prediction by boolean mask."""
    return take_prediction_by_indices(prediction, mask_to_indices(mask))


def split_key_point_prediction(
    prediction: Union[TensorNativeDetections, KeyPointPrediction],
) -> Tuple[Optional[KeyPoints], TensorNativeDetections]:
    """Normalise a block input to its bounding-box component.

    Returns ``(key_points, detections)`` where ``key_points`` is the
    ``KeyPoints`` component when the input is a keypoint-detection tuple
    (``(KeyPoints, Detections)``), otherwise ``None``. Blocks that only need
    bounding boxes operate on the returned ``detections`` and re-wrap with the
    returned ``key_points`` to preserve keypoints downstream. Raises if a
    keypoint tuple lacks its bbox component.
    """
    if isinstance(prediction, tuple):
        key_points, detections = prediction
        if detections is None:
            raise ValueError(
                "Keypoint prediction is missing the bounding-box component "
                "required by this block."
            )
        return key_points, detections
    return None, prediction


def instance_mask_to_numpy(
    detections: InstanceDetections,
    index: int,
) -> np.ndarray:
    """Materialise a single instance's mask as a 2-D ``np.ndarray`` of bool
    ``(H, W)``. RLE masks are decoded one instance at a time so the full stack
    is never materialised at once (the same convention as the serialiser)."""
    mask = detections.mask
    if isinstance(mask, InstancesRLEMasks):
        return coco_rle_masks_to_numpy_mask(
            InstancesRLEMasks(image_size=mask.image_size, masks=[mask.masks[index]])
        )[0]
    return mask[index].detach().to("cpu").numpy().astype(bool)
