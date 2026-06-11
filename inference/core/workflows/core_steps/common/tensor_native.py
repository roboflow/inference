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

from typing import Dict, List, Optional, Sequence, Tuple, Union
from uuid import uuid4

import numpy as np
import torch

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.rle_utils import coco_rle_masks_to_numpy_mask

from inference.core.workflows.execution_engine.constants import (
    CLASS_ID_KEY,
    CLASS_NAME_KEY,
    CLASS_NAMES_KEY,
    CONFIDENCE_KEY,
    DETECTION_ID_KEY,
    HEIGHT_KEY,
    IMAGE_DIMENSIONS_KEY,
    INFERENCE_ID_KEY,
    PARENT_COORDINATES_KEY,
    PARENT_DIMENSIONS_KEY,
    PARENT_ID_KEY,
    PREDICTION_TYPE_KEY,
    ROOT_PARENT_COORDINATES_KEY,
    ROOT_PARENT_DIMENSIONS_KEY,
    ROOT_PARENT_ID_KEY,
    WIDTH_KEY,
    X_KEY,
    Y_KEY,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData

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


def build_native_image_metadata(
    image: WorkflowImageData,
    class_names: Dict[int, str],
    prediction_type: str,
    inference_id: Optional[str] = None,
) -> dict:
    """Build the per-image ``image_metadata`` dict carried by a tensor-native
    ``Detections`` prediction produced by a model block.

    Holds the ``class_id -> name`` map (required by the tensor-native serialiser),
    the image dimensions, and the parent/root lineage needed for crop-aware
    coordinate recovery downstream. Mirrors the convention in
    ``formatters/vlm_as_detector/v1_tensor.py``, but with a parametrised
    ``prediction_type``. The image shape is read without forcing a device->host
    materialization (so tensor-only inputs stay on device).
    """
    height, width = image._read_shape_without_materialization()
    parent = image.parent_metadata
    root = image.workflow_root_ancestor_metadata
    parent_coordinates = parent.origin_coordinates
    root_coordinates = root.origin_coordinates
    metadata = {
        CLASS_NAMES_KEY: class_names,
        PREDICTION_TYPE_KEY: prediction_type,
        IMAGE_DIMENSIONS_KEY: [height, width],
        PARENT_ID_KEY: parent.parent_id,
        PARENT_COORDINATES_KEY: [
            parent_coordinates.left_top_x,
            parent_coordinates.left_top_y,
        ],
        PARENT_DIMENSIONS_KEY: [
            parent_coordinates.origin_height,
            parent_coordinates.origin_width,
        ],
        ROOT_PARENT_ID_KEY: root.parent_id,
        ROOT_PARENT_COORDINATES_KEY: [
            root_coordinates.left_top_x,
            root_coordinates.left_top_y,
        ],
        ROOT_PARENT_DIMENSIONS_KEY: [
            root_coordinates.origin_height,
            root_coordinates.origin_width,
        ],
    }
    if inference_id is not None:
        metadata[INFERENCE_ID_KEY] = inference_id
    return metadata


def attach_native_detection_metadata(
    detections: Detections,
    image: WorkflowImageData,
    class_names: Dict[int, str],
    prediction_type: str,
    inference_id: Optional[str] = None,
) -> Detections:
    """LOCAL-path helper for model blocks that get a native ``Detections`` straight
    from an inference_models adapter's ``run_tensor_native_inference``.

    The adapter fills ``xyxy`` / ``class_id`` / ``confidence`` (and sometimes a few
    per-detection ``bboxes_metadata`` fields) but knows nothing about the workflow
    image lineage. This attaches the workflow ``image_metadata`` and guarantees each
    detection carries a ``detection_id`` (generated when missing), preserving any
    keys the model already set (e.g. EasyOCR's per-box ``text``). Mutates and
    returns the same object.
    """
    detections.image_metadata = build_native_image_metadata(
        image=image,
        class_names=class_names,
        prediction_type=prediction_type,
        inference_id=inference_id,
    )
    number_of_detections = int(detections.xyxy.shape[0])
    if number_of_detections == 0:
        detections.bboxes_metadata = None
        return detections
    existing = detections.bboxes_metadata
    bboxes_metadata = []
    for index in range(number_of_detections):
        entry = (
            dict(existing[index])
            if existing is not None and index < len(existing)
            else {}
        )
        entry.setdefault(DETECTION_ID_KEY, str(uuid4()))
        bboxes_metadata.append(entry)
    detections.bboxes_metadata = bboxes_metadata
    return detections


def native_detections_from_inference_predictions(
    image: WorkflowImageData,
    predictions: List[dict],
    prediction_type: str,
    class_names: Optional[Dict[int, str]] = None,
    inference_id: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Detections:
    """REMOTE-path helper: build a native ``Detections`` from standard inference
    object-detection prediction dicts (center ``x``/``y``/``width``/``height``,
    ``confidence``, ``class_id``, optional ``class`` name and ``detection_id``).

    Boxes are converted from center form to corner ``xyxy``. When ``class_names``
    is not supplied it is derived from the predictions' own ``class_id``/``class``
    pairs so the serialiser can resolve every id. A ``detection_id`` is preserved
    when present, otherwise generated.
    """
    xyxy: List[List[float]] = []
    class_id: List[int] = []
    confidence: List[float] = []
    bboxes_metadata: List[dict] = []
    derived_class_names: Dict[int, str] = {}
    for prediction in predictions:
        center_x = float(prediction[X_KEY])
        center_y = float(prediction[Y_KEY])
        box_width = float(prediction[WIDTH_KEY])
        box_height = float(prediction[HEIGHT_KEY])
        xyxy.append(
            [
                center_x - box_width / 2,
                center_y - box_height / 2,
                center_x + box_width / 2,
                center_y + box_height / 2,
            ]
        )
        prediction_class_id = int(prediction.get(CLASS_ID_KEY, 0))
        class_id.append(prediction_class_id)
        confidence.append(float(prediction.get(CONFIDENCE_KEY, 1.0)))
        if CLASS_NAME_KEY in prediction:
            derived_class_names[prediction_class_id] = str(prediction[CLASS_NAME_KEY])
        bboxes_metadata.append(
            {DETECTION_ID_KEY: str(prediction.get(DETECTION_ID_KEY) or uuid4())}
        )
    number_of_detections = len(xyxy)
    resolved_class_names = (
        class_names if class_names is not None else derived_class_names
    )
    image_metadata = build_native_image_metadata(
        image=image,
        class_names=resolved_class_names,
        prediction_type=prediction_type,
        inference_id=inference_id,
    )
    return Detections(
        xyxy=torch.as_tensor(xyxy, dtype=torch.float32, device=device).reshape(-1, 4),
        class_id=torch.as_tensor(class_id, dtype=torch.long, device=device).reshape(
            -1
        ),
        confidence=torch.as_tensor(
            confidence, dtype=torch.float32, device=device
        ).reshape(-1),
        image_metadata=image_metadata,
        bboxes_metadata=bboxes_metadata if number_of_detections > 0 else None,
    )
