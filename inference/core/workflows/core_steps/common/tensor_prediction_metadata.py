"""Attach workflows-level metadata to inference_models native prediction
types.

Per `[ITERATE PRED.2]` in the tensor-data-representation plan: per-prediction
global state (inference_id, model_id, prediction_type, class_names, image
dimensions, parent_* and root_parent_* coordinates and ids) lives on the
`image_metadata` dict carried by `inference_models.Detections`,
`InstanceDetections`, `MultiLabelClassificationPrediction`, and
`SemanticSegmentationResult`. This collapses what numpy-mode sv.Detections
replicates per detection into one dict per prediction.

`ClassificationPrediction` is intentionally not handled here — its metadata
slot is plural (`images_metadata: List[dict]`) because single-label
classification returns one prediction object for the whole batch. A
dedicated helper lands with the classification tensor block.
"""

import uuid
from typing import Dict, Optional, Union

from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.semantic_segmentation import (
    SemanticSegmentationResult,
)

from inference.core.workflows.execution_engine.constants import (
    IMAGE_DIMENSIONS_KEY,
    INFERENCE_ID_KEY,
    PARENT_COORDINATES_KEY,
    PARENT_DIMENSIONS_KEY,
    PARENT_ID_KEY,
    PREDICTION_TYPE_KEY,
    ROOT_PARENT_COORDINATES_KEY,
    ROOT_PARENT_DIMENSIONS_KEY,
    ROOT_PARENT_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData

CLASS_NAMES_KEY = "class_names"
MODEL_ID_KEY = "model_id"

PredictionWithSingularMetadata = Union[
    Detections,
    InstanceDetections,
    KeyPoints,
    MultiLabelClassificationPrediction,
    SemanticSegmentationResult,
]


def attach_prediction_metadata(
    prediction: PredictionWithSingularMetadata,
    *,
    image: WorkflowImageData,
    model_id: str,
    prediction_type: str,
    class_names: Optional[Dict[int, str]] = None,
    inference_id: Optional[str] = None,
) -> str:
    """Populate `prediction.image_metadata` from `WorkflowImageData`'s
    parent / root-parent metadata. Mutates `prediction` in place.

    Returns the resolved inference_id — read from existing
    `image_metadata[INFERENCE_ID_KEY]`, then from the `inference_id`
    argument, then minted via `uuid.uuid4()` if neither is set.

    `class_names` is a `class_id -> class_name` mapping. Local mode
    passes the full mapping built from the model's class list; remote
    mode passes a partial mapping built by walking response predictions
    (only class_ids that appeared in the response are present). `None`
    omits the key entirely.

    Raises `TypeError` for `ClassificationPrediction` (plural
    `images_metadata` requires a dedicated helper).
    """
    if isinstance(prediction, ClassificationPrediction):
        raise TypeError(
            "ClassificationPrediction uses plural `images_metadata` "
            "(per-image list for a single-label batch). Use a dedicated "
            "attach helper when the classification tensor block lands."
        )
    existing = prediction.image_metadata or {}
    resolved_inference_id = (
        existing.get(INFERENCE_ID_KEY) or inference_id or str(uuid.uuid4())
    )
    h, w = image._read_shape_without_materialization()
    parent = image.parent_metadata
    root = image.workflow_root_ancestor_metadata
    new_metadata = {
        **existing,
        INFERENCE_ID_KEY: resolved_inference_id,
        MODEL_ID_KEY: model_id,
        PREDICTION_TYPE_KEY: prediction_type,
        IMAGE_DIMENSIONS_KEY: (h, w),
        PARENT_ID_KEY: parent.parent_id,
        PARENT_DIMENSIONS_KEY: (
            parent.origin_coordinates.origin_height,
            parent.origin_coordinates.origin_width,
        ),
        PARENT_COORDINATES_KEY: (
            parent.origin_coordinates.left_top_x,
            parent.origin_coordinates.left_top_y,
        ),
        ROOT_PARENT_ID_KEY: root.parent_id,
        ROOT_PARENT_DIMENSIONS_KEY: (
            root.origin_coordinates.origin_height,
            root.origin_coordinates.origin_width,
        ),
        ROOT_PARENT_COORDINATES_KEY: (
            root.origin_coordinates.left_top_x,
            root.origin_coordinates.left_top_y,
        ),
    }
    if class_names is not None:
        new_metadata[CLASS_NAMES_KEY] = dict(class_names)
    prediction.image_metadata = new_metadata
    return resolved_inference_id


def attach_classification_prediction_metadata(
    prediction: ClassificationPrediction,
    *,
    images,
    model_id: str,
    prediction_type: str,
    class_names: Optional[Dict[int, str]] = None,
    inference_ids: Optional[List[str]] = None,
) -> List[str]:
    """Populate `prediction.images_metadata` (plural list) for a batch-shaped
    `ClassificationPrediction`. Single-label classification returns one
    prediction object for the whole batch; this helper writes one
    metadata dict per image in the batch and returns the list of
    resolved inference_ids.

    `images` is a Batch[WorkflowImageData] (or any iterable). `inference_ids`,
    when provided, must align with the batch length; missing entries are
    minted via uuid4.
    """
    images_list = list(images)
    bs = len(images_list)
    existing = list(prediction.images_metadata or [{} for _ in range(bs)])
    if len(existing) < bs:
        existing.extend({} for _ in range(bs - len(existing)))
    resolved_ids: List[str] = []
    for i, image in enumerate(images_list):
        per_image_existing = existing[i] or {}
        explicit = inference_ids[i] if inference_ids and i < len(inference_ids) else None
        resolved_id = (
            per_image_existing.get(INFERENCE_ID_KEY) or explicit or str(uuid.uuid4())
        )
        h, w = image._read_shape_without_materialization()
        parent = image.parent_metadata
        root = image.workflow_root_ancestor_metadata
        new_meta = {
            **per_image_existing,
            INFERENCE_ID_KEY: resolved_id,
            MODEL_ID_KEY: model_id,
            PREDICTION_TYPE_KEY: prediction_type,
            IMAGE_DIMENSIONS_KEY: (h, w),
            PARENT_ID_KEY: parent.parent_id,
            PARENT_DIMENSIONS_KEY: (
                parent.origin_coordinates.origin_height,
                parent.origin_coordinates.origin_width,
            ),
            PARENT_COORDINATES_KEY: (
                parent.origin_coordinates.left_top_x,
                parent.origin_coordinates.left_top_y,
            ),
            ROOT_PARENT_ID_KEY: root.parent_id,
            ROOT_PARENT_DIMENSIONS_KEY: (
                root.origin_coordinates.origin_height,
                root.origin_coordinates.origin_width,
            ),
            ROOT_PARENT_COORDINATES_KEY: (
                root.origin_coordinates.left_top_x,
                root.origin_coordinates.left_top_y,
            ),
        }
        if class_names is not None:
            new_meta[CLASS_NAMES_KEY] = dict(class_names)
        existing[i] = new_meta
        resolved_ids.append(resolved_id)
    prediction.images_metadata = existing
    return resolved_ids
