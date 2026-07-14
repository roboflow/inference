"""
Tensor-native sibling of ``stitch_detections_batch.py``: shifts each crop's ``xyxy``
by the origin the native OD producer stored per-image in
``image_metadata[PARENT_COORDINATES_KEY]``, then concatenates the per-crop
``Detections``.

This is just example, test implementation, please do not assume it being fully functional.
"""

from copy import deepcopy
from typing import List, Type

import torch

from inference.core.workflows.core_steps.common.query_language.operations.detections.base import (
    _concatenate_detections,
    _copy_detections,
)
from inference.core.workflows.execution_engine.constants import PARENT_COORDINATES_KEY
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    WorkflowImageData,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from tests.workflows.integration_tests.execution.stub_plugins.dimensionality_manipulation_plugin.stitch_detections_batch import (
    BlockManifest,
)
from inference_models.models.base.object_detection import Detections


def _shift_native_to_parent(prediction: Detections) -> Detections:
    """Return a copy of the prediction with ``xyxy`` shifted by the crop origin
    stored in ``image_metadata[PARENT_COORDINATES_KEY]`` as ``[x, y]``."""
    prediction_copy = _copy_detections(prediction)
    image_metadata = prediction_copy.image_metadata or {}
    coords = image_metadata.get(PARENT_COORDINATES_KEY, [0, 0])
    shift_x, shift_y = float(coords[0]), float(coords[1])
    shift = torch.as_tensor(
        [shift_x, shift_y, shift_x, shift_y],
        dtype=prediction_copy.xyxy.dtype,
        device=prediction_copy.xyxy.device,
    )
    prediction_copy.xyxy = prediction_copy.xyxy + shift
    return prediction_copy


def _empty_like(prediction: Detections) -> Detections:
    """Build an empty native ``Detections`` on the same device/dtype as ``prediction``."""
    xyxy = prediction.xyxy
    class_id = prediction.class_id
    confidence = prediction.confidence
    return Detections(
        xyxy=xyxy.new_zeros((0, 4)),
        class_id=class_id.new_zeros((0,)),
        confidence=confidence.new_zeros((0,)),
        image_metadata=deepcopy(prediction.image_metadata),
        bboxes_metadata=[],
    )


def merge_native_predictions(image_predictions: List[Detections]) -> Detections:
    non_empty = [_shift_native_to_parent(p) for p in image_predictions if len(p)]
    if not non_empty:
        return _empty_like(image_predictions[0])
    merged = non_empty[0]
    for prediction in non_empty[1:]:
        merged = _concatenate_detections(merged, prediction)
    return merged


class StitchDetectionsBatchBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        images: Batch[WorkflowImageData],
        images_predictions: Batch[Batch[Detections]],
    ) -> BlockResult:
        result = []
        for image, image_predictions in zip(images, images_predictions):
            merged_prediction = merge_native_predictions(list(image_predictions))
            result.append({"predictions": merged_prediction})
        return result
