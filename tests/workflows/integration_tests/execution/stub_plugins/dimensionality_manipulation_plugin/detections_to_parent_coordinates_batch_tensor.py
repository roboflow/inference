"""
Tensor-native sibling of ``detections_to_parent_coordinates_batch.py``.

Operates on native ``inference_models.Detections`` (xyxy/class_id/confidence are
torch tensors; per-box state lives in ``bboxes_metadata`` — the native equivalent
of sv ``.data``). The numpy block writes ``parent_id`` / ``parent_coordinates`` /
``parent_dimensions`` via sv ``.data`` item-assignment; here we write the SAME key
strings (``PARENT_ID_KEY`` / ``PARENT_COORDINATES_KEY`` / ``PARENT_DIMENSIONS_KEY``)
into each per-box ``bboxes_metadata`` dict instead. The ``BlockManifest`` (the
``type`` Literal + I/O contract) is reused verbatim from the numpy module.

This is just example, test implementation, please do not assume it being fully functional.
"""

from typing import Type

from inference.core.workflows.core_steps.common.query_language.operations.detections.base import (
    _copy_detections,
)
from inference.core.workflows.execution_engine.constants import (
    PARENT_COORDINATES_KEY,
    PARENT_DIMENSIONS_KEY,
    PARENT_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    WorkflowImageData,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from tests.workflows.integration_tests.execution.stub_plugins.dimensionality_manipulation_plugin.detections_to_parent_coordinates_batch import (
    BlockManifest,
)
from inference_models.models.base.object_detection import Detections


def _write_parent_metadata_native(
    prediction: Detections,
    parent_id: str,
    parent_coordinates,
) -> Detections:
    """Return a COPY of the native prediction with parent lineage written per-box
    into ``bboxes_metadata`` (mirroring the numpy block's sv ``.data`` writes)."""
    prediction_copy = _copy_detections(prediction)
    number_of_boxes = len(prediction_copy)
    bboxes_metadata = prediction_copy.bboxes_metadata
    if bboxes_metadata is None:
        bboxes_metadata = [{} for _ in range(number_of_boxes)]
    else:
        bboxes_metadata = [dict(entry) for entry in bboxes_metadata]
    offset = [0, 0]
    dimensions = (
        [parent_coordinates.origin_height, parent_coordinates.origin_width]
        if parent_coordinates
        else None
    )
    for entry in bboxes_metadata:
        entry[PARENT_ID_KEY] = parent_id
        if parent_coordinates:
            entry[PARENT_COORDINATES_KEY] = list(offset)
            entry[PARENT_DIMENSIONS_KEY] = list(dimensions)
    prediction_copy.bboxes_metadata = bboxes_metadata
    return prediction_copy


class DetectionsToParentCoordinatesBatchBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return True

    def run(
        self,
        images: Batch[WorkflowImageData],
        images_predictions: Batch[Batch[Detections]],
    ) -> BlockResult:
        result = []
        for image, image_predictions in zip(images, images_predictions):
            parent_id = image.parent_metadata.parent_id
            parent_coordinates = image.parent_metadata.origin_coordinates
            transformed_predictions = []
            for prediction in image_predictions:
                prediction_copy = _write_parent_metadata_native(
                    prediction=prediction,
                    parent_id=parent_id,
                    parent_coordinates=parent_coordinates,
                )
                transformed_predictions.append({"predictions": prediction_copy})
            result.append(transformed_predictions)
        return result
