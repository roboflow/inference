"""
Tensor-native sibling of ``detections_to_parent_coordinates_non_batch.py``: writes
``parent_id`` / ``parent_coordinates`` / ``parent_dimensions`` per-box into
``bboxes_metadata`` (the native counterpart of sv ``.data``).

This is just example, test implementation, please do not assume it being fully functional.
"""

from typing import Type

from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    WorkflowImageData,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from tests.workflows.integration_tests.execution.stub_plugins.dimensionality_manipulation_plugin.detections_to_parent_coordinates_non_batch import (
    BlockManifest,
)
from tests.workflows.integration_tests.execution.stub_plugins.dimensionality_manipulation_plugin.detections_to_parent_coordinates_batch_tensor import (
    _write_parent_metadata_native,
)
from inference_models.models.base.object_detection import Detections


class DetectionsToParentCoordinatesNonBatchBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        image: WorkflowImageData,
        image_predictions: Batch[Detections],
    ) -> BlockResult:
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
        return transformed_predictions
