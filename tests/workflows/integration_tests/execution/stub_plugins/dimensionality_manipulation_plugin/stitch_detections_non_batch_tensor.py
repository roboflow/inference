"""
Tensor-native sibling of ``stitch_detections_non_batch.py``.

Same native logic as the batch sibling: read the crop origin from each prediction's
``image_metadata[PARENT_COORDINATES_KEY]``, shift ``xyxy`` with torch arithmetic,
concatenate the per-crop native ``Detections`` into one. The ``BlockManifest`` is
reused verbatim from the numpy module.

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
from tests.workflows.integration_tests.execution.stub_plugins.dimensionality_manipulation_plugin.stitch_detections_non_batch import (
    BlockManifest,
)
from tests.workflows.integration_tests.execution.stub_plugins.dimensionality_manipulation_plugin.stitch_detections_batch_tensor import (
    merge_native_predictions,
)
from inference_models.models.base.object_detection import Detections


class StitchDetectionsNonBatchBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        image: WorkflowImageData,
        image_predictions: Batch[Detections],
    ) -> BlockResult:
        merged_prediction = merge_native_predictions(list(image_predictions))
        return {"predictions": merged_prediction}
