"""
Tensor-native sibling of ``tile_detections_non_batch.py``.

Same as the batch tile sibling: convert each native ``Detections`` to ``sv.Detections``
via ``.to_supervision()`` ONLY to feed the numpy ``sv.BoxAnnotator``; the output is
the IMAGE tiles (unchanged). The ``BlockManifest`` is reused verbatim from the numpy
module.

This is just example, test implementation, please do not assume it being fully functional.
"""

from typing import Type

from inference.core.utils.drawing import create_tiles
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    WorkflowImageData,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from tests.workflows.integration_tests.execution.stub_plugins.dimensionality_manipulation_plugin.tile_detections_non_batch import (
    BlockManifest,
)
from inference_models.models.base.object_detection import Detections

import supervision as sv


class TileDetectionsNonBatchBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        crops: Batch[WorkflowImageData],
        crops_predictions: Batch[Detections],
    ) -> BlockResult:
        annotator = sv.BoxAnnotator()
        visualisations = []
        for image, prediction in zip(crops, crops_predictions):
            annotated_image = annotator.annotate(
                image.numpy_image.copy(),
                prediction.to_supervision(),
            )
            visualisations.append(annotated_image)
        tile = create_tiles(visualisations)
        return {"visualisations": tile}
