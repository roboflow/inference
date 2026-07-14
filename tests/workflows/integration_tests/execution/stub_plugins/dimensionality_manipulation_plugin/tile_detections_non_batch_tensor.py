"""
Tensor-native sibling of ``tile_detections_non_batch.py``: native ``Detections`` is
converted via ``.to_supervision()`` only to feed ``sv.BoxAnnotator``; the block
outputs image tiles, not detections.

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
