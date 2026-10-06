"""
Tensor-native sibling of ``tile_detections_batch.py``: native ``Detections`` is
converted via ``.to_supervision()`` only to feed ``sv.BoxAnnotator``; the block
outputs image tiles, not detections.

This is just example, test implementation, please do not assume it being fully functional.
"""

from typing import Type

import supervision as sv

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
from inference_models.models.base.object_detection import Detections
from tests.workflows.integration_tests.execution.stub_plugins.dimensionality_manipulation_plugin.tile_detections_batch import (
    BlockManifest,
)


class TileDetectionsBatchBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        images_crops: Batch[Batch[WorkflowImageData]],
        crops_predictions: Batch[Batch[Detections]],
    ) -> BlockResult:
        annotator = sv.BoxAnnotator()
        visualisations = []
        for image_crops, crop_predictions in zip(images_crops, crops_predictions):
            visualisations_batch_element = []
            for image, prediction in zip(image_crops, crop_predictions):
                annotated_image = annotator.annotate(
                    image.numpy_image.copy(),
                    prediction.to_supervision(),
                )
                visualisations_batch_element.append(annotated_image)
            tile = create_tiles(visualisations_batch_element)
            visualisations.append({"visualisations": tile})
        return visualisations
