"""
Tensor-native sibling of ``tile_detections_batch.py``.

Drawing is inherently numpy (``sv.BoxAnnotator``), so the native ``Detections`` is
converted to ``sv.Detections`` via ``.to_supervision()`` ONLY to feed ``annotate()``.
The output is the IMAGE tiles (unchanged) — this conversion is acceptable because it
feeds a numpy renderer, not a detection output. The ``BlockManifest`` is reused
verbatim from the numpy module.

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
from tests.workflows.integration_tests.execution.stub_plugins.dimensionality_manipulation_plugin.tile_detections_batch import (
    BlockManifest,
)
from inference_models.models.base.object_detection import Detections

import supervision as sv


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
