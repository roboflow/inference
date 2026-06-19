from typing import List, Type

from inference.core.env import ENABLE_TENSOR_DATA_REPRESENTATION
from inference.core.workflows.prototypes.block import WorkflowBlock

# numpy / sv-shaped blocks (the original stub implementations)
from tests.workflows.integration_tests.execution.stub_plugins.dimensionality_manipulation_plugin.detections_to_parent_coordinates_batch import (
    DetectionsToParentCoordinatesBatchBlock,
)
from tests.workflows.integration_tests.execution.stub_plugins.dimensionality_manipulation_plugin.detections_to_parent_coordinates_non_batch import (
    DetectionsToParentCoordinatesNonBatchBlock,
)
from tests.workflows.integration_tests.execution.stub_plugins.dimensionality_manipulation_plugin.stitch_detections_batch import (
    StitchDetectionsBatchBlock,
)
from tests.workflows.integration_tests.execution.stub_plugins.dimensionality_manipulation_plugin.stitch_detections_non_batch import (
    StitchDetectionsNonBatchBlock,
)
from tests.workflows.integration_tests.execution.stub_plugins.dimensionality_manipulation_plugin.tile_detections_batch import (
    TileDetectionsBatchBlock,
)
from tests.workflows.integration_tests.execution.stub_plugins.dimensionality_manipulation_plugin.tile_detections_non_batch import (
    TileDetectionsNonBatchBlock,
)

# tensor-native siblings (operate on inference_models.Detections)
from tests.workflows.integration_tests.execution.stub_plugins.dimensionality_manipulation_plugin.detections_to_parent_coordinates_batch_tensor import (
    DetectionsToParentCoordinatesBatchBlock as DetectionsToParentCoordinatesBatchBlockTensor,
)
from tests.workflows.integration_tests.execution.stub_plugins.dimensionality_manipulation_plugin.detections_to_parent_coordinates_non_batch_tensor import (
    DetectionsToParentCoordinatesNonBatchBlock as DetectionsToParentCoordinatesNonBatchBlockTensor,
)
from tests.workflows.integration_tests.execution.stub_plugins.dimensionality_manipulation_plugin.stitch_detections_batch_tensor import (
    StitchDetectionsBatchBlock as StitchDetectionsBatchBlockTensor,
)
from tests.workflows.integration_tests.execution.stub_plugins.dimensionality_manipulation_plugin.stitch_detections_non_batch_tensor import (
    StitchDetectionsNonBatchBlock as StitchDetectionsNonBatchBlockTensor,
)
from tests.workflows.integration_tests.execution.stub_plugins.dimensionality_manipulation_plugin.tile_detections_batch_tensor import (
    TileDetectionsBatchBlock as TileDetectionsBatchBlockTensor,
)
from tests.workflows.integration_tests.execution.stub_plugins.dimensionality_manipulation_plugin.tile_detections_non_batch_tensor import (
    TileDetectionsNonBatchBlock as TileDetectionsNonBatchBlockTensor,
)


def load_blocks() -> List[Type[WorkflowBlock]]:
    if ENABLE_TENSOR_DATA_REPRESENTATION:
        return [
            DetectionsToParentCoordinatesBatchBlockTensor,
            DetectionsToParentCoordinatesNonBatchBlockTensor,
            StitchDetectionsBatchBlockTensor,
            StitchDetectionsNonBatchBlockTensor,
            TileDetectionsBatchBlockTensor,
            TileDetectionsNonBatchBlockTensor,
        ]
    return [
        DetectionsToParentCoordinatesBatchBlock,
        DetectionsToParentCoordinatesNonBatchBlock,
        StitchDetectionsBatchBlock,
        StitchDetectionsNonBatchBlock,
        TileDetectionsBatchBlock,
        TileDetectionsNonBatchBlock,
    ]
