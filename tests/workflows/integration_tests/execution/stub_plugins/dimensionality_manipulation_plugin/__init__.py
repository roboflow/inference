from typing import List, Type

from inference.core.workflows.prototypes.block import WorkflowBlock
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


def load_blocks() -> List[Type[WorkflowBlock]]:
    return [
        DetectionsToParentCoordinatesBatchBlock,
        DetectionsToParentCoordinatesNonBatchBlock,
        StitchDetectionsBatchBlock,
        StitchDetectionsNonBatchBlock,
        TileDetectionsBatchBlock,
        TileDetectionsNonBatchBlock,
    ]
