from typing import List, Type

from inference.core.workflows.prototypes.block import WorkflowBlock
from tests.workflows.integration_tests.dimensionality_manipulation_plugin.detections_to_parent_coordinates_batch import (
    DetectionsToParentCoordinatesBatchBlock,
)
from tests.workflows.integration_tests.dimensionality_manipulation_plugin.detections_to_parent_coordinates_non_batch import (
    DetectionsToParentCoordinatesNonBatchBlock,
)


def load_blocks() -> List[Type[WorkflowBlock]]:
    return [
        DetectionsToParentCoordinatesBatchBlock,
        DetectionsToParentCoordinatesNonBatchBlock,
    ]
