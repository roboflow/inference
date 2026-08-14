"""
Tensor-native mirrors of the image-shape-reading helpers in
`common/utils.py`. Created per the plan's Step 5b.

The functions here have the same logical contract as their numpy
counterparts. They are currently thin wrappers that delegate to the
numpy implementations because the underlying `attach_parent_coordinates_to_detections`
reads `ImageParentMetadata.origin_coordinates` (populated by
`WorkflowImageData.parent_metadata` /
`workflow_root_ancestor_metadata`, which already use
`_read_shape_without_materialization` to avoid forcing
device->host on tensor-only inputs). Keeping the mirror in its own
module gives tensor-specific optimisations a landing spot that the
loader can swap to without touching the numpy file.
"""

from typing import Iterable, List

import supervision as sv

from inference.core.workflows.core_steps.common.utils import (
    attach_parent_coordinates_to_detections,
    attach_parents_coordinates_to_batch_of_sv_detections,
    attach_parents_coordinates_to_sv_detections,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


def attach_parents_coordinates_to_batch_of_sv_detections_tensor(
    predictions: List[sv.Detections],
    images: Iterable[WorkflowImageData],
) -> List[sv.Detections]:
    return attach_parents_coordinates_to_batch_of_sv_detections(
        predictions=predictions, images=images
    )


def attach_parents_coordinates_to_sv_detections_tensor(
    detections: sv.Detections,
    image: WorkflowImageData,
) -> sv.Detections:
    return attach_parents_coordinates_to_sv_detections(
        detections=detections, image=image
    )


def attach_parent_coordinates_to_detections_tensor(
    detections: sv.Detections,
    parent_metadata: ImageParentMetadata,
    parent_id_key: str,
    coordinates_key: str,
    dimensions_key: str,
) -> sv.Detections:
    return attach_parent_coordinates_to_detections(
        detections=detections,
        parent_metadata=parent_metadata,
        parent_id_key=parent_id_key,
        coordinates_key=coordinates_key,
        dimensions_key=dimensions_key,
    )


__all__ = [
    "attach_parents_coordinates_to_batch_of_sv_detections_tensor",
    "attach_parents_coordinates_to_sv_detections_tensor",
    "attach_parent_coordinates_to_detections_tensor",
]
