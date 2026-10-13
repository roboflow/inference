"""Prediction transform blocks: UQL filter/transformation and per-class confidence.

``create_catalogue()`` registers every block in ``BLOCKS``.
"""

from roboflow_workflows.v2.blocks.transforms.operations_chain import (
    DetectionsFilterBlock,
    DetectionsTransformationBlock,
)
from roboflow_workflows.v2.blocks.transforms.per_class_confidence import (
    PerClassConfidenceFilterBlock,
)

from roboflow_workflows.v2.blocks.transforms.similarity import (
    CosineSimilarityBlock,
)

BLOCKS = (
    DetectionsFilterBlock,
    DetectionsTransformationBlock,
    PerClassConfidenceFilterBlock,
    CosineSimilarityBlock,
)

__all__ = [
    "BLOCKS",
    "DetectionsFilterBlock",
    "DetectionsTransformationBlock",
    "PerClassConfidenceFilterBlock",
    "CosineSimilarityBlock",
]
