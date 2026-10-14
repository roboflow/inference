"""Prediction transform blocks: UQL filter/transformation and per-class confidence.

``create_catalogue()`` registers every block in ``BLOCKS``.
"""

from roboflow_workflows.v2.blocks.transforms.detections_filter.v1 import DetectionsFilterBlock
from roboflow_workflows.v2.blocks.transforms.detections_transformation.v1 import DetectionsTransformationBlock
from roboflow_workflows.v2.blocks.transforms.per_class_confidence_filter.v1 import PerClassConfidenceFilterBlock

from roboflow_workflows.v2.blocks.transforms.cosine_similarity.v1 import CosineSimilarityBlock

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
