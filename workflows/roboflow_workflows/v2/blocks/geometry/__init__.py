"""Region and parent-mapping blocks: stitch, offset, rotated rectangles,
class replacement from crops and collapse of a nesting axis.

``create_catalogue()`` registers every block in ``BLOCKS``.
"""

from roboflow_workflows.v2.blocks.geometry.bounding_rect import BoundingRectBlock
from roboflow_workflows.v2.blocks.geometry.class_replacement import (
    DetectionsClassesReplacementBlock,
)
from roboflow_workflows.v2.blocks.geometry.collapse import DimensionCollapseBlock
from roboflow_workflows.v2.blocks.geometry.offset import DetectionOffsetBlock
from roboflow_workflows.v2.blocks.geometry.stitch import DetectionsStitchBlock

BLOCKS = (
    DetectionsStitchBlock,
    DetectionOffsetBlock,
    BoundingRectBlock,
    DetectionsClassesReplacementBlock,
    DimensionCollapseBlock,
)

__all__ = [
    "BLOCKS",
    "BoundingRectBlock",
    "DetectionOffsetBlock",
    "DetectionsClassesReplacementBlock",
    "DetectionsStitchBlock",
    "DimensionCollapseBlock",
]
