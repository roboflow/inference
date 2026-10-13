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

from roboflow_workflows.v2.blocks.geometry.calibration import (
    CameraCalibrationBlock,
)
from roboflow_workflows.v2.blocks.geometry.contours import (
    ContoursDetectionBlock,
)
from roboflow_workflows.v2.blocks.geometry.dynamic_zone import (
    DynamicZoneBlock,
)
from roboflow_workflows.v2.blocks.geometry.geotag import (
    GeotagDetectionBlock,
)
from roboflow_workflows.v2.blocks.geometry.nearest_neighbor import (
    DetectionsNearestNeighborBlock,
)
from roboflow_workflows.v2.blocks.geometry.perspective import (
    PerspectiveCorrectionBlock,
)
from roboflow_workflows.v2.blocks.geometry.sift import (
    SiftBlock,
)
from roboflow_workflows.v2.blocks.geometry.sift_comparison import (
    SiftComparisonBlock,
)
from roboflow_workflows.v2.blocks.geometry.template_matching import (
    TemplateMatchingBlock,
)

BLOCKS = (
    DetectionsStitchBlock,
    DetectionOffsetBlock,
    BoundingRectBlock,
    DetectionsClassesReplacementBlock,
    DimensionCollapseBlock,
    SiftBlock,
    PerspectiveCorrectionBlock,
    TemplateMatchingBlock,
    GeotagDetectionBlock,
    DetectionsNearestNeighborBlock,
    SiftComparisonBlock,
    CameraCalibrationBlock,
    ContoursDetectionBlock,
    DynamicZoneBlock,
)

__all__ = [
    "BLOCKS",
    "BoundingRectBlock",
    "DetectionOffsetBlock",
    "DetectionsClassesReplacementBlock",
    "DetectionsStitchBlock",
    "DimensionCollapseBlock",
    "SiftBlock",
    "PerspectiveCorrectionBlock",
    "TemplateMatchingBlock",
    "GeotagDetectionBlock",
    "DetectionsNearestNeighborBlock",
    "SiftComparisonBlock",
    "CameraCalibrationBlock",
    "ContoursDetectionBlock",
    "DynamicZoneBlock",
]
