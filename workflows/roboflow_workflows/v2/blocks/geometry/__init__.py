"""Region and parent-mapping blocks: stitch, offset, rotated rectangles,
class replacement from crops and collapse of a nesting axis.

``create_catalogue()`` registers every block in ``BLOCKS``.
"""

from roboflow_workflows.v2.blocks.geometry.bounding_rect.v1 import BoundingRectBlock
from roboflow_workflows.v2.blocks.geometry.detections_classes_replacement.v1 import DetectionsClassesReplacementBlock
from roboflow_workflows.v2.blocks.geometry.dimension_collapse.v1 import DimensionCollapseBlock
from roboflow_workflows.v2.blocks.geometry.detection_offset.v1 import DetectionOffsetBlock
from roboflow_workflows.v2.blocks.geometry.detections_stitch.v1 import DetectionsStitchBlock

from roboflow_workflows.v2.blocks.geometry.camera_calibration.v1 import CameraCalibrationBlock
from roboflow_workflows.v2.blocks.geometry.contours_detection.v1 import ContoursDetectionBlock
from roboflow_workflows.v2.blocks.geometry.dynamic_zone.v1 import DynamicZoneBlock
from roboflow_workflows.v2.blocks.geometry.geotag_detection.v1 import GeotagDetectionBlock
from roboflow_workflows.v2.blocks.geometry.detections_nearest_neighbor.v1 import DetectionsNearestNeighborBlock
from roboflow_workflows.v2.blocks.geometry.perspective_correction.v1 import PerspectiveCorrectionBlock
from roboflow_workflows.v2.blocks.geometry.sift.v1 import SiftBlock
from roboflow_workflows.v2.blocks.geometry.sift_comparison.v2 import SiftComparisonBlock
from roboflow_workflows.v2.blocks.geometry.template_matching.v1 import TemplateMatchingBlock

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
