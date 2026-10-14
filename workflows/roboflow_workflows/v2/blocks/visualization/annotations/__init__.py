"""V2 ports of the latest V1 visualization blocks, plus a semantic grid painter.

====================================================  ===================================================
V2 type                                               port of
====================================================  ===================================================
roboflow_core/bounding_box_visualization@v1           roboflow_core/bounding_box_visualization@v1
roboflow_core/label_visualization@v2                  roboflow_core/label_visualization@v2
roboflow_core/polygon_visualization@v2                roboflow_core/polygon_visualization@v2
roboflow_core/mask_visualization@v1                   roboflow_core/mask_visualization@v1
roboflow_core/keypoint_visualization@v1               roboflow_core/keypoint_visualization@v1
roboflow_core/classification_label_visualization@v1   roboflow_core/classification_label_visualization@v1
roboflow_core/crop_visualization@v1                   roboflow_core/crop_visualization@v1
roboflow_core/blur_visualization@v1                   roboflow_core/blur_visualization@v1
roboflow_core/pixelate_visualization@v1               roboflow_core/pixelate_visualization@v1
roboflow_core/semantic_segmentation_visualization@v1  (new) native semantic segmentation grids
roboflow_core/circle_visualization@v1                 roboflow_core/circle_visualization@v1
roboflow_core/corner_visualization@v1                 roboflow_core/corner_visualization@v1
roboflow_core/dot_visualization@v1                    roboflow_core/dot_visualization@v1
roboflow_core/ellipse_visualization@v1                roboflow_core/ellipse_visualization@v1
roboflow_core/halo_visualization@v2                   roboflow_core/halo_visualization@v2
roboflow_core/triangle_visualization@v1               roboflow_core/triangle_visualization@v1
roboflow_core/heatmap_visualization@v1                roboflow_core/heatmap_visualization@v1
roboflow_core/trace_visualization@v1                  roboflow_core/trace_visualization@v1
roboflow_core/polygon_zone_visualization@v1           roboflow_core/polygon_zone_visualization@v1
roboflow_core/line_counter_visualization@v1           roboflow_core/line_counter_visualization@v1
roboflow_core/reference_path_visualization@v1         roboflow_core/reference_path_visualization@v1
roboflow_core/background_color_visualization@v1       roboflow_core/background_color_visualization@v1
roboflow_core/color_visualization@v1                  roboflow_core/color_visualization@v1
roboflow_core/grid_visualization@v1                   roboflow_core/grid_visualization@v1
roboflow_core/icon_visualization@v1                   roboflow_core/icon_visualization@v1
roboflow_core/model_comparison_visualization@v1       roboflow_core/model_comparison_visualization@v1
roboflow_core/rich_label_visualization@v1             roboflow_core/rich_label_visualization@v1
roboflow_core/text_display@v1                         roboflow_core/text_display@v1
====================================================  ===================================================

Each block runs once per image on the image's device and returns a new
image with the input's identity, frame, source and time; inputs are never
changed. Box readers accept detections, instance segmentation (dense, RLE or
a mask view, which they never upscale) and keypoints with boxes. Mask readers
take ``mask_quality`` (``accurate`` or ``coarse``; see ``rows``). Where work
runs, and what equals V1 pixel for pixel, is stated in each block's module.

V1 ``copy_image`` is rejected with either value; chain blocks with
``image: $steps.<previous>.image`` instead. Predictions of another frame of
the same workflow input are mapped into the drawn image and clipped to it.

Retained annotations are selected explicitly by this package; retired siblings
are excluded from default discovery.

``create_catalogue()`` registers every block in ``BLOCKS``. The package
``roboflow_workflows.v2.blocks`` exports this package's
``LabelVisualizationBlock`` as ``AnnotationLabelVisualizationBlock``; its own
``LabelVisualizationBlock`` stays the ``roboflow_core/simple_label_visualization@v1`` prototype.

The older painters in ``roboflow_workflows.v2.blocks.visualization``
(``roboflow_core/simple_box_visualization@v1``, ``roboflow_core/simple_label_visualization@v1`` and friends) are
unchanged and remain available. They are limited fast prototypes, not V1
ports: for V1 options use ``roboflow_core/bounding_box_visualization@v1`` and
``roboflow_core/label_visualization@v2``.
"""

from roboflow_workflows.v2.blocks.visualization.annotations.blur_visualization.v1 import BlurVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.annotations.bounding_box_visualization.v1 import BoundingBoxVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.annotations.circle_visualization.v1 import CircleVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.annotations.classification_label_visualization.v1 import ClassificationLabelVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.annotations.corner_visualization.v1 import CornerVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.annotations.crop_visualization.v1 import CropVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.annotations.dot_visualization.v1 import DotVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.annotations.ellipse_visualization.v1 import EllipseVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.annotations.halo_visualization.v2 import HaloVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.annotations.keypoint_visualization.v1 import KeypointVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.annotations.label_visualization.v2 import LabelVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.annotations.mask_visualization.v1 import MaskVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.annotations.pixelate_visualization.v1 import PixelateVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.annotations.polygon_visualization.v2 import PolygonVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.annotations.semantic_segmentation_visualization.v1 import SemanticSegmentationVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.annotations.triangle_visualization.v1 import TriangleVisualizationBlock

from roboflow_workflows.v2.blocks.visualization.annotations.heatmap_visualization.v1 import HeatmapVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.annotations.trace_visualization.v1 import TraceVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.annotations.line_counter_visualization.v1 import LineCounterVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.annotations.polygon_zone_visualization.v1 import PolygonZoneVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.annotations.reference_path_visualization.v1 import ReferencePathVisualizationBlock

from roboflow_workflows.v2.blocks.visualization.annotations.background_color_visualization.v1 import BackgroundColorVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.annotations.color_visualization.v1 import ColorVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.annotations.grid_visualization.v1 import GridVisualizationBlock

from roboflow_workflows.v2.blocks.visualization.annotations.icon_visualization.v1 import IconVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.annotations.model_comparison_visualization.v1 import ModelComparisonVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.annotations.rich_label_visualization.v1 import RichLabelVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.annotations.text_display.v1 import TextDisplayBlock

BLOCKS = (
    BoundingBoxVisualizationBlock,
    LabelVisualizationBlock,
    PolygonVisualizationBlock,
    MaskVisualizationBlock,
    KeypointVisualizationBlock,
    ClassificationLabelVisualizationBlock,
    CropVisualizationBlock,
    BlurVisualizationBlock,
    PixelateVisualizationBlock,
    SemanticSegmentationVisualizationBlock,
    CircleVisualizationBlock,
    CornerVisualizationBlock,
    DotVisualizationBlock,
    EllipseVisualizationBlock,
    HaloVisualizationBlock,
    TriangleVisualizationBlock,
    HeatmapVisualizationBlock,
    TraceVisualizationBlock,
    PolygonZoneVisualizationBlock,
    LineCounterVisualizationBlock,
    ReferencePathVisualizationBlock,
    BackgroundColorVisualizationBlock,
    ColorVisualizationBlock,
    GridVisualizationBlock,
    IconVisualizationBlock,
    RichLabelVisualizationBlock,
    ModelComparisonVisualizationBlock,
    TextDisplayBlock,
)

__all__ = [
    "BLOCKS",
    "BlurVisualizationBlock",
    "BoundingBoxVisualizationBlock",
    "CircleVisualizationBlock",
    "ClassificationLabelVisualizationBlock",
    "CornerVisualizationBlock",
    "CropVisualizationBlock",
    "DotVisualizationBlock",
    "EllipseVisualizationBlock",
    "HaloVisualizationBlock",
    "KeypointVisualizationBlock",
    "LabelVisualizationBlock",
    "MaskVisualizationBlock",
    "PixelateVisualizationBlock",
    "PolygonVisualizationBlock",
    "SemanticSegmentationVisualizationBlock",
    "TriangleVisualizationBlock",
    "HeatmapVisualizationBlock",
    "TraceVisualizationBlock",
    "PolygonZoneVisualizationBlock",
    "LineCounterVisualizationBlock",
    "ReferencePathVisualizationBlock",
    "BackgroundColorVisualizationBlock",
    "ColorVisualizationBlock",
    "GridVisualizationBlock",
    "IconVisualizationBlock",
    "RichLabelVisualizationBlock",
    "ModelComparisonVisualizationBlock",
    "TextDisplayBlock",
]
