"""V2 ports of the latest V1 visualization blocks, plus a semantic grid painter.

=============================================  ==============================================
V2 type                                        port of
=============================================  ==============================================
v2/bounding_box_visualization@v1               roboflow_core/bounding_box_visualization@v1
v2/label_visualization@v2                      roboflow_core/label_visualization@v2
v2/polygon_visualization@v2                    roboflow_core/polygon_visualization@v2
v2/mask_visualization@v1                       roboflow_core/mask_visualization@v1
v2/keypoint_visualization@v1                   roboflow_core/keypoint_visualization@v1
v2/classification_label_visualization@v1       roboflow_core/classification_label_visualization@v1
v2/crop_visualization@v1                       roboflow_core/crop_visualization@v1
v2/blur_visualization@v1                       roboflow_core/blur_visualization@v1
v2/pixelate_visualization@v1                   roboflow_core/pixelate_visualization@v1
v2/semantic_segmentation_visualization@v1      (new) native semantic segmentation grids
v2/circle_visualization@v1                     roboflow_core/circle_visualization@v1
v2/corner_visualization@v1                     roboflow_core/corner_visualization@v1
v2/dot_visualization@v1                        roboflow_core/dot_visualization@v1
v2/ellipse_visualization@v1                    roboflow_core/ellipse_visualization@v1
v2/halo_visualization@v2                       roboflow_core/halo_visualization@v2
v2/triangle_visualization@v1                   roboflow_core/triangle_visualization@v1
v2/heatmap_visualization@v1                   roboflow_core/heatmap_visualization@v1
v2/trace_visualization@v1                     roboflow_core/trace_visualization@v1
v2/polygon_zone_visualization@v1              roboflow_core/polygon_zone_visualization@v1
v2/line_counter_visualization@v1              roboflow_core/line_counter_visualization@v1
v2/reference_path_visualization@v1            roboflow_core/reference_path_visualization@v1
v2/background_color_visualization@v1          roboflow_core/background_color_visualization@v1
v2/color_visualization@v1                     roboflow_core/color_visualization@v1
v2/grid_visualization@v1                      roboflow_core/grid_visualization@v1
=============================================  ==============================================

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
``LabelVisualizationBlock`` stays the ``v2/label_visualization@v1`` prototype.

The older painters in ``roboflow_workflows.v2.blocks.visualization``
(``v2/box_visualization@v1``, ``v2/label_visualization@v1`` and friends) are
unchanged and remain available. They are limited fast prototypes, not V1
ports: for V1 options use ``v2/bounding_box_visualization@v1`` and
``v2/label_visualization@v2``.
"""

from roboflow_workflows.v2.blocks.visualization.annotations.blur import (
    BlurVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.bounding_box import (
    BoundingBoxVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.circle import (
    CircleVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.classification_label import (
    ClassificationLabelVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.corner import (
    CornerVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.crop import (
    CropVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.dot import (
    DotVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.ellipse import (
    EllipseVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.halo import (
    HaloVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.keypoint import (
    KeypointVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.label import (
    LabelVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.mask import (
    MaskVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.pixelate import (
    PixelateVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.polygon import (
    PolygonVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.semantic_segmentation import (
    SemanticSegmentationVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.triangle import (
    TriangleVisualizationBlock,
)

from roboflow_workflows.v2.blocks.visualization.annotations.heatmap import (
    HeatmapVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.trace import (
    TraceVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.zones import (
    LineCounterVisualizationBlock,
    PolygonZoneVisualizationBlock,
    ReferencePathVisualizationBlock,
)

from roboflow_workflows.v2.blocks.visualization.annotations.background_color import (
    BackgroundColorVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.color import (
    ColorVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.grid import (
    GridVisualizationBlock,
)

from roboflow_workflows.v2.blocks.visualization.annotations.icon import (
    IconVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.model_comparison import (
    ModelComparisonVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.rich_label import (
    RichLabelVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.text_display import (
    TextDisplayBlock,
)

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
