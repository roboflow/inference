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

Not in this slice (queued for later waves): background_color, circle, color,
corner, dot, ellipse, grid, halo@v2, heatmap, icon, line_counter,
model_comparison, polygon_zone, reference_path, rich_label, text_display,
trace and triangle visualizations (18 V1 families).

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
from roboflow_workflows.v2.blocks.visualization.annotations.classification_label import (
    ClassificationLabelVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.crop import (
    CropVisualizationBlock,
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
)

__all__ = [
    "BLOCKS",
    "BlurVisualizationBlock",
    "BoundingBoxVisualizationBlock",
    "ClassificationLabelVisualizationBlock",
    "CropVisualizationBlock",
    "KeypointVisualizationBlock",
    "LabelVisualizationBlock",
    "MaskVisualizationBlock",
    "PixelateVisualizationBlock",
    "PolygonVisualizationBlock",
    "SemanticSegmentationVisualizationBlock",
]
