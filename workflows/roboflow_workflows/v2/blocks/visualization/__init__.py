"""Opt-in batched detection painters for V2 workflows.

Five fast prototype blocks in ``native`` (see ``_shared.blocks`` for routes
and costs). They are not V1 ports; the 27 V1 ports and the semantic
segmentation painter live in ``annotations``. ``create_catalogue()``
registers both sets::

    roboflow_core/detection_drawing@v1                    predictions -> drawing (one host transfer)
    roboflow_core/simple_box_visualization@v1             image, predictions -> new image with boxes
    roboflow_core/simple_label_visualization@v1           image, predictions -> new image with labels
    roboflow_core/detection_visualization@v1              image, predictions -> new image, both (fused)
    roboflow_core/simple_label_visualization_in_place@v1  labels on a workflow-created image (mutates)

``predictions`` of a painter is native ``Detections`` or a ``DetectionDrawing``.
``BLOCKS`` and ``KINDS`` are what a catalogue needs::

    Catalogue(visualization.BLOCKS, kinds=[*visualization.KINDS, ...])

Importing this package loads torch, supervision and the V1 tensor drawing
primitives.
"""

from roboflow_workflows.v2.blocks.visualization._catalogue_blocks import BLOCKS
from roboflow_workflows.v2.blocks.visualization.native.simple_box_visualization.v1 import BoxVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.native.detection_drawing.v1 import DetectionDrawingBlock
from roboflow_workflows.v2.blocks.visualization.native.detection_visualization.v1 import DetectionVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.native.simple_label_visualization_in_place.v1 import InPlaceLabelVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.native.simple_label_visualization.v1 import LabelVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.drawing import (
    DETECTION_DRAWING_KIND,
    DetectionDrawing,
    prepare_drawings,
)
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.predictions import (
    OBJECT_DETECTION_PREDICTION_KIND,
)

KINDS = (IMAGE_KIND, OBJECT_DETECTION_PREDICTION_KIND, DETECTION_DRAWING_KIND)

__all__ = [
    "BLOCKS",
    "BoxVisualizationBlock",
    "DETECTION_DRAWING_KIND",
    "DetectionDrawing",
    "DetectionDrawingBlock",
    "DetectionVisualizationBlock",
    "InPlaceLabelVisualizationBlock",
    "KINDS",
    "LabelVisualizationBlock",
    "prepare_drawings",
]
