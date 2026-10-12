"""Opt-in batched detection painters for V2 workflows.

Five blocks (see ``blocks`` for routes and costs)::

    v2/detection_drawing@v1              predictions -> drawing (one host transfer)
    v2/box_visualization@v1              image, predictions -> new image with boxes
    v2/label_visualization@v1            image, predictions -> new image with labels
    v2/detection_visualization@v1        image, predictions -> new image, both (fused)
    v2/label_visualization_in_place@v1   labels on a workflow-created image (mutates)

``predictions`` of a painter is native ``Detections`` or a ``DetectionDrawing``.
``BLOCKS`` and ``KINDS`` are what a catalogue needs::

    Catalogue(visualization.BLOCKS, kinds=[*visualization.KINDS, ...])

Importing this package loads torch, supervision and the V1 tensor drawing
primitives, never Triton: the ``triton`` backend imports it on first use.
"""

from roboflow_workflows.execution_engine.v2.blocks.kinds import IMAGE_KIND
from roboflow_workflows.execution_engine.v2.blocks.predictions import (
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.execution_engine.v2.blocks.visualization.blocks import (
    BLOCKS,
    BoxVisualizationBlock,
    DetectionDrawingBlock,
    DetectionVisualizationBlock,
    InPlaceLabelVisualizationBlock,
    LabelVisualizationBlock,
)
from roboflow_workflows.execution_engine.v2.blocks.visualization.drawing import (
    DETECTION_DRAWING_KIND,
    DetectionDrawing,
    prepare_drawings,
)
from roboflow_workflows.execution_engine.v2.blocks.visualization.kernels import (
    BACKENDS,
    DrawingBackendUnavailable,
)

KINDS = (IMAGE_KIND, OBJECT_DETECTION_PREDICTION_KIND, DETECTION_DRAWING_KIND)

__all__ = [
    "BACKENDS",
    "BLOCKS",
    "BoxVisualizationBlock",
    "DETECTION_DRAWING_KIND",
    "DetectionDrawing",
    "DetectionDrawingBlock",
    "DetectionVisualizationBlock",
    "DrawingBackendUnavailable",
    "InPlaceLabelVisualizationBlock",
    "KINDS",
    "LabelVisualizationBlock",
    "prepare_drawings",
]
