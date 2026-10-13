"""Deprecated import path for ``roboflow_workflows.v2.blocks.visualization``.

Kept so existing V2 alpha callers keep working; it re-exports the same
objects. Import from the canonical module instead.
"""

from roboflow_workflows.v2.blocks.visualization import (
    BLOCKS,
    DETECTION_DRAWING_KIND,
    KINDS,
    BoxVisualizationBlock,
    DetectionDrawing,
    DetectionDrawingBlock,
    DetectionVisualizationBlock,
    InPlaceLabelVisualizationBlock,
    LabelVisualizationBlock,
    prepare_drawings,
)

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
