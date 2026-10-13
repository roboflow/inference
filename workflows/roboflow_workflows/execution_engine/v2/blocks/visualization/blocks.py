"""Deprecated import path for ``roboflow_workflows.v2.blocks.visualization.blocks``.

Kept so existing V2 alpha callers keep working; it re-exports the same
objects. Import from the canonical module instead.
"""

from roboflow_workflows.v2.blocks.visualization.blocks import (
    BLOCKS,
    BoxVisualizationBlock,
    DetectionDrawingBlock,
    DetectionVisualizationBlock,
    InPlaceLabelVisualizationBlock,
    LabelVisualizationBlock,
)

__all__ = [
    "BLOCKS",
    "BoxVisualizationBlock",
    "DetectionDrawingBlock",
    "DetectionVisualizationBlock",
    "InPlaceLabelVisualizationBlock",
    "LabelVisualizationBlock",
]
