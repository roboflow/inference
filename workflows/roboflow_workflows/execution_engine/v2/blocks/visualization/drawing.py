"""Deprecated import path for ``roboflow_workflows.v2.blocks.visualization.drawing``.

Kept so existing V2 alpha callers keep working; it re-exports the same
objects. Import from the canonical module instead.
"""

from roboflow_workflows.v2.blocks.visualization.drawing import (
    BOX_THICKNESS,
    DETECTION_DRAWING_KIND,
    PALETTE,
    DetectionDrawing,
    prepare_drawings,
)

__all__ = [
    "BOX_THICKNESS",
    "DETECTION_DRAWING_KIND",
    "DetectionDrawing",
    "PALETTE",
    "prepare_drawings",
]
