"""Canonical V2 visualization blocks.

Bounding boxes and labels offer accurate and fast implementations under their
existing block identifiers. All blocks accept tensor-backed images; there is
one catalogue entry per visualization capability.

Accurate rendering is the default. Select the fast implementation at compile
time through the workflow definition, without changing either block's type::

    "execution": {
        "step_quality": {"$steps.boxes": "fast", "$steps.labels": "fast"}
    }

Fast square boxes use approximate outline rasterization; rounded boxes use
accurate rendering. Fast labels cache opaque
sprites and fall back to exact patches for other styles. Both retain the full
parameter contract and return a copy; they do not mutate the input image.
"""

from roboflow_workflows.v2.blocks.visualization.annotations import (
    BLOCKS,
    BoundingBoxVisualizationBlock,
    LabelVisualizationBlock,
)

__all__ = ["BLOCKS", "BoundingBoxVisualizationBlock", "LabelVisualizationBlock"]
