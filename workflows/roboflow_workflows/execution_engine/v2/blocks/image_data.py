"""Deprecated import path for ``roboflow_workflows.v2.data.image``.

Kept so existing V2 alpha callers keep working; it re-exports the same
objects. Import from the canonical module instead.
"""

from roboflow_workflows.v2.data.image import (
    INTERPOLATIONS,
    SUPPORTED_CHANNELS,
    CompositeSource,
    FrameMapping,
    ImageData,
    Pair,
    Rectangle,
    SizeHW,
)

__all__ = [
    "CompositeSource",
    "FrameMapping",
    "INTERPOLATIONS",
    "ImageData",
    "Pair",
    "Rectangle",
    "SUPPORTED_CHANNELS",
    "SizeHW",
]
