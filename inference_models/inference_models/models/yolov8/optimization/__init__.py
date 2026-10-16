"""Selectable postprocessing for YOLOv8 TensorRT object detection.

Importing this package does not import Triton.
"""

from .base import BasePostprocessor
from .contracts import PostprocessRequest, PostprocessResult
from .registry import (
    create_postprocess_registry,
    postprocess_runtime_components,
    select_postprocessor,
    validate_postprocess_implementation,
)
from .segmented import KernelInitialization, SegmentedPostprocessor, decode_padded

__all__ = [
    "BasePostprocessor",
    "KernelInitialization",
    "PostprocessRequest",
    "PostprocessResult",
    "SegmentedPostprocessor",
    "create_postprocess_registry",
    "decode_padded",
    "postprocess_runtime_components",
    "select_postprocessor",
    "validate_postprocess_implementation",
]
