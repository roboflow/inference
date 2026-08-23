"""YOLO26 depth-estimation inference-path alternatives."""

from inference_models.models.yolo26.optimization.execution_plan import (
    YOLO26DepthExecutionPlan,
)
from inference_models.models.yolo26.optimization.ids import (
    YOLO26_DEPTH_POSTPROCESSOR_BASE,
    YOLO26_DEPTH_POSTPROCESSOR_TRITON_AA_RESIZE_EXACT_V2,
    YOLO26_DEPTH_POSTPROCESSOR_TRITON_AA_RESIZE_V1,
)

__all__ = [
    "YOLO26DepthExecutionPlan",
    "YOLO26_DEPTH_POSTPROCESSOR_BASE",
    "YOLO26_DEPTH_POSTPROCESSOR_TRITON_AA_RESIZE_V1",
    "YOLO26_DEPTH_POSTPROCESSOR_TRITON_AA_RESIZE_EXACT_V2",
]
