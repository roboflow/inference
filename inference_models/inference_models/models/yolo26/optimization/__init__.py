"""YOLO26 inference-path optimization contracts."""

from inference_models.models.yolo26.optimization.execution_plan import (
    YOLO26DepthOnnxExecutionPlan,
)
from inference_models.models.yolo26.optimization.ids import (
    YOLO26_DEPTH_ONNX_SCHEDULER_BASE,
    YOLO26_DEPTH_ONNX_SCHEDULER_ENV_NAME,
    YOLO26_DEPTH_ONNX_SCHEDULER_ORT_CUDA_GRAPH_V1,
)

__all__ = [
    "YOLO26DepthOnnxExecutionPlan",
    "YOLO26_DEPTH_ONNX_SCHEDULER_BASE",
    "YOLO26_DEPTH_ONNX_SCHEDULER_ENV_NAME",
    "YOLO26_DEPTH_ONNX_SCHEDULER_ORT_CUDA_GRAPH_V1",
]
