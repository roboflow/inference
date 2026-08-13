"""YOLO26 depth ONNX scheduler registry construction."""

import torch

from inference_models.models.optimization.registry import ImplementationRegistry
from inference_models.models.yolo26.optimization.schedulers import (
    BaseOnnxExecutionScheduler,
    OrtCudaGraphExecutionScheduler,
    build_base_scheduler_metadata,
)


def build_yolo26_depth_onnx_scheduler_registry(
    *,
    session,
    input_name: str,
    input_batch_size,
    device: torch.device,
) -> ImplementationRegistry:
    """Build the explicit scheduler catalog for one loaded model."""
    registry = ImplementationRegistry(scope_name="YOLO26 depth ONNX")
    device_kind = "gpu" if device.type == "cuda" else "cpu"
    base_metadata = build_base_scheduler_metadata(device_kind=device_kind)
    registry.register_factory(
        metadata=base_metadata,
        factory=lambda: BaseOnnxExecutionScheduler(
            session=session,
            input_name=input_name,
            input_batch_size=input_batch_size,
            device=device,
            metadata=base_metadata,
        ),
    )
    registry.register_factory(
        metadata=OrtCudaGraphExecutionScheduler.metadata,
        factory=lambda: OrtCudaGraphExecutionScheduler(
            session=session,
            input_name=input_name,
            device=device,
        ),
    )
    return registry
