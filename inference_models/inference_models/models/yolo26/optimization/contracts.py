"""YOLO26 depth ONNX scheduler protocol."""

from typing import Protocol

import torch

from inference_models.models.optimization.contracts import (
    ExecutionContext,
    InferenceStage,
    OptimizationMetadata,
)


class OnnxExecutionScheduler(InferenceStage, Protocol):
    """Execute the unchanged ONNX model with a selected submission strategy."""

    metadata: OptimizationMetadata

    def is_compatible(self, context: ExecutionContext) -> bool:
        """Return whether this scheduler supports the runtime context."""

    def execute(self, pre_processed_images: torch.Tensor) -> torch.Tensor:
        """Run the protected ONNX computation and return its single output."""
