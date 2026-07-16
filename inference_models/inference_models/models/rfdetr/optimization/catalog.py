"""RF-DETR implementation catalog and registry construction."""

from types import MappingProxyType
from typing import Mapping

import torch

from inference_models.models.rfdetr.optimization.contracts import OptimizationMetadata
from inference_models.models.rfdetr.optimization.postprocessors import (
    BasePostprocessor,
    TritonFusedPostprocessor,
)
from inference_models.models.rfdetr.optimization.preprocessors import (
    BasePreprocessor,
    ThreadedExactPreprocessor,
    TritonUniversalPreprocessor,
)
from inference_models.models.rfdetr.optimization.registry import ImplementationRegistry

RFDETR_PREPROCESSOR_IMPLEMENTATIONS: Mapping[str, OptimizationMetadata] = (
    MappingProxyType(
        {
            implementation.metadata.implementation_id: implementation.metadata
            for implementation in (
                BasePreprocessor,
                ThreadedExactPreprocessor,
                TritonUniversalPreprocessor,
            )
        }
    )
)

RFDETR_POSTPROCESSOR_IMPLEMENTATIONS: Mapping[str, OptimizationMetadata] = (
    MappingProxyType(
        {
            implementation.metadata.implementation_id: implementation.metadata
            for implementation in (BasePostprocessor, TritonFusedPostprocessor)
        }
    )
)


def build_rfdetr_implementation_registry(
    *,
    device: torch.device,
    preprocessor_max_workers: int,
) -> ImplementationRegistry:
    """Build the complete RF-DETR stage implementation registry.

    Args:
        device: CUDA target selected for the TensorRT model.
        preprocessor_max_workers: Bounded threaded preprocessing worker limit.

    Returns:
        Registry containing every available preprocessing and postprocessing choice.
    """
    registry = ImplementationRegistry()
    registry.register(BasePreprocessor(max_workers=preprocessor_max_workers))
    registry.register(ThreadedExactPreprocessor(max_workers=preprocessor_max_workers))
    registry.register(TritonUniversalPreprocessor(device=device))
    registry.register(BasePostprocessor())
    registry.register(TritonFusedPostprocessor(device=device))

    return registry
