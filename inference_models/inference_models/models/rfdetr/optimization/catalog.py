"""RF-DETR implementation catalog and registry construction."""

from types import MappingProxyType
from typing import Mapping

import torch

from inference_models.models.optimization.contracts import (
    OptimizationMetadata,
    OptimizationStage,
)
from inference_models.models.optimization.registry import ImplementationRegistry
from inference_models.models.rfdetr.optimization.buffer_strategies import (
    BaseBufferStrategy,
)
from inference_models.models.rfdetr.optimization.engine_plugins import (
    BaseEngineAdjacentPlugin,
)
from inference_models.models.rfdetr.optimization.postprocessors import (
    BasePostprocessor,
    TritonFusedPostprocessor,
)
from inference_models.models.rfdetr.optimization.preprocessors import (
    BasePreprocessor,
    ThreadedExactPreprocessor,
    TritonUniversalPreprocessor,
)
from inference_models.models.rfdetr.optimization.schedulers import (
    BaseExecutionScheduler,
)

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

RFDETR_BUFFER_STRATEGY_IMPLEMENTATIONS: Mapping[str, OptimizationMetadata] = (
    MappingProxyType(
        {BaseBufferStrategy.metadata.implementation_id: (BaseBufferStrategy.metadata)}
    )
)

RFDETR_SCHEDULER_IMPLEMENTATIONS: Mapping[str, OptimizationMetadata] = MappingProxyType(
    {
        BaseExecutionScheduler.metadata.implementation_id: (
            BaseExecutionScheduler.metadata
        )
    }
)

RFDETR_ENGINE_PLUGIN_IMPLEMENTATIONS: Mapping[str, OptimizationMetadata] = (
    MappingProxyType(
        {
            BaseEngineAdjacentPlugin.metadata.implementation_id: (
                BaseEngineAdjacentPlugin.metadata
            )
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
    registry = ImplementationRegistry(scope_name="RF-DETR")
    registry.register_factory(
        metadata=BasePreprocessor.metadata,
        factory=lambda: BasePreprocessor(max_workers=preprocessor_max_workers),
    )
    registry.register_factory(
        metadata=ThreadedExactPreprocessor.metadata,
        factory=lambda: ThreadedExactPreprocessor(max_workers=preprocessor_max_workers),
    )
    registry.register_factory(
        metadata=TritonUniversalPreprocessor.metadata,
        factory=lambda: TritonUniversalPreprocessor(device=device),
    )
    registry.register_factory(
        metadata=BaseBufferStrategy.metadata,
        factory=BaseBufferStrategy,
    )
    registry.register_factory(
        metadata=BaseExecutionScheduler.metadata,
        factory=lambda: BaseExecutionScheduler(device=device),
    )
    registry.register_factory(
        metadata=BasePostprocessor.metadata,
        factory=BasePostprocessor,
    )
    registry.register_factory(
        metadata=TritonFusedPostprocessor.metadata,
        factory=lambda: TritonFusedPostprocessor(device=device),
    )
    registry.register_factory(
        metadata=BaseEngineAdjacentPlugin.metadata,
        factory=BaseEngineAdjacentPlugin,
    )
    registry.set_auto_preferences(
        stage=OptimizationStage.PREPROCESS,
        implementation_ids=(
            TritonUniversalPreprocessor.metadata.implementation_id,
            ThreadedExactPreprocessor.metadata.implementation_id,
        ),
    )
    registry.set_auto_preferences(
        stage=OptimizationStage.POSTPROCESS,
        implementation_ids=(TritonFusedPostprocessor.metadata.implementation_id,),
    )

    return registry
