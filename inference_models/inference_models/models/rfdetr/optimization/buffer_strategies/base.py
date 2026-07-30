"""Preserved RF-DETR intermediate-buffer ownership strategy."""

from inference_models.models.optimization.contracts import (
    DeviceCompatibility,
    ExecutionContext,
    InputCompatibility,
    OptimizationMetadata,
    OptimizationStage,
    immutable_mapping,
    metadata_supports_context,
)
from inference_models.models.rfdetr.optimization.contracts import (
    EngineInputBuffer,
    PreprocessResult,
)
from inference_models.models.rfdetr.optimization.ids import RFDETR_BUFFER_STRATEGY_BASE


class BaseBufferStrategy:
    """Preserve the framework-owned engine-input tensor without copying it.

    This RF-DETR implementation covers only the preprocessing-to-engine boundary.
    Extending buffer ownership or reuse across engine outputs and postprocessing would
    require a broader model-path contract.
    """

    metadata = OptimizationMetadata(
        implementation_id=RFDETR_BUFFER_STRATEGY_BASE,
        stage=OptimizationStage.BUFFER_STRATEGY,
        version="1",
        target=DeviceCompatibility(device_kind="gpu"),
        inputs=InputCompatibility(
            scenarios=("*",),
            axis_constraints=immutable_mapping({"batch": ">=1"}),
            dtypes=("float32",),
            layouts=("contiguous NCHW",),
        ),
        dependencies=("torch",),
        fallback_id=RFDETR_BUFFER_STRATEGY_BASE,
        changes_numerics=False,
        supports_concurrency=True,
        supports_cuda_graphs=True,
        output_contract=immutable_mapping(
            {
                "device": "unchanged",
                "dtype": "unchanged",
                "layout": "unchanged",
                "ownership": "framework tensor retained by caller",
                "aliasing": "exact preprocessing tensor; no copy",
                "lifetime": "through TensorRT consumption",
            }
        ),
        numerical_behavior="identity; does not inspect or modify tensor values",
        stream_behavior="preserves the preprocessing readiness event",
    )

    def is_compatible(self, context: ExecutionContext) -> bool:
        """Return whether framework-owned buffers support the runtime context.

        Args:
            context: Runtime target and request context.

        Returns:
            Whether the target is compatible.
        """
        compatible = metadata_supports_context(self.metadata, context)

        return compatible

    def prepare_engine_input(
        self,
        result: PreprocessResult,
        context: ExecutionContext,
    ) -> EngineInputBuffer:
        """Preserve the exact preprocessing tensor as the engine input.

        Args:
            result: Typed preprocessing result.
            context: Runtime context containing the producer stream.

        Returns:
            Engine-input buffer aliasing the original preprocessing tensor.
        """
        del context
        engine_input_buffer = EngineInputBuffer(
            tensor=result.tensor,
            ready_event=result.ready_event,
            input_kind=result.input_kind,
            preprocessor_implementation_id=result.implementation_id,
            fallback_reason=result.fallback_reason,
        )

        return engine_input_buffer
