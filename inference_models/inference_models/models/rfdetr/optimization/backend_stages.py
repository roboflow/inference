"""Base stage adapters preserving Torch and ONNX object-detection operations.

The protected forward and backend-specific postprocessing remain callbacks owned
by the model. These adapters describe and execute their boundaries without
assuming TensorRT output types or importing TensorRT/PyCUDA.
"""

from dataclasses import replace
from typing import Callable, TypeVar

import torch

from inference_models.models.optimization.contracts import (
    DeviceCompatibility,
    ExecutionContext,
    InputCompatibility,
    OptimizationMetadata,
    OptimizationStage,
    immutable_mapping,
    metadata_supports_context,
)
from inference_models.models.rfdetr.optimization.readiness import (
    PreprocessReadinessTracker,
)

ResultT = TypeVar("ResultT")


class BackendEnginePlugin:
    """Execute the model's existing semantic forward operation."""

    metadata = OptimizationMetadata(
        implementation_id="base",
        stage=OptimizationStage.ENGINE_PLUGIN,
        version="1",
        target=DeviceCompatibility(device_kind="any"),
        inputs=InputCompatibility(scenarios=("*",)),
        dependencies=("torch",),
        fallback_id="base",
        changes_numerics=False,
        supports_concurrency=True,
        supports_cuda_graphs=False,
        output_contract=immutable_mapping({"ownership": "backend-owned model output"}),
        numerical_behavior="preserves the backend semantic forward",
        stream_behavior="preserves the model's existing forward stream and locks",
    )

    def is_compatible(self, context: ExecutionContext) -> bool:
        """Check the stage's declared runtime requirements.

        Args:
            context (ExecutionContext): Device and installed runtime components.

        Returns:
            bool: Whether the stage supports the selected context.
        """
        compatible = metadata_supports_context(self.metadata, context)

        return compatible

    def execute(self, operation: Callable[[], ResultT]) -> ResultT:
        """Invoke the backend operation without changing its semantics.

        Args:
            operation (Callable): Zero-argument backend operation.

        Returns:
            ResultT: The operation's unchanged result.
        """
        result = operation()

        return result


class BackendPostprocessor(BackendEnginePlugin):
    """Preserve backend-specific detection conversion and class remapping."""

    metadata = replace(
        BackendEnginePlugin.metadata,
        stage=OptimizationStage.POSTPROCESS,
        numerical_behavior="preserves backend-specific reference postprocessing",
        stream_behavior="preserves the model's existing postprocessing synchronization",
        output_contract=immutable_mapping({"type": "list[Detections]"}),
    )


class BackendExecutionScheduler(BackendEnginePlugin):
    """Transfer exact-tensor readiness to the Torch or ONNX consumer stream."""

    metadata = replace(
        BackendEnginePlugin.metadata,
        stage=OptimizationStage.SCHEDULER,
        numerical_behavior="does not inspect or change tensor values",
        stream_behavior=(
            "standalone preprocessing synchronizes; composed inference "
            "waits on the completion event on the backend consumer stream"
        ),
        output_contract=immutable_mapping(
            {"ownership": "unchanged input/output tensors"}
        ),
    )

    def __init__(self):
        self._readiness = PreprocessReadinessTracker()

    def finalize_preprocess(
        self, engine_input, *, context, independent_stage_execution
    ):
        """Synchronize standalone inputs or retain readiness for composed inference.

        Args:
            engine_input (EngineInputBuffer): Preprocessed tensor and completion event.
            context (ExecutionContext): Device and preprocessing stream.
            independent_stage_execution (bool): Whether to synchronize before return.

        Returns:
            torch.Tensor: The unchanged tensor to pass to the backend.
        """
        if independent_stage_execution:
            if engine_input.ready_event is not None:
                engine_input.ready_event.synchronize()
            elif context.current_stream is not None:
                context.current_stream.synchronize()
        else:
            event = engine_input.ready_event
            if event is None and context.current_stream is not None:
                event = torch.cuda.Event()
                event.record(context.current_stream)

            self._readiness.record(
                engine_input.tensor,
                ready_event=event,
                input_kind=engine_input.input_kind,
                implementation_id=engine_input.preprocessor_implementation_id,
                fallback_reason=engine_input.fallback_reason,
            )

        return engine_input.tensor

    def execute_engine(self, tensor, *, stream, operation):
        """Wait for this tensor's preprocessing before invoking the backend.

        Args:
            tensor (torch.Tensor): Exact tensor returned from preprocessing.
            stream (torch.cuda.Stream, optional): Backend consumer stream.
            operation (Callable): Zero-argument backend forward operation.

        Returns:
            Any: The backend operation's unchanged result.
        """
        readiness = self._readiness.consume(tensor)
        if readiness is not None and readiness.ready_event is not None:
            stream.wait_event(readiness.ready_event)
        if stream is not None:
            tensor.record_stream(stream)

        result = operation()

        return result
