"""Preserved RF-DETR CUDA stream and dependency scheduler."""

import threading
from typing import List, Tuple

import torch

from inference_models import Detections
from inference_models.errors import ModelRuntimeError
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
    EngineOperation,
    PostprocessOperation,
)
from inference_models.models.rfdetr.optimization.ids import RFDETR_SCHEDULER_BASE
from inference_models.models.rfdetr.optimization.readiness import (
    PreprocessReadinessTracker,
)


class BaseExecutionScheduler:
    """Preserve RF-DETR stream reuse, event handoff, and serialized forward."""

    metadata = OptimizationMetadata(
        implementation_id=RFDETR_SCHEDULER_BASE,
        stage=OptimizationStage.SCHEDULER,
        version="1",
        target=DeviceCompatibility(device_kind="gpu"),
        inputs=InputCompatibility(
            scenarios=("*",),
            axis_constraints=immutable_mapping({"batch": ">=1"}),
            dtypes=("float32",),
            layouts=("contiguous NCHW",),
        ),
        dependencies=("torch",),
        fallback_id=RFDETR_SCHEDULER_BASE,
        changes_numerics=False,
        supports_concurrency=True,
        supports_cuda_graphs=True,
        output_contract=immutable_mapping(
            {
                "preprocess_stream": "one reusable CUDA stream per caller thread",
                "engine_stream": "one reusable CUDA stream per model instance",
                "postprocess_stream": "one reusable CUDA stream per caller thread",
                "engine_concurrency": "serialized per model instance",
            }
        ),
        numerical_behavior="does not inspect or modify tensor values",
        stream_behavior=(
            "records preprocessing readiness by exact tensor identity, waits on the "
            "engine stream, and synchronizes public postprocessing results"
        ),
    )

    def __init__(self, *, device: torch.device) -> None:
        self._device = device
        self._engine_lock = threading.Lock()
        self._inference_stream = torch.cuda.Stream(device=device)
        self._preprocess_readiness = PreprocessReadinessTracker()
        self._thread_local_storage = threading.local()

    def is_compatible(self, context: ExecutionContext) -> bool:
        """Return whether the scheduler supports the runtime context.

        Args:
            context: Runtime target and request context.

        Returns:
            Whether the target is compatible.
        """
        compatible = metadata_supports_context(self.metadata, context)

        return compatible

    def preprocess_stream(self) -> torch.cuda.Stream:
        """Return the caller thread's reusable preprocessing stream.

        Returns:
            CUDA stream assigned to preprocessing work.
        """
        if not hasattr(self._thread_local_storage, "preprocess_stream"):
            self._thread_local_storage.preprocess_stream = torch.cuda.Stream(
                device=self._device
            )

        return self._thread_local_storage.preprocess_stream

    def finalize_preprocess(
        self,
        engine_input: EngineInputBuffer,
        *,
        context: ExecutionContext,
        independent_stage_execution: bool,
    ) -> torch.Tensor:
        """Publish or synchronize preprocessing output for forward consumption.

        Args:
            engine_input: Tensor and readiness state from the buffer strategy.
            context: Runtime context containing the preprocessing stream.
            independent_stage_execution: Whether the tensor must be ready on return.

        Returns:
            Exact tensor that the engine scheduler will consume.

        Raises:
            ModelRuntimeError: If the preprocessing context has no CUDA stream.
        """
        stream = context.current_stream
        if stream is None:
            raise ModelRuntimeError(
                message="RF-DETR base scheduler requires a preprocessing CUDA stream.",
                help_url=(
                    "https://inference-models.roboflow.com/errors/models-runtime/"
                    "#modelruntimeerror"
                ),
            )

        if engine_input.ready_event is None:
            stream.synchronize()
        elif independent_stage_execution:
            engine_input.ready_event.synchronize()

        if not independent_stage_execution:
            self._preprocess_readiness.record(
                engine_input.tensor,
                ready_event=engine_input.ready_event,
                input_kind=engine_input.input_kind,
                implementation_id=engine_input.preprocessor_implementation_id,
                fallback_reason=engine_input.fallback_reason,
            )

        return engine_input.tensor

    def execute_engine(
        self,
        pre_processed_images: torch.Tensor,
        *,
        operation: EngineOperation,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute serialized engine work after waiting on preprocessing.

        Args:
            pre_processed_images: Exact tensor returned by preprocessing.
            operation: Engine-boundary operation accepting the assigned CUDA stream.

        Returns:
            Raw TensorRT detection boxes and logits.
        """
        with self._engine_lock:
            readiness = self._preprocess_readiness.consume(pre_processed_images)
            if readiness is not None and readiness.ready_event is not None:
                self._inference_stream.wait_event(readiness.ready_event)
            model_results = operation(self._inference_stream)

        return model_results

    def execute_postprocess(
        self,
        model_results: Tuple[torch.Tensor, torch.Tensor],
        *,
        operation: PostprocessOperation,
    ) -> List[Detections]:
        """Run postprocessing on a reusable stream and synchronize its results.

        Args:
            model_results: Raw TensorRT detection boxes and logits.
            operation: Postprocessing operation accepting the assigned CUDA stream.

        Returns:
            Per-image detections ready for public consumption.
        """
        stream = self._postprocess_stream
        with torch.cuda.stream(stream):
            for result_element in model_results:
                result_element.record_stream(stream)
            results = operation(stream)
        stream.synchronize()

        return results

    @property
    def _postprocess_stream(self) -> torch.cuda.Stream:
        if not hasattr(self._thread_local_storage, "postprocess_stream"):
            self._thread_local_storage.postprocess_stream = torch.cuda.Stream(
                device=self._device
            )

        return self._thread_local_storage.postprocess_stream
