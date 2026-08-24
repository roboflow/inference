"""Selectable YOLO26 depth-estimation CUDA execution schedulers."""

from __future__ import annotations

import threading
from typing import Callable

import torch

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
from inference_models.models.optimization.torch_readiness import TensorReadinessTracker
from inference_models.models.yolo26.optimization.ids import (
    YOLO26_DEPTH_SCHEDULER_BASE,
    YOLO26_DEPTH_SCHEDULER_CUDA_EVENT_HANDOFF_V1,
)

EngineOperation = Callable[[torch.cuda.Stream], torch.Tensor]


class BaseYOLO26DepthExecutionScheduler:
    """Preserve preprocessing synchronization and serialized TensorRT execution."""

    metadata = OptimizationMetadata(
        implementation_id=YOLO26_DEPTH_SCHEDULER_BASE,
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
        fallback_id=YOLO26_DEPTH_SCHEDULER_BASE,
        changes_numerics=False,
        supports_concurrency=True,
        supports_cuda_graphs=True,
        output_contract=immutable_mapping(
            {
                "preprocess_stream": "one reusable CUDA stream per caller thread",
                "engine_stream": "one reusable CUDA stream per model instance",
                "engine_concurrency": "serialized per model instance",
            }
        ),
        numerical_behavior="does not inspect or modify tensor values",
        stream_behavior=(
            "synchronizes the caller preprocessing stream before returning and "
            "serializes TensorRT execution on the model-instance stream"
        ),
    )

    def __init__(self, *, device: torch.device) -> None:
        self._device = device
        self._engine_lock = threading.Lock()
        self._inference_stream = torch.cuda.Stream(device=device)
        self._thread_local_storage = threading.local()

    def is_compatible(self, context: ExecutionContext) -> bool:
        return metadata_supports_context(self.metadata, context)

    def preprocess_stream(self) -> torch.cuda.Stream:
        """Return the caller thread's reusable preprocessing stream."""
        if not hasattr(self._thread_local_storage, "preprocess_stream"):
            self._thread_local_storage.preprocess_stream = torch.cuda.Stream(
                device=self._device
            )

        return self._thread_local_storage.preprocess_stream

    def finalize_preprocess(
        self,
        tensor: torch.Tensor,
        *,
        context: ExecutionContext,
        independent_stage_execution: bool,
    ) -> torch.Tensor:
        """Return a host-ready tensor using the preserved synchronization."""
        del independent_stage_execution
        stream = self._require_stream(context)
        with torch.cuda.nvtx.range(
            "yolo26-depth.preprocess[phase=synchronize,effective_scheduler=base]"
        ):
            stream.synchronize()

        return tensor

    def execute_engine(
        self,
        tensor: torch.Tensor,
        *,
        operation: EngineOperation,
    ) -> torch.Tensor:
        """Execute TensorRT under the preserved model-instance lock."""
        del tensor
        with self._engine_lock:
            result = operation(self._inference_stream)

        return result

    @staticmethod
    def _require_stream(context: ExecutionContext) -> torch.cuda.Stream:
        stream = context.current_stream
        if stream is None:
            raise ModelRuntimeError(
                message="YOLO26 depth scheduler requires a preprocessing CUDA stream.",
                help_url=(
                    "https://inference-models.roboflow.com/errors/"
                    "models-runtime/#modelruntimeerror"
                ),
            )

        return stream


class CUDAEventHandoffYOLO26DepthExecutionScheduler(BaseYOLO26DepthExecutionScheduler):
    """Transfer exact-tensor readiness from preprocessing to TensorRT."""

    metadata = OptimizationMetadata(
        implementation_id=YOLO26_DEPTH_SCHEDULER_CUDA_EVENT_HANDOFF_V1,
        stage=OptimizationStage.SCHEDULER,
        version="1",
        target=DeviceCompatibility(device_kind="gpu"),
        inputs=InputCompatibility(
            scenarios=(
                "camera_640x480_batch_1_base",
                "camera_3840x2160_batch_1_high",
            ),
            axis_constraints=immutable_mapping({"batch": 1}),
            dtypes=("float32",),
            layouts=("contiguous NCHW",),
        ),
        dependencies=("torch",),
        fallback_id=YOLO26_DEPTH_SCHEDULER_BASE,
        changes_numerics=False,
        supports_concurrency=True,
        supports_cuda_graphs=True,
        output_contract=immutable_mapping(
            {
                "preprocess_stream": "one reusable CUDA stream per caller thread",
                "engine_stream": "one reusable CUDA stream per model instance",
                "readiness": "one-shot CUDA event keyed by exact tensor identity",
                "engine_concurrency": "serialized per model instance",
            }
        ),
        numerical_behavior="does not inspect or modify tensor values",
        stream_behavior=(
            "records preprocessing readiness, synchronizes direct stage calls, and "
            "uses a CUDA stream wait for composed inference"
        ),
    )

    def __init__(self, *, device: torch.device) -> None:
        super().__init__(device=device)
        self._preprocess_readiness = TensorReadinessTracker[torch.cuda.Event]()

    def finalize_preprocess(
        self,
        tensor: torch.Tensor,
        *,
        context: ExecutionContext,
        independent_stage_execution: bool,
    ) -> torch.Tensor:
        """Synchronize direct calls or publish readiness for composed inference."""
        stream = self._require_stream(context)
        with torch.cuda.nvtx.range(
            "yolo26-depth.preprocess["
            "phase=publish,effective_scheduler=cuda-event-handoff-v1]"
        ):
            ready_event = torch.cuda.Event()
            ready_event.record(stream)
            if independent_stage_execution:
                ready_event.synchronize()
            else:
                self._preprocess_readiness.record(tensor, state=ready_event)

        return tensor

    def execute_engine(
        self,
        tensor: torch.Tensor,
        *,
        operation: EngineOperation,
    ) -> torch.Tensor:
        """Wait on the exact producer event before TensorRT submission."""
        with self._engine_lock:
            ready_event = self._preprocess_readiness.consume(tensor)
            if ready_event is not None:
                with torch.cuda.nvtx.range(
                    "yolo26-depth.forward["
                    "phase=preprocess-wait,"
                    "effective_scheduler=cuda-event-handoff-v1]"
                ):
                    self._inference_stream.wait_event(ready_event)
            result = operation(self._inference_stream)

        return result
