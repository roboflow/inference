"""RF-DETR-specific requests, results, and stage protocols."""

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Protocol, Tuple, Union

import numpy as np
import torch

from inference_models import Detections, PreProcessingOverrides
from inference_models.entities import ColorFormat
from inference_models.models.common.roboflow.model_packages import (
    ImagePreProcessing,
    NetworkInputDefinition,
    PreProcessingMetadata,
)
from inference_models.models.optimization.contracts import (
    CompatibilityResult,
    ExecutionContext,
    InferenceStage,
    OptimizationMetadata,
)
from inference_models.models.rfdetr.class_remapping import ClassesReMapping

ImageInput = Union[np.ndarray, torch.Tensor]


@dataclass(frozen=True)
class PreprocessRequest:
    """Inputs required by an RF-DETR preprocessing implementation."""

    images: Union[ImageInput, List[ImageInput]]
    input_color_format: Optional[ColorFormat]
    image_pre_processing: ImagePreProcessing
    network_input: NetworkInputDefinition
    pre_processing_overrides: Optional[PreProcessingOverrides]


@dataclass(frozen=True)
class PreprocessResult:
    """Typed preprocessing output and asynchronous readiness state."""

    tensor: torch.Tensor
    metadata: List[PreProcessingMetadata]
    implementation_id: str
    ready_event: Optional[torch.cuda.Event] = None
    input_kind: str = "reference"
    fallback_reason: Optional[str] = None


@dataclass(frozen=True)
class EngineInputBuffer:
    """Engine-ready tensor and producer readiness state."""

    tensor: torch.Tensor
    ready_event: Optional[torch.cuda.Event]
    input_kind: str
    preprocessor_implementation_id: str
    fallback_reason: Optional[str] = None


@dataclass(frozen=True)
class PostprocessRequest:
    """Inputs required by an RF-DETR postprocessing implementation."""

    bboxes: torch.Tensor
    logits: torch.Tensor
    pre_processing_meta: List[PreProcessingMetadata]
    threshold: Union[float, torch.Tensor]
    num_classes: int
    classes_re_mapping: Optional[ClassesReMapping]


@dataclass(frozen=True)
class EngineExecutionRequest:
    """Inputs and runtime objects required at the TensorRT engine boundary."""

    pre_processed_images: torch.Tensor
    trt_config: Any
    engine: Any
    trt_execution_context: Any
    device: torch.device
    input_name: str
    output_names: List[str]
    trt_cuda_graph_cache: Optional[Any]


class Preprocessor(InferenceStage, Protocol):
    """RF-DETR preprocessing stage interface."""

    metadata: OptimizationMetadata

    def is_compatible(self, context: ExecutionContext) -> bool:
        """Return whether the preprocessor supports a runtime context.

        Args:
            context: Runtime target and request context.

        Returns:
            Whether the preprocessor is compatible.
        """

    def check_model_compatibility(
        self,
        *,
        image_pre_processing: ImagePreProcessing,
        network_input: NetworkInputDefinition,
    ) -> CompatibilityResult:
        """Check compatibility with static model preprocessing configuration.

        Args:
            image_pre_processing: Model-package image transformations.
            network_input: Model-package network input definition.

        Returns:
            Compatibility result with actionable reasons.
        """

    def check_request_compatibility(
        self,
        *,
        request: PreprocessRequest,
        context: ExecutionContext,
    ) -> CompatibilityResult:
        """Check compatibility with one concrete preprocessing request.

        Args:
            request: Typed preprocessing request.
            context: Runtime target and request context.

        Returns:
            Compatibility result with actionable reasons.
        """

    def check_runtime_compatibility(
        self,
        *,
        request: PreprocessRequest,
        context: ExecutionContext,
    ) -> CompatibilityResult:
        """Check whether runtime state supports this preprocessing request.

        Args:
            request: Typed preprocessing request.
            context: Runtime target and request context.

        Returns:
            Compatibility result with any recorded runtime failure reason.
        """

    def preprocess(
        self,
        request: PreprocessRequest,
        context: ExecutionContext,
    ) -> PreprocessResult:
        """Preprocess one request on the context stream.

        Args:
            request: Typed preprocessing request.
            context: Runtime context containing the selected stream.

        Returns:
            Typed preprocessing result.
        """


class BufferStrategy(InferenceStage, Protocol):
    """RF-DETR buffer-strategy interface.

    The current object-detection TensorRT path applies this stage only at the
    preprocessing-to-engine boundary. Supporting ownership or reuse across the full
    inference path may require extending this contract.
    """

    metadata: OptimizationMetadata

    def is_compatible(self, context: ExecutionContext) -> bool:
        """Return whether the buffer strategy supports a runtime context.

        Args:
            context: Runtime target and request context.

        Returns:
            Whether the buffer strategy is compatible.
        """

    def prepare_engine_input(
        self,
        result: PreprocessResult,
        context: ExecutionContext,
    ) -> EngineInputBuffer:
        """Prepare preprocessing output for scheduler and engine consumption.

        Args:
            result: Typed preprocessing result.
            context: Runtime context containing the producer stream.

        Returns:
            Engine-ready tensor with explicit readiness and ownership state.
        """


EngineOperation = Callable[
    [torch.cuda.Stream],
    Tuple[torch.Tensor, torch.Tensor],
]
PostprocessOperation = Callable[[torch.cuda.Stream], List[Detections]]


class ExecutionScheduler(InferenceStage, Protocol):
    """RF-DETR stream, dependency, and request-ordering stage interface."""

    metadata: OptimizationMetadata

    def is_compatible(self, context: ExecutionContext) -> bool:
        """Return whether the scheduler supports a runtime context.

        Args:
            context: Runtime target and request context.

        Returns:
            Whether the scheduler is compatible.
        """

    def preprocess_stream(self) -> torch.cuda.Stream:
        """Return the reusable stream assigned to preprocessing.

        Returns:
            CUDA stream for preprocessing work in the current caller thread.
        """

    def finalize_preprocess(
        self,
        engine_input: EngineInputBuffer,
        *,
        context: ExecutionContext,
        independent_stage_execution: bool,
    ) -> torch.Tensor:
        """Publish or synchronize a preprocessed tensor for its consumer.

        Args:
            engine_input: Tensor and readiness state from the engine-input buffer
                strategy.
            context: Runtime context containing the producer stream.
            independent_stage_execution: Whether the tensor must be ready on return.

        Returns:
            The exact tensor to pass to the protected forward stage.
        """

    def execute_engine(
        self,
        pre_processed_images: torch.Tensor,
        *,
        operation: EngineOperation,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute engine work after satisfying producer dependencies.

        Args:
            pre_processed_images: Exact tensor returned by preprocessing.
            operation: Engine-boundary operation accepting the assigned CUDA stream.

        Returns:
            Raw TensorRT detection boxes and logits.
        """

    def execute_postprocess(
        self,
        model_results: Tuple[torch.Tensor, torch.Tensor],
        *,
        operation: PostprocessOperation,
    ) -> List[Detections]:
        """Execute postprocessing with output lifetime and synchronization handling.

        Args:
            model_results: Raw TensorRT detection boxes and logits.
            operation: Postprocessing operation accepting the assigned CUDA stream.

        Returns:
            Per-image detections ready for public consumption.
        """


class Postprocessor(InferenceStage, Protocol):
    """RF-DETR postprocessing stage interface."""

    metadata: OptimizationMetadata

    def is_compatible(self, context: ExecutionContext) -> bool:
        """Return whether the postprocessor supports a runtime context.

        Args:
            context: Runtime target and request context.

        Returns:
            Whether the postprocessor is compatible.
        """

    def check_request_compatibility(
        self,
        *,
        request: PostprocessRequest,
        context: ExecutionContext,
    ) -> CompatibilityResult:
        """Check compatibility with one concrete postprocessing request.

        Args:
            request: Typed postprocessing request.
            context: Runtime target and request context.

        Returns:
            Compatibility result with actionable reasons.
        """

    def check_runtime_compatibility(
        self,
        *,
        request: PostprocessRequest,
        context: ExecutionContext,
    ) -> CompatibilityResult:
        """Check whether runtime state supports this postprocessing request.

        Args:
            request: Typed postprocessing request.
            context: Runtime target and request context.

        Returns:
            Compatibility result with any recorded runtime failure reason.
        """

    def postprocess(
        self,
        request: PostprocessRequest,
        context: ExecutionContext,
    ) -> List[Detections]:
        """Postprocess one request on the context stream.

        Args:
            request: Typed postprocessing request.
            context: Runtime context containing the selected stream.

        Returns:
            Per-image detections.
        """


class EngineAdjacentPlugin(InferenceStage, Protocol):
    """RF-DETR TensorRT engine-boundary implementation interface."""

    metadata: OptimizationMetadata

    def is_compatible(self, context: ExecutionContext) -> bool:
        """Return whether the engine plugin supports a runtime context.

        Args:
            context: Runtime target and request context.

        Returns:
            Whether the engine plugin is compatible.
        """

    def execute(
        self,
        request: EngineExecutionRequest,
        context: ExecutionContext,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute the protected TensorRT forward boundary.

        Args:
            request: TensorRT input and engine runtime objects.
            context: Runtime context containing TensorRT's execution stream.

        Returns:
            Raw TensorRT detection boxes and logits.
        """
