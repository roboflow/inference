"""RF-DETR-specific requests, results, and stage protocols."""

from dataclasses import dataclass
from typing import List, Optional, Protocol, Union

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
class PostprocessRequest:
    """Inputs required by an RF-DETR postprocessing implementation."""

    bboxes: torch.Tensor
    logits: torch.Tensor
    pre_processing_meta: List[PreProcessingMetadata]
    threshold: Union[float, torch.Tensor]
    num_classes: int
    classes_re_mapping: Optional[ClassesReMapping]


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
