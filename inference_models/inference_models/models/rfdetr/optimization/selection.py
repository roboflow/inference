"""RF-DETR preprocessing selection with declared compatibility fallback."""

from dataclasses import dataclass
from typing import Callable, Optional, cast

from inference_models.errors import ModelRuntimeError
from inference_models.models.common.roboflow.model_packages import (
    ImagePreProcessing,
    NetworkInputDefinition,
)
from inference_models.models.optimization.contracts import (
    CompatibilityResult,
    ExecutionContext,
    OptimizationStage,
)
from inference_models.models.optimization.registry import ImplementationRegistry
from inference_models.models.rfdetr.optimization.contracts import (
    Preprocessor,
    PreprocessRequest,
)


@dataclass(frozen=True)
class PreprocessorSelection:
    """Requested and effective RF-DETR preprocessing selection."""

    implementation: Preprocessor
    requested_id: str
    fallback_reason: Optional[str] = None

    @property
    def effective_id(self) -> str:
        """Return the implementation ID that will execute.

        Returns:
            Effective preprocessing implementation ID.
        """
        return self.implementation.metadata.implementation_id

    @property
    def used_fallback(self) -> bool:
        """Return whether selection followed the declared fallback.

        Returns:
            Whether compatibility resolution followed a declared fallback.
        """
        return self.fallback_reason is not None


def resolve_preprocessor_for_model(
    *,
    registry: ImplementationRegistry,
    requested_id: str,
    context: ExecutionContext,
    image_pre_processing: ImagePreProcessing,
    network_input: NetworkInputDefinition,
) -> PreprocessorSelection:
    """Resolve preprocessing against static model-package configuration.

    Args:
        registry: RF-DETR implementation registry.
        requested_id: Requested preprocessing implementation ID.
        context: Runtime target context.
        image_pre_processing: Model-package image transformations.
        network_input: Model-package network input definition.

    Returns:
        Effective implementation and optional fallback reason.

    Raises:
        ModelRuntimeError: If neither requested nor fallback implementation supports
            the model configuration.
    """
    implementation = cast(
        Preprocessor,
        registry.resolve(
            stage=OptimizationStage.PREPROCESS,
            requested_id=requested_id,
            context=context,
        ),
    )

    def check(candidate: Preprocessor) -> CompatibilityResult:
        result = candidate.check_model_compatibility(
            image_pre_processing=image_pre_processing,
            network_input=network_input,
        )

        return result

    selection = _apply_declared_fallback(
        registry=registry,
        implementation=implementation,
        requested_id=requested_id,
        context=context,
        check_compatibility=check,
    )

    return selection


def resolve_preprocessor_for_request(
    *,
    registry: ImplementationRegistry,
    implementation: Preprocessor,
    request: PreprocessRequest,
    context: ExecutionContext,
) -> PreprocessorSelection:
    """Resolve preprocessing against one concrete inference request.

    Args:
        registry: RF-DETR implementation registry.
        implementation: Model-level selected preprocessor.
        request: Typed preprocessing request.
        context: Runtime target and request context.

    Returns:
        Effective request implementation and optional fallback reason.

    Raises:
        ModelRuntimeError: If neither selected nor fallback implementation supports
            the request.
    """
    requested_id = implementation.metadata.implementation_id

    def check(candidate: Preprocessor) -> CompatibilityResult:
        result = candidate.check_request_compatibility(
            request=request,
            context=context,
        )

        return result

    selection = _apply_declared_fallback(
        registry=registry,
        implementation=implementation,
        requested_id=requested_id,
        context=context,
        check_compatibility=check,
    )

    return selection


def _apply_declared_fallback(
    *,
    registry: ImplementationRegistry,
    implementation: Preprocessor,
    requested_id: str,
    context: ExecutionContext,
    check_compatibility: Callable[[Preprocessor], CompatibilityResult],
) -> PreprocessorSelection:
    compatibility = check_compatibility(implementation)
    if compatibility.supported:
        return PreprocessorSelection(
            implementation=implementation,
            requested_id=requested_id,
        )

    fallback_id = implementation.metadata.fallback_id
    if fallback_id == implementation.metadata.implementation_id:
        raise _unsupported_preprocessor_error(
            requested_id=requested_id,
            requested_reason=compatibility.reason,
        )
    fallback = cast(
        Preprocessor,
        registry.resolve(
            stage=OptimizationStage.PREPROCESS,
            requested_id=fallback_id,
            context=context,
        ),
    )
    fallback_compatibility = check_compatibility(fallback)
    if not fallback_compatibility.supported:
        raise _unsupported_preprocessor_error(
            requested_id=requested_id,
            requested_reason=compatibility.reason,
            fallback_id=fallback_id,
            fallback_reason=fallback_compatibility.reason,
        )

    selection = PreprocessorSelection(
        implementation=fallback,
        requested_id=requested_id,
        fallback_reason=compatibility.reason,
    )

    return selection


def _unsupported_preprocessor_error(
    *,
    requested_id: str,
    requested_reason: str,
    fallback_id: Optional[str] = None,
    fallback_reason: Optional[str] = None,
) -> ModelRuntimeError:
    details = f"{requested_id!r} is unsupported: {requested_reason}."
    if fallback_id is not None:
        details += f" Fallback {fallback_id!r} is unsupported: {fallback_reason}."
    error = ModelRuntimeError(
        message=f"RF-DETR preprocessing cannot execute this contract. {details}",
        help_url=(
            "https://inference-models.roboflow.com/errors/models-runtime/"
            "#modelruntimeerror"
        ),
    )

    return error
