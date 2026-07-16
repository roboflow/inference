"""RF-DETR stage selection with declared compatibility fallback."""

from dataclasses import dataclass
from typing import Callable, Dict, Generic, Optional, TypeVar, cast

from inference_models.errors import ModelRuntimeError
from inference_models.models.common.roboflow.model_packages import (
    ImagePreProcessing,
    NetworkInputDefinition,
)
from inference_models.models.optimization.contracts import (
    CompatibilityResult,
    ExecutionContext,
    InferenceStage,
    OptimizationStage,
)
from inference_models.models.optimization.registry import ImplementationRegistry
from inference_models.models.rfdetr.optimization.contracts import (
    Postprocessor,
    PostprocessRequest,
    Preprocessor,
    PreprocessRequest,
)

StageT = TypeVar("StageT", bound=InferenceStage)


@dataclass(frozen=True)
class ImplementationSelection(Generic[StageT]):
    """Requested and effective RF-DETR stage selection."""

    implementation: StageT
    requested_id: str
    fallback_reason: Optional[str] = None

    @property
    def effective_id(self) -> str:
        """Return the implementation ID that will execute.

        Returns:
            Effective stage implementation ID.
        """
        return self.implementation.metadata.implementation_id

    @property
    def used_fallback(self) -> bool:
        """Return whether selection followed the declared fallback.

        Returns:
            Whether compatibility resolution followed a declared fallback.
        """
        return self.fallback_reason is not None

    def to_dict(self) -> Dict[str, Optional[str]]:
        """Serialize requested and effective selection metadata.

        Returns:
            JSON-compatible selection metadata.
        """
        serialized = {
            "requested_id": self.requested_id,
            "effective_id": self.effective_id,
            "fallback_reason": self.fallback_reason,
        }

        return serialized


def resolve_preprocessor_for_model(
    *,
    registry: ImplementationRegistry,
    requested_id: str,
    context: ExecutionContext,
    image_pre_processing: ImagePreProcessing,
    network_input: NetworkInputDefinition,
    allow_fallback: bool,
) -> ImplementationSelection[Preprocessor]:
    """Resolve preprocessing against static model-package configuration.

    Args:
        registry: RF-DETR implementation registry.
        requested_id: Requested preprocessing implementation ID.
        context: Runtime target context.
        image_pre_processing: Model-package image transformations.
        network_input: Model-package network input definition.
        allow_fallback: Whether declared compatibility fallback may be used.

    Returns:
        Effective implementation and optional fallback reason.

    Raises:
        ModelRuntimeError: If the requested implementation is incompatible and no
            permitted compatible fallback exists.
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
        stage=OptimizationStage.PREPROCESS,
        implementation=implementation,
        requested_id=requested_id,
        context=context,
        check_compatibility=check,
        allow_fallback=allow_fallback,
    )

    return selection


def resolve_preprocessor_for_request(
    *,
    registry: ImplementationRegistry,
    implementation: Preprocessor,
    request: PreprocessRequest,
    context: ExecutionContext,
    allow_fallback: bool,
) -> ImplementationSelection[Preprocessor]:
    """Resolve preprocessing against one concrete inference request.

    Args:
        registry: RF-DETR implementation registry.
        implementation: Model-level selected preprocessor.
        request: Typed preprocessing request.
        context: Runtime target and request context.
        allow_fallback: Whether declared compatibility fallback may be used.

    Returns:
        Effective request implementation and optional fallback reason.

    Raises:
        ModelRuntimeError: If the selected implementation is incompatible and no
            permitted compatible fallback exists.
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
        stage=OptimizationStage.PREPROCESS,
        implementation=implementation,
        requested_id=requested_id,
        context=context,
        check_compatibility=check,
        allow_fallback=allow_fallback,
    )

    return selection


def resolve_postprocessor_for_request(
    *,
    registry: ImplementationRegistry,
    implementation: Postprocessor,
    request: PostprocessRequest,
    context: ExecutionContext,
    allow_fallback: bool,
) -> ImplementationSelection[Postprocessor]:
    """Resolve postprocessing against one concrete inference request.

    Args:
        registry: RF-DETR implementation registry.
        implementation: Model-level selected postprocessor.
        request: Typed postprocessing request.
        context: Runtime target and request context.
        allow_fallback: Whether declared compatibility fallback may be used.

    Returns:
        Effective request implementation and optional fallback reason.

    Raises:
        ModelRuntimeError: If the selected implementation is incompatible and no
            permitted compatible fallback exists.
    """
    requested_id = implementation.metadata.implementation_id

    def check(candidate: Postprocessor) -> CompatibilityResult:
        result = candidate.check_request_compatibility(
            request=request,
            context=context,
        )

        return result

    selection = _apply_declared_fallback(
        registry=registry,
        stage=OptimizationStage.POSTPROCESS,
        implementation=implementation,
        requested_id=requested_id,
        context=context,
        check_compatibility=check,
        allow_fallback=allow_fallback,
    )

    return selection


def _apply_declared_fallback(
    *,
    registry: ImplementationRegistry,
    stage: OptimizationStage,
    implementation: StageT,
    requested_id: str,
    context: ExecutionContext,
    check_compatibility: Callable[[StageT], CompatibilityResult],
    allow_fallback: bool,
) -> ImplementationSelection[StageT]:
    compatibility = check_compatibility(implementation)
    if compatibility.supported:
        selection = ImplementationSelection(
            implementation=implementation,
            requested_id=requested_id,
        )

        return selection
    if not allow_fallback:
        raise _unsupported_implementation_error(
            stage=stage,
            requested_id=requested_id,
            requested_reason=compatibility.reason,
            fallback_disabled=True,
        )

    fallback_id = implementation.metadata.fallback_id
    if fallback_id == implementation.metadata.implementation_id:
        raise _unsupported_implementation_error(
            stage=stage,
            requested_id=requested_id,
            requested_reason=compatibility.reason,
        )
    fallback = cast(
        StageT,
        registry.resolve(
            stage=stage,
            requested_id=fallback_id,
            context=context,
        ),
    )
    fallback_compatibility = check_compatibility(fallback)
    if not fallback_compatibility.supported:
        raise _unsupported_implementation_error(
            stage=stage,
            requested_id=requested_id,
            requested_reason=compatibility.reason,
            fallback_id=fallback_id,
            fallback_reason=fallback_compatibility.reason,
        )

    selection = ImplementationSelection(
        implementation=fallback,
        requested_id=requested_id,
        fallback_reason=compatibility.reason,
    )

    return selection


def _unsupported_implementation_error(
    *,
    stage: OptimizationStage,
    requested_id: str,
    requested_reason: str,
    fallback_id: Optional[str] = None,
    fallback_reason: Optional[str] = None,
    fallback_disabled: bool = False,
) -> ModelRuntimeError:
    details = f"{requested_id!r} is unsupported: {requested_reason}."
    if fallback_disabled:
        details += " Compatibility fallback is disabled by the execution plan."
    elif fallback_id is not None:
        details += f" Fallback {fallback_id!r} is unsupported: {fallback_reason}."
    error = ModelRuntimeError(
        message=f"RF-DETR {stage.value} cannot execute this contract. {details}",
        help_url=(
            "https://inference-models.roboflow.com/errors/models-runtime/"
            "#modelruntimeerror"
        ),
    )

    return error
