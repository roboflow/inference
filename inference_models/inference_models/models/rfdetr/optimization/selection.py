"""RF-DETR stage selection with declared compatibility fallback."""

from typing import Callable, Optional, TypeVar, cast

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
from inference_models.models.optimization.errors import RecoverableStageExecutionError
from inference_models.models.optimization.registry import (
    ImplementationRegistry,
    ImplementationSelection,
)
from inference_models.models.rfdetr.optimization.contracts import (
    Postprocessor,
    PostprocessRequest,
    Preprocessor,
    PreprocessRequest,
)

StageT = TypeVar("StageT", bound=InferenceStage)


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
    static_selection = registry.resolve_selection(
        stage=OptimizationStage.PREPROCESS,
        requested_id=requested_id,
        context=context,
        allow_fallback=allow_fallback,
    )
    implementation = cast(
        Preprocessor,
        static_selection.implementation,
    )

    def check(candidate: Preprocessor) -> CompatibilityResult:
        result = candidate.check_model_compatibility(
            image_pre_processing=image_pre_processing,
            network_input=network_input,
        )

        return result

    compatibility = check(implementation)
    if compatibility.supported:
        selection = cast(
            ImplementationSelection[Preprocessor],
            static_selection,
        )

        return selection

    selection = _apply_declared_fallback(
        registry=registry,
        stage=OptimizationStage.PREPROCESS,
        implementation=implementation,
        requested_id=static_selection.requested_id,
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


def resolve_preprocessor_runtime_fallback(
    *,
    registry: ImplementationRegistry,
    selection: ImplementationSelection[Preprocessor],
    request: PreprocessRequest,
    context: ExecutionContext,
    allow_fallback: bool,
) -> ImplementationSelection[Preprocessor]:
    """Resolve whether preprocessing must follow a runtime failure fallback.

    Args:
        registry: RF-DETR implementation registry.
        selection: Request-compatible preprocessing selection.
        request: Typed preprocessing request.
        context: Runtime target and request context.
        allow_fallback: Whether a recorded runtime failure may use the declared
            fallback.

    Returns:
        Original selection when its runtime remains available, otherwise its
        declared compatible fallback.

    Raises:
        RecoverableStageExecutionError: If execution failed and fallback is
            unavailable or disabled.
    """
    implementation = selection.implementation
    runtime_compatibility = implementation.check_runtime_compatibility(
        request=request,
        context=context,
    )
    if runtime_compatibility.supported:
        return selection
    if not allow_fallback:
        raise RecoverableStageExecutionError(
            message=(
                "RF-DETR preprocess implementation cannot execute after a "
                f"recoverable runtime failure: {runtime_compatibility.reason}. "
                "Runtime failure fallback is disabled by the execution plan."
            ),
            help_url=(
                "https://inference-models.roboflow.com/errors/models-runtime/"
                "#modelruntimeerror"
            ),
        )

    def check(candidate: Preprocessor) -> CompatibilityResult:
        request_compatibility = candidate.check_request_compatibility(
            request=request,
            context=context,
        )
        candidate_runtime_compatibility = candidate.check_runtime_compatibility(
            request=request,
            context=context,
        )
        if (
            request_compatibility.supported
            and candidate_runtime_compatibility.supported
        ):
            return CompatibilityResult.compatible()

        return CompatibilityResult.incompatible(
            *request_compatibility.reasons,
            *candidate_runtime_compatibility.reasons,
        )

    fallback_selection = _apply_declared_fallback(
        registry=registry,
        stage=OptimizationStage.PREPROCESS,
        implementation=implementation,
        requested_id=selection.requested_id,
        context=context,
        check_compatibility=check,
        allow_fallback=allow_fallback,
    )

    return fallback_selection


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


def resolve_postprocessor_runtime_fallback(
    *,
    registry: ImplementationRegistry,
    selection: ImplementationSelection[Postprocessor],
    request: PostprocessRequest,
    context: ExecutionContext,
    allow_fallback: bool,
) -> ImplementationSelection[Postprocessor]:
    """Resolve whether postprocessing must follow a runtime failure fallback.

    Args:
        registry: RF-DETR implementation registry.
        selection: Request-compatible postprocessing selection.
        request: Typed postprocessing request.
        context: Runtime target and request context.
        allow_fallback: Whether a recorded runtime failure may use the declared
            fallback.

    Returns:
        Original selection when its runtime remains available, otherwise its
        declared compatible fallback.

    Raises:
        RecoverableStageExecutionError: If execution failed and fallback is
            unavailable or disabled.
    """
    implementation = selection.implementation
    runtime_compatibility = implementation.check_runtime_compatibility(
        request=request,
        context=context,
    )
    if runtime_compatibility.supported:
        return selection
    if not allow_fallback:
        raise RecoverableStageExecutionError(
            message=(
                "RF-DETR postprocess implementation cannot execute after a "
                f"recoverable runtime failure: {runtime_compatibility.reason}. "
                "Runtime failure fallback is disabled by the execution plan."
            ),
            help_url=(
                "https://inference-models.roboflow.com/errors/models-runtime/"
                "#modelruntimeerror"
            ),
        )

    def check(candidate: Postprocessor) -> CompatibilityResult:
        request_compatibility = candidate.check_request_compatibility(
            request=request,
            context=context,
        )
        candidate_runtime_compatibility = candidate.check_runtime_compatibility(
            request=request,
            context=context,
        )
        if (
            request_compatibility.supported
            and candidate_runtime_compatibility.supported
        ):
            return CompatibilityResult.compatible()

        return CompatibilityResult.incompatible(
            *request_compatibility.reasons,
            *candidate_runtime_compatibility.reasons,
        )

    fallback_selection = _apply_declared_fallback(
        registry=registry,
        stage=OptimizationStage.POSTPROCESS,
        implementation=implementation,
        requested_id=selection.requested_id,
        context=context,
        check_compatibility=check,
        allow_fallback=allow_fallback,
    )

    return fallback_selection


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
