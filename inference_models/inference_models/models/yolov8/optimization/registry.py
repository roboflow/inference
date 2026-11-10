"""Postprocess selection for YOLOv8 TensorRT object detection.

Selection order, each step falling back to ``base`` with a readable reason::

    static     registry: device, packages, model class   (no Triton import)
    request    shape, dtype, batch, options, metadata    (no GPU reads)
    runtime    one-time kernel calibration per device    (imports Triton)

``base`` and ``auto`` select the original decoder. Import, compilation and CUDA
errors propagate; only an inconclusive calibration falls back.
"""

from typing import Mapping, Optional

from inference_models.errors import ModelRuntimeError
from inference_models.models.optimization.contracts import (
    ExecutionContext,
    OptimizationStage,
    immutable_mapping,
)
from inference_models.models.optimization.ids import AUTO_IMPLEMENTATION_ID
from inference_models.models.optimization.registry import (
    ImplementationRegistry,
    ImplementationSelection,
)

from .base import BasePostprocessor
from .contracts import PostprocessRequest
from .segmented import (
    MODEL_COMPONENT,
    SegmentedPostprocessor,
    segmented_runtime_components,
)

POSTPROCESS_SCOPE = "YOLOv8 TensorRT object detection"


def create_postprocess_registry(
    *, segmented: Optional[SegmentedPostprocessor] = None
) -> ImplementationRegistry:
    """Create the postprocess registry of one model instance.

    Args:
        segmented: Segmented stage to register; None creates one that builds
            its Triton kernels on first use.

    Returns:
        Registry holding ``base`` and ``segmented``; ``auto`` resolves to base.
    """
    if segmented is None:
        segmented = SegmentedPostprocessor()

    registry = ImplementationRegistry(scope_name=POSTPROCESS_SCOPE)
    registry.register(BasePostprocessor())
    registry.register(segmented)

    return registry


def validate_postprocess_implementation(implementation_id: str) -> None:
    """Reject an unknown postprocess implementation ID before model loading.

    Args:
        implementation_id: Requested ID, ``auto`` or a registered ID.

    Raises:
        ModelRuntimeError: If the ID is not registered.
    """
    if implementation_id == AUTO_IMPLEMENTATION_ID:
        return

    create_postprocess_registry().metadata(
        stage=OptimizationStage.POSTPROCESS, implementation_id=implementation_id
    )


def postprocess_runtime_components(*, model_qualified: bool) -> Mapping[str, bool]:
    """Collect the static facts the segmented stage depends on.

    Args:
        model_qualified: Whether the model is exactly the qualified class;
            subclasses with other exports are not qualified.

    Returns:
        Read-only component availability for ``ExecutionContext``.
    """
    components = immutable_mapping(
        {**segmented_runtime_components(), MODEL_COMPONENT: model_qualified}
    )

    return components


def select_postprocessor(
    *,
    registry: ImplementationRegistry,
    requested_id: str,
    request: PostprocessRequest,
    context: ExecutionContext,
) -> ImplementationSelection:
    """Select the stage for one request: static, then request, then runtime.

    Args:
        registry: Registry from ``create_postprocess_registry``.
        requested_id: ``base``, ``auto`` or a registered ID.
        request: Resolved output, metadata and options.
        context: Target, caller stream and runtime components.

    Returns:
        Requested and effective selection, with the fallback reason if any.

    Raises:
        ModelRuntimeError: If the ID is unknown or no fallback is compatible.
    """
    selection = registry.resolve_selection(
        stage=OptimizationStage.POSTPROCESS,
        requested_id=requested_id,
        context=context,
        allow_fallback=True,
    )
    if selection.used_fallback:
        return selection

    implementation = selection.implementation
    compatibility = implementation.check_request_compatibility(
        request=request, context=context
    )
    if compatibility.supported:
        compatibility = implementation.check_runtime_compatibility(
            request=request, context=context
        )
    if compatibility.supported:
        return selection

    fallback_id = implementation.metadata.fallback_id
    if fallback_id == selection.effective_id:
        raise ModelRuntimeError(
            message=(
                f"{POSTPROCESS_SCOPE} postprocess implementation "
                f"{fallback_id!r} rejected the request: {compatibility.reason}."
            ),
            help_url="https://inference-models.roboflow.com/errors/models-runtime/#modelruntimeerror",
        )

    fallback = registry.resolve_selection(
        stage=OptimizationStage.POSTPROCESS,
        requested_id=fallback_id,
        context=context,
        allow_fallback=False,
    )
    selection = ImplementationSelection(
        implementation=fallback.implementation,
        requested_id=requested_id,
        fallback_reason=compatibility.reason,
    )

    return selection
