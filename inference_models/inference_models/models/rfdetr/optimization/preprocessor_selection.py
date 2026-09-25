"""Model-owned preprocessing eligibility cache and request-local fallback traversal.

The target, dependency snapshot, registry and model configuration are fixed for
the selector's lifetime. Recreate it if any of those change. Request inputs,
overrides, CUDA streams and runtime-failure state are never cached.
"""

import threading
from dataclasses import dataclass, replace
from typing import Dict, Optional, cast

from inference_models.errors import ModelRuntimeError
from inference_models.models.common.roboflow.model_packages import (
    ImagePreProcessing,
    NetworkInputDefinition,
)
from inference_models.models.optimization.contracts import (
    CompatibilityResult,
    ExecutionContext,
    OptimizationMetadata,
    OptimizationStage,
)
from inference_models.models.optimization.errors import RecoverableStageExecutionError
from inference_models.models.optimization.registry import (
    ImplementationRegistry,
    ImplementationSelection,
)
from inference_models.models.rfdetr.optimization.contracts import (
    Preprocessor,
    PreprocessRequest,
)


@dataclass(frozen=True)
class _Candidate:
    metadata: OptimizationMetadata
    implementation: Optional[Preprocessor]
    compatibility: CompatibilityResult


class PreprocessorSelector:
    """Cache static/model eligibility once per model, not request compatibility.

    Args:
        registry (ImplementationRegistry): Model-owned lazy implementation registry.
        context (ExecutionContext): Fixed device and dependency snapshot.
        image_pre_processing (ImagePreProcessing): Fixed model transformations.
        network_input (NetworkInputDefinition): Fixed model input configuration.
    """

    def __init__(
        self,
        *,
        registry: ImplementationRegistry,
        context: ExecutionContext,
        image_pre_processing: ImagePreProcessing,
        network_input: NetworkInputDefinition,
    ) -> None:
        self._registry = registry
        self._context = replace(context, current_stream=None)
        self._image_pre_processing = image_pre_processing
        self._network_input = network_input
        self._candidates: Dict[str, _Candidate] = {}
        self._lock = threading.Lock()

    def _candidate(self, implementation_id: str) -> _Candidate:
        candidate = self._candidates.get(implementation_id)
        if candidate is not None:
            return candidate

        # Serialize cold construction only. Warm dispatch never takes this lock.
        with self._lock:
            candidate = self._candidates.get(implementation_id)
            if candidate is not None:
                return candidate

            metadata = self._registry.metadata(
                stage=OptimizationStage.PREPROCESS, implementation_id=implementation_id
            )
            implementation, compatibility = self._registry.inspect_candidate(
                stage=OptimizationStage.PREPROCESS,
                implementation_id=implementation_id,
                context=self._context,
            )
            implementation = cast(Optional[Preprocessor], implementation)
            if compatibility.supported:
                compatibility = implementation.check_model_compatibility(
                    image_pre_processing=self._image_pre_processing,
                    network_input=self._network_input,
                )
            candidate = _Candidate(metadata, implementation, compatibility)
            self._candidates[implementation_id] = candidate

        return candidate

    def resolve_model(
        self, *, requested_id: str, allow_fallback: bool
    ) -> ImplementationSelection[Preprocessor]:
        """Choose a model-compatible primary without preparing unused fallbacks.

        Args:
            requested_id (str): Explicit implementation or auto.
            allow_fallback (bool): Permit fallback for an incompatible explicit ID.

        Returns:
            ImplementationSelection: Model-level selection.

        Raises:
            ModelRuntimeError: If no eligible implementation can be selected.
        """
        if requested_id == "auto":
            for implementation_id in self._registry.auto_preferences(
                stage=OptimizationStage.PREPROCESS
            ):
                candidate = self._candidate(implementation_id)
                if candidate.compatibility.supported:
                    selection = ImplementationSelection(
                        candidate.implementation, requested_id
                    )

                    return selection

            raise self._error(requested_id, "No model-compatible auto candidate")

        selection = self._walk(
            implementation_id=requested_id,
            requested_id=requested_id,
            allow_fallback=allow_fallback,
        )

        return selection

    def resolve_request(
        self,
        *,
        implementation: Preprocessor,
        request: PreprocessRequest,
        context: ExecutionContext,
        allow_fallback: bool,
    ) -> ImplementationSelection[Preprocessor]:
        """Check this request from the stored primary, without sticky fallback.

        Args:
            implementation (Preprocessor): Model-selected primary stage.
            request (PreprocessRequest): Current inputs and overrides.
            context (ExecutionContext): Current stream on the model's fixed target.
            allow_fallback (bool): Permit walking compatibility fallbacks.

        Returns:
            ImplementationSelection: Request-compatible stage.

        Raises:
            ModelRuntimeError: If the chain is invalid or no candidate is compatible.
        """
        selection = self._walk(
            implementation_id=implementation.metadata.implementation_id,
            requested_id=implementation.metadata.implementation_id,
            allow_fallback=allow_fallback,
            request=request,
            context=context,
        )

        return selection

    def resolve_runtime_fallback(
        self,
        *,
        selection: ImplementationSelection[Preprocessor],
        request: PreprocessRequest,
        context: ExecutionContext,
        allow_fallback: bool,
    ) -> ImplementationSelection[Preprocessor]:
        """Check fresh failure state without repeating the selected request check.

        Args:
            selection (ImplementationSelection): Already request-validated stage.
            request (PreprocessRequest): Current inputs and overrides.
            context (ExecutionContext): Current execution stream and target.
            allow_fallback (bool): Permit recovery from recorded runtime failures.

        Returns:
            ImplementationSelection: Unchanged stage or fully validated fallback.

        Raises:
            RecoverableStageExecutionError: If failure recovery is disabled.
            ModelRuntimeError: If no valid fallback exists.
        """
        implementation = selection.implementation
        compatibility = implementation.check_runtime_compatibility(
            request=request, context=context
        )
        if compatibility.supported:
            return selection

        if not allow_fallback:
            raise RecoverableStageExecutionError(
                message=f"RF-DETR preprocess runtime failure: {compatibility.reason}. Runtime failure fallback is disabled by the execution plan."
            )

        reasons = [
            reason
            for reason in (selection.fallback_reason, compatibility.reason)
            if reason
        ]
        fallback = self._walk(
            implementation_id=implementation.metadata.fallback_id,
            requested_id=selection.requested_id,
            allow_fallback=True,
            request=request,
            context=context,
            check_runtime=True,
            reasons=reasons,
            visited={implementation.metadata.implementation_id},
        )

        return fallback

    def _walk(
        self,
        *,
        implementation_id,
        requested_id,
        allow_fallback,
        request=None,
        context=None,
        check_runtime=False,
        reasons=None,
        visited=None,
    ):
        reasons = [] if reasons is None else list(reasons)
        visited = set() if visited is None else set(visited)
        while True:
            if implementation_id in visited:
                raise self._error(
                    requested_id, f"fallback cycle detected at {implementation_id!r}"
                )

            visited.add(implementation_id)
            candidate = self._candidate(implementation_id)
            compatibility = candidate.compatibility
            if compatibility.supported and request is not None:
                compatibility = candidate.implementation.check_request_compatibility(
                    request=request, context=context
                )
            if compatibility.supported and check_runtime:
                compatibility = candidate.implementation.check_runtime_compatibility(
                    request=request, context=context
                )
            if compatibility.supported:
                selection = ImplementationSelection(
                    candidate.implementation,
                    requested_id,
                    fallback_reason="; ".join(reasons) or None,
                )
                return selection

            reasons.append(compatibility.reason)
            if not allow_fallback:
                raise self._error(
                    requested_id,
                    "; ".join(reasons) + ". Compatibility fallback is disabled",
                )

            fallback_id = candidate.metadata.fallback_id
            if fallback_id == implementation_id:
                raise self._error(
                    requested_id,
                    f"Fallback {implementation_id!r} is unsupported: "
                    + "; ".join(reasons),
                )

            implementation_id = fallback_id

    @staticmethod
    def _error(requested_id, reason):
        error = ModelRuntimeError(
            message=f"RF-DETR preprocess {requested_id!r} is incompatible with this contract: {reason}.",
            help_url="https://inference-models.roboflow.com/errors/models-runtime/#modelruntimeerror",
        )
        return error
