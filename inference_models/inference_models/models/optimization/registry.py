"""Context-aware registry for selectable inference-stage implementations."""

from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, Tuple

from inference_models.errors import ModelRuntimeError
from inference_models.models.optimization.contracts import (
    ExecutionContext,
    InferenceStage,
    OptimizationStage,
)
from inference_models.models.optimization.ids import (
    AUTO_IMPLEMENTATION_ID,
    BASE_IMPLEMENTATION_ID,
)


class ImplementationRegistry:
    """Register and resolve typed inference-stage implementations."""

    def __init__(
        self,
        *,
        scope_name: str,
        base_id: str = BASE_IMPLEMENTATION_ID,
        auto_id: str = AUTO_IMPLEMENTATION_ID,
    ) -> None:
        self._scope_name = scope_name
        self._base_id = base_id
        self._auto_id = auto_id
        self._implementations: DefaultDict[
            OptimizationStage, Dict[str, InferenceStage]
        ] = defaultdict(dict)

    def register(self, implementation: InferenceStage) -> None:
        """Register one implementation by stage and stable ID.

        Args:
            implementation: Typed stage implementation.

        Raises:
            ValueError: If the stage and ID are already registered.
        """
        metadata = implementation.metadata
        stage_implementations = self._implementations[metadata.stage]
        if metadata.implementation_id in stage_implementations:
            raise ValueError(
                f"Duplicate {metadata.stage.value} implementation "
                f"{metadata.implementation_id!r}."
            )
        stage_implementations[metadata.implementation_id] = implementation

    def resolve(
        self,
        *,
        stage: OptimizationStage,
        requested_id: str,
        context: ExecutionContext,
    ) -> InferenceStage:
        """Resolve one compatible explicit or automatic implementation.

        Args:
            stage: Stage category being selected.
            requested_id: Stable implementation ID or the automatic-selection ID.
            context: Runtime target and request context.

        Returns:
            Compatible implementation instance.

        Raises:
            ModelRuntimeError: If the ID is unknown or incompatible.
        """
        stage_implementations = self._implementations.get(stage, {})
        if requested_id == self._auto_id:
            implementation = self._resolve_auto(
                stage=stage,
                implementations=stage_implementations.values(),
                context=context,
            )

            return implementation

        implementation = stage_implementations.get(requested_id)
        if implementation is None:
            available = ", ".join(sorted([self._auto_id, *stage_implementations]))
            raise ModelRuntimeError(
                message=(
                    f"Unknown {self._scope_name} {stage.value} implementation "
                    f"{requested_id!r}. Available implementations: {available}."
                ),
                help_url=(
                    "https://inference-models.roboflow.com/errors/models-runtime/"
                    "#modelruntimeerror"
                ),
            )
        if not implementation.is_compatible(context):
            raise ModelRuntimeError(
                message=(
                    f"{self._scope_name} {stage.value} implementation "
                    f"{requested_id!r} is not compatible with "
                    f"device={context.device!r}, "
                    f"device_kind={context.device_kind!r}, "
                    f"device_family={context.device_family!r}."
                ),
                help_url=(
                    "https://inference-models.roboflow.com/errors/models-runtime/"
                    "#modelruntimeerror"
                ),
            )

        return implementation

    def implementations(
        self,
        stage: OptimizationStage,
    ) -> Tuple[InferenceStage, ...]:
        """Return registered implementations for one stage.

        Args:
            stage: Stage category to inspect.

        Returns:
            Implementations in registration order.
        """
        registered = tuple(self._implementations.get(stage, {}).values())

        return registered

    def _resolve_auto(
        self,
        *,
        stage: OptimizationStage,
        implementations: Iterable[InferenceStage],
        context: ExecutionContext,
    ) -> InferenceStage:
        implementations = tuple(implementations)
        for implementation in implementations:
            metadata = implementation.metadata
            if metadata.implementation_id == self._base_id:
                continue
            validated = any(
                environment.matches(context)
                for environment in metadata.validated_environments
            )
            if validated and implementation.is_compatible(context):
                return implementation

        base = next(
            (
                implementation
                for implementation in implementations
                if implementation.metadata.implementation_id == self._base_id
            ),
            None,
        )
        if base is None:
            raise ModelRuntimeError(
                message=(
                    f"{self._scope_name} {stage.value} registry has no "
                    f"{self._base_id!r} implementation."
                ),
                help_url=(
                    "https://inference-models.roboflow.com/errors/models-runtime/"
                    "#modelruntimeerror"
                ),
            )

        return base
