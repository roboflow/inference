"""Context-aware registry for selectable inference-stage implementations."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, DefaultDict, Dict, Generic, Optional, Tuple, TypeVar

from inference_models.errors import ModelRuntimeError
from inference_models.models.optimization.contracts import (
    ExecutionContext,
    InferenceStage,
    OptimizationMetadata,
    OptimizationStage,
    metadata_compatibility,
)
from inference_models.models.optimization.ids import (
    AUTO_IMPLEMENTATION_ID,
    BASE_IMPLEMENTATION_ID,
)

StageT = TypeVar("StageT", bound=InferenceStage)


@dataclass(frozen=True)
class ImplementationSelection(Generic[StageT]):
    """Requested and effective inference-stage selection."""

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


@dataclass
class _ImplementationRegistration:
    metadata: OptimizationMetadata
    factory: Callable[[], InferenceStage]
    implementation: Optional[InferenceStage] = None

    def materialize(self) -> InferenceStage:
        if self.implementation is None:
            self.implementation = self.factory()

        return self.implementation


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
        self._registrations: DefaultDict[
            OptimizationStage, Dict[str, _ImplementationRegistration]
        ] = defaultdict(dict)

    def register(self, implementation: InferenceStage) -> None:
        """Register one implementation by stage and stable ID.

        Args:
            implementation: Typed stage implementation.

        Raises:
            ValueError: If the stage and ID are already registered.
        """
        metadata = implementation.metadata
        self._register(
            _ImplementationRegistration(
                metadata=metadata,
                factory=lambda: implementation,
                implementation=implementation,
            )
        )

    def register_factory(
        self,
        *,
        metadata: OptimizationMetadata,
        factory: Callable[[], InferenceStage],
    ) -> None:
        """Register a lazily constructed implementation.

        Args:
            metadata: Static implementation metadata available before construction.
            factory: Zero-argument constructor for the implementation.
        """
        self._register(
            _ImplementationRegistration(
                metadata=metadata,
                factory=factory,
            )
        )

    def _register(self, registration: _ImplementationRegistration) -> None:
        metadata = registration.metadata
        stage_registrations = self._registrations[metadata.stage]
        if metadata.implementation_id in stage_registrations:
            raise ValueError(
                f"Duplicate {metadata.stage.value} implementation "
                f"{metadata.implementation_id!r}."
            )
        stage_registrations[metadata.implementation_id] = registration

    def resolve(
        self,
        *,
        stage: OptimizationStage,
        requested_id: str,
        context: ExecutionContext,
    ) -> InferenceStage:
        """Resolve one compatible implementation without compatibility fallback.

        Args:
            stage: Stage category being selected.
            requested_id: Stable implementation ID or the automatic-selection ID.
            context: Runtime target and request context.

        Returns:
            Compatible implementation instance.

        Raises:
            ModelRuntimeError: If the ID is unknown or incompatible.
        """
        selection = self.resolve_selection(
            stage=stage,
            requested_id=requested_id,
            context=context,
            allow_fallback=False,
        )

        return selection.implementation

    def resolve_selection(
        self,
        *,
        stage: OptimizationStage,
        requested_id: str,
        context: ExecutionContext,
        allow_fallback: bool,
    ) -> ImplementationSelection[InferenceStage]:
        """Resolve static compatibility, fallback, and lazy construction.

        Args:
            stage: Stage category being selected.
            requested_id: Stable implementation ID or the automatic-selection ID.
            context: Runtime target and available-component context.
            allow_fallback: Whether an incompatible explicit implementation may follow
                its declared fallback.

        Returns:
            Requested/effective selection and optional fallback reason.

        Raises:
            ModelRuntimeError: If no compatible implementation can be selected.
        """
        if requested_id == self._auto_id:
            registration = self._resolve_auto_registration(
                stage=stage,
                context=context,
            )
            implementation = registration.materialize()
            selection = ImplementationSelection(
                implementation=implementation,
                requested_id=requested_id,
            )

            return selection

        selection = self._resolve_explicit_selection(
            stage=stage,
            requested_id=requested_id,
            context=context,
            allow_fallback=allow_fallback,
            visited=(),
        )

        return selection

    def implementations(
        self,
        stage: OptimizationStage,
    ) -> Tuple[InferenceStage, ...]:
        """Return registered implementations for one stage.

        Args:
            stage: Stage category to inspect.

        Returns:
            Materialized implementations in registration order.
        """
        registered = tuple(
            registration.materialize()
            for registration in self._registrations.get(stage, {}).values()
        )

        return registered

    def _resolve_explicit_selection(
        self,
        *,
        stage: OptimizationStage,
        requested_id: str,
        context: ExecutionContext,
        allow_fallback: bool,
        visited: Tuple[str, ...],
    ) -> ImplementationSelection[InferenceStage]:
        stage_registrations = self._registrations.get(stage, {})
        registration = stage_registrations.get(requested_id)
        if registration is None:
            available = ", ".join(sorted([self._auto_id, *stage_registrations]))
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
        if requested_id in visited:
            chain = " -> ".join((*visited, requested_id))
            raise ModelRuntimeError(
                message=(
                    f"{self._scope_name} {stage.value} fallback cycle detected: "
                    f"{chain}."
                ),
                help_url=(
                    "https://inference-models.roboflow.com/errors/models-runtime/"
                    "#modelruntimeerror"
                ),
            )

        incompatibility_reason = self._registration_incompatibility_reason(
            registration=registration,
            context=context,
        )
        if incompatibility_reason is None:
            implementation = registration.materialize()
            if implementation.is_compatible(context):
                selection = ImplementationSelection(
                    implementation=implementation,
                    requested_id=requested_id,
                )

                return selection
            incompatibility_reason = "implementation rejected the runtime context"

        fallback_id = registration.metadata.fallback_id
        if not allow_fallback or fallback_id == requested_id:
            raise ModelRuntimeError(
                message=(
                    f"{self._scope_name} {stage.value} implementation "
                    f"{requested_id!r} is incompatible: {incompatibility_reason}."
                ),
                help_url=(
                    "https://inference-models.roboflow.com/errors/models-runtime/"
                    "#modelruntimeerror"
                ),
            )

        try:
            fallback_selection = self._resolve_explicit_selection(
                stage=stage,
                requested_id=fallback_id,
                context=context,
                allow_fallback=True,
                visited=(*visited, requested_id),
            )
        except ModelRuntimeError as error:
            raise ModelRuntimeError(
                message=(
                    f"{self._scope_name} {stage.value} implementation "
                    f"{requested_id!r} is incompatible: {incompatibility_reason}. "
                    f"Fallback {fallback_id!r} could not be selected: {error}"
                ),
                help_url=(
                    "https://inference-models.roboflow.com/errors/models-runtime/"
                    "#modelruntimeerror"
                ),
            ) from error

        selection = ImplementationSelection(
            implementation=fallback_selection.implementation,
            requested_id=requested_id,
            fallback_reason=incompatibility_reason,
        )

        return selection

    def _resolve_auto_registration(
        self,
        *,
        stage: OptimizationStage,
        context: ExecutionContext,
    ) -> _ImplementationRegistration:
        registrations = tuple(self._registrations.get(stage, {}).values())
        for registration in registrations:
            metadata = registration.metadata
            if metadata.implementation_id == self._base_id:
                continue
            validated = any(
                environment.matches(context)
                for environment in metadata.validated_environments
            )
            if (
                validated
                and self._registration_incompatibility_reason(
                    registration=registration,
                    context=context,
                )
                is None
            ):
                implementation = registration.materialize()
                if implementation.is_compatible(context):
                    return registration

        base = next(
            (
                registration
                for registration in registrations
                if registration.metadata.implementation_id == self._base_id
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
        base_reason = self._registration_incompatibility_reason(
            registration=base,
            context=context,
        )
        if base_reason is None:
            base_implementation = base.materialize()
            if not base_implementation.is_compatible(context):
                base_reason = "implementation rejected the runtime context"
        if base_reason is not None:
            raise ModelRuntimeError(
                message=(
                    f"{self._scope_name} {stage.value} base implementation is "
                    f"incompatible: {base_reason}."
                ),
                help_url=(
                    "https://inference-models.roboflow.com/errors/models-runtime/"
                    "#modelruntimeerror"
                ),
            )

        return base

    @staticmethod
    def _registration_incompatibility_reason(
        *,
        registration: _ImplementationRegistration,
        context: ExecutionContext,
    ) -> Optional[str]:
        compatibility = metadata_compatibility(
            metadata=registration.metadata,
            context=context,
        )
        if not compatibility.supported:
            return compatibility.reason
        if (
            registration.implementation is not None
            and not registration.implementation.is_compatible(context)
        ):
            return "implementation rejected the runtime context"

        return None
