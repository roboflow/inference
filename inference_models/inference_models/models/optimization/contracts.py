"""Reusable metadata and runtime contracts for inference-path implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, Literal, Mapping, Optional, Protocol, Tuple


def immutable_mapping(values: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
    """Create an immutable shallow copy of a metadata mapping.

    Args:
        values: Optional source mapping.

    Returns:
        Read-only mapping detached from the source.
    """
    immutable = MappingProxyType(dict(values or {}))

    return immutable


@dataclass(frozen=True)
class CompatibilityResult:
    """Result of checking an implementation against a concrete contract."""

    supported: bool
    reasons: Tuple[str, ...] = ()

    @classmethod
    def compatible(cls) -> "CompatibilityResult":
        """Create a successful compatibility result.

        Returns:
            Result indicating that the implementation may execute.
        """
        result = cls(supported=True)

        return result

    @classmethod
    def incompatible(cls, *reasons: str) -> "CompatibilityResult":
        """Create an unsupported compatibility result.

        Args:
            *reasons: Human-readable incompatibility reasons.

        Returns:
            Result indicating that a fallback or error is required.
        """
        result = cls(supported=False, reasons=tuple(reasons))

        return result

    @property
    def reason(self) -> str:
        """Return all incompatibility reasons as one readable value.

        Returns:
            Comma-separated reasons, or an empty string when compatible.
        """
        reason = ", ".join(self.reasons)

        return reason


class OptimizationStage(str, Enum):
    """Common selectable stages in an inference path."""

    PREPROCESS = "preprocess"
    BUFFER_STRATEGY = "buffer_strategy"
    SCHEDULER = "scheduler"
    POSTPROCESS = "postprocess"
    ENGINE_PLUGIN = "engine_plugin"


@dataclass(frozen=True)
class DeviceCompatibility:
    """Hardware compatibility declared by one implementation."""

    device_kind: Literal["cpu", "gpu"]
    device_families: Tuple[str, ...] = ()
    minimum_compute_capability: Optional[Tuple[int, int]] = None


@dataclass(frozen=True)
class InputCompatibility:
    """Input constraints declared by one implementation."""

    scenarios: Tuple[str, ...]
    axis_constraints: Mapping[str, Any] = field(default_factory=immutable_mapping)
    dtypes: Tuple[str, ...] = ()
    layouts: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "axis_constraints",
            immutable_mapping(self.axis_constraints),
        )


@dataclass(frozen=True)
class ValidationRecord:
    """Reproducible identity of one successfully validated workload.

    ``docker_image`` should contain an immutable image digest rather than a mutable
    tag. Failed and inconclusive attempts are not validation records.
    """

    device_kind: Literal["cpu", "gpu"]
    device_name: str
    scenario: str
    profiler_commit: str
    runtime_commit: str
    docker_image: str
    model_id: str
    backend: str
    quantization: str


@dataclass(frozen=True)
class OptimizationMetadata:
    """Stable metadata for one inference-stage implementation."""

    implementation_id: str
    stage: OptimizationStage
    version: str
    target: DeviceCompatibility
    inputs: InputCompatibility
    dependencies: Tuple[str, ...]
    fallback_id: str
    changes_numerics: bool
    supports_concurrency: bool
    supports_cuda_graphs: bool
    output_contract: Mapping[str, Any] = field(default_factory=immutable_mapping)
    numerical_behavior: str = ""
    stream_behavior: str = ""
    validation_records: Tuple[ValidationRecord, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "output_contract",
            immutable_mapping(self.output_contract),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata for logs and runtime-profile records.

        Returns:
            JSON-compatible metadata dictionary.
        """
        validation_records = [
            {
                "device_kind": record.device_kind,
                "device_name": record.device_name,
                "scenario": record.scenario,
                "profiler_commit": record.profiler_commit,
                "runtime_commit": record.runtime_commit,
                "docker_image": record.docker_image,
                "model_id": record.model_id,
                "backend": record.backend,
                "quantization": record.quantization,
            }
            for record in self.validation_records
        ]
        serialized = {
            "implementation_id": self.implementation_id,
            "stage": self.stage.value,
            "version": self.version,
            "target": {
                "device_kind": self.target.device_kind,
                "device_families": list(self.target.device_families),
                "minimum_compute_capability": self.target.minimum_compute_capability,
            },
            "inputs": {
                "scenarios": list(self.inputs.scenarios),
                "axis_constraints": dict(self.inputs.axis_constraints),
                "dtypes": list(self.inputs.dtypes),
                "layouts": list(self.inputs.layouts),
            },
            "dependencies": list(self.dependencies),
            "fallback_id": self.fallback_id,
            "changes_numerics": self.changes_numerics,
            "supports_concurrency": self.supports_concurrency,
            "supports_cuda_graphs": self.supports_cuda_graphs,
            "output_contract": dict(self.output_contract),
            "numerical_behavior": self.numerical_behavior,
            "stream_behavior": self.stream_behavior,
            "validation_records": validation_records,
        }

        return serialized


@dataclass(frozen=True)
class ExecutionContext:
    """Runtime target and request context used for stage resolution."""

    device_kind: Literal["cpu", "gpu"]
    device: str
    device_name: str
    machine_type: str
    scenario: str
    resolved_axes: Mapping[str, Any] = field(default_factory=immutable_mapping)
    current_stream: Optional[Any] = None
    device_family: Optional[str] = None
    compute_capability: Optional[Tuple[int, int]] = None
    runtime_components: Mapping[str, bool] = field(default_factory=immutable_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "resolved_axes", immutable_mapping(self.resolved_axes))
        object.__setattr__(
            self,
            "runtime_components",
            immutable_mapping(self.runtime_components),
        )


def metadata_compatibility(
    metadata: OptimizationMetadata,
    context: ExecutionContext,
) -> CompatibilityResult:
    """Check declared target and dependency constraints against a runtime context.

    Runtime components are rejected only when the context explicitly reports them as
    unavailable. Components absent from the context remain unknown so existing model
    paths can adopt capability reporting incrementally.

    Args:
        metadata: Implementation compatibility metadata.
        context: Runtime target and available-component context.

    Returns:
        Compatibility result with actionable static-runtime reasons.
    """
    reasons = []
    target = metadata.target
    if target.device_kind != context.device_kind:
        reasons.append(
            f"requires device_kind={target.device_kind!r}, "
            f"received {context.device_kind!r}"
        )
    if (
        target.device_families
        and context.device_family is not None
        and context.device_family not in target.device_families
    ):
        reasons.append(
            f"device_family={context.device_family!r} is not in "
            f"{target.device_families!r}"
        )
    if (
        target.minimum_compute_capability is not None
        and context.compute_capability is not None
        and context.compute_capability < target.minimum_compute_capability
    ):
        reasons.append(
            f"compute_capability={context.compute_capability!r} is below "
            f"{target.minimum_compute_capability!r}"
        )
    unavailable_dependencies = [
        dependency
        for dependency in metadata.dependencies
        if context.runtime_components.get(dependency) is False
    ]
    if unavailable_dependencies:
        reasons.append(
            "unavailable runtime components: "
            + ", ".join(sorted(unavailable_dependencies))
        )

    if reasons:
        compatibility = CompatibilityResult.incompatible(*reasons)
    else:
        compatibility = CompatibilityResult.compatible()

    return compatibility


def metadata_supports_context(
    metadata: OptimizationMetadata,
    context: ExecutionContext,
) -> bool:
    """Return whether declared target constraints support a runtime context.

    Args:
        metadata: Implementation compatibility metadata.
        context: Runtime target and request context.

    Returns:
        Whether the target constraints match.
    """
    compatibility = metadata_compatibility(metadata=metadata, context=context)

    return compatibility.supported


class InferenceStage(Protocol):
    """Common interface implemented by every selectable stage."""

    metadata: OptimizationMetadata

    def is_compatible(self, context: ExecutionContext) -> bool:
        """Return whether the stage supports a runtime context.

        Args:
            context: Runtime target and request context.

        Returns:
            Whether the stage is compatible.
        """
