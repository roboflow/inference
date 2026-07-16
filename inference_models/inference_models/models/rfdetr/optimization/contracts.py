"""Shared RF-DETR optimization metadata and stage contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, List, Literal, Mapping, Optional, Protocol, Tuple, Union

import numpy as np
import torch

from inference_models import Detections, PreProcessingOverrides
from inference_models.entities import ColorFormat
from inference_models.models.common.roboflow.model_packages import (
    ImagePreProcessing,
    NetworkInputDefinition,
    PreProcessingMetadata,
)
from inference_models.models.rfdetr.class_remapping import ClassesReMapping


def immutable_mapping(values: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
    """Create an immutable shallow copy of a metadata mapping.

    Args:
        values: Optional source mapping.

    Returns:
        Read-only mapping detached from the source.
    """
    immutable = MappingProxyType(dict(values or {}))

    return immutable


class OptimizationStage(str, Enum):
    """Selectable stage in the RF-DETR inference path."""

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
class ValidationEnvironment:
    """One measured environment for an implementation."""

    machine_type: str
    device_kind: Literal["cpu", "gpu"]
    device_name: str
    scenario: str
    resolved_axes: Mapping[str, Any]
    runtime_versions: Mapping[str, str]
    source_commit: str
    profiling_bundle: str
    status: Literal["validated", "failed", "inconclusive"]

    def __post_init__(self) -> None:
        object.__setattr__(self, "resolved_axes", immutable_mapping(self.resolved_axes))
        object.__setattr__(
            self,
            "runtime_versions",
            immutable_mapping(self.runtime_versions),
        )

    def matches(self, context: "ExecutionContext") -> bool:
        """Return whether this validation record matches a runtime context.

        Args:
            context: Runtime context being resolved.

        Returns:
            Whether target and scenario identity match.
        """
        target_matches = (
            self.status == "validated"
            and self.device_kind == context.device_kind
            and self.device_name == context.device_name
        )
        scenario_matches = self.scenario in {context.scenario, "*"}

        return target_matches and scenario_matches


@dataclass(frozen=True)
class OptimizationMetadata:
    """Stable metadata for one RF-DETR stage implementation."""

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
    validated_environments: Tuple[ValidationEnvironment, ...] = ()

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
        environments = [
            {
                "machine_type": environment.machine_type,
                "device_kind": environment.device_kind,
                "device_name": environment.device_name,
                "scenario": environment.scenario,
                "resolved_axes": dict(environment.resolved_axes),
                "runtime_versions": dict(environment.runtime_versions),
                "source_commit": environment.source_commit,
                "profiling_bundle": environment.profiling_bundle,
                "status": environment.status,
            }
            for environment in self.validated_environments
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
            "validated_environments": environments,
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
    current_stream: Optional[torch.cuda.Stream] = None
    device_family: Optional[str] = None
    compute_capability: Optional[Tuple[int, int]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "resolved_axes", immutable_mapping(self.resolved_axes))


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
    target = metadata.target
    if target.device_kind != context.device_kind:
        return False
    if (
        target.device_families
        and context.device_family is not None
        and context.device_family not in target.device_families
    ):
        return False
    if (
        target.minimum_compute_capability is not None
        and context.compute_capability is not None
        and context.compute_capability < target.minimum_compute_capability
    ):
        return False

    return True


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
    ready_event: Optional[torch.cuda.Event] = None
    input_kind: str = "reference"


@dataclass(frozen=True)
class PostprocessRequest:
    """Inputs required by an RF-DETR postprocessing implementation."""

    bboxes: torch.Tensor
    logits: torch.Tensor
    pre_processing_meta: List[PreProcessingMetadata]
    threshold: Union[float, torch.Tensor]
    num_classes: int
    classes_re_mapping: Optional[ClassesReMapping]


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


class Preprocessor(Protocol):
    """RF-DETR preprocessing stage interface."""

    metadata: OptimizationMetadata

    def is_compatible(self, context: ExecutionContext) -> bool:
        """Return whether the preprocessor supports a runtime context.

        Args:
            context: Runtime target and request context.

        Returns:
            Whether the preprocessor is compatible.
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


class Postprocessor(Protocol):
    """RF-DETR postprocessing stage interface."""

    metadata: OptimizationMetadata

    def is_compatible(self, context: ExecutionContext) -> bool:
        """Return whether the postprocessor supports a runtime context.

        Args:
            context: Runtime target and request context.

        Returns:
            Whether the postprocessor is compatible.
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
