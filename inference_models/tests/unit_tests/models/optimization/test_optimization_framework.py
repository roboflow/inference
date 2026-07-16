from dataclasses import dataclass

import pytest
import torch

from inference_models.errors import ModelRuntimeError
from inference_models.models.optimization.contracts import (
    CompatibilityResult,
    DeviceCompatibility,
    ExecutionContext,
    InputCompatibility,
    OptimizationMetadata,
    OptimizationStage,
    ValidationEnvironment,
)
from inference_models.models.optimization.execution_plan import InferenceExecutionPlan
from inference_models.models.optimization.registry import ImplementationRegistry
from inference_models.models.optimization.torch_readiness import TensorReadinessTracker


@dataclass(frozen=True)
class _State:
    implementation_id: str


class _Stage:
    def __init__(
        self,
        implementation_id: str,
        *,
        compatible: bool = True,
        validated_environments=(),
    ) -> None:
        self.metadata = OptimizationMetadata(
            implementation_id=implementation_id,
            stage=OptimizationStage.PREPROCESS,
            version="1",
            target=DeviceCompatibility(device_kind="gpu"),
            inputs=InputCompatibility(scenarios=("*",)),
            dependencies=(),
            fallback_id="base",
            changes_numerics=False,
            supports_concurrency=True,
            supports_cuda_graphs=False,
            validated_environments=validated_environments,
        )
        self._compatible = compatible

    def is_compatible(self, context: ExecutionContext) -> bool:
        return self._compatible


def _context() -> ExecutionContext:
    return ExecutionContext(
        device_kind="gpu",
        device="cuda:0",
        device_name="test-gpu",
        machine_type="test-machine",
        scenario="batch",
        resolved_axes={"batch": 1, "height": 640},
    )


def test_inference_execution_plan_defaults_and_serializes() -> None:
    plan = InferenceExecutionPlan()

    assert plan.to_dict() == {
        "preprocessor": "base",
        "buffer_strategy": "base",
        "scheduler": "base",
        "postprocessor": "base",
        "engine_plugin": "base",
    }


def test_compatibility_result_preserves_actionable_reasons() -> None:
    result = CompatibilityResult.incompatible("static crop", "grayscale")

    assert not result.supported
    assert result.reasons == ("static crop", "grayscale")
    assert result.reason == "static crop, grayscale"


def test_validation_environment_matches_target_and_scenario() -> None:
    validation = ValidationEnvironment(
        machine_type="test-machine",
        device_kind="gpu",
        device_name="test-gpu",
        scenario="batch",
        resolved_axes={"batch": 1},
        runtime_versions={"torch": "test"},
        source_commit="test",
        profiling_bundle="test-bundle",
        status="validated",
    )

    assert validation.matches(_context())
    assert not validation.matches(
        ExecutionContext(
            device_kind="gpu",
            device="cuda:0",
            device_name="other-gpu",
            machine_type="test-machine",
            scenario="batch",
            resolved_axes={"batch": 1},
        )
    )


def test_registry_uses_scope_in_actionable_errors() -> None:
    registry = ImplementationRegistry(scope_name="Example model")
    registry.register(_Stage("base"))

    with pytest.raises(ModelRuntimeError, match="Unknown Example model preprocess"):
        registry.resolve(
            stage=OptimizationStage.PREPROCESS,
            requested_id="missing",
            context=_context(),
        )


def test_registry_auto_selects_only_matching_validated_candidate() -> None:
    validation = ValidationEnvironment(
        machine_type="test-machine",
        device_kind="gpu",
        device_name="test-gpu",
        scenario="batch",
        resolved_axes={"batch": 1},
        runtime_versions={},
        source_commit="test",
        profiling_bundle="test-bundle",
        status="validated",
    )
    registry = ImplementationRegistry(scope_name="Example model")
    base = _Stage("base")
    candidate = _Stage("candidate", validated_environments=(validation,))
    registry.register(base)
    registry.register(candidate)

    selected = registry.resolve(
        stage=OptimizationStage.PREPROCESS,
        requested_id="auto",
        context=_context(),
    )

    assert selected is candidate


def test_tensor_readiness_tracker_uses_exact_tensor_identity() -> None:
    tracker = TensorReadinessTracker[_State]()
    tensor = torch.zeros(1)
    other = torch.zeros(1)
    tracker.record(tensor, state=_State(implementation_id="candidate"))

    assert tracker.consume(other) is None
    assert tracker.consume(tensor) == _State(implementation_id="candidate")
    assert tracker.consume(tensor) is None
