from dataclasses import dataclass

import pytest
import torch

from inference_models.errors import ModelRuntimeError
from inference_models.models.optimization import (
    runtime_components as runtime_components_module,
)
from inference_models.models.optimization.contracts import (
    CompatibilityResult,
    DeviceCompatibility,
    ExecutionContext,
    InputCompatibility,
    OptimizationMetadata,
    OptimizationStage,
    ValidationRecord,
)
from inference_models.models.optimization.execution_plan import InferenceExecutionPlan
from inference_models.models.optimization.registry import ImplementationRegistry
from inference_models.models.optimization.runtime_components import (
    get_runtime_components,
)
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
        dependencies=(),
        validation_records=(),
    ) -> None:
        self.metadata = OptimizationMetadata(
            implementation_id=implementation_id,
            stage=OptimizationStage.PREPROCESS,
            version="1",
            target=DeviceCompatibility(device_kind="gpu"),
            inputs=InputCompatibility(scenarios=("*",)),
            dependencies=dependencies,
            fallback_id="base",
            changes_numerics=False,
            supports_concurrency=True,
            supports_cuda_graphs=False,
            validation_records=validation_records,
        )
        self._compatible = compatible

    def is_compatible(self, context: ExecutionContext) -> bool:
        return self._compatible


def _context() -> ExecutionContext:
    return ExecutionContext(
        device_kind="gpu",
        device="cuda:0",
    )


def test_inference_execution_plan_defaults_and_serializes() -> None:
    plan = InferenceExecutionPlan()

    assert plan.to_dict() == {
        "preprocessor": "base",
        "buffer_strategy": "base",
        "scheduler": "base",
        "postprocessor": "base",
        "engine_plugin": "base",
        "allow_compatibility_fallback": True,
        "allow_runtime_failure_fallback": True,
    }


def test_compatibility_result_preserves_actionable_reasons() -> None:
    result = CompatibilityResult.incompatible("static crop", "grayscale")

    assert not result.supported
    assert result.reasons == ("static crop", "grayscale")
    assert result.reason == "static crop, grayscale"


def test_runtime_component_discovery_centralizes_package_import_checks(
    monkeypatch,
) -> None:
    def import_module(module_name: str):
        if module_name == "triton":
            raise ImportError("triton is unavailable")

        return object()

    get_runtime_components.cache_clear()
    monkeypatch.setattr(
        runtime_components_module.importlib, "import_module", import_module
    )
    try:
        components = get_runtime_components()
    finally:
        get_runtime_components.cache_clear()

    assert components == {
        "Pillow": True,
        "TensorRT": True,
        "torch": True,
        "torchvision": True,
        "triton": False,
    }


def test_validation_record_is_serialized_as_informational_metadata() -> None:
    validation = ValidationRecord(
        device_kind="gpu",
        device_name="test-gpu",
        scenario="batch",
        profiler_commit="profiler-commit",
        runtime_commit="runtime-commit",
        docker_image="roboflow/inference:validation",
        model_id="model",
        backend="tensorrt",
        quantization="none",
    )
    metadata = _Stage("candidate", validation_records=(validation,)).metadata

    assert metadata.validation_records == (validation,)
    assert metadata.to_dict()["validation_records"] == [
        {
            "device_kind": "gpu",
            "device_name": "test-gpu",
            "scenario": "batch",
            "profiler_commit": "profiler-commit",
            "runtime_commit": "runtime-commit",
            "docker_image": "roboflow/inference:validation",
            "model_id": "model",
            "backend": "tensorrt",
            "quantization": "none",
        }
    ]


def test_registry_uses_scope_in_actionable_errors() -> None:
    registry = ImplementationRegistry(scope_name="Example model")
    registry.register(_Stage("base"))

    with pytest.raises(ModelRuntimeError, match="Unknown Example model preprocess"):
        registry.resolve(
            stage=OptimizationStage.PREPROCESS,
            requested_id="missing",
            context=_context(),
        )


def test_registry_auto_selects_first_compatible_preference() -> None:
    registry = ImplementationRegistry(scope_name="Example model")
    base = _Stage("base")
    first = _Stage("first", compatible=False)
    candidate = _Stage("candidate")
    registry.register(base)
    registry.register(first)
    registry.register(candidate)
    registry.set_auto_preferences(
        stage=OptimizationStage.PREPROCESS,
        implementation_ids=("first", "candidate"),
    )

    selected = registry.resolve(
        stage=OptimizationStage.PREPROCESS,
        requested_id="auto",
        context=_context(),
    )

    assert selected is candidate


@pytest.mark.parametrize(
    "implementation_ids",
    [
        ("base",),
        ("auto",),
        ("candidate", "candidate"),
        ("missing",),
    ],
)
def test_registry_rejects_invalid_auto_preferences(implementation_ids) -> None:
    registry = ImplementationRegistry(scope_name="Example model")
    registry.register(_Stage("base"))
    registry.register(_Stage("candidate"))

    with pytest.raises(ValueError):
        registry.set_auto_preferences(
            stage=OptimizationStage.PREPROCESS,
            implementation_ids=implementation_ids,
        )


def test_registry_static_dependency_fallback_is_lazy() -> None:
    registry = ImplementationRegistry(scope_name="Example model")
    constructed = []
    base_metadata = _Stage("base").metadata
    candidate_metadata = _Stage(
        "candidate",
        dependencies=("triton",),
    ).metadata
    registry.register_factory(
        metadata=base_metadata,
        factory=lambda: constructed.append("base") or _Stage("base"),
    )
    registry.register_factory(
        metadata=candidate_metadata,
        factory=lambda: constructed.append("candidate")
        or _Stage(
            "candidate",
            dependencies=("triton",),
        ),
    )
    context = ExecutionContext(
        device_kind="gpu",
        device="cuda:0",
        runtime_components={"triton": False},
    )

    selection = registry.resolve_selection(
        stage=OptimizationStage.PREPROCESS,
        requested_id="candidate",
        context=context,
        allow_fallback=True,
    )

    assert selection.effective_id == "base"
    assert selection.fallback_reason == "unavailable runtime components: triton"
    assert constructed == ["base"]


def test_registry_can_reject_static_dependency_fallback() -> None:
    registry = ImplementationRegistry(scope_name="Example model")
    registry.register(_Stage("base"))
    registry.register(_Stage("candidate", dependencies=("triton",)))
    context = ExecutionContext(
        device_kind="gpu",
        device="cuda:0",
        runtime_components={"triton": False},
    )

    with pytest.raises(ModelRuntimeError, match="unavailable runtime components"):
        registry.resolve_selection(
            stage=OptimizationStage.PREPROCESS,
            requested_id="candidate",
            context=context,
            allow_fallback=False,
        )


def test_tensor_readiness_tracker_uses_exact_tensor_identity() -> None:
    tracker = TensorReadinessTracker[_State]()
    tensor = torch.zeros(1)
    other = torch.zeros(1)
    tracker.record(tensor, state=_State(implementation_id="candidate"))

    assert tracker.consume(other) is None
    assert tracker.consume(tensor) == _State(implementation_id="candidate")
    assert tracker.consume(tensor) is None
