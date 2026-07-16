import json
from dataclasses import FrozenInstanceError

import pytest
import torch

from inference_models.errors import ModelRuntimeError
from inference_models.models.rfdetr.optimization.catalog import (
    RFDETR_POSTPROCESSOR_IMPLEMENTATIONS,
    RFDETR_PREPROCESSOR_IMPLEMENTATIONS,
)
from inference_models.models.rfdetr.optimization.contracts import (
    DeviceCompatibility,
    ExecutionContext,
    InputCompatibility,
    OptimizationMetadata,
    OptimizationStage,
    ValidationEnvironment,
    immutable_mapping,
)
from inference_models.models.rfdetr.optimization.execution_plan import (
    RFDetrExecutionPlan,
)
from inference_models.models.rfdetr.optimization.ids import (
    RFDETR_POSTPROCESSOR_BASE,
    RFDETR_POSTPROCESSOR_TRITON_FUSED_V1,
    RFDETR_PREPROCESSOR_BASE,
    RFDETR_PREPROCESSOR_TRITON_UNIVERSAL_V1,
)
from inference_models.models.rfdetr.optimization.readiness import (
    PreprocessReadinessTracker,
)
from inference_models.models.rfdetr.optimization.registry import ImplementationRegistry


class _Stage:
    def __init__(
        self,
        implementation_id: str,
        *,
        compatible: bool = True,
        validated: bool = False,
    ) -> None:
        validation_environments = (
            (
                ValidationEnvironment(
                    machine_type="test",
                    device_kind="gpu",
                    device_name="test-gpu",
                    scenario="runtime",
                    resolved_axes={},
                    runtime_versions={},
                    source_commit="test",
                    profiling_bundle="test-bundle",
                    status="validated",
                ),
            )
            if validated
            else ()
        )
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
            validated_environments=validation_environments,
        )
        self._compatible = compatible

    def is_compatible(self, context: ExecutionContext) -> bool:
        return self._compatible


def _context() -> ExecutionContext:
    return ExecutionContext(
        device_kind="gpu",
        device="cuda:0",
        device_name="test-gpu",
        machine_type="test",
        scenario="runtime",
    )


def test_execution_plan_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv(
        "INFERENCE_MODELS_RFDETR_PREPROCESSOR",
        RFDETR_PREPROCESSOR_TRITON_UNIVERSAL_V1,
    )
    monkeypatch.setenv(
        "INFERENCE_MODELS_RFDETR_POSTPROCESSOR",
        RFDETR_POSTPROCESSOR_TRITON_FUSED_V1,
    )

    plan = RFDetrExecutionPlan.resolve()

    assert plan.preprocessor_id == RFDETR_PREPROCESSOR_TRITON_UNIVERSAL_V1
    assert plan.postprocessor_id == RFDETR_POSTPROCESSOR_TRITON_FUSED_V1


def test_explicit_plan_ignores_environment(monkeypatch) -> None:
    monkeypatch.setenv(
        "INFERENCE_MODELS_RFDETR_PREPROCESSOR",
        RFDETR_PREPROCESSOR_TRITON_UNIVERSAL_V1,
    )
    plan = RFDetrExecutionPlan(
        preprocessor_id=RFDETR_PREPROCESSOR_BASE,
        postprocessor_id=RFDETR_POSTPROCESSOR_BASE,
    )

    resolved = RFDetrExecutionPlan.resolve(execution_plan=plan)

    assert resolved is plan


def test_execution_plan_rejects_ambiguous_legacy_selection() -> None:
    with pytest.raises(ModelRuntimeError, match="cannot be combined"):
        RFDetrExecutionPlan.resolve(
            execution_plan=RFDetrExecutionPlan(),
            preprocessor_id=RFDETR_PREPROCESSOR_BASE,
        )


def test_execution_plan_rejects_unimplemented_stage_category() -> None:
    with pytest.raises(ModelRuntimeError, match="does not yet provide"):
        RFDetrExecutionPlan.resolve(
            execution_plan=RFDetrExecutionPlan(scheduler_id="future-scheduler")
        )


def test_registry_resolves_explicit_and_auto_base() -> None:
    registry = ImplementationRegistry()
    base = _Stage("base")
    candidate = _Stage("candidate")
    registry.register(base)
    registry.register(candidate)

    assert (
        registry.resolve(
            stage=OptimizationStage.PREPROCESS,
            requested_id="candidate",
            context=_context(),
        )
        is candidate
    )
    assert (
        registry.resolve(
            stage=OptimizationStage.PREPROCESS,
            requested_id="auto",
            context=_context(),
        )
        is base
    )


def test_registry_auto_selects_a_validated_compatible_candidate() -> None:
    registry = ImplementationRegistry()
    base = _Stage("base")
    candidate = _Stage("candidate", validated=True)
    registry.register(base)
    registry.register(candidate)

    assert (
        registry.resolve(
            stage=OptimizationStage.PREPROCESS,
            requested_id="auto",
            context=_context(),
        )
        is candidate
    )


def test_registry_rejects_unknown_and_incompatible_explicit_selection() -> None:
    registry = ImplementationRegistry()
    registry.register(_Stage("base"))
    registry.register(_Stage("incompatible", compatible=False))

    with pytest.raises(ModelRuntimeError, match="Unknown RF-DETR preprocess"):
        registry.resolve(
            stage=OptimizationStage.PREPROCESS,
            requested_id="unknown",
            context=_context(),
        )
    with pytest.raises(ModelRuntimeError, match="not compatible"):
        registry.resolve(
            stage=OptimizationStage.PREPROCESS,
            requested_id="incompatible",
            context=_context(),
        )


def test_implementation_metadata_is_typed_and_immutable() -> None:
    preprocessor = RFDETR_PREPROCESSOR_IMPLEMENTATIONS[
        RFDETR_PREPROCESSOR_TRITON_UNIVERSAL_V1
    ]
    postprocessor = RFDETR_POSTPROCESSOR_IMPLEMENTATIONS[
        RFDETR_POSTPROCESSOR_TRITON_FUSED_V1
    ]

    assert preprocessor.stage is OptimizationStage.PREPROCESS
    assert postprocessor.stage is OptimizationStage.POSTPROCESS
    assert preprocessor.inputs.axis_constraints["channels"] == 3
    with pytest.raises(FrozenInstanceError):
        preprocessor.version = "2"
    with pytest.raises(TypeError):
        preprocessor.inputs.axis_constraints["channels"] = 4
    assert json.loads(json.dumps(preprocessor.to_dict()))["stage"] == "preprocess"


def test_readiness_tracker_consumes_only_the_exact_tensor() -> None:
    tracker = PreprocessReadinessTracker()
    tensor = torch.zeros(1)
    other = torch.zeros(1)
    tracker.record(
        tensor,
        ready_event=None,
        input_kind="test",
        implementation_id="candidate",
    )

    assert tracker.consume(other) is None
    readiness = tracker.consume(tensor)
    assert readiness is not None
    assert readiness.input_kind == "test"
    assert readiness.implementation_id == "candidate"
    assert tracker.consume(tensor) is None


def test_immutable_mapping_detaches_from_source() -> None:
    source = {"batch": 1}
    immutable = immutable_mapping(source)
    source["batch"] = 2

    assert immutable["batch"] == 1
