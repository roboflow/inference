import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch

from inference_models.errors import ModelRuntimeError
from inference_models.models.common.roboflow.model_packages import (
    ColorMode,
    ImagePreProcessing,
    NetworkInputDefinition,
    ResizeMode,
    TrainingInputSize,
)
from inference_models.models.optimization.contracts import (
    CompatibilityResult,
    DeviceCompatibility,
    ExecutionContext,
    InputCompatibility,
    OptimizationMetadata,
    OptimizationStage,
    ValidationEnvironment,
    immutable_mapping,
)
from inference_models.models.optimization.registry import ImplementationRegistry
from inference_models.models.rfdetr.optimization.catalog import (
    RFDETR_POSTPROCESSOR_IMPLEMENTATIONS,
    RFDETR_PREPROCESSOR_IMPLEMENTATIONS,
)
from inference_models.models.rfdetr.optimization.contracts import PreprocessRequest
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
from inference_models.models.rfdetr.optimization.selection import (
    resolve_preprocessor_for_model,
    resolve_preprocessor_for_request,
)


class _Stage:
    def __init__(
        self,
        implementation_id: str,
        *,
        compatible: bool = True,
        validated: bool = False,
        model_supported: bool = True,
        request_supported: bool = True,
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
        self._model_supported = model_supported
        self._request_supported = request_supported

    def is_compatible(self, context: ExecutionContext) -> bool:
        return self._compatible

    def check_model_compatibility(
        self,
        *,
        image_pre_processing: ImagePreProcessing,
        network_input: NetworkInputDefinition,
    ) -> CompatibilityResult:
        del image_pre_processing, network_input
        if self._model_supported:
            return CompatibilityResult.compatible()

        return CompatibilityResult.incompatible("static crop")

    def check_request_compatibility(
        self,
        *,
        request: PreprocessRequest,
        context: ExecutionContext,
    ) -> CompatibilityResult:
        del request, context
        if self._request_supported:
            return CompatibilityResult.compatible()

        return CompatibilityResult.incompatible("heterogeneous source dimensions")


def _context() -> ExecutionContext:
    return ExecutionContext(
        device_kind="gpu",
        device="cuda:0",
        device_name="test-gpu",
        machine_type="test",
        scenario="runtime",
    )


def _network_input() -> NetworkInputDefinition:
    return NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=64),
        dataset_version_resize_dimensions=None,
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
        scaling_factor=255,
        normalization=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
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


def test_execution_plan_rejects_unimplemented_stage_category() -> None:
    with pytest.raises(ModelRuntimeError, match="does not yet provide"):
        RFDetrExecutionPlan.resolve(
            execution_plan=RFDetrExecutionPlan(scheduler_id="future-scheduler")
        )


def test_registry_resolves_explicit_and_auto_base() -> None:
    registry = ImplementationRegistry(scope_name="RF-DETR")
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
    registry = ImplementationRegistry(scope_name="RF-DETR")
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
    registry = ImplementationRegistry(scope_name="RF-DETR")
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


def test_model_contract_incompatibility_resolves_declared_base_fallback() -> None:
    registry = ImplementationRegistry(scope_name="RF-DETR")
    base = _Stage("base")
    candidate = _Stage("candidate", model_supported=False)
    registry.register(base)
    registry.register(candidate)

    selection = resolve_preprocessor_for_model(
        registry=registry,
        requested_id="candidate",
        context=_context(),
        image_pre_processing=ImagePreProcessing(),
        network_input=_network_input(),
    )

    assert selection.implementation is base
    assert selection.effective_id == "base"
    assert selection.fallback_reason == "static crop"


def test_auto_selection_is_not_reported_as_fallback_when_candidate_is_supported() -> (
    None
):
    registry = ImplementationRegistry(scope_name="RF-DETR")
    registry.register(_Stage("base"))
    candidate = _Stage("candidate", validated=True)
    registry.register(candidate)

    selection = resolve_preprocessor_for_model(
        registry=registry,
        requested_id="auto",
        context=_context(),
        image_pre_processing=ImagePreProcessing(),
        network_input=_network_input(),
    )

    assert selection.implementation is candidate
    assert selection.effective_id == "candidate"
    assert not selection.used_fallback


def test_request_incompatibility_resolves_declared_base_fallback() -> None:
    registry = ImplementationRegistry(scope_name="RF-DETR")
    base = _Stage("base")
    candidate = _Stage("candidate", request_supported=False)
    registry.register(base)
    registry.register(candidate)
    request = PreprocessRequest(
        images=np.zeros((8, 9, 3), dtype=np.uint8),
        input_color_format=ColorMode.RGB,
        image_pre_processing=ImagePreProcessing(),
        network_input=_network_input(),
        pre_processing_overrides=None,
    )

    selection = resolve_preprocessor_for_request(
        registry=registry,
        implementation=candidate,
        request=request,
        context=_context(),
    )

    assert selection.implementation is base
    assert selection.effective_id == "base"
    assert selection.fallback_reason == "heterogeneous source dimensions"


def test_fallback_is_rejected_when_base_is_also_incompatible() -> None:
    registry = ImplementationRegistry(scope_name="RF-DETR")
    registry.register(_Stage("base", model_supported=False))
    registry.register(_Stage("candidate", model_supported=False))

    with pytest.raises(ModelRuntimeError, match="Fallback 'base' is unsupported"):
        resolve_preprocessor_for_model(
            registry=registry,
            requested_id="candidate",
            context=_context(),
            image_pre_processing=ImagePreProcessing(),
            network_input=_network_input(),
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
