import dataclasses
import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch

from inference_models import PreProcessingOverrides
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
    immutable_mapping,
)
from inference_models.models.optimization.errors import RecoverableStageExecutionError
from inference_models.models.optimization.ids import AUTO_IMPLEMENTATION_ID
from inference_models.models.optimization.registry import ImplementationRegistry
from inference_models.models.rfdetr import triton_universal_preprocess_runtime
from inference_models.models.rfdetr.optimization.catalog import (
    RFDETR_BUFFER_STRATEGY_IMPLEMENTATIONS,
    RFDETR_ENGINE_PLUGIN_IMPLEMENTATIONS,
    RFDETR_POSTPROCESSOR_IMPLEMENTATIONS,
    RFDETR_PREPROCESSOR_IMPLEMENTATIONS,
    RFDETR_SCHEDULER_IMPLEMENTATIONS,
    build_rfdetr_implementation_registry,
)
from inference_models.models.rfdetr.optimization.contracts import (
    PostprocessRequest,
    PreprocessRequest,
    PreprocessResult,
)
from inference_models.models.rfdetr.optimization.execution_plan import (
    RFDetrExecutionPlan,
)
from inference_models.models.rfdetr.optimization.ids import (
    RFDETR_ALLOW_COMPATIBILITY_FALLBACK_ENV_NAME,
    RFDETR_ALLOW_RUNTIME_FAILURE_FALLBACK_ENV_NAME,
    RFDETR_POSTPROCESSOR_BASE,
    RFDETR_POSTPROCESSOR_TRITON_FUSED_V1,
    RFDETR_PREPROCESSOR_BASE,
    RFDETR_PREPROCESSOR_TRITON_UNIVERSAL_V1,
)
from inference_models.models.rfdetr.optimization.readiness import (
    PreprocessReadinessTracker,
)
from inference_models.models.rfdetr.optimization.selection import (
    resolve_postprocessor_for_request,
    resolve_postprocessor_runtime_fallback,
    resolve_preprocessor_for_model,
    resolve_preprocessor_for_request,
    resolve_preprocessor_runtime_fallback,
)


class _Stage:
    def __init__(
        self,
        implementation_id: str,
        *,
        compatible: bool = True,
        model_supported: bool = True,
        request_supported: bool = True,
        runtime_supported: bool = True,
        stage: OptimizationStage = OptimizationStage.PREPROCESS,
    ) -> None:
        self.metadata = OptimizationMetadata(
            implementation_id=implementation_id,
            stage=stage,
            version="1",
            target=DeviceCompatibility(device_kind="gpu"),
            inputs=InputCompatibility(scenarios=("*",)),
            dependencies=(),
            fallback_id="base",
            changes_numerics=False,
            supports_concurrency=True,
            supports_cuda_graphs=False,
        )
        self._compatible = compatible
        self._model_supported = model_supported
        self._request_supported = request_supported
        self._runtime_supported = runtime_supported
        self.preprocess_calls = 0
        self.postprocess_calls = 0

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

    def check_runtime_compatibility(
        self,
        *,
        request,
        context: ExecutionContext,
    ) -> CompatibilityResult:
        del request, context
        if self._runtime_supported:
            return CompatibilityResult.compatible()

        return CompatibilityResult.incompatible(
            "implementation runtime failed during an earlier request"
        )

    def preprocess(
        self,
        request: PreprocessRequest,
        context: ExecutionContext,
    ) -> PreprocessResult:
        del request, context
        self.preprocess_calls += 1

        return PreprocessResult(
            tensor=torch.zeros((1, 3, 8, 9)),
            metadata=[],
            implementation_id=self.metadata.implementation_id,
        )

    def postprocess(
        self,
        request: PostprocessRequest,
        context: ExecutionContext,
    ):
        del request, context
        self.postprocess_calls += 1

        return []


def _context() -> ExecutionContext:
    return ExecutionContext(
        device_kind="gpu",
        device="cuda:0",
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


def test_execution_plan_defaults_to_auto_selection(monkeypatch) -> None:
    monkeypatch.delenv("INFERENCE_MODELS_RFDETR_PREPROCESSOR", raising=False)
    monkeypatch.delenv("INFERENCE_MODELS_RFDETR_POSTPROCESSOR", raising=False)

    default_plan = RFDetrExecutionPlan()
    resolved_plan = RFDetrExecutionPlan.resolve()

    for plan in (default_plan, resolved_plan):
        assert plan.preprocessor_id == AUTO_IMPLEMENTATION_ID
        assert plan.postprocessor_id == AUTO_IMPLEMENTATION_ID


def test_execution_plan_reads_environment_overrides(monkeypatch) -> None:
    monkeypatch.setenv(
        "INFERENCE_MODELS_RFDETR_PREPROCESSOR",
        RFDETR_PREPROCESSOR_BASE,
    )
    monkeypatch.setenv(
        "INFERENCE_MODELS_RFDETR_POSTPROCESSOR",
        RFDETR_POSTPROCESSOR_BASE,
    )

    plan = RFDetrExecutionPlan.resolve()

    assert plan.preprocessor_id == RFDETR_PREPROCESSOR_BASE
    assert plan.postprocessor_id == RFDETR_POSTPROCESSOR_BASE


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


def test_execution_plan_preserves_all_explicit_stage_ids() -> None:
    plan = RFDetrExecutionPlan(
        buffer_strategy_id="future-buffer",
        scheduler_id="future-scheduler",
        engine_plugin_id="future-plugin",
        allow_compatibility_fallback=False,
        allow_runtime_failure_fallback=True,
    )

    resolved = RFDetrExecutionPlan.resolve(execution_plan=plan)

    assert resolved is plan
    assert resolved.buffer_strategy_id == "future-buffer"
    assert resolved.scheduler_id == "future-scheduler"
    assert resolved.engine_plugin_id == "future-plugin"
    assert not resolved.allow_compatibility_fallback
    assert resolved.allow_runtime_failure_fallback


def test_profiling_execution_plan_is_explicit_and_forbids_fallback(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "INFERENCE_MODELS_RFDETR_PREPROCESSOR",
        RFDETR_PREPROCESSOR_TRITON_UNIVERSAL_V1,
    )
    profiling_plan = RFDetrExecutionPlan(
        preprocessor_id=RFDETR_PREPROCESSOR_BASE,
        buffer_strategy_id="base",
        scheduler_id="base",
        postprocessor_id=RFDETR_POSTPROCESSOR_BASE,
        engine_plugin_id="base",
        allow_compatibility_fallback=False,
        allow_runtime_failure_fallback=False,
    )

    resolved = RFDetrExecutionPlan.from_dict(profiling_plan.to_dict())

    assert isinstance(resolved, RFDetrExecutionPlan)
    assert resolved.preprocessor_id == RFDETR_PREPROCESSOR_BASE
    assert resolved.buffer_strategy_id == "base"
    assert resolved.scheduler_id == "base"
    assert resolved.postprocessor_id == RFDETR_POSTPROCESSOR_BASE
    assert resolved.engine_plugin_id == "base"
    assert not resolved.allow_compatibility_fallback
    assert not resolved.allow_runtime_failure_fallback


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


def test_registry_auto_selects_a_preferred_compatible_candidate() -> None:
    registry = ImplementationRegistry(scope_name="RF-DETR")
    base = _Stage("base")
    candidate = _Stage("candidate")
    registry.register(base)
    registry.register(candidate)
    registry.set_auto_preferences(
        stage=OptimizationStage.PREPROCESS,
        implementation_ids=("candidate",),
    )

    assert (
        registry.resolve(
            stage=OptimizationStage.PREPROCESS,
            requested_id="auto",
            context=_context(),
        )
        is candidate
    )


def test_rfdetr_auto_preferences_skip_unavailable_triton_and_arm_simd() -> None:
    registry = build_rfdetr_implementation_registry(
        device=torch.device("cuda:0"),
    )
    context = ExecutionContext(
        device_kind="gpu",
        device="cuda:0",
        runtime_components={"triton": False},
        host_architecture="aarch64",
    )

    preprocessor = registry.resolve(
        stage=OptimizationStage.PREPROCESS,
        requested_id="auto",
        context=context,
    )
    postprocessor = registry.resolve(
        stage=OptimizationStage.POSTPROCESS,
        requested_id="auto",
        context=context,
    )

    assert preprocessor.metadata.implementation_id == "base"
    assert postprocessor.metadata.implementation_id == RFDETR_POSTPROCESSOR_BASE


@pytest.mark.parametrize("backend", ["torch", "onnx", "trt"])
@pytest.mark.parametrize("simd_available", [True, False])
def test_auto_and_full_request_chain_with_real_preprocessors(
    monkeypatch, backend, simd_available
):
    from inference_models.models.rfdetr.optimization.preprocessor_selection import (
        PreprocessorSelector,
    )
    from inference_models.models.rfdetr.optimization.preprocessors import pillow_simd

    def load():
        if not simd_available:
            raise ImportError("SIMD not installed")
        return object()

    monkeypatch.setattr(pillow_simd, "load_pillow_simd_image", load)
    monkeypatch.setattr(triton_universal_preprocess_runtime, "TRITON_AVAILABLE", True)
    registry = build_rfdetr_implementation_registry(
        device=torch.device("cuda"), backend=backend
    )
    context = ExecutionContext(
        device_kind="gpu",
        device="cuda",
        host_architecture="x86_64",
        runtime_components={"triton": True},
    )
    selector = PreprocessorSelector(
        registry=registry,
        context=context,
        image_pre_processing=ImagePreProcessing(),
        network_input=_network_input(),
    )
    primary = selector.resolve_model(
        requested_id="auto", allow_fallback=True
    ).implementation
    assert primary.metadata.implementation_id == "triton-universal-v1"
    request = PreprocessRequest(
        images=np.zeros((32, 32, 3), dtype=np.uint8),
        input_color_format="rgb",
        image_pre_processing=ImagePreProcessing(),
        network_input=_network_input(),
        pre_processing_overrides=None,
        image_size_wh=(48, 48),
    )
    selected = selector.resolve_request(
        implementation=primary,
        request=request,
        context=context,
        allow_fallback=True,
    )
    assert selected.effective_id == ("pillow-simd-v1" if simd_available else "base")
    # Same shape, but tensor inputs must skip SIMD. Neither fallback is sticky.
    tensor_request = dataclasses.replace(request, images=torch.zeros((3, 32, 32)))
    selected = selector.resolve_request(
        implementation=primary,
        request=tensor_request,
        context=context,
        allow_fallback=True,
    )
    assert selected.effective_id == "base"
    selected = selector.resolve_request(
        implementation=primary,
        request=dataclasses.replace(request, image_size_wh=None),
        context=context,
        allow_fallback=True,
    )
    assert selected.effective_id == "triton-universal-v1"
    # Model-level auto selection also checks native availability, not just metadata.
    no_triton = PreprocessorSelector(
        registry=registry,
        context=dataclasses.replace(context, runtime_components={"triton": False}),
        image_pre_processing=ImagePreProcessing(),
        network_input=_network_input(),
    )
    assert no_triton.resolve_model(
        requested_id="auto", allow_fallback=True
    ).effective_id == ("pillow-simd-v1" if simd_available else "base")


@pytest.mark.parametrize("backend", ["trt", "torch", "onnx"])
def test_registry_rejects_removed_threaded_preprocessor(backend) -> None:
    """Reject the removed implementation rather than silently selecting base.

    Args:
        backend (str): Object-detection registry to inspect.
    """
    registry = build_rfdetr_implementation_registry(
        device=torch.device("cpu"), backend=backend
    )
    with pytest.raises(ModelRuntimeError, match="Unknown RF-DETR preprocess"):
        registry.resolve_selection(
            stage=OptimizationStage.PREPROCESS,
            requested_id="threaded-exact-v1",
            context=ExecutionContext(device_kind="cpu", device="cpu"),
            allow_fallback=True,
        )


@pytest.mark.parametrize("allow_fallback", [False, True])
def test_model_eval_request_retains_triton_preprocessor(
    monkeypatch: pytest.MonkeyPatch,
    allow_fallback: bool,
) -> None:
    """Resolve model-eval metadata and disable flags through the real stage registry.

    Args:
        monkeypatch: Fixture simulating Triton availability without GPU execution.
        allow_fallback: Whether an incompatible selection may fall back to base.
    """
    monkeypatch.setattr(triton_universal_preprocess_runtime, "TRITON_AVAILABLE", True)
    registry = build_rfdetr_implementation_registry(
        device=torch.device("cuda:0"),
    )
    context = _context()
    network = _network_input().model_copy(
        update={
            "dataset_version_resize_dimensions": TrainingInputSize(height=32, width=48)
        }
    )
    transforms = ImagePreProcessing.model_validate({"auto-orient": {"enabled": True}})
    model_selection = resolve_preprocessor_for_model(
        registry=registry,
        requested_id="auto",
        context=context,
        image_pre_processing=transforms,
        network_input=network,
        allow_fallback=allow_fallback,
    )
    request_selection = resolve_preprocessor_for_request(
        registry=registry,
        implementation=model_selection.implementation,
        request=PreprocessRequest(
            images=np.zeros((48, 80, 3), dtype=np.uint8),
            input_color_format=ColorMode.BGR,
            image_pre_processing=transforms,
            network_input=network,
            pre_processing_overrides=PreProcessingOverrides(
                disable_contrast_enhancement=True,
                disable_grayscale=True,
                disable_static_crop=True,
            ),
        ),
        context=context,
        allow_fallback=allow_fallback,
    )

    for selection in (model_selection, request_selection):
        assert selection.effective_id == RFDETR_PREPROCESSOR_TRITON_UNIVERSAL_V1
        assert not selection.used_fallback
        assert selection.fallback_reason is None


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
    with pytest.raises(ModelRuntimeError, match="is incompatible"):
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
        allow_fallback=True,
    )

    assert selection.implementation is base
    assert selection.effective_id == "base"
    assert selection.fallback_reason == "static crop"


def test_auto_selection_is_not_reported_as_fallback_when_candidate_is_supported() -> (
    None
):
    registry = ImplementationRegistry(scope_name="RF-DETR")
    registry.register(_Stage("base"))
    candidate = _Stage("candidate")
    registry.register(candidate)
    registry.set_auto_preferences(
        stage=OptimizationStage.PREPROCESS,
        implementation_ids=("candidate",),
    )

    selection = resolve_preprocessor_for_model(
        registry=registry,
        requested_id="auto",
        context=_context(),
        image_pre_processing=ImagePreProcessing(),
        network_input=_network_input(),
        allow_fallback=True,
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
        allow_fallback=True,
    )

    assert selection.implementation is base
    assert selection.effective_id == "base"
    assert selection.fallback_reason == "heterogeneous source dimensions"


def test_recorded_runtime_failure_resolves_declared_base_fallback() -> None:
    registry = ImplementationRegistry(scope_name="RF-DETR")
    base = _Stage("base")
    candidate = _Stage("candidate", runtime_supported=False)
    registry.register(base)
    registry.register(candidate)
    request = PreprocessRequest(
        images=np.zeros((8, 9, 3), dtype=np.uint8),
        input_color_format=ColorMode.RGB,
        image_pre_processing=ImagePreProcessing(),
        network_input=_network_input(),
        pre_processing_overrides=None,
    )
    request_selection = resolve_preprocessor_for_request(
        registry=registry,
        implementation=candidate,
        request=request,
        context=_context(),
        allow_fallback=False,
    )
    first_selection = resolve_preprocessor_runtime_fallback(
        registry=registry,
        selection=request_selection,
        request=request,
        context=_context(),
        allow_fallback=True,
    )
    second_selection = resolve_preprocessor_runtime_fallback(
        registry=registry,
        selection=request_selection,
        request=request,
        context=_context(),
        allow_fallback=True,
    )

    assert first_selection.effective_id == "base"
    assert second_selection.effective_id == "base"
    assert (
        first_selection.fallback_reason
        == "implementation runtime failed during an earlier request"
    )


def test_recorded_runtime_failure_is_raised_when_fallback_is_disabled() -> None:
    registry = ImplementationRegistry(scope_name="RF-DETR")
    base = _Stage("base")
    candidate = _Stage("candidate", runtime_supported=False)
    registry.register(base)
    registry.register(candidate)
    request = PreprocessRequest(
        images=np.zeros((8, 9, 3), dtype=np.uint8),
        input_color_format=ColorMode.RGB,
        image_pre_processing=ImagePreProcessing(),
        network_input=_network_input(),
        pre_processing_overrides=None,
    )

    request_selection = resolve_preprocessor_for_request(
        registry=registry,
        implementation=candidate,
        request=request,
        context=_context(),
        allow_fallback=True,
    )

    with pytest.raises(
        RecoverableStageExecutionError,
        match="Runtime failure fallback is disabled",
    ):
        resolve_preprocessor_runtime_fallback(
            registry=registry,
            selection=request_selection,
            request=request,
            context=_context(),
            allow_fallback=False,
        )

    assert base.preprocess_calls == 0


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
            allow_fallback=True,
        )


def test_strict_plan_rejects_compatibility_fallback() -> None:
    registry = ImplementationRegistry(scope_name="RF-DETR")
    registry.register(_Stage("base"))
    registry.register(_Stage("candidate", model_supported=False))

    with pytest.raises(ModelRuntimeError, match="fallback is disabled"):
        resolve_preprocessor_for_model(
            registry=registry,
            requested_id="candidate",
            context=_context(),
            image_pre_processing=ImagePreProcessing(),
            network_input=_network_input(),
            allow_fallback=False,
        )


def test_postprocessor_contract_uses_same_declared_fallback_policy() -> None:
    registry = ImplementationRegistry(scope_name="RF-DETR")
    base = _Stage("base", stage=OptimizationStage.POSTPROCESS)
    candidate = _Stage(
        "candidate",
        request_supported=False,
        stage=OptimizationStage.POSTPROCESS,
    )
    registry.register(base)
    registry.register(candidate)
    request = PostprocessRequest(
        bboxes=torch.zeros((1, 2, 4)),
        logits=torch.zeros((1, 2, 3)),
        pre_processing_meta=[],
        threshold=0.5,
        num_classes=3,
        classes_re_mapping=None,
    )

    selection = resolve_postprocessor_for_request(
        registry=registry,
        implementation=candidate,
        request=request,
        context=_context(),
        allow_fallback=True,
    )

    assert selection.implementation is base
    assert selection.effective_id == "base"
    assert selection.fallback_reason == "heterogeneous source dimensions"

    with pytest.raises(ModelRuntimeError, match="fallback is disabled"):
        resolve_postprocessor_for_request(
            registry=registry,
            implementation=candidate,
            request=request,
            context=_context(),
            allow_fallback=False,
        )


def test_recorded_postprocessor_runtime_failure_resolves_base_fallback() -> None:
    registry = ImplementationRegistry(scope_name="RF-DETR")
    base = _Stage("base", stage=OptimizationStage.POSTPROCESS)
    candidate = _Stage(
        "candidate",
        runtime_supported=False,
        stage=OptimizationStage.POSTPROCESS,
    )
    registry.register(base)
    registry.register(candidate)
    request = PostprocessRequest(
        bboxes=torch.zeros((1, 2, 4)),
        logits=torch.zeros((1, 2, 3)),
        pre_processing_meta=[],
        threshold=0.5,
        num_classes=3,
        classes_re_mapping=None,
    )
    request_selection = resolve_postprocessor_for_request(
        registry=registry,
        implementation=candidate,
        request=request,
        context=_context(),
        allow_fallback=False,
    )

    fallback_selection = resolve_postprocessor_runtime_fallback(
        registry=registry,
        selection=request_selection,
        request=request,
        context=_context(),
        allow_fallback=True,
    )

    assert fallback_selection.implementation is base
    assert fallback_selection.effective_id == "base"
    assert (
        fallback_selection.fallback_reason
        == "implementation runtime failed during an earlier request"
    )


def test_recorded_postprocessor_runtime_failure_respects_strict_plan() -> None:
    registry = ImplementationRegistry(scope_name="RF-DETR")
    base = _Stage("base", stage=OptimizationStage.POSTPROCESS)
    candidate = _Stage(
        "candidate",
        runtime_supported=False,
        stage=OptimizationStage.POSTPROCESS,
    )
    registry.register(base)
    registry.register(candidate)
    request = PostprocessRequest(
        bboxes=torch.zeros((1, 2, 4)),
        logits=torch.zeros((1, 2, 3)),
        pre_processing_meta=[],
        threshold=0.5,
        num_classes=3,
        classes_re_mapping=None,
    )
    request_selection = resolve_postprocessor_for_request(
        registry=registry,
        implementation=candidate,
        request=request,
        context=_context(),
        allow_fallback=True,
    )

    with pytest.raises(
        RecoverableStageExecutionError,
        match="Runtime failure fallback is disabled",
    ):
        resolve_postprocessor_runtime_fallback(
            registry=registry,
            selection=request_selection,
            request=request,
            context=_context(),
            allow_fallback=False,
        )

    assert base.postprocess_calls == 0


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


def test_all_execution_plan_stages_publish_base_metadata() -> None:
    stage_catalogs = {
        OptimizationStage.PREPROCESS: RFDETR_PREPROCESSOR_IMPLEMENTATIONS,
        OptimizationStage.BUFFER_STRATEGY: RFDETR_BUFFER_STRATEGY_IMPLEMENTATIONS,
        OptimizationStage.SCHEDULER: RFDETR_SCHEDULER_IMPLEMENTATIONS,
        OptimizationStage.POSTPROCESS: RFDETR_POSTPROCESSOR_IMPLEMENTATIONS,
        OptimizationStage.ENGINE_PLUGIN: RFDETR_ENGINE_PLUGIN_IMPLEMENTATIONS,
    }

    for stage, catalog in stage_catalogs.items():
        assert "base" in catalog
        assert catalog["base"].stage is stage
        assert json.loads(json.dumps(catalog["base"].to_dict()))["stage"] == stage.value


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
    assert readiness.ready_event is None
    assert readiness.input_kind == "test"
    assert readiness.implementation_id == "candidate"
    assert tracker.consume(tensor) is None


def test_immutable_mapping_detaches_from_source() -> None:
    source = {"batch": 1}
    immutable = immutable_mapping(source)
    source["batch"] = 2

    assert immutable["batch"] == 1


@pytest.mark.parametrize(
    "compatibility,runtime",
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_execution_plan_reads_fallback_policy_from_environment(
    monkeypatch, compatibility, runtime
) -> None:
    monkeypatch.setenv(RFDETR_ALLOW_COMPATIBILITY_FALLBACK_ENV_NAME, str(compatibility))
    monkeypatch.setenv(RFDETR_ALLOW_RUNTIME_FAILURE_FALLBACK_ENV_NAME, str(runtime))

    plan = RFDetrExecutionPlan.resolve()

    assert plan.allow_compatibility_fallback is compatibility
    assert plan.allow_runtime_failure_fallback is runtime


def test_execution_plan_fallback_policy_defaults_to_permissive(monkeypatch) -> None:
    monkeypatch.delenv(RFDETR_ALLOW_COMPATIBILITY_FALLBACK_ENV_NAME, raising=False)
    monkeypatch.delenv(RFDETR_ALLOW_RUNTIME_FAILURE_FALLBACK_ENV_NAME, raising=False)

    plan = RFDetrExecutionPlan.resolve()

    assert plan.allow_compatibility_fallback is True
    assert plan.allow_runtime_failure_fallback is True


def test_explicit_plan_ignores_environment_fallback_policy(monkeypatch) -> None:
    monkeypatch.setenv(RFDETR_ALLOW_COMPATIBILITY_FALLBACK_ENV_NAME, "false")
    monkeypatch.setenv(RFDETR_ALLOW_RUNTIME_FAILURE_FALLBACK_ENV_NAME, "false")
    explicit = RFDetrExecutionPlan(
        allow_compatibility_fallback=True, allow_runtime_failure_fallback=True
    )

    assert RFDetrExecutionPlan.resolve(execution_plan=explicit) is explicit


def test_serialized_plan_ignores_environment_fallback_policy(monkeypatch) -> None:
    monkeypatch.setenv(RFDETR_ALLOW_COMPATIBILITY_FALLBACK_ENV_NAME, "false")
    monkeypatch.setenv(RFDETR_ALLOW_RUNTIME_FAILURE_FALLBACK_ENV_NAME, "false")
    serialized = RFDetrExecutionPlan().to_dict()

    resolved = RFDetrExecutionPlan.resolve(execution_plan=serialized)

    assert resolved.allow_compatibility_fallback is True
    assert resolved.allow_runtime_failure_fallback is True
