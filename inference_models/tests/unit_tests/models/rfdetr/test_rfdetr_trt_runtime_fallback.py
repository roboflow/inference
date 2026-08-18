import importlib
import sys
import threading
from importlib.machinery import ModuleSpec
from types import MethodType, ModuleType, SimpleNamespace
from typing import List

import numpy as np
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
)
from inference_models.models.optimization.errors import RecoverableStageExecutionError
from inference_models.models.optimization.registry import ImplementationRegistry
from inference_models.models.rfdetr.optimization.contracts import (
    PostprocessRequest,
    PreprocessRequest,
    PreprocessResult,
)
from inference_models.models.rfdetr.optimization.ids import (
    RFDETR_POSTPROCESSOR_BASE,
    RFDETR_POSTPROCESSOR_TRITON_FUSED_V1,
    RFDETR_PREPROCESSOR_BASE,
    RFDETR_PREPROCESSOR_TRITON_UNIVERSAL_V1,
)

_MODEL_MODULE = "inference_models.models.rfdetr.rfdetr_object_detection_trt"
_TRT_DEPENDENCY_MODULES = (
    "inference_models.models.common.cuda",
    "inference_models.models.common.trt",
    _MODEL_MODULE,
)
_MISSING = object()


@pytest.fixture
def rfdetr_trt_model_class(monkeypatch):
    """Import the TRT model with inert annotation-only dependency doubles."""

    class ILogger:
        class Severity:
            pass

    fake_trt = ModuleType("tensorrt")
    fake_trt.__spec__ = ModuleSpec("tensorrt", loader=None)
    fake_trt.ILogger = ILogger
    fake_trt.ICudaEngine = type("ICudaEngine", (), {})
    fake_trt.IExecutionContext = type("IExecutionContext", (), {})

    fake_cuda = ModuleType("pycuda.driver")
    fake_cuda.__spec__ = ModuleSpec("pycuda.driver", loader=None)
    fake_cuda.Context = type("Context", (), {})
    fake_cuda.Device = type("Device", (), {})
    fake_pycuda = ModuleType("pycuda")
    fake_pycuda.__spec__ = ModuleSpec(
        "pycuda",
        loader=None,
        is_package=True,
    )
    fake_pycuda.__path__ = []
    fake_pycuda.driver = fake_cuda

    previous_parent_attributes = {}
    for name in _TRT_DEPENDENCY_MODULES:
        parent_name, attribute = name.rsplit(".", 1)
        parent = importlib.import_module(parent_name)
        previous_parent_attributes[name] = (
            parent,
            attribute,
            getattr(parent, attribute, _MISSING),
        )
    previous_modules = {
        name: sys.modules.pop(name, None) for name in _TRT_DEPENDENCY_MODULES
    }
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)
    monkeypatch.setitem(sys.modules, "pycuda", fake_pycuda)
    monkeypatch.setitem(sys.modules, "pycuda.driver", fake_cuda)

    try:
        module = importlib.import_module(_MODEL_MODULE)
        yield module.RFDetrForObjectDetectionTRT
    finally:
        for name in _TRT_DEPENDENCY_MODULES:
            sys.modules.pop(name, None)
            previous = previous_modules[name]
            if previous is not None:
                sys.modules[name] = previous
            parent, attribute, previous_attribute = previous_parent_attributes[name]
            if previous_attribute is _MISSING:
                if hasattr(parent, attribute):
                    delattr(parent, attribute)
            else:
                setattr(parent, attribute, previous_attribute)


class _RuntimeStage:
    def __init__(
        self,
        implementation_id: str,
        *,
        stage: OptimizationStage,
        fail_recoverably: bool = False,
        disable_after_failure: bool = True,
    ) -> None:
        fallback_id = (
            RFDETR_PREPROCESSOR_BASE
            if stage is OptimizationStage.PREPROCESS
            else RFDETR_POSTPROCESSOR_BASE
        )
        self.metadata = OptimizationMetadata(
            implementation_id=implementation_id,
            stage=stage,
            version="1",
            target=DeviceCompatibility(device_kind="gpu"),
            inputs=InputCompatibility(scenarios=("*",)),
            dependencies=(),
            fallback_id=fallback_id,
            changes_numerics=False,
            supports_concurrency=True,
            supports_cuda_graphs=False,
        )
        self._runtime_supported = True
        self._fail_recoverably = fail_recoverably
        self._disable_after_failure = disable_after_failure
        self.calls = 0

    def is_compatible(self, context: ExecutionContext) -> bool:
        del context
        return True

    def check_request_compatibility(
        self,
        *,
        request,
        context: ExecutionContext,
    ) -> CompatibilityResult:
        del request, context
        return CompatibilityResult.compatible()

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
            f"{self.metadata.implementation_id} failed during an earlier request"
        )

    def preprocess(
        self,
        request: PreprocessRequest,
        context: ExecutionContext,
    ) -> PreprocessResult:
        del request, context
        self.calls += 1
        self._raise_if_configured()

        return PreprocessResult(
            tensor=torch.full((1, 3, 2, 2), self.calls, dtype=torch.float32),
            metadata=[],
            implementation_id=self.metadata.implementation_id,
        )

    def postprocess(
        self,
        request: PostprocessRequest,
        context: ExecutionContext,
    ) -> List[str]:
        del request, context
        self.calls += 1
        self._raise_if_configured()

        return [self.metadata.implementation_id]

    def _raise_if_configured(self) -> None:
        if not self._fail_recoverably:
            return
        if self._disable_after_failure:
            self._runtime_supported = False
        raise RecoverableStageExecutionError(
            message=f"{self.metadata.implementation_id} recoverable failure",
            help_url=(
                "https://inference-models.roboflow.com/errors/models-runtime/"
                "#modelruntimeerror"
            ),
        )


class _Scheduler:
    def preprocess_stream(self):
        return object()

    def finalize_preprocess(
        self,
        engine_input,
        *,
        context: ExecutionContext,
        independent_stage_execution: bool,
    ) -> torch.Tensor:
        del context, independent_stage_execution
        return engine_input.tensor

    def execute_postprocess(self, model_results, *, operation):
        del model_results
        return operation(object())


class _BufferStrategy:
    def prepare_engine_input(
        self,
        result: PreprocessResult,
        context: ExecutionContext,
    ) -> PreprocessResult:
        del context
        return result


def _context() -> ExecutionContext:
    return ExecutionContext(device_kind="gpu", device="cuda:0")


def _build_model(
    model_class,
    *,
    selected_stage: _RuntimeStage,
    base_stage: _RuntimeStage,
    allow_runtime_failure_fallback: bool,
    allow_compatibility_fallback: bool,
):
    registry = ImplementationRegistry(scope_name="RF-DETR")
    registry.register(base_stage)
    registry.register(selected_stage)
    model = model_class.__new__(model_class)
    model._implementation_registry = registry
    model._rfdetr_execution_plan = SimpleNamespace(
        allow_compatibility_fallback=allow_compatibility_fallback,
        allow_runtime_failure_fallback=allow_runtime_failure_fallback,
    )
    model._scheduler = _Scheduler()
    model._request_fallback_warnings = SimpleNamespace(claim=lambda **kwargs: False)
    model._thread_local_storage = threading.local()
    model._execution_stage_context = MethodType(
        lambda self, *, current_stream: _context(),
        model,
    )
    model._record_static_stage_execution = MethodType(
        lambda self, *, stage: None,
        model,
    )

    return model


def _build_preprocess_model(
    model_class,
    *,
    candidate: _RuntimeStage,
    base: _RuntimeStage,
    allow_runtime_failure_fallback: bool = True,
    allow_compatibility_fallback: bool = True,
):
    model = _build_model(
        model_class,
        selected_stage=candidate,
        base_stage=base,
        allow_runtime_failure_fallback=allow_runtime_failure_fallback,
        allow_compatibility_fallback=allow_compatibility_fallback,
    )
    model._preprocessor = candidate
    model._buffer_strategy = _BufferStrategy()
    model._inference_config = SimpleNamespace(
        image_pre_processing=object(),
        network_input=object(),
    )

    return model


def _build_postprocess_model(
    model_class,
    *,
    candidate: _RuntimeStage,
    base: _RuntimeStage,
    allow_runtime_failure_fallback: bool = True,
    allow_compatibility_fallback: bool = True,
):
    model = _build_model(
        model_class,
        selected_stage=candidate,
        base_stage=base,
        allow_runtime_failure_fallback=allow_runtime_failure_fallback,
        allow_compatibility_fallback=allow_compatibility_fallback,
    )
    model._postprocessor = candidate
    model._classes_re_mapping = None
    model._class_names = ["class"]
    model.recommended_parameters = None

    return model


def _preprocess_stages(
    *,
    disable_after_failure: bool = True,
):
    candidate = _RuntimeStage(
        RFDETR_PREPROCESSOR_TRITON_UNIVERSAL_V1,
        stage=OptimizationStage.PREPROCESS,
        fail_recoverably=True,
        disable_after_failure=disable_after_failure,
    )
    base = _RuntimeStage(
        RFDETR_PREPROCESSOR_BASE,
        stage=OptimizationStage.PREPROCESS,
    )

    return candidate, base


def _postprocess_stages():
    candidate = _RuntimeStage(
        RFDETR_POSTPROCESSOR_TRITON_FUSED_V1,
        stage=OptimizationStage.POSTPROCESS,
        fail_recoverably=True,
    )
    base = _RuntimeStage(
        RFDETR_POSTPROCESSOR_BASE,
        stage=OptimizationStage.POSTPROCESS,
    )

    return candidate, base


def test_preprocess_retries_base_then_short_circuits_recorded_failure(
    rfdetr_trt_model_class,
) -> None:
    candidate, base = _preprocess_stages()
    model = _build_preprocess_model(
        rfdetr_trt_model_class,
        candidate=candidate,
        base=base,
    )

    first_tensor, first_metadata = model.pre_process(
        images=np.zeros((2, 2, 3), dtype=np.uint8)
    )
    first_selection = model._thread_local_storage.last_preprocessor_selection
    second_tensor, second_metadata = model.pre_process(
        images=np.zeros((2, 2, 3), dtype=np.uint8)
    )

    assert first_tensor.shape == (1, 3, 2, 2)
    assert second_tensor.shape == (1, 3, 2, 2)
    assert first_metadata == second_metadata == []
    assert candidate.calls == 1
    assert base.calls == 2
    assert first_selection["requested_id"] == RFDETR_PREPROCESSOR_TRITON_UNIVERSAL_V1
    assert first_selection["effective_id"] == RFDETR_PREPROCESSOR_BASE
    assert first_selection["fallback_reason"] is not None


@pytest.mark.parametrize(
    ("allow_compatibility_fallback", "allow_runtime_failure_fallback"),
    [(False, True), (True, False)],
)
def test_preprocess_disabled_fallback_exposes_model_runtime_error(
    rfdetr_trt_model_class,
    allow_compatibility_fallback: bool,
    allow_runtime_failure_fallback: bool,
) -> None:
    candidate, base = _preprocess_stages()
    model = _build_preprocess_model(
        rfdetr_trt_model_class,
        candidate=candidate,
        base=base,
        allow_compatibility_fallback=allow_compatibility_fallback,
        allow_runtime_failure_fallback=allow_runtime_failure_fallback,
    )

    with pytest.raises(ModelRuntimeError) as error:
        model.pre_process(images=np.zeros((2, 2, 3), dtype=np.uint8))

    assert type(error.value) is ModelRuntimeError
    assert candidate.calls == 1
    assert base.calls == 0


def test_preprocess_same_implementation_guard_does_not_retry(
    rfdetr_trt_model_class,
) -> None:
    candidate, base = _preprocess_stages(disable_after_failure=False)
    model = _build_preprocess_model(
        rfdetr_trt_model_class,
        candidate=candidate,
        base=base,
    )

    with pytest.raises(ModelRuntimeError) as error:
        model.pre_process(images=np.zeros((2, 2, 3), dtype=np.uint8))

    assert type(error.value) is ModelRuntimeError
    assert candidate.calls == 1
    assert base.calls == 0


def test_postprocess_retries_base_then_short_circuits_recorded_failure(
    rfdetr_trt_model_class,
) -> None:
    candidate, base = _postprocess_stages()
    model = _build_postprocess_model(
        rfdetr_trt_model_class,
        candidate=candidate,
        base=base,
    )
    model_results = (
        torch.zeros((1, 2, 4)),
        torch.zeros((1, 2, 2)),
    )

    first_results = model.post_process(
        model_results=model_results,
        pre_processing_meta=[],
        confidence=0.5,
    )
    first_selection = model._thread_local_storage.last_postprocessor_selection
    second_results = model.post_process(
        model_results=model_results,
        pre_processing_meta=[],
        confidence=0.5,
    )

    assert first_results == second_results == [RFDETR_POSTPROCESSOR_BASE]
    assert candidate.calls == 1
    assert base.calls == 2
    assert first_selection["requested_id"] == RFDETR_POSTPROCESSOR_TRITON_FUSED_V1
    assert first_selection["effective_id"] == RFDETR_POSTPROCESSOR_BASE
    assert first_selection["fallback_reason"] is not None


@pytest.mark.parametrize(
    ("allow_compatibility_fallback", "allow_runtime_failure_fallback"),
    [(False, True), (True, False)],
)
def test_postprocess_disabled_fallback_exposes_model_runtime_error(
    rfdetr_trt_model_class,
    allow_compatibility_fallback: bool,
    allow_runtime_failure_fallback: bool,
) -> None:
    candidate, base = _postprocess_stages()
    model = _build_postprocess_model(
        rfdetr_trt_model_class,
        candidate=candidate,
        base=base,
        allow_compatibility_fallback=allow_compatibility_fallback,
        allow_runtime_failure_fallback=allow_runtime_failure_fallback,
    )

    with pytest.raises(ModelRuntimeError) as error:
        model.post_process(
            model_results=(
                torch.zeros((1, 2, 4)),
                torch.zeros((1, 2, 2)),
            ),
            pre_processing_meta=[],
            confidence=0.5,
        )

    assert type(error.value) is ModelRuntimeError
    assert candidate.calls == 1
    assert base.calls == 0
