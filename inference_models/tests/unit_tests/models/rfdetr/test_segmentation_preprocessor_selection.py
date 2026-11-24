"""Segmentation adapter selection and exact-tensor readiness without a GPU."""

import importlib
import sys
import threading
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from inference_models.models.optimization.contracts import CompatibilityResult
from inference_models.models.rfdetr.optimization.execution_plan import (
    RFDetrExecutionPlan,
)
from inference_models.models.rfdetr.optimization.readiness import (
    PreprocessReadinessTracker,
)
from inference_models.models.rfdetr.optimization.segmentation_preprocessing import (
    _resolve_segmentation_plan,
    _SegmentationUniversalPreprocessor,
)
from tests.unit_tests.models.rfdetr import (
    test_rfdetr_trt_runtime_fallback as dependencies,
)


@pytest.fixture
def model_class(monkeypatch):
    name = "inference_models.models.rfdetr.rfdetr_instance_segmentation_trt"
    monkeypatch.setattr(dependencies, "_MODEL_MODULE", name)
    monkeypatch.setattr(
        dependencies,
        "_TRT_DEPENDENCY_MODULES",
        (
            "inference_models.models.common.cuda",
            "inference_models.models.common.trt",
            name,
        ),
    )
    # Reuse the tested isolated import fixture's dependency doubles, selecting
    # the segmentation class under its expected yield attribute only temporarily.
    original_import = importlib.import_module

    def import_module(module_name, *args, **kwargs):
        module = original_import(module_name, *args, **kwargs)
        if module_name == name:
            monkeypatch.setattr(
                module,
                "RFDetrForObjectDetectionTRT",
                module.RFDetrForInstanceSegmentationTRT,
                raising=False,
            )
        return module

    monkeypatch.setattr(importlib, "import_module", import_module)
    yield from dependencies.rfdetr_trt_model_class.__wrapped__(monkeypatch)


def test_default_environment_and_explicit_precedence(monkeypatch):
    monkeypatch.delenv("INFERENCE_MODELS_RFDETR_PREPROCESSOR", raising=False)
    assert (
        _resolve_segmentation_plan(None, legacy_enabled=False).preprocessor_id == "base"
    )
    assert (
        _resolve_segmentation_plan(None, legacy_enabled=True).preprocessor_id
        == "triton-universal-v1"
    )
    monkeypatch.setenv("INFERENCE_MODELS_RFDETR_PREPROCESSOR", "base")
    assert (
        _resolve_segmentation_plan(None, legacy_enabled=True).preprocessor_id == "base"
    )
    plan = RFDetrExecutionPlan(
        preprocessor_id="triton-universal-v1", postprocessor_id="base"
    )
    assert _resolve_segmentation_plan(plan.to_dict(), legacy_enabled=False) == plan
    with pytest.raises(ValueError, match="existing base postprocessor"):
        _resolve_segmentation_plan(
            RFDetrExecutionPlan(postprocessor_id="triton-fused-v1"),
            legacy_enabled=False,
        )
    with pytest.raises(ValueError, match="exactly"):
        _resolve_segmentation_plan({}, legacy_enabled=False)


def test_float_dtype_and_device_guard(monkeypatch):
    from inference_models.models.rfdetr.optimization.preprocessors.triton_universal import (
        TritonUniversalPreprocessor,
    )

    monkeypatch.setattr(
        TritonUniversalPreprocessor,
        "check_request_compatibility",
        lambda *args, **kwargs: CompatibilityResult.compatible(),
    )
    implementation = object.__new__(_SegmentationUniversalPreprocessor)
    context = SimpleNamespace(device="cuda:0")
    for scores in (torch.zeros(3, 8, 8), torch.zeros(3, 8, 8).half()):
        result = implementation.check_request_compatibility(
            request=SimpleNamespace(images=scores), context=context
        )
        assert not result.supported and "dtype/device" in result.reason
    assert implementation.check_request_compatibility(
        request=SimpleNamespace(images=np.zeros((8, 8, 3), dtype=np.uint8)),
        context=context,
    ).supported


def test_preprocess_twice_then_forward_first_uses_its_event(model_class, monkeypatch):
    model = object.__new__(model_class)
    model._preprocess_readiness = PreprocessReadinessTracker()
    model._inference_stream = Mock()
    model._lock = threading.Lock()
    model._trt_cuda_graph_cache = None
    model._cuda_context = None
    model._trt_config = model._engine = model._execution_context = None
    model._device = torch.device("cuda:0")
    model._input_name, model._output_names = "images", ["boxes", "labels", "masks"]
    first, second = torch.empty(1), torch.empty(1)
    events = [Mock(), Mock()]
    for tensor, event in zip((first, second), events):
        model._preprocess_readiness.record(
            tensor,
            ready_event=event,
            input_kind="uint8-triton",
            implementation_id="triton-universal-v1",
        )
    recorded = []
    monkeypatch.setattr(
        torch.Tensor, "record_stream", lambda tensor, stream: recorded.append(tensor)
    )
    module = sys.modules[model_class.__module__]
    monkeypatch.setattr(module, "use_cuda_context", lambda **kwargs: nullcontext())
    infer = Mock(return_value=(first, first, first))
    monkeypatch.setattr(module, "infer_from_trt_engine", infer)
    model.forward(first)
    assert model._inference_stream.wait_event.call_args.args == (events[0],)
    assert recorded[0] is first
    model.forward(second)
    assert model._inference_stream.wait_event.call_args.args == (events[1],)
    assert model._preprocess_readiness.consume(first) is None


def test_async_forward_consumes_exact_input(model_class, monkeypatch):
    model = object.__new__(model_class)
    model._trt_cuda_graph_cache = object()
    model._lock = threading.Lock()
    model._inference_stream = Mock()
    model._cuda_context = None
    model._trt_config = model._engine = model._execution_context = None
    model._device = torch.device("cuda:0")
    model._input_name, model._output_names = "images", ["boxes", "labels", "masks"]
    model._preprocess_readiness = PreprocessReadinessTracker()
    tensor, event = torch.empty(1), Mock()
    model._preprocess_readiness.record(
        tensor,
        ready_event=event,
        input_kind="uint8-triton",
        implementation_id="triton-universal-v1",
    )
    monkeypatch.setattr(torch.Tensor, "record_stream", Mock())
    module = sys.modules[model_class.__module__]
    monkeypatch.setattr(module, "use_cuda_context", lambda **kwargs: nullcontext())
    monkeypatch.setattr(
        module, "infer_from_trt_engine", Mock(return_value=(tensor, tensor, tensor))
    )
    model.forward_async(tensor, [])
    model._inference_stream.wait_event.assert_called_once_with(event)


@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("failure", [None, "recoverable", "strict", "unknown"])
def test_adapter_forwards_overrides_and_publishes_per_call_readiness(
    model_class, monkeypatch, independent, failure
):
    from inference_models.models.optimization.contracts import ExecutionContext
    from inference_models.models.rfdetr.optimization.contracts import PreprocessResult

    model = object.__new__(model_class)
    model._device = torch.device("cuda:0")
    model._thread_local_storage = threading.local()
    model._stream_pipeline_enabled = True
    stream, event = Mock(), Mock()
    stream.device = model._device
    model._pre_process_cuda_stream = stream
    model._preprocess_context = ExecutionContext(device_kind="gpu", device="cuda:0")
    model._preprocess_readiness = PreprocessReadinessTracker()
    model._inference_config = SimpleNamespace(
        image_pre_processing=object(), network_input=object()
    )
    model._rfdetr_execution_plan = RFDetrExecutionPlan(
        preprocessor_id="triton-universal-v1"
    )
    model._preprocessor_model_fallback_reason = None
    tensor = torch.empty(1)
    result = PreprocessResult(
        tensor=tensor,
        metadata=[],
        ready_event=event,
        implementation_id="base",
        input_kind="reference",
    )
    implementation = Mock()
    implementation.preprocess.return_value = result
    selection = SimpleNamespace(
        implementation=implementation,
        effective_id="base",
        fallback_reason="custom image_size override",
    )
    model._preprocessor = object()
    selector = Mock()
    selector.resolve_request.return_value = selection
    selector.resolve_runtime_fallback.return_value = selection
    model._preprocessor_selector = selector
    monkeypatch.setattr(torch.cuda, "current_stream", Mock(return_value=object()))
    monkeypatch.setattr(torch.Tensor, "record_stream", Mock())
    if failure is not None:
        from dataclasses import replace

        from inference_models.errors import ModelRuntimeError
        from inference_models.models.optimization.errors import (
            RecoverableStageExecutionError,
        )

        failing_stage = Mock()
        error = (
            ValueError("unknown failure")
            if failure == "unknown"
            else RecoverableStageExecutionError(message="known JIT failure")
        )
        failing_stage.preprocess.side_effect = error
        initial = SimpleNamespace(
            implementation=failing_stage,
            effective_id="triton-universal-v1",
            fallback_reason=None,
        )
        selector.resolve_runtime_fallback.side_effect = [initial, selection]
        if failure == "strict":
            model._rfdetr_execution_plan = replace(
                model._rfdetr_execution_plan, allow_runtime_failure_fallback=False
            )
        if failure in ("strict", "unknown"):
            with pytest.raises(
                ModelRuntimeError if failure == "strict" else ValueError
            ):
                model.pre_process(np.zeros((8, 8, 3), dtype=np.uint8))
            implementation.preprocess.assert_not_called()
            return

    overrides = object()
    actual, metadata = model.pre_process(
        np.zeros((8, 8, 3), dtype=np.uint8),
        image_size=(16, 24),
        pre_processing_overrides=overrides,
        independent_stage_execution=independent,
    )
    request = selector.resolve_request.call_args.kwargs["request"]
    assert request.image_size_wh == (16, 24)
    assert request.pre_processing_overrides is overrides
    assert actual is tensor and metadata == []
    assert (
        model.last_preprocessor_selection["fallback_reason"]
        == "custom image_size override"
    )
    readiness = model._preprocess_readiness.consume(tensor)
    if independent:
        event.synchronize.assert_called_once()
        assert readiness is None
    else:
        event.synchronize.assert_not_called()
        assert readiness.ready_event is event


def test_universal_custom_size_is_incompatible_and_disabling_overrides_are_allowed(
    monkeypatch,
):
    from inference_models import PreProcessingOverrides
    from inference_models.models.common.roboflow.model_packages import (
        ImagePreProcessing,
    )
    from inference_models.models.optimization.contracts import ExecutionContext
    from inference_models.models.rfdetr import (
        triton_universal_preprocess_runtime as runtime,
    )
    from inference_models.models.rfdetr.optimization.contracts import PreprocessRequest
    from tests.unit_tests.models.rfdetr.test_triton_universal_preprocess_runtime import (
        _network_input,
    )

    monkeypatch.setattr(runtime, "TRITON_AVAILABLE", True)
    stage = _SegmentationUniversalPreprocessor(device=torch.device("cuda:0"))
    context = ExecutionContext(device_kind="gpu", device="cuda:0")
    request = PreprocessRequest(
        images=np.zeros((8, 8, 3), dtype=np.uint8),
        image_pre_processing=ImagePreProcessing(),
        network_input=_network_input(),
        input_color_format="bgr",
        pre_processing_overrides=PreProcessingOverrides(),
    )
    assert stage.check_request_compatibility(request=request, context=context).supported
    from dataclasses import replace

    refused = stage.check_request_compatibility(
        request=replace(request, image_size_wh=(16, 16)), context=context
    )
    assert not refused.supported and "image_size" in refused.reason


@pytest.mark.parametrize(
    "means,stds",
    [
        ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ((0.0, 0.25, -0.5), (1.0, 0.5, 2.0)),
    ],
)
def test_exact_uint8_lut_covers_every_byte_in_reference_order(means, stds):
    import torchvision.transforms.functional as TF
    from inference_models.models.rfdetr.triton_universal_preprocess_runtime import (
        _exact_uint8_normalization_lut,
    )

    image = np.repeat(np.arange(256, dtype=np.uint8).reshape(16, 16, 1), 3, axis=2)
    expected = TF.normalize(
        TF.to_tensor(image), mean=list(means), std=list(stds)
    ).reshape(3, 256)
    actual = _exact_uint8_normalization_lut(means=means, stds=stds)
    assert torch.equal(actual, expected)
    assert actual.dtype == torch.float32 and actual.is_contiguous()
