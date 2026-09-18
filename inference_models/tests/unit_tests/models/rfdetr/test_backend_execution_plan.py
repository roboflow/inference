"""Cross-backend selection, fallback and readiness without model downloads."""

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from inference_models.errors import ModelRuntimeError
from inference_models.models.common.roboflow.model_packages import (
    AnySizePadding,
    ColorMode,
    InferenceConfig,
    NetworkInputDefinition,
    ResizeMode,
    TrainingInputSize,
)
from inference_models.models.optimization.contracts import (
    CompatibilityResult,
    ExecutionContext,
)
from inference_models.models.optimization.errors import RecoverableStageExecutionError
from inference_models.models.rfdetr.optimization.backend_path import RFDetrBackendPath
from inference_models.models.rfdetr.optimization.backend_stages import (
    BackendExecutionScheduler,
)
from inference_models.models.rfdetr.optimization.contracts import (
    EngineInputBuffer,
    PreprocessRequest,
)
from inference_models.models.rfdetr.optimization.execution_plan import (
    RFDetrExecutionPlan,
)
from inference_models.models.rfdetr.optimization.preprocessors.base import (
    BasePreprocessor,
)
from inference_models.models.rfdetr.optimization.preprocessors.triton_universal import (
    TritonUniversalPreprocessor,
)
from inference_models.models.rfdetr.pre_processing import pre_process_network_input


def config():
    """Build a small dynamic-size RF-DETR configuration.

    Returns:
        InferenceConfig: Reference settings shared by backend and SIMD tests.
    """
    inference_config = InferenceConfig(
        network_input=NetworkInputDefinition(
            training_input_size=TrainingInputSize(height=32, width=32),
            dynamic_spatial_size_supported=True,
            dynamic_spatial_size_mode=AnySizePadding(type="any-size"),
            color_mode=ColorMode.RGB,
            resize_mode=ResizeMode.STRETCH_TO,
            input_channels=3,
            scaling_factor=255,
            normalization=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        )
    )

    return inference_config


@pytest.mark.parametrize("backend", ["torch", "onnx"])
def test_cpu_triton_request_falls_back_and_preserves_dynamic_size(backend):
    """Verify CPU fallback retains dynamic-size reference behavior.

    Args:
        backend (str): Backend registry to exercise.
    """
    cfg = config()
    path = RFDetrBackendPath(
        device=torch.device("cpu"),
        inference_config=cfg,
        backend=backend,
        execution_plan=RFDetrExecutionPlan(preprocessor_id="triton-universal-v1"),
    )
    image = np.random.default_rng(9).integers(0, 256, (95, 77, 3), dtype=np.uint8)
    actual, metadata = path.preprocess(image, image_size=(24, 40))
    expected, expected_meta = pre_process_network_input(
        image,
        cfg.image_pre_processing,
        cfg.network_input,
        torch.device("cpu"),
        image_size_wh=(24, 40),
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert metadata == expected_meta
    assert actual.shape == (1, 3, 40, 24)
    assert path.plan.preprocessor_id == "base"
    assert (
        "device_kind"
        in path.runtime_metadata["model_selection"]["preprocessor"]["fallback_reason"]
    )
    assert (
        path.forward(actual, stream=None, operation=lambda: {"boxes": actual})["boxes"]
        is actual
    )
    assert path.postprocess(lambda: ["detections"]) == ["detections"]
    assert set(path.runtime_metadata["last_execution"]) == {
        "preprocessor",
        "buffer_strategy",
        "scheduler",
        "engine_plugin",
        "postprocessor",
    }


@pytest.mark.parametrize(
    "stage",
    ["preprocessor", "scheduler", "buffer_strategy", "engine_plugin", "postprocessor"],
)
def test_unknown_stage_id_never_silently_falls_back(stage):
    """Reject misspelled implementation IDs for every stage.

    Args:
        stage (str): Execution-plan stage receiving an invalid ID.
    """
    plan = replace(RFDetrExecutionPlan(), **{stage + "_id": "typo"})
    with pytest.raises(ModelRuntimeError, match="Unknown"):
        RFDetrBackendPath(
            device=torch.device("cpu"),
            inference_config=config(),
            backend="onnx",
            execution_plan=plan,
        )


def test_cpu_strict_triton_selection_raises():
    """Reject incompatible CPU requests when fallback is disabled."""
    with pytest.raises(ModelRuntimeError, match="incompatible"):
        RFDetrBackendPath(
            device=torch.device("cpu"),
            inference_config=config(),
            backend="torch",
            execution_plan=RFDetrExecutionPlan(
                preprocessor_id="triton-universal-v1",
                allow_compatibility_fallback=False,
            ),
        )


def test_explicit_plan_overrides_environment(monkeypatch):
    """Prefer explicit plans over environment-derived implementation IDs.

    Args:
        monkeypatch (pytest.MonkeyPatch): Environment override fixture.
    """
    monkeypatch.setenv("INFERENCE_MODELS_RFDETR_PREPROCESSOR", "unknown")
    path = RFDetrBackendPath(
        device=torch.device("cpu"),
        inference_config=config(),
        backend="torch",
        execution_plan=RFDetrExecutionPlan(preprocessor_id="base"),
    )
    assert path.plan.preprocessor_id == "base"
    with pytest.raises(ModelRuntimeError, match="Unknown"):
        RFDetrBackendPath(
            device=torch.device("cpu"), inference_config=config(), backend="torch"
        )


def test_universal_size_override_declares_fallback_without_gpu_work():
    """Reject custom input sizes before attempting CUDA preprocessing."""
    cfg = config()
    stage = object.__new__(TritonUniversalPreprocessor)
    request = PreprocessRequest(
        np.zeros((50, 60, 3), dtype=np.uint8),
        "bgr",
        cfg.image_pre_processing,
        cfg.network_input,
        None,
        (16, 16),
    )
    result = stage.check_request_compatibility(
        request=request, context=ExecutionContext("gpu", "cuda:0")
    )
    assert not result.supported
    assert "image_size" in result.reason


def test_non_cuda_gpu_declares_fallback_before_constructing_cuda_runtime():
    """Reject MPS targets without constructing a CUDA implementation."""
    path = RFDetrBackendPath(
        device=torch.device("mps"),
        inference_config=config(),
        backend="torch",
        execution_plan=RFDetrExecutionPlan(preprocessor_id="triton-universal-v1"),
    )
    assert path.plan.preprocessor_id == "base"
    assert (
        "device"
        in path.runtime_metadata["model_selection"]["preprocessor"]["fallback_reason"]
    )


def test_reference_adapter_records_caller_cuda_storage(monkeypatch):
    """Keep caller-owned CUDA tensor storage alive on the preprocessing stream.

    Args:
        monkeypatch (pytest.MonkeyPatch): CUDA adapter replacement fixture.
    """
    from inference_models.models.rfdetr.optimization.preprocessors import common

    image = MagicMock(spec=torch.Tensor)
    image.device = torch.device("cuda:0")
    stream = SimpleNamespace(device=image.device)
    output = torch.zeros((1, 3, 32, 32))
    monkeypatch.setattr(common, "use_cuda_stream", lambda _: nullcontext())
    monkeypatch.setattr(
        common, "pre_process_network_input", lambda **kwargs: (output, [])
    )
    cfg = config()
    result = common.run_reference_preprocessor(
        PreprocessRequest(
            image, "rgb", cfg.image_pre_processing, cfg.network_input, None
        ),
        ExecutionContext("gpu", "cuda:0", current_stream=stream),
        implementation_id="base",
        max_workers=1,
    )
    image.record_stream.assert_called_once_with(stream)
    assert result.tensor is output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime required")
@pytest.mark.parametrize("backend", ["torch", "onnx"])
@pytest.mark.parametrize("kind", ["numpy", "uint8-cuda", "float-cuda"])
@pytest.mark.parametrize("preprocessor_id", ["base", "triton-universal-v1"])
def test_cuda_preprocessing_parity_and_event_handoff(backend, kind, preprocessor_id):
    """Compare reference numerics after a cross-stream input handoff.

    Args:
        backend (str): Backend registry to exercise.
        kind (str): NumPy, uint8 CUDA, or floating CUDA input representation.
        preprocessor_id (str): Explicit preprocessing implementation to select.
    """
    from inference_models.models.rfdetr.triton_preprocess import TRITON_AVAILABLE

    if not TRITON_AVAILABLE:
        pytest.skip("Triton required")

    device = torch.device("cuda:0")
    cfg = config()
    path = RFDetrBackendPath(
        device=device,
        inference_config=cfg,
        backend=backend,
        execution_plan=RFDetrExecutionPlan(
            preprocessor_id=preprocessor_id, allow_compatibility_fallback=False
        ),
    )
    image = np.random.default_rng(11).integers(0, 256, (160, 200, 3), dtype=np.uint8)
    if kind != "numpy":
        image = torch.from_numpy(image).permute(2, 0, 1).to(device)
    if kind == "float-cuda":
        image = image.float() / 255
    actual, meta = path.preprocess(
        image, input_color_format="bgr", independent_stage_execution=False
    )
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        consumed = path.forward(actual, stream=stream, operation=lambda: actual.clone())
    stream.synchronize()
    expected, expected_meta = pre_process_network_input(
        image,
        cfg.image_pre_processing,
        cfg.network_input,
        device,
        input_color_format="bgr",
    )
    torch.testing.assert_close(consumed, expected, rtol=0, atol=1e-6)
    assert meta == expected_meta
    assert (
        path.runtime_metadata["last_execution"]["preprocessor"]["effective_id"]
        == preprocessor_id
    )


class _FailingPreprocessor(BasePreprocessor):
    metadata = replace(BasePreprocessor.metadata, implementation_id="failing")

    def __init__(self, error):
        super().__init__(max_workers=1)
        self.error = error
        self.failed = False
        self.calls = 0

    def preprocess(self, request, context):
        """Record one attempted call and raise the configured test error.

        Args:
            request (PreprocessRequest): Ignored test request.
            context (ExecutionContext): Ignored test runtime context.

        Raises:
            Exception: The error supplied when constructing the double.
        """
        self.calls += 1
        self.failed = True
        raise self.error

    def check_runtime_compatibility(self, *, request, context):
        """Report a permanent runtime failure after the first attempted call.

        Args:
            request (PreprocessRequest): Ignored test request.
            context (ExecutionContext): Ignored runtime context.

        Returns:
            CompatibilityResult: Whether this double has not failed yet.
        """
        result = (
            CompatibilityResult.incompatible("JIT failed")
            if self.failed
            else CompatibilityResult.compatible()
        )

        return result


@pytest.mark.parametrize("allow_runtime", [True, False])
def test_runtime_failure_policy_and_future_short_circuit(allow_runtime):
    """Recover only when policy permits and avoid retrying failed implementations.

    Args:
        allow_runtime (bool): Whether recoverable failures may use the base stage.
    """
    path = RFDetrBackendPath(
        device=torch.device("cpu"),
        inference_config=config(),
        backend="torch",
        execution_plan=RFDetrExecutionPlan(
            preprocessor_id="base", allow_runtime_failure_fallback=allow_runtime
        ),
    )
    candidate = _FailingPreprocessor(
        RecoverableStageExecutionError(message="JIT failed")
    )
    path.preprocessor = candidate
    image = np.zeros((50, 60, 3), dtype=np.uint8)
    if not allow_runtime:
        with pytest.raises(ModelRuntimeError, match="JIT failed"):
            path.preprocess(image)
    else:
        first, _ = path.preprocess(image)
        second, _ = path.preprocess(image)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        assert candidate.calls == 1
        assert (
            path.runtime_metadata["last_execution"]["preprocessor"]["effective_id"]
            == "base"
        )


def test_unclassified_failure_propagates():
    """Propagate unexpected errors without silently selecting another stage."""
    path = RFDetrBackendPath(
        device=torch.device("cpu"), inference_config=config(), backend="onnx"
    )
    path.preprocessor = _FailingPreprocessor(ValueError("unexpected"))
    with pytest.raises(ValueError, match="unexpected"):
        path.preprocess(np.zeros((50, 60, 3), dtype=np.uint8))


def test_scheduler_waits_for_the_exact_tensor_and_standalone_synchronizes():
    """Synchronize standalone inputs and consume readiness by exact tensor identity."""
    calls = []
    event = SimpleNamespace(synchronize=lambda: calls.append("synchronize"))
    stream = SimpleNamespace(wait_event=lambda value: calls.append(("wait", value)))
    scheduler = BackendExecutionScheduler()
    tensor = torch.zeros(1)
    buffer = EngineInputBuffer(tensor, event, "uint8", "test")
    context = ExecutionContext("gpu", "cuda:0", current_stream=stream)
    scheduler.finalize_preprocess(
        buffer, context=context, independent_stage_execution=True
    )
    assert calls == ["synchronize"]
    scheduler.finalize_preprocess(
        buffer, context=context, independent_stage_execution=False
    )

    # Tensor doubles retain weakref identity and model CUDA allocator recording.
    class _Tensor:
        def record_stream(self, stream):
            """Record allocator-lifetime tracking without requiring CUDA.

            Args:
                stream (SimpleNamespace): Fake consumer stream.
            """
            calls.append("record")

    cuda_tensor = _Tensor()
    cuda_buffer = replace(buffer, tensor=cuda_tensor)
    scheduler.finalize_preprocess(
        cuda_buffer, context=context, independent_stage_execution=False
    )
    scheduler.execute_engine(
        cuda_tensor, stream=stream, operation=lambda: calls.append("forward")
    )
    assert calls[-3:] == [("wait", event), "record", "forward"]
    calls.clear()
    scheduler.execute_engine(
        cuda_tensor, stream=stream, operation=lambda: calls.append("forward")
    )
    assert calls == ["record", "forward"]


def test_torch_model_runs_selected_stages_with_real_cpu_tensors():
    """Exercise all composed stages around a lightweight Torch model."""
    from inference_models.models.rfdetr.post_processor import PostProcess
    from inference_models.models.rfdetr.rfdetr_object_detection_pytorch import (
        RFDetrForObjectDetectionTorch,
    )

    class _TinyModel(torch.nn.Module):
        def forward(self, tensor):
            """Produce predictable raw predictions for a real tensor batch.

            Args:
                tensor (torch.Tensor): Preprocessed input batch.

            Returns:
                dict: Constant boxes and low-confidence class logits.
            """
            return {
                "pred_boxes": torch.full((len(tensor), 300, 4), 0.5),
                "pred_logits": torch.full((len(tensor), 300, 2), -5.0),
            }

    model = RFDetrForObjectDetectionTorch(
        model=_TinyModel(),
        inference_config=config(),
        class_names=["object"],
        classes_re_mapping=None,
        device=torch.device("cpu"),
        post_processor=PostProcess(),
        resolution=32,
        rfdetr_execution_plan=RFDetrExecutionPlan(preprocessor_id="base"),
    )
    image = np.zeros((50, 60, 3), dtype=np.uint8)
    result = model(image, image_size=(48, 40))
    assert len(result) == 1
    assert result[0].xyxy.shape == (0, 4)
    assert set(model.optimization_runtime_metadata["last_execution"]) == {
        "preprocessor",
        "buffer_strategy",
        "scheduler",
        "engine_plugin",
        "postprocessor",
    }


def test_onnx_model_delegates_forward_and_postprocess_through_plan(monkeypatch):
    """Exercise all composed stages around an ONNX-session double.

    Args:
        monkeypatch (pytest.MonkeyPatch): ONNX execution replacement fixture.
    """
    from inference_models.models.rfdetr import rfdetr_object_detection_onnx as adapter

    received = []

    def _infer(**kwargs):
        received.append(kwargs["inputs"]["images"])
        boxes = torch.full((1, 300, 4), 0.5)
        logits = torch.full((1, 300, 2), -5.0)

        return boxes, logits

    monkeypatch.setattr(adapter, "run_onnx_session_with_batch_size_limit", _infer)
    model = adapter.RFDetrForObjectDetectionONNX(
        session=object(),
        input_name="images",
        class_names=["object"],
        classes_re_mapping=None,
        inference_config=config(),
        device=torch.device("cpu"),
        input_batch_size=1,
        rfdetr_execution_plan=RFDetrExecutionPlan(
            preprocessor_id="triton-universal-v1"
        ),
    )
    results = model(np.zeros((50, 60, 3), dtype=np.uint8))
    assert len(results) == 1
    assert received[0].shape == (1, 3, 32, 32)
    assert model.preprocessor_implementation_id == "base"
    assert set(model.optimization_runtime_metadata["last_execution"]) == {
        "preprocessor",
        "buffer_strategy",
        "scheduler",
        "engine_plugin",
        "postprocessor",
    }
