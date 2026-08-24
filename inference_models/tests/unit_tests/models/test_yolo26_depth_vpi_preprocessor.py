from contextlib import nullcontext
from types import SimpleNamespace

import cv2
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
    ExecutionContext,
    OptimizationStage,
)
from inference_models.models.yolo26.optimization import preprocessors
from inference_models.models.yolo26.optimization.ids import (
    YOLO26_DEPTH_PREPROCESSOR_VPI_CUDA_LETTERBOX_FUSED_CONVERT_V3,
)
from inference_models.models.yolo26.optimization.postprocessors import (
    build_yolo26_depth_implementation_registry,
)
from inference_models.models.yolo26.optimization.preprocessors import (
    VPICUDALetterboxFusedConvertYOLO26DepthPreprocessor,
    _prepare_exact_vpi_letterbox_request,
    _prepare_large_numpy_image,
)
from inference_models.models.yolo26.vpi_depth_preprocess import (
    VPICUDALetterboxResizer,
)


def _network_input(*, resize_mode=ResizeMode.LETTERBOX):
    return NetworkInputDefinition(
        training_input_size=TrainingInputSize(width=3, height=3),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=resize_mode,
        padding_value=127,
        input_channels=3,
        scaling_factor=255,
        normalization=None,
    )


def _use_small_exact_geometry(monkeypatch):
    monkeypatch.setattr(preprocessors, "_VPI_EXACT_SOURCE_SIZE", (10, 15))
    monkeypatch.setattr(preprocessors, "_VPI_EXACT_TARGET_SIZE", (3, 3))
    monkeypatch.setattr(preprocessors, "_VPI_EXACT_RESIZED_SIZE", (2, 3))


def test_exact_vpi_request_matches_opencv_integer_letterbox_contract(
    monkeypatch,
):
    _use_small_exact_geometry(monkeypatch)
    image = np.arange(10 * 15 * 3, dtype=np.uint8).reshape((10, 15, 3))
    image_pre_processing = ImagePreProcessing()
    network_input = _network_input()

    source, geometry, padding_value, metadata = _prepare_exact_vpi_letterbox_request(
        image=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        input_color_mode=ColorMode.BGR,
        pre_processing_overrides=None,
    )
    opencv_image, opencv_metadata = _prepare_large_numpy_image(
        image=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        input_color_mode=ColorMode.BGR,
        pre_processing_overrides=None,
    )
    expected_content = cv2.resize(image, (3, 2))

    assert source is image
    assert np.array_equal(expected_content, image[2::5, 2::5])
    assert np.array_equal(opencv_image[:2], expected_content)
    assert np.all(opencv_image[2] == padding_value)
    assert geometry.new_height == 2
    assert geometry.new_width == 3
    assert (geometry.pad_top, geometry.pad_bottom) == (0, 1)
    assert metadata == opencv_metadata


def test_exact_vpi_request_rejects_non_letterbox_resize(monkeypatch):
    _use_small_exact_geometry(monkeypatch)

    with pytest.raises(ModelRuntimeError, match="resize mode must be letterbox"):
        _prepare_exact_vpi_letterbox_request(
            image=np.zeros((10, 15, 3), dtype=np.uint8),
            image_pre_processing=ImagePreProcessing(),
            network_input=_network_input(resize_mode=ResizeMode.STRETCH_TO),
            input_color_mode=ColorMode.BGR,
            pre_processing_overrides=None,
        )


def test_registry_rejects_explicit_vpi_candidate_when_runtime_is_missing():
    registry = build_yolo26_depth_implementation_registry(
        device=torch.device("cuda:0"),
    )
    context = ExecutionContext(
        device_kind="gpu",
        device="cuda:0",
        compute_capability=(8, 7),
        runtime_components={"VPI": False, "torch": True, "triton": True},
    )

    with pytest.raises(ModelRuntimeError, match="unavailable runtime components"):
        registry.resolve_selection(
            stage=OptimizationStage.PREPROCESS,
            requested_id=(
                YOLO26_DEPTH_PREPROCESSOR_VPI_CUDA_LETTERBOX_FUSED_CONVERT_V3
            ),
            context=context,
            allow_fallback=False,
        )


class _FakeEvent:
    def __init__(self, calls):
        self._calls = calls

    def record(self, stream):
        self._calls.append(("event-record", stream))

    def synchronize(self):
        self._calls.append("event-synchronize")


class _FakeLock:
    def __init__(self, calls, buffer):
        self._calls = calls
        self._buffer = buffer

    def __enter__(self):
        self._calls.append("lock-enter")
        return self._buffer

    def __exit__(self, exc_type, exc_value, traceback):
        self._calls.append("lock-exit")


class _FakeVPIImage:
    def __init__(self, calls, buffer):
        self._calls = calls
        self._buffer = buffer

    def rlock_cuda(self):
        self._calls.append("lock-create")
        return _FakeLock(self._calls, self._buffer)


class _FakeVPIStream:
    def __init__(self, calls):
        self._calls = calls

    def sync(self):
        self._calls.append("vpi-stream-sync")


class _FakeVPISource:
    def __init__(self, calls):
        self._calls = calls

    def rescale(self, output, *, interp, backend, stream):
        self._calls.append(("rescale", output, interp, backend, stream))


class _FakeVPIModule:
    __version__ = "3.2.4"
    Format = SimpleNamespace(BGR8="BGR8")
    Interp = SimpleNamespace(LINEAR="LINEAR")
    Backend = SimpleNamespace(CUDA="CUDA")

    def __init__(self, calls, buffer):
        self._calls = calls
        self._buffer = buffer

    def Image(self, size, image_format):
        self._calls.append(("image-create", size, image_format))
        return _FakeVPIImage(self._calls, self._buffer)

    def Stream(self):
        self._calls.append("stream-create")
        return _FakeVPIStream(self._calls)

    def asimage(self, image, image_format):
        self._calls.append(("source-wrap", image.shape, image_format))
        return _FakeVPISource(self._calls)


def test_vpi_output_lock_remains_until_torch_consumer_event(monkeypatch):
    calls = []
    buffer = object()
    vpi_module = _FakeVPIModule(calls, buffer)
    fake_tensor = torch.zeros((2, 3, 3), dtype=torch.uint8)
    monkeypatch.setattr(torch.cuda, "Event", lambda: _FakeEvent(calls))
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda _: nullcontext())
    monkeypatch.setattr(torch, "as_tensor", lambda *args, **kwargs: fake_tensor)
    monkeypatch.setattr(
        VPICUDALetterboxResizer,
        "_validate_zero_copy_output",
        lambda self, **kwargs: None,
    )
    resizer = VPICUDALetterboxResizer(
        device=torch.device("cuda:0"),
        vpi_module=vpi_module,
    )
    consumer_stream = object()

    borrowed = resizer.resize(
        image=np.zeros((10, 15, 3), dtype=np.uint8),
        output_height=2,
        output_width=3,
        target_device=torch.device("cuda:0"),
    )
    borrowed.mark_consumed_and_release(stream=consumer_stream)
    acquired = resizer._pool.acquire()
    resizer._pool.release(acquired)

    assert borrowed.tensor is fake_tensor
    assert "vpi-stream-sync" in calls
    assert calls.index("lock-enter") < calls.index(("event-record", consumer_stream))
    assert calls.index(("event-record", consumer_stream)) < calls.index(
        "event-synchronize"
    )
    assert calls.index("event-synchronize") < calls.index("lock-exit")


class _FakeBorrow:
    def __init__(self):
        self.tensor = torch.zeros((2, 3, 3), dtype=torch.uint8)
        self.completed_stream = None
        self.aborted_stream = None

    def mark_consumed_and_release(self, *, stream):
        self.completed_stream = stream

    def abort_and_release(self, *, stream):
        self.aborted_stream = stream


class _FakeResizer:
    def __init__(self, borrow):
        self.borrow = borrow
        self.request = None

    def resize(self, **kwargs):
        self.request = kwargs
        return self.borrow


class _FakeConverter:
    def __init__(self):
        self.request = None
        self.output = torch.ones((1, 3, 3, 3))

    def convert_letterbox(self, **kwargs):
        self.request = kwargs
        return self.output


def test_vpi_preprocessor_composes_resize_padding_and_stream_lifetime(monkeypatch):
    _use_small_exact_geometry(monkeypatch)
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda _: nullcontext())
    borrow = _FakeBorrow()
    resizer = _FakeResizer(borrow)
    converter = _FakeConverter()
    preprocessor = object.__new__(VPICUDALetterboxFusedConvertYOLO26DepthPreprocessor)
    preprocessor._resizer = resizer
    preprocessor._converter = converter
    stream = object()

    output, metadata = preprocessor._prepare_and_convert(
        image=np.zeros((10, 15, 3), dtype=np.uint8),
        image_pre_processing=ImagePreProcessing(),
        network_input=_network_input(),
        target_device=torch.device("cuda:0"),
        input_color_mode=ColorMode.BGR,
        pre_processing_overrides=None,
        context=ExecutionContext(
            device_kind="gpu",
            device="cuda:0",
            current_stream=stream,
        ),
    )

    assert output is converter.output
    assert resizer.request["output_height"] == 2
    assert resizer.request["output_width"] == 3
    assert converter.request["target_size"] == (3, 3)
    assert converter.request["padding"] == (0, 0, 1, 0)
    assert converter.request["padding_value"] == 127
    assert converter.request["reverse_channels"]
    assert metadata.pad_bottom == 1
    assert borrow.completed_stream is stream
    assert borrow.aborted_stream is None
