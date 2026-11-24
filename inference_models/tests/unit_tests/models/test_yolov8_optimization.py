"""CPU contract tests for opt-in YOLOv8 postprocessing.

CUDA is never executed; stream-wiring tests only observe fake stream objects.
"""

import copy
import importlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace

import pytest
import torch

from inference_models import Detections
from inference_models.entities import ImageDimensions
from inference_models.errors import ModelRuntimeError
from inference_models.models.common.roboflow.model_packages import (
    PreProcessingMetadata,
    StaticCropOffset,
)
from inference_models.models.common.roboflow.post_processing import (
    post_process_nms_fused_model_output,
    rescale_detections,
    run_nms_for_object_detection,
)
from inference_models.models.optimization.contracts import ExecutionContext
from inference_models.models.yolov8.optimization import segmented
from inference_models.models.yolov8.optimization.base import BasePostprocessor
from inference_models.models.yolov8.optimization.contracts import PostprocessRequest


def _metadata(*, offset=0):
    size = ImageDimensions(width=640, height=480)
    metadata = PreProcessingMetadata(
        pad_left=7,
        pad_top=11,
        pad_right=0,
        pad_bottom=0,
        original_size=size,
        size_after_pre_processing=size,
        inference_size=ImageDimensions(width=640, height=640),
        scale_width=0.75,
        scale_height=1.25,
        static_crop_offset=StaticCropOffset(
            offset_x=offset, offset_y=offset, crop_width=640, crop_height=480
        ),
    )

    return metadata


def _raw():
    # Tied scores, duplicated boxes in distinct classes, threshold equality,
    # coordinates requiring clipping, and an empty second image.
    raw = torch.tensor(
        [
            [
                [20, 20, 80, 800],
                [20, 20, 60, 800],
                [20, 20, 10, 50],
                [20, 20, 10, 50],
                [0.75, 0.0, 0.5, 0.75],
                [0, 0.75, 0, 0],
            ]
        ],
        dtype=torch.float32,
    ).repeat(2, 1, 1)
    raw[1, 4:] = 0

    return raw


def _request(**overrides):
    values = dict(
        model_results=_raw(),
        pre_processing_meta=[_metadata(), _metadata(offset=17)],
        threshold=0.5,
        iou_threshold=0.45,
        max_detections=300,
        class_agnostic_nms=False,
        fused=False,
    )
    values.update(overrides)
    request = PostprocessRequest(**values)

    return request


def _context(*, device="cpu", components=None):
    context = ExecutionContext(
        device_kind="gpu" if device.startswith("cuda") else "cpu",
        device=device,
        current_stream=(
            SimpleNamespace(device=torch.device(device))
            if device.startswith("cuda")
            else None
        ),
        runtime_components=components or {},
    )

    return context


def _original_postprocess(request):
    # The pre-integration model body is deliberately the oracle, independent of
    # either new stage's execution or output-conversion implementation.
    if request.fused:
        rows = post_process_nms_fused_model_output(
            output=request.model_results, conf_thresh=request.threshold
        )
    else:
        rows = run_nms_for_object_detection(
            output=request.model_results,
            conf_thresh=request.threshold,
            iou_thresh=request.iou_threshold,
            max_detections=request.max_detections,
            class_agnostic=request.class_agnostic_nms,
        )
    rows = rescale_detections(
        detections=rows, images_metadata=request.pre_processing_meta
    )
    results = [
        Detections(
            xyxy=row[:, :4].round().int(),
            class_id=row[:, 5].int(),
            confidence=row[:, 4],
        )
        for row in rows
    ]

    return results


def _assert_exact(actual, expected):
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        for name in ("xyxy", "class_id", "confidence"):
            left, right = getattr(got, name), getattr(want, name)
            assert left.dtype == right.dtype
            assert left.shape == right.shape
            assert left.device == right.device
            assert torch.equal(left, right), name


@pytest.mark.parametrize("agnostic", [False, True])
@pytest.mark.parametrize("max_detections", [-1, 0, 1, 300])
@pytest.mark.parametrize("threshold", [0.5, torch.tensor([0.5, 0.8])])
def test_base_preserves_original_numerics_and_input_ownership(
    agnostic, max_detections, threshold
):
    request = _request(
        class_agnostic_nms=agnostic,
        max_detections=max_detections,
        threshold=threshold,
    )
    raw_before = request.model_results.clone()
    metadata_before = copy.deepcopy(request.pre_processing_meta)
    stage = BasePostprocessor()

    result = stage.postprocess(request, context=_context())
    actual = result.wait()
    _assert_exact(actual, _original_postprocess(request))
    assert torch.equal(request.model_results, raw_before)
    assert request.pre_processing_meta == metadata_before
    snapshots = [
        (one.xyxy.clone(), one.class_id.clone(), one.confidence.clone())
        for one in actual
    ]

    changed = replace(request, model_results=torch.zeros_like(raw_before))
    stage.postprocess(changed, context=_context()).wait()
    for one, (xyxy, class_id, confidence) in zip(actual, snapshots):
        assert torch.equal(one.xyxy, xyxy)
        assert torch.equal(one.class_id, class_id)
        assert torch.equal(one.confidence, confidence)


def test_base_fused_export_keeps_inclusive_confidence_boundary():
    raw = torch.tensor(
        [[[1, 2, 3, 4, 0.5, 0], [2, 3, 4, 5, 0.49, 1]]],
        dtype=torch.float32,
    )
    request = _request(model_results=raw, pre_processing_meta=[_metadata()], fused=True)
    before = raw.clone()
    actual = BasePostprocessor().postprocess(request, context=_context()).wait()

    _assert_exact(actual, _original_postprocess(request))
    assert len(actual[0].confidence) == 1
    assert torch.equal(raw, before)


@pytest.mark.parametrize("requested", ["base", "auto"])
def test_default_and_auto_resolve_base_without_importing_kernels(requested):
    from inference_models.models.yolov8.optimization.registry import (
        create_postprocess_registry,
        select_postprocessor,
    )

    selection = select_postprocessor(
        registry=create_postprocess_registry(),
        requested_id=requested,
        request=_request(),
        context=_context(),
    )

    assert selection.effective_id == "base"
    assert selection.requested_id == requested
    assert not selection.used_fallback


@pytest.mark.parametrize(
    "context",
    [
        _context(),
        _context(device="cuda:0", components={"triton": False}),
        _context(device="cuda:0", components={"yolov8_trt": False}),
    ],
)
def test_segmented_static_fallback_is_inspectable(context):
    from inference_models.models.yolov8.optimization.registry import (
        create_postprocess_registry,
        select_postprocessor,
    )

    selection = select_postprocessor(
        registry=create_postprocess_registry(),
        requested_id="segmented",
        request=_request(),
        context=context,
    )

    assert selection.effective_id == "base"
    assert selection.used_fallback
    assert selection.fallback_reason
    assert selection.to_dict()["requested_id"] == "segmented"
    _assert_exact(
        selection.implementation.postprocess(_request(), context=_context()).wait(),
        _original_postprocess(_request()),
    )


def test_unknown_implementation_is_not_silently_accepted():
    from inference_models.models.yolov8.optimization.registry import (
        create_postprocess_registry,
        select_postprocessor,
    )

    with pytest.raises(
        ModelRuntimeError, match="unknown|Unknown|registered|implementation"
    ):
        select_postprocessor(
            registry=create_postprocess_registry(),
            requested_id="segmentd",
            request=_request(),
            context=_context(),
        )


class _CudaMetadataTensor(torch.Tensor):
    """Expose CUDA metadata only; never execute this tensor on a CUDA device."""

    @property
    def is_cuda(self):
        return True

    @property
    def device(self):
        return torch.device("cuda:0")


class _NoValueReadsTensor(torch.Tensor):
    def _forbidden_read(self, *args, **kwargs):
        raise AssertionError("Unsupported tensor options must not read values")

    __repr__ = _forbidden_read
    __float__ = _forbidden_read
    __bool__ = _forbidden_read
    __int__ = _forbidden_read
    item = _forbidden_read


@pytest.mark.parametrize("option", ["max_detections", "iou_threshold"])
def test_tensor_option_rejection_never_formats_or_reads_values(option):
    value = torch.tensor(1.0).as_subclass(_NoValueReadsTensor)
    request = _request(
        model_results=_raw().as_subclass(_CudaMetadataTensor), **{option: value}
    )

    compatibility = segmented.SegmentedPostprocessor().check_request_compatibility(
        request=request, context=_context(device="cuda:0")
    )

    assert not compatibility.supported
    assert option in compatibility.reason


@pytest.mark.parametrize(
    "overrides",
    [
        {"fused": True},
        {"model_results": torch.zeros((1, 6, 4))},
        {"model_results": torch.zeros((2, 6, 4), dtype=torch.float16)},
        {"model_results": torch.zeros((2, 6, 4), dtype=torch.float64)},
        {"model_results": torch.zeros((2, 6))},
        {"model_results": torch.zeros((2, 4, 4))},
        {"pre_processing_meta": [_metadata()]},
        {"max_detections": -1},
        {"threshold": torch.tensor([0.1, 0.2, 0.3])},
        {"iou_threshold": float("nan")},
        {"iou_threshold": -0.1},
        {"max_detections": 1.5},
        {"threshold": torch.tensor(0.5)},
    ],
)
def test_segmented_rejects_unqualified_requests_before_launch(overrides):
    request = _request(**overrides)
    request = replace(
        request,
        model_results=request.model_results.as_subclass(_CudaMetadataTensor),
    )
    compatibility = segmented.SegmentedPostprocessor().check_request_compatibility(
        request=request, context=_context(device="cuda:0")
    )

    assert not compatibility.supported
    assert compatibility.reason


def test_segmented_rejects_cpu_tensor_even_in_cuda_context():
    compatibility = segmented.SegmentedPostprocessor().check_request_compatibility(
        request=_request(), context=_context(device="cuda:0")
    )

    assert not compatibility.supported
    assert compatibility.reason


def test_qualified_tensor_metadata_passes_request_checks():
    request = _request(model_results=_raw().as_subclass(_CudaMetadataTensor))
    compatibility = segmented.SegmentedPostprocessor().check_request_compatibility(
        request=request, context=_context(device="cuda:0")
    )

    assert compatibility.supported, compatibility.reason


def test_segmented_rejects_nondefault_dtype_without_changing_global_state(monkeypatch):
    monkeypatch.setattr(torch, "get_default_dtype", lambda: torch.float64)
    request = _request(model_results=_raw().as_subclass(_CudaMetadataTensor))
    compatibility = segmented.SegmentedPostprocessor().check_request_compatibility(
        request=request, context=_context(device="cuda:0")
    )

    assert not compatibility.supported
    assert "dtype" in compatibility.reason
    assert torch.get_default_dtype() == torch.float64


def test_metadata_compatibility_does_not_read_tensor_scalars(monkeypatch):
    def _forbidden_read(*args, **kwargs):
        raise AssertionError("Compatibility must not synchronize tensor metadata")

    request = _request(
        model_results=_raw().as_subclass(_CudaMetadataTensor),
        pre_processing_meta=[
            _metadata()._replace(scale_width=torch.tensor(0.75)),
            _metadata(),
        ],
    )
    monkeypatch.setattr(torch.Tensor, "item", _forbidden_read)
    monkeypatch.setattr(torch.Tensor, "__float__", _forbidden_read)
    monkeypatch.setattr(torch.Tensor, "__bool__", _forbidden_read)

    compatibility = segmented.SegmentedPostprocessor().check_request_compatibility(
        request=request, context=_context(device="cuda:0")
    )

    assert not compatibility.supported
    assert compatibility.reason


def test_result_wait_completes_and_materializes_only_once():
    from inference_models.models.yolov8.optimization.contracts import PostprocessResult

    calls = []
    event = SimpleNamespace(synchronize=lambda: calls.append("wait"))

    def _materialize():
        calls.append("materialize")
        return ["owned-output"]

    result = PostprocessResult(ready_event=event, _materialize=_materialize)

    first = result.wait()
    second = result.wait()

    assert first is second
    assert calls == ["wait", "materialize"]


def test_result_wait_propagates_cuda_failure_without_materializing():
    from inference_models.models.yolov8.optimization.contracts import PostprocessResult

    def _failure():
        raise RuntimeError("CUDA illegal memory access")

    def _must_not_materialize():
        pytest.fail("Output must not be consumed after a failed CUDA wait")

    result = PostprocessResult(
        ready_event=SimpleNamespace(synchronize=_failure),
        _materialize=_must_not_materialize,
    )

    with pytest.raises(RuntimeError, match="CUDA illegal memory access"):
        result.wait()


class _CalibrationUnqualified(Exception):
    def __init__(self):
        super().__init__("no arithmetic variant matched")
        self.report = {"matching": []}


def _mock_calibration(monkeypatch, *, build):
    # Exercise real stage initialization/selection while replacing only the
    # optional GPU calibration dependency and CUDA context managers.
    monkeypatch.setitem(
        sys.modules,
        "inference_models.models.yolov8.optimization.calibration",
        SimpleNamespace(CalibrationError=_CalibrationUnqualified, build_kernels=build),
    )
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(segmented, "segmented_runtime_components", lambda: {})


def test_inconclusive_calibration_falls_back_once_with_inspectable_reason(monkeypatch):
    from inference_models.models.yolov8.optimization.registry import (
        create_postprocess_registry,
        select_postprocessor,
    )

    calls = []

    def _build(device):
        calls.append(device)
        raise _CalibrationUnqualified()

    _mock_calibration(monkeypatch, build=_build)
    stage = segmented.SegmentedPostprocessor()
    registry = create_postprocess_registry(segmented=stage)
    request = _request(model_results=_raw().as_subclass(_CudaMetadataTensor))

    for _ in range(2):
        selection = select_postprocessor(
            registry=registry,
            requested_id="segmented",
            request=request,
            context=_context(device="cuda:0"),
        )
        assert selection.effective_id == "base"
        assert "calibration" in selection.fallback_reason
    assert calls == [torch.device("cuda:0")]
    assert (
        stage.initialization_status(device="cuda:0").status == "calibration_unqualified"
    )


@pytest.mark.parametrize(
    "error",
    [RuntimeError("CUDA illegal memory access"), ImportError("compiler dependency")],
)
def test_initialization_failure_propagates_without_recording_fallback(
    monkeypatch, error
):
    from inference_models.models.yolov8.optimization.registry import (
        create_postprocess_registry,
        select_postprocessor,
    )

    def _build(device):
        raise error

    _mock_calibration(monkeypatch, build=_build)
    stage = segmented.SegmentedPostprocessor()
    registry = create_postprocess_registry(segmented=stage)
    request = _request(model_results=_raw().as_subclass(_CudaMetadataTensor))

    with pytest.raises(type(error), match=str(error)) as raised:
        select_postprocessor(
            registry=registry,
            requested_id="segmented",
            request=request,
            context=_context(device="cuda:0"),
        )

    assert raised.value is error
    assert stage.initialization_status(device="cuda:0") is None


def test_shared_stage_initializes_once_for_concurrent_requests(monkeypatch):
    calls = []
    barrier = threading.Barrier(2)
    kernels = SimpleNamespace(identity=lambda: {"backend": "test"})

    def _build(device):
        calls.append(device)
        return kernels, {"matching": ["test"]}

    _mock_calibration(monkeypatch, build=_build)
    stage = segmented.SegmentedPostprocessor()
    request = _request(model_results=_raw().as_subclass(_CudaMetadataTensor))

    def _check(_):
        barrier.wait(timeout=10)
        result = stage.check_runtime_compatibility(
            request=request, context=_context(device="cuda:0")
        )
        return result.supported

    with ThreadPoolExecutor(max_workers=2) as pool:
        supported = list(pool.map(_check, range(8)))

    assert all(supported)
    assert calls == [torch.device("cuda:0")]
    assert stage.initialization_status(device="cuda:0").kernels is kernels


def test_decoder_kernel_failure_propagates(monkeypatch):
    kernels = SimpleNamespace(identity=lambda: {"backend": "test"})
    stage = segmented.SegmentedPostprocessor(kernels=kernels)

    def _decode(*args, **kwargs):
        raise RuntimeError("kernel launch failed")

    monkeypatch.setattr(segmented, "decode_padded", _decode)

    with pytest.raises(RuntimeError, match="kernel launch failed"):
        stage.postprocess(_request(), context=_context())


def test_missing_triton_falls_back_before_kernel_import(monkeypatch):
    from inference_models.models.yolov8.optimization.registry import (
        create_postprocess_registry,
        select_postprocessor,
    )

    def _must_not_initialize(*args, **kwargs):
        pytest.fail("Missing Triton must be rejected before kernel initialization")

    monkeypatch.setitem(sys.modules, "triton", None)
    segmented.segmented_runtime_components.cache_clear()
    try:
        stage = segmented.SegmentedPostprocessor()
        monkeypatch.setattr(stage, "_initialize", _must_not_initialize)
        components = segmented.segmented_runtime_components()
        assert components["triton"] is False
        selection = select_postprocessor(
            registry=create_postprocess_registry(segmented=stage),
            requested_id="segmented",
            request=_request(model_results=_raw().as_subclass(_CudaMetadataTensor)),
            context=_context(device="cuda:0", components=components),
        )
    finally:
        segmented.segmented_runtime_components.cache_clear()

    assert selection.effective_id == "base"
    assert "triton" in selection.fallback_reason


def test_request_fallback_precedes_initialization(monkeypatch):
    from inference_models.models.yolov8.optimization.registry import (
        create_postprocess_registry,
        select_postprocessor,
    )

    def _must_not_initialize(*args, **kwargs):
        pytest.fail("Unsupported batch must fall back before initialization")

    stage = segmented.SegmentedPostprocessor()
    monkeypatch.setattr(segmented, "segmented_runtime_components", lambda: {})
    monkeypatch.setattr(stage, "_initialize", _must_not_initialize)
    request = _request(
        model_results=_raw()[:1].as_subclass(_CudaMetadataTensor),
        pre_processing_meta=[_metadata()],
    )

    selection = select_postprocessor(
        registry=create_postprocess_registry(segmented=stage),
        requested_id="segmented",
        request=request,
        context=_context(device="cuda:0"),
    )

    assert selection.effective_id == "base"
    assert "batch" in selection.fallback_reason


_MODEL_MODULE = "inference_models.models.yolov8.yolov8_object_detection_trt"
_TRT_DEPENDENCY_MODULES = (
    "inference_models.models.common.cuda",
    "inference_models.models.common.trt",
    _MODEL_MODULE,
)
_MISSING = object()


@pytest.fixture
def model_module(monkeypatch):
    """Import the TRT model with inert TensorRT and PyCUDA doubles.

    Mirrors the RF-DETR TRT unit-test fixture: only the import boundary is
    replaced, and both ``sys.modules`` and parent attributes are restored.
    """

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
    fake_pycuda.__spec__ = ModuleSpec("pycuda", loader=None, is_package=True)
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
        yield importlib.import_module(_MODEL_MODULE)
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


class _FakeStream:
    def __init__(self, device=None):
        self.device = torch.device(device)


class _FakeEvent:
    def __init__(self):
        self.recorded_on = None

    def record(self, stream):
        self.recorded_on = stream


class _RecordedCudaTensor(_CudaMetadataTensor):
    def record_stream(self, stream):
        self.recorded_streams.append(stream)


def _recorded_output(values):
    tensor = torch.tensor(values).as_subclass(_RecordedCudaTensor)
    tensor.recorded_streams = []

    return tensor


def _stream_wiring_model(model_module, monkeypatch, *, caller_stream):
    # Fake CUDA objects only observe which streams and events the model wires;
    # the real CUDA ordering is covered by the integration suite.
    monkeypatch.setattr(torch.cuda, "Stream", _FakeStream)
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device=None: caller_stream(),
    )
    calls = []

    def _postprocess(request, context):
        detections = [
            Detections(
                xyxy=_recorded_output([[1, 2, 3, 4]]),
                class_id=_recorded_output([0]),
                confidence=_recorded_output([0.5]),
            )
        ]
        calls.append(
            SimpleNamespace(
                request=request,
                context=context,
                thread=threading.get_ident(),
                detections=detections,
            )
        )
        result = SimpleNamespace(wait=lambda: detections, details={}, ready_event=None)

        return result

    selection = SimpleNamespace(
        requested_id="segmented",
        effective_id="segmented",
        fallback_reason=None,
        implementation=SimpleNamespace(postprocess=_postprocess),
        to_dict=lambda: {"effective_id": "segmented"},
    )
    monkeypatch.setattr(
        model_module, "select_postprocessor", lambda **kwargs: selection
    )
    model = model_module.YOLOv8ForObjectDetectionTRT(
        engine=None,
        input_name="images",
        output_name="output",
        class_names=["first", "second"],
        inference_config=SimpleNamespace(post_processing=SimpleNamespace(fused=False)),
        trt_config=None,
        device=torch.device("cuda:0"),
        cuda_context=None,
        execution_context=None,
        trt_cuda_graph_cache=None,
        postprocess_implementation="segmented",
    )

    return model, calls


def test_post_process_runs_on_owned_thread_stream_after_caller_event(
    model_module, monkeypatch
):
    caller_streams = threading.local()

    def _caller_stream():
        if not hasattr(caller_streams, "stream"):
            caller_streams.stream = _FakeStream("cuda:0")
        return caller_streams.stream

    model, calls = _stream_wiring_model(
        model_module, monkeypatch, caller_stream=_caller_stream
    )
    raw = _raw().as_subclass(_CudaMetadataTensor)
    metadata = [_metadata(), _metadata(offset=17)]
    barrier = threading.Barrier(2)

    def _run(_):
        barrier.wait(timeout=10)
        for _ in range(2):
            model.post_process(raw, metadata, confidence=0.5)
        return model._post_process_stream, _caller_stream()

    with ThreadPoolExecutor(max_workers=2) as pool:
        observed = list(pool.map(_run, range(2)))

    assert len(calls) == 4
    owned = {id(stream) for stream, _ in observed}
    callers = {id(stream) for _, stream in observed}
    assert len(owned) == 2
    assert owned.isdisjoint(callers)
    for model_stream, caller_stream in observed:
        assert model_stream.device == torch.device("cuda:0")
        thread_calls = [
            call for call in calls if call.context.current_stream is model_stream
        ]
        assert len(thread_calls) == 2
        assert len({call.thread for call in thread_calls}) == 1
        for call in thread_calls:
            assert call.context.device == "cuda:0"
            assert call.request.model_results is raw
            assert call.request.ready_event.recorded_on is caller_stream
            for detection in call.detections:
                for tensor in (
                    detection.xyxy,
                    detection.class_id,
                    detection.confidence,
                ):
                    assert tensor.recorded_streams == [caller_stream]


def test_post_process_inside_owned_stream_needs_no_cross_stream_wiring(
    model_module, monkeypatch
):
    model, calls = _stream_wiring_model(
        model_module, monkeypatch, caller_stream=lambda: model._post_process_stream
    )

    model.post_process(_raw().as_subclass(_CudaMetadataTensor), [_metadata()] * 2)

    assert calls[0].context.current_stream is model._post_process_stream
    assert calls[0].request.ready_event is None
    assert calls[0].detections[0].xyxy.recorded_streams == []
