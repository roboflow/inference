"""CUDA tests of the production model boundary with deterministic raw outputs.

No engine, weights, downloads, or inference-service dependencies are needed.
The oracle retains the model's original NMS/rescale/output conversion sequence.
"""

import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from inference_models import Detections
from inference_models.configuration import (
    INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_CLASS_AGNOSTIC_NMS,
    INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_CONFIDENCE,
    INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_IOU_THRESHOLD,
    INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_MAX_DETECTIONS,
)
from inference_models.entities import ImageDimensions
from inference_models.models.common.roboflow.model_packages import (
    PreProcessingMetadata,
    StaticCropOffset,
)
from inference_models.models.common.roboflow.post_processing import (
    ConfidenceFilter,
    post_process_nms_fused_model_output,
    rescale_detections,
    run_nms_for_object_detection,
)
from inference_models.models.optimization.contracts import ExecutionContext
from inference_models.models.yolov8.optimization.contracts import PostprocessRequest
from inference_models.models.yolov8.optimization.segmented import SegmentedPostprocessor
from inference_models.weights_providers.entities import RecommendedParameters

pytestmark = [
    pytest.mark.gpu_only,
    pytest.mark.trt_extras,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


@pytest.fixture(scope="module")
def model():
    pytest.importorskip("triton")
    pytest.importorskip("tensorrt")
    pytest.importorskip("pycuda.driver")
    from inference_models.models.yolov8.yolov8_object_detection_trt import (
        YOLOv8ForObjectDetectionTRT,
    )

    instance = YOLOv8ForObjectDetectionTRT(
        engine=None,
        input_name="images",
        output_name="output",
        class_names=["first", "second", "third"],
        inference_config=SimpleNamespace(post_processing=SimpleNamespace(fused=False)),
        trt_config=None,
        device=torch.device("cuda:0"),
        cuda_context=None,
        execution_context=None,
        trt_cuda_graph_cache=None,
        postprocess_implementation="segmented",
    )

    return instance


def _metadata(batch):
    results = []
    for index in range(batch):
        width, height = (1920, 1080) if index % 2 else (640, 480)
        size = ImageDimensions(width=width, height=height)
        results.append(
            PreProcessingMetadata(
                pad_left=7 * index,
                pad_top=11 * index,
                pad_right=0,
                pad_bottom=0,
                original_size=size,
                size_after_pre_processing=size,
                inference_size=ImageDimensions(width=640, height=640),
                scale_width=640 / width,
                scale_height=480 / height,
                static_crop_offset=StaticCropOffset(
                    offset_x=17 if index % 2 else 0,
                    offset_y=9 if index % 2 else 0,
                    crop_width=width,
                    crop_height=height,
                ),
            )
        )

    return results


def _raw(*, batch=3, anchors=96):
    generator = torch.Generator().manual_seed(20261016)
    raw = torch.zeros((batch, 7, anchors), dtype=torch.float32)
    raw[:, :2] = torch.rand((batch, 2, anchors), generator=generator) * 700 - 20
    raw[:, 2:4] = torch.rand((batch, 2, anchors), generator=generator) * 120 + 1
    raw[:, 4:] = torch.rand((batch, 3, anchors), generator=generator)
    # Score ties, duplicated boxes across classes, threshold equality and
    # adjacent float32 scores make sort/threshold changes observable.
    raw[:, :, 1] = raw[:, :, 0]
    raw[:, 4:, :4] = 0
    raw[:, 4, 0] = 0.75
    raw[:, 5, 1] = 0.75
    raw[:, 4, 2] = 0.5
    raw[:, 4, 3] = torch.nextafter(torch.tensor(0.5), torch.tensor(1.0))
    raw[-1, 4:] = 0
    if batch > 2:
        raw[1, 4:, anchors // 3 :] = 0
    result = raw.cuda()
    torch.cuda.current_stream().synchronize()

    return result


def _original(model, raw, metadata, **options):
    confidence = ConfidenceFilter(
        confidence=options.get("confidence", "default"),
        recommended_parameters=model.recommended_parameters,
        default_confidence=INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_CONFIDENCE,
    ).get_threshold(model.class_names)
    if model._inference_config.post_processing.fused:
        rows = post_process_nms_fused_model_output(raw, conf_thresh=confidence)
    else:
        rows = run_nms_for_object_detection(
            raw,
            conf_thresh=confidence,
            iou_thresh=options.get(
                "iou_threshold", INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_IOU_THRESHOLD
            ),
            max_detections=options.get(
                "max_detections",
                INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_MAX_DETECTIONS,
            ),
            class_agnostic=options.get(
                "class_agnostic_nms",
                INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_CLASS_AGNOSTIC_NMS,
            ),
        )
    rows = rescale_detections(rows, images_metadata=metadata)
    results = [
        Detections(
            xyxy=row[:, :4].round().int(),
            class_id=row[:, 5].int(),
            confidence=row[:, 4],
        )
        for row in rows
    ]
    torch.cuda.current_stream().synchronize()

    return results


def _assert_exact(actual, expected):
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        for field in ("xyxy", "class_id", "confidence"):
            left, right = getattr(got, field), getattr(want, field)
            assert left.dtype == right.dtype, field
            assert left.shape == right.shape, field
            assert left.device == right.device, field
            if left.is_floating_point():
                left = left.contiguous().view(torch.int32)
                right = right.contiguous().view(torch.int32)
            assert torch.equal(left, right), field


@pytest.mark.parametrize("confidence", [0.0, 0.5, 0.99])
@pytest.mark.parametrize("iou", [0.0, 0.45, 1.0])
@pytest.mark.parametrize("max_detections", [0, 1, 300])
@pytest.mark.parametrize("agnostic", [False, True])
def test_integrated_segmented_matches_original(
    model, confidence, iou, max_detections, agnostic
):
    raw = _raw()
    metadata = _metadata(raw.shape[0])
    before = raw.clone()
    metadata_before = copy.deepcopy(metadata)
    options = dict(
        confidence=confidence,
        iou_threshold=iou,
        max_detections=max_detections,
        class_agnostic_nms=agnostic,
    )
    expected = _original(model, raw, metadata, **options)

    actual = model.post_process(raw, metadata, **options)

    assert model.postprocess_selection["effective_id"] == "segmented"
    _assert_exact(actual, expected)
    assert torch.equal(raw.view(torch.int32), before.view(torch.int32))
    assert metadata == metadata_before


@pytest.mark.parametrize("iou", [0.3, 0.45, 0.5, 0.7])
def test_integrated_iou_boundary_pairs(model, iou):
    # Equal-height squares shifted so their mathematical IoU is exactly the
    # threshold; float32 conversion decides either side as in torchvision.
    raw = torch.zeros((2, 7, 96), dtype=torch.float32)
    for pair in range(48):
        width = 4.0 + pair
        x = pair * 13.17
        shift = width * (1 - iou) / (1 + iou)
        raw[:, :4, 2 * pair] = torch.tensor([x, x, width, width])
        raw[:, :4, 2 * pair + 1] = torch.tensor([x + shift, x, width, width])
        raw[:, 4, 2 * pair] = 0.9
        raw[:, 4, 2 * pair + 1] = 0.8
    raw = raw.cuda()
    metadata = _metadata(2)
    expected = _original(model, raw, metadata, iou_threshold=iou, confidence=0.25)

    actual = model.post_process(raw, metadata, iou_threshold=iou, confidence=0.25)

    assert model.postprocess_selection["effective_id"] == "segmented"
    _assert_exact(actual, expected)


@pytest.mark.parametrize("confidence", ["default", "best", 0.75])
def test_integrated_confidence_resolution(model, confidence, monkeypatch):
    monkeypatch.setattr(
        model,
        "recommended_parameters",
        RecommendedParameters(
            confidence=0.4,
            per_class_confidence={"first": 0.8, "second": 0.2, "third": 0.6},
        ),
    )
    raw = _raw()
    metadata = _metadata(3)
    expected = _original(model, raw, metadata, confidence=confidence)

    actual = model.post_process(raw, metadata, confidence=confidence)

    assert model.postprocess_selection["effective_id"] == "segmented"
    _assert_exact(actual, expected)


@pytest.mark.parametrize("case", ["batch_one", "float64", "negative_limit", "fused"])
def test_integrated_request_fallback_matches_original(model, case, monkeypatch):
    raw = _raw(batch=1 if case == "batch_one" else 3)
    if case == "float64":
        raw = raw.double()
    if case == "fused":
        monkeypatch.setattr(model._inference_config.post_processing, "fused", True)
        raw = torch.tensor(
            [[[1, 2, 3, 4, 0.5, 0], [2, 3, 4, 5, 0.49, 1]]],
            device="cuda",
            dtype=torch.float32,
        ).repeat(3, 1, 1)
    options = {"confidence": 0.5}
    if case == "negative_limit":
        options["max_detections"] = -1
    metadata = _metadata(raw.shape[0])
    expected = _original(model, raw, metadata, **options)

    actual = model.post_process(raw, metadata, **options)

    assert model.postprocess_selection["effective_id"] == "base"
    assert model.postprocess_selection["fallback_reason"]
    _assert_exact(actual, expected)


def test_integrated_results_ready_owned_and_concurrent(model, monkeypatch):
    from inference_models.models.yolov8.optimization import segmented

    raw = _raw()
    metadata = _metadata(3)
    before = raw.clone()
    metadata_before = copy.deepcopy(metadata)
    expected = _original(model, raw, metadata, confidence=0.5)
    local = threading.local()
    original_decode = segmented.decode_padded

    def _witness(*args, **kwargs):
        result = original_decode(*args, **kwargs)
        local.ready = torch.cuda.Event()
        local.ready.record(torch.cuda.current_stream())

        return result

    monkeypatch.setattr(segmented, "decode_padded", _witness)
    # Initialize calibration once; threads below exercise shared initialized
    # kernels, while each request must keep independent scratch/output storage.
    retained = model.post_process(raw, metadata, confidence=0.5)
    assert local.ready.query()
    barrier = threading.Barrier(2)

    def _run(index):
        torch.cuda.set_device(0)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for _ in range(5):
                barrier.wait(timeout=30)
                actual = model.post_process(raw, metadata, confidence=0.5)
                # Check before any comparison or synchronize can conceal a race.
                assert local.ready.query()
                assert model.postprocess_selection["effective_id"] == "segmented"
                _assert_exact(actual, expected)
                actual[0].xyxy.fill_(index + 1)
                actual[0].confidence.zero_()
        return stream.cuda_stream

    with ThreadPoolExecutor(max_workers=2) as pool:
        streams = list(pool.map(_run, range(2)))

    assert len(set(streams)) == 2
    _assert_exact(retained, expected)
    assert torch.equal(raw.view(torch.int32), before.view(torch.int32))
    assert metadata == metadata_before


def test_request_ready_event_orders_producer_and_consumer(model):
    source = _raw()
    metadata = _metadata(source.shape[0])
    expected = _original(model, source, metadata, confidence=0.5, iou_threshold=0.45)
    output = torch.zeros_like(source)
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    ready = torch.cuda.Event()
    context = ExecutionContext(
        device_kind="gpu", device="cuda:0", current_stream=consumer
    )
    request = PostprocessRequest(
        model_results=output,
        pre_processing_meta=metadata,
        threshold=0.5,
        iou_threshold=0.45,
        max_detections=300,
        class_agnostic_nms=False,
        ready_event=ready,
    )
    stage = SegmentedPostprocessor()
    # Finish calibration and allocation before the delayed producer starts so
    # initialization cannot accidentally mask a missing consumer event wait.
    assert stage.check_runtime_compatibility(request=request, context=context).supported
    torch.cuda.synchronize()

    with torch.cuda.stream(producer):
        torch.cuda._sleep(50_000_000)
        output.copy_(source)
        ready.record(producer)
    assert not ready.query(), "Producer must still be pending when decoding starts"

    result = stage.postprocess(request, context=context)
    actual = result.wait()

    assert ready.query()
    assert result.ready_event.query()
    _assert_exact(actual, expected)
    assert torch.equal(output, source)


def test_shared_stage_concurrent_cold_calibration_on_cuda(model, monkeypatch):
    from inference_models.models.yolov8.optimization import calibration

    raw = _raw()
    metadata = _metadata(raw.shape[0])
    expected = _original(model, raw, metadata, confidence=0.5, iou_threshold=0.45)
    request = PostprocessRequest(
        model_results=raw,
        pre_processing_meta=metadata,
        threshold=0.5,
        iou_threshold=0.45,
        max_detections=300,
        class_agnostic_nms=False,
    )
    stage = SegmentedPostprocessor()
    assert stage.initialization_status(device="cuda:0") is None
    original_build = calibration.build_kernels
    calibration_streams = []
    lock = threading.Lock()
    barrier = threading.Barrier(2)

    def _record_real_calibration(device):
        with lock:
            calibration_streams.append(torch.cuda.current_stream(device).cuda_stream)
        built = original_build(device)
        return built

    monkeypatch.setattr(calibration, "build_kernels", _record_real_calibration)

    def _run(_):
        torch.cuda.set_device(0)
        stream = torch.cuda.Stream()
        context = ExecutionContext(
            device_kind="gpu", device="cuda:0", current_stream=stream
        )
        barrier.wait(timeout=30)
        result = stage.postprocess(request, context=context)
        actual = result.wait()
        assert result.ready_event.query()
        _assert_exact(actual, expected)
        return stream.cuda_stream

    with ThreadPoolExecutor(max_workers=2) as pool:
        streams = list(pool.map(_run, range(2)))

    assert len(set(streams)) == 2
    assert len(calibration_streams) == 1
    assert calibration_streams[0] in streams
    assert stage.initialization_status(device="cuda:0").status == "calibrated"
