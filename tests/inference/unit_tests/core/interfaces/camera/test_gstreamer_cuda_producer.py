from types import SimpleNamespace

import numpy as np
import torch

from inference.core.interfaces.camera.gstreamer_cuda_producer import (
    GstreamerCudaVideoFrameProducer,
    build_gstreamer_cuda_pipeline,
    required_gstreamer_cuda_elements,
)


class _NativePipeline:
    def __init__(self, frame=None, factories=()) -> None:
        self.grab_calls = 0
        self.retrieve_calls = 0
        self.frame = frame if frame is not None else object()
        self.factories = set(factories)
        self.factory_queries = []
        self.last_grab_timeout_ns = None

    def grab(self, timeout_ns=None) -> bool:
        self.grab_calls += 1
        self.last_grab_timeout_ns = timeout_ns
        return True

    def retrieve(self):
        self.retrieve_calls += 1
        return self.frame

    def has_factory(self, factory: str) -> bool:
        self.factory_queries.append(factory)
        return factory in self.factories

    def frame_info(self):
        return SimpleNamespace(
            width=320,
            height=180,
            fps_numerator=30,
            fps_denominator=1,
            duration_ns=0,
        )


def _producer(
    native_pipeline: _NativePipeline, *, output_tensor: bool = True
) -> GstreamerCudaVideoFrameProducer:
    producer = GstreamerCudaVideoFrameProducer.__new__(GstreamerCudaVideoFrameProducer)
    producer._source_ref = "rtsp://camera.example.test/live"
    producer._native_pipeline = native_pipeline
    producer._output_tensor = output_tensor
    producer._decoder_validated = True
    producer._prerolled_frame_pending = False
    producer._cached_source_properties = None
    producer._grab_timeout_ns = 5_000_000_000
    producer._closed = False
    producer._eos = False
    return producer


def test_metadata_preroll_is_consumable_once_and_later_grabs_advance() -> None:
    native_pipeline = _NativePipeline()
    producer = _producer(native_pipeline)

    first_properties = producer.discover_source_properties()
    second_properties = producer.discover_source_properties()

    assert first_properties is second_properties
    assert native_pipeline.grab_calls == 1
    assert producer.grab()
    assert native_pipeline.grab_calls == 1
    assert producer.grab()
    assert native_pipeline.grab_calls == 2
    # The producer must hand the native pull a finite deadline so a stalled
    # source raises instead of blocking VideoSource forever (FQ-1).
    assert native_pipeline.last_grab_timeout_ns == producer._grab_timeout_ns


def test_retrieving_preroll_requires_next_grab_to_advance_native_pipeline() -> None:
    native_pipeline = _NativePipeline()
    producer = _producer(native_pipeline)

    producer.discover_source_properties()
    success, _ = producer.retrieve()

    assert success
    assert native_pipeline.retrieve_calls == 1
    assert producer.grab()
    assert native_pipeline.grab_calls == 2


def test_numpy_retrieve_materializes_bgr_hwc_from_native_rgb_tensor() -> None:
    rgb_tensor = torch.tensor(
        [
            [[10, 20], [30, 40]],
            [[50, 60], [70, 80]],
            [[90, 100], [110, 120]],
        ],
        dtype=torch.uint8,
    )
    producer = _producer(_NativePipeline(frame=rgb_tensor), output_tensor=False)

    success, image = producer.retrieve()

    assert success
    assert isinstance(image, np.ndarray)
    assert image.dtype == np.uint8
    assert image.flags.c_contiguous
    np.testing.assert_array_equal(
        image,
        np.array(
            [
                [[90, 50, 10], [100, 60, 20]],
                [[110, 70, 30], [120, 80, 40]],
            ],
            dtype=np.uint8,
        ),
    )


def test_numpy_mode_validates_cuda_conversion_and_hardware_decoder() -> None:
    native_pipeline = _NativePipeline(factories={"cudaconvertscale", "nvh264dec"})
    producer = _producer(native_pipeline, output_tensor=False)
    producer._decoder_validated = False

    assert producer.grab()

    assert producer._decoder_validated
    assert "cudaconvertscale" in native_pipeline.factory_queries
    assert "nvh264dec" in native_pipeline.factory_queries


def test_tensor_pipeline_keeps_frames_in_cuda_memory() -> None:
    pipeline = build_gstreamer_cuda_pipeline(
        "rtsps://camera.example.test/live", device_id=2
    )

    assert 'caps="video/x-raw(memory:CUDAMemory)"' in pipeline
    assert "cudaconvertscale cuda-device-id=2" in pipeline
    assert "video/x-raw(memory:CUDAMemory),format=RGBP" in pipeline
    assert "appsink name=rf_tensor_sink" in pipeline
    assert "cudaupload" not in pipeline
    assert "cudadownload" not in pipeline
    assert "videoconvert" not in pipeline


def test_rtsps_element_contract_includes_tls_capable_rtsp_source() -> None:
    elements = set(required_gstreamer_cuda_elements("rtsps://camera.example.test/live"))

    assert {
        "cudaconvertscale",
        "h264parse",
        "h265parse",
        "rtph264depay",
        "rtph265depay",
        "rtspsrc",
        "uridecodebin",
    }.issubset(elements)


def test_local_mp4_contract_includes_demuxer() -> None:
    elements = set(required_gstreamer_cuda_elements("sample.mp4"))

    assert "qtdemux" in elements


def test_v4l2_device_is_not_treated_as_a_regular_file() -> None:
    try:
        GstreamerCudaVideoFrameProducer("/dev/video0")
    except TypeError:
        pass
    else:
        raise AssertionError("V4L2 device path must not use the URI producer")
