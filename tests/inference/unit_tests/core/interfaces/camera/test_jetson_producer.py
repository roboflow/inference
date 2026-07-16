from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from inference.core.interfaces.camera.jetson_producer import (
    JetsonVideoFrameProducer,
    build_gstreamer_pipeline,
    required_gstreamer_elements,
)


class _NativePipeline:
    def __init__(self, frame=None, factories=()) -> None:
        self.grab_calls = 0
        self.retrieve_calls = 0
        self.interrupt_calls = 0
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

    def interrupt(self) -> None:
        self.interrupt_calls += 1


def _native_producer(
    native_pipeline: _NativePipeline,
    *,
    output_tensor: bool = True,
    source_ref: str = "rtsp://camera.example.test/live",
) -> JetsonVideoFrameProducer:
    producer = JetsonVideoFrameProducer.__new__(JetsonVideoFrameProducer)
    producer._source_ref = source_ref
    producer._native_pipeline = native_pipeline
    producer._capture = None
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
    producer = _native_producer(native_pipeline)

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


def test_retrieving_preroll_clears_pending_grab_and_interrupts_native_wait() -> None:
    native_pipeline = _NativePipeline()
    producer = _native_producer(native_pipeline)

    producer.discover_source_properties()
    success, _ = producer.retrieve()
    producer.interrupt()

    assert success
    assert native_pipeline.retrieve_calls == 1
    assert native_pipeline.interrupt_calls == 1
    assert not producer.isOpened()


def test_numpy_retrieve_materializes_bgr_hwc_from_native_rgb_tensor() -> None:
    rgb_tensor = torch.tensor(
        [
            [[10, 20], [30, 40]],
            [[50, 60], [70, 80]],
            [[90, 100], [110, 120]],
        ],
        dtype=torch.uint8,
    )
    producer = _native_producer(_NativePipeline(frame=rgb_tensor), output_tensor=False)

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


def test_numpy_mode_validates_the_instantiated_hardware_decoder() -> None:
    native_pipeline = _NativePipeline(factories={"nvv4l2decoder"})
    producer = _native_producer(native_pipeline, output_tensor=False)
    producer._decoder_validated = False

    assert producer.grab()

    assert producer._decoder_validated
    assert "nvv4l2decoder" in native_pipeline.factory_queries


def test_raw_v4l2_source_does_not_require_a_decoder_element() -> None:
    producer = _native_producer(
        _NativePipeline(), output_tensor=False, source_ref="/dev/video0"
    )
    producer._decoder_validated = False

    assert producer.grab()

    assert producer._decoder_validated


def test_compressed_v4l2_source_rejects_a_software_decoder() -> None:
    producer = _native_producer(
        _NativePipeline(factories={"jpegdec"}),
        output_tensor=False,
        source_ref="/dev/video0",
    )
    producer._decoder_validated = False

    with pytest.raises(RuntimeError, match="software decoders: jpegdec"):
        producer.grab()


def test_rtsps_source_uses_live_rtsp_pipeline() -> None:
    pipeline = build_gstreamer_pipeline(
        "rtsps://camera.example.test:7441/live?token=secret"
    )

    # Explicit video-only chain: a codec-specific depayloader never links the
    # audio track, so an audio-muxing camera cannot poison the pipeline. The
    # decoder's NV12 NVMM output feeds the appsink directly (the bridge
    # converts NV12->RGB in CUDA) — no nvvidconv VIC pass, and the queue
    # buffers compressed data before the depayloader instead of leaking
    # decoded frames.
    assert pipeline.startswith(
        'rtspsrc location="rtsps://camera.example.test:7441/live?token=secret" '
        "protocols=tcp latency=200 ! queue ! "
    )
    assert (
        "rtph264depay ! h264parse ! nvv4l2decoder enable-max-performance=1" in pipeline
    )
    assert "uridecodebin" not in pipeline
    assert "nvvidconv" not in pipeline
    assert "video/x-raw(memory:NVMM),format=NV12" in pipeline
    assert "appsink name=rf_tensor_sink" in pipeline
    assert "max-buffers=4 drop=false sync=false" in pipeline
    assert "leaky" not in pipeline


def test_rtspt_source_is_recognised_as_rtsp() -> None:
    pipeline = build_gstreamer_pipeline("rtspt://camera.example.test/live")

    assert pipeline.startswith('rtspsrc location="rtspt://camera.example.test/live"')


def test_rtsp_codec_env_selects_h265_chain(monkeypatch) -> None:
    monkeypatch.setenv("ROBOFLOW_RTSP_VIDEO_CODEC", "h265")

    pipeline = build_gstreamer_pipeline("rtsp://camera.example.test/live")

    assert "rtph265depay ! h265parse ! nvv4l2decoder" in pipeline


def test_rtsp_codec_env_rejects_unsupported_codec(monkeypatch) -> None:
    monkeypatch.setenv("ROBOFLOW_RTSP_VIDEO_CODEC", "mjpeg")

    with pytest.raises(ValueError, match="Unsupported RTSP video codec"):
        build_gstreamer_pipeline("rtsp://camera.example.test/live")


def test_rtsp_transport_env_overrides_protocols_and_latency(monkeypatch) -> None:
    monkeypatch.setenv("ROBOFLOW_RTSP_PROTOCOLS", "tcp+udp")
    monkeypatch.setenv("ROBOFLOW_RTSP_LATENCY_MS", "1000")

    pipeline = build_gstreamer_pipeline("rtsp://camera.example.test/live")

    assert "protocols=tcp+udp latency=1000 ! " in pipeline


def test_tensor_rtsps_pipeline_keeps_nvmm_at_named_appsink() -> None:
    pipeline = build_gstreamer_pipeline(
        "rtsps://camera.example.test:7441/live",
        output_tensor=True,
    )

    assert "video/x-raw(memory:NVMM),format=NV12" in pipeline
    assert "appsink name=rf_tensor_sink" in pipeline
    assert "videoconvert" not in pipeline
    assert "video/x-raw,format=BGR" not in pipeline


def test_rtsps_source_requires_rtsp_and_nvidia_decode_elements() -> None:
    elements = set(required_gstreamer_elements("rtsps://camera.example.test/live"))

    assert {
        "appsink",
        "h264parse",
        "h265parse",
        "nvvidconv",
        "nvv4l2decoder",
        "rtph264depay",
        "rtph265depay",
        "rtspsrc",
    } <= elements
    # The explicit rtspsrc chain does not autoplug, so the uridecodebin stack
    # is no longer part of the RTSP requirements.
    assert "uridecodebin" not in elements
    assert "videoconvert" not in elements


def test_tensor_source_requires_hardware_jpeg_without_cpu_converter() -> None:
    elements = set(required_gstreamer_elements("/tmp/sample.mp4", output_tensor=True))

    assert "nvjpegdec" in elements
    assert "nvv4l2decoder" in elements
    assert "videoconvert" not in elements


def test_file_source_uses_non_dropping_sink_and_matching_demuxer() -> None:
    pipeline = build_gstreamer_pipeline("/tmp/sample.mp4")
    elements = set(required_gstreamer_elements("/tmp/sample.mp4"))

    assert pipeline.startswith(
        f'uridecodebin uri="{Path("/tmp/sample.mp4").resolve().as_uri()}"'
    )
    assert "max-buffers=4 drop=false sync=false" in pipeline
    assert "qtdemux" in elements


def test_csi_source_uses_nvargus_camera() -> None:
    pipeline = build_gstreamer_pipeline("csi://2")

    assert pipeline.startswith("nvarguscamerasrc sensor-id=2")
    assert "video/x-raw(memory:NVMM),format=NV12" in pipeline


def test_v4l2_device_path_uses_live_sink() -> None:
    pipeline = build_gstreamer_pipeline("/dev/video3")

    assert pipeline.startswith('v4l2src device="/dev/video3" ! decodebin !')
    assert "max-buffers=1 drop=true sync=false" in pipeline


def test_v4l2_decodebin_can_negotiate_raw_mjpeg_and_h264_sources() -> None:
    pipeline = build_gstreamer_pipeline("/dev/video3", output_tensor=True)
    elements = set(required_gstreamer_elements("/dev/video3", output_tensor=True))

    assert 'v4l2src device="/dev/video3" ! decodebin ! queue' in pipeline
    assert "video/x-raw(memory:NVMM),format=RGBA" in pipeline
    assert {
        "decodebin",
        "h264parse",
        "jpegparse",
        "nvjpegdec",
        "nvv4l2decoder",
        "v4l2src",
    } <= elements
