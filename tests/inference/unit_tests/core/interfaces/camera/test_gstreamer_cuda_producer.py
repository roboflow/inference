from types import SimpleNamespace

import numpy as np
import pytest
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

    def stats(self):
        return {"frames": self.grab_calls}


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
    producer._last_grabbed_at_ns = None
    producer._grab_gap_count = 0
    producer._grab_gap_sum_ns = 0
    producer._grab_gap_max_ns = 0
    producer._grab_gap_under_half_period = 0
    producer._grab_gap_over_one_and_half_period = 0
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


def test_live_appsink_sync_is_staging_selectable(monkeypatch) -> None:
    monkeypatch.setenv("ROBOFLOW_GSTREAMER_CUDA_APPSINK_SYNC", "true")

    pipeline = build_gstreamer_cuda_pipeline("rtsp://camera.example.test/live")

    assert (
        "appsink name=rf_tensor_sink max-buffers=1 drop=true sync=true" in pipeline
    )


def test_live_appsink_sync_preserves_low_latency_default(monkeypatch) -> None:
    monkeypatch.delenv("ROBOFLOW_GSTREAMER_CUDA_APPSINK_SYNC", raising=False)

    pipeline = build_gstreamer_cuda_pipeline("rtsp://camera.example.test/live")

    assert (
        "appsink name=rf_tensor_sink max-buffers=1 drop=true sync=false" in pipeline
    )


def test_grab_cadence_stats_count_short_and_long_gaps(monkeypatch) -> None:
    timestamps = iter((0, 10_000_000, 80_000_000))
    monkeypatch.setattr(
        "inference.core.interfaces.camera.gstreamer_cuda_producer.time.monotonic_ns",
        lambda: next(timestamps),
    )
    producer = _producer(_NativePipeline())

    producer.discover_source_properties()
    assert producer.grab()  # consume the preroll without recording a duplicate
    assert producer.grab()
    assert producer.grab()

    assert producer.tensor_bridge_stats == {
        "frames": 3,
        "grab_gap_count": 2,
        "grab_gap_under_half_period": 1,
        "grab_gap_over_one_and_half_period": 1,
        "grab_gap_max_us": 70_000,
        "grab_gap_mean_us": 40_000,
    }


def test_rtsps_element_contract_includes_tls_capable_rtsp_source() -> None:
    elements = set(required_gstreamer_cuda_elements("rtsps://camera.example.test/live"))

    assert {
        "cudaconvertscale",
        "decodebin",
        "h264parse",
        "h265parse",
        "rtph264depay",
        "rtph265depay",
        "rtspsrc",
    }.issubset(elements)
    # The explicit rtspsrc chain does not autoplug the source, so the
    # uridecodebin stack is no longer part of the RTSP requirements.
    assert "uridecodebin" not in elements


def test_rtsp_source_uses_explicit_video_only_pipeline() -> None:
    pipeline = build_gstreamer_cuda_pipeline(
        "rtsps://camera.example.test:7441/live?token=secret", device_id=1
    )

    # Explicit video-only chain: the media=video caps filter pins rtspsrc's
    # delayed link to the video stream (first-pad-wins would otherwise let an
    # audio-first camera wire audio into the chain), so an audio-muxing camera
    # cannot poison the pipeline with missing-decoder or not-linked bus errors.
    assert pipeline.startswith(
        'rtspsrc location="rtsps://camera.example.test:7441/live?token=secret" '
        "protocols=tcp latency=200 ! application/x-rtp,media=video ! queue ! "
    )
    assert "rtph264depay ! h264parse ! " in pipeline
    # decodebin is caps-pinned to CUDAMemory so only an NVIDIA decoder can
    # terminate autoplugging — software decoders cannot satisfy the caps.
    assert 'decodebin caps="video/x-raw(memory:CUDAMemory)"' in pipeline
    assert "uridecodebin" not in pipeline
    assert "cudaconvertscale cuda-device-id=1" in pipeline
    assert "appsink name=rf_tensor_sink" in pipeline


def test_rtspt_source_is_recognised_as_rtsp() -> None:
    pipeline = build_gstreamer_cuda_pipeline("rtspt://camera.example.test/live")

    assert pipeline.startswith(
        'rtspsrc location="rtspt://camera.example.test/live"'
    )


def test_rtsp_codec_env_selects_h265_chain(monkeypatch) -> None:
    monkeypatch.setenv("ROBOFLOW_RTSP_VIDEO_CODEC", "h265")

    pipeline = build_gstreamer_cuda_pipeline("rtsp://camera.example.test/live")

    assert "rtph265depay ! h265parse ! " in pipeline


def test_rtsp_codec_env_rejects_unsupported_codec(monkeypatch) -> None:
    monkeypatch.setenv("ROBOFLOW_RTSP_VIDEO_CODEC", "mjpeg")

    with pytest.raises(ValueError, match="Unsupported RTSP video codec"):
        build_gstreamer_cuda_pipeline("rtsp://camera.example.test/live")


def test_rtsp_transport_env_overrides_protocols_and_latency(monkeypatch) -> None:
    monkeypatch.setenv("ROBOFLOW_RTSP_PROTOCOLS", "tcp+udp")
    monkeypatch.setenv("ROBOFLOW_RTSP_LATENCY_MS", "1000")

    pipeline = build_gstreamer_cuda_pipeline("rtsp://camera.example.test/live")

    assert "protocols=tcp+udp latency=1000 ! " in pipeline


def test_file_source_keeps_uridecodebin_pipeline() -> None:
    pipeline = build_gstreamer_cuda_pipeline("sample.mp4", device_id=0)

    assert pipeline.startswith("uridecodebin uri=")
    assert 'caps="video/x-raw(memory:CUDAMemory)"' in pipeline
    assert "rtspsrc" not in pipeline


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
