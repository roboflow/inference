from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from streamvision.camera.jetson_producer import (
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

    # Explicit video-only chain: the media=video caps filter pins rtspsrc's
    # delayed link to the video stream (first-pad-wins would otherwise let an
    # audio-first camera wire audio into the chain), so an audio-muxing camera
    # cannot poison the pipeline. The decoder's NV12 NVMM output feeds the
    # appsink directly (the bridge converts NV12->RGB in CUDA) — no nvvidconv
    # VIC pass, and the queue buffers compressed data before the depayloader
    # instead of leaking decoded frames.
    assert pipeline.startswith(
        'rtspsrc location="rtsps://camera.example.test:7441/live?token=secret" '
        "protocols=tcp latency=200 ! application/x-rtp,media=video ! queue ! "
    )
    assert "application/x-rtp,media=video" in pipeline
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


@pytest.mark.parametrize("video", [0, "/dev/video0"])
@pytest.mark.parametrize("fourcc", ["MJPG", "JPEG", 1196444237, "1196444237.0"])
def test_v4l2_requested_mjpeg_uses_camera_caps_and_hardware_decode(video, fourcc):
    pipeline = build_gstreamer_pipeline(
        video,
        output_tensor=True,
        video_source_properties={
            "fourcc": fourcc,
            "frame_width": 1920,
            "frame_height": 1080,
            "fps": 30,
        },
    )

    assert pipeline.startswith(
        'v4l2src device="/dev/video0" ! '
        "image/jpeg,width=1920,height=1080,framerate=30/1 ! "
        "jpegparse ! nvv4l2decoder mjpeg=1 ! queue"
    )
    assert "decodebin" not in pipeline
    assert "video/x-raw(memory:NVMM),format=RGBA ! appsink" in pipeline


@pytest.mark.parametrize(
    "fourcc,chain",
    [
        ("YUYV", "video/x-raw,format=YUY2"),
        ("NV12", "video/x-raw,format=NV12"),
        ("H264", "video/x-h264 ! h264parse ! nvv4l2decoder"),
        ("HEVC", "video/x-h265 ! h265parse ! nvv4l2decoder"),
    ],
)
def test_v4l2_explicit_formats_do_not_autoplug_a_different_decoder(fourcc, chain):
    pipeline = build_gstreamer_pipeline(
        "/dev/video1", video_source_properties={"fourcc": fourcc}
    )

    assert f'v4l2src device="/dev/video1" ! {chain} ! queue' in pipeline
    assert "decodebin" not in pipeline


def test_v4l2_unspecified_format_keeps_size_and_fractional_rate_on_each_media_type():
    pipeline = build_gstreamer_pipeline(
        2,
        video_source_properties={"FRAME_WIDTH": "1280", "FPS": 30000 / 1001},
    )

    caps_filter = pipeline.split(" ! ")[1]
    assert caps_filter.startswith('capsfilter caps="')
    assert caps_filter.endswith('"')
    caps = caps_filter.removeprefix('capsfilter caps="').removesuffix('"').split(";")
    assert set(caps) == {
        f"{media},width=1280,framerate=30000/1001"
        for media in ("video/x-raw", "image/jpeg", "video/x-h264", "video/x-h265")
    }
    assert " ! decodebin ! queue" in pipeline
    assert "height=" not in pipeline


@pytest.mark.parametrize(
    "properties",
    [{"fps": 30}, {"frame_width": 1280, "frame_height": 720}],
)
def test_v4l2_unspecified_format_builds_real_gstreamer_pipeline(properties):
    gi = pytest.importorskip("gi", reason="GStreamer Python bindings unavailable")
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    if not all(Gst.ElementFactory.find(name) for name in ("v4l2src", "decodebin")):
        pytest.skip("GStreamer camera/decode plugins unavailable")

    pipeline = build_gstreamer_pipeline(0, video_source_properties=properties)
    # Parse the real capture/decode chain without Jetson-only conversion elements.
    capture = pipeline.split(" ! queue", 1)[0]
    parsed = Gst.parse_launch(f"{capture} ! fakesink")
    try:
        filters = parsed.iterate_recurse()
        camera_caps = None
        while True:
            status, element = filters.next()
            if status != Gst.IteratorResult.OK:
                break

            if element.get_factory().get_name() == "capsfilter":
                camera_caps = element.get_property("caps")
                break

        assert camera_caps is not None
        assert camera_caps.get_size() == 4
        assert {camera_caps.get_structure(i).get_name() for i in range(4)} == {
            "video/x-raw",
            "image/jpeg",
            "video/x-h264",
            "video/x-h265",
        }
        expected = Gst.Caps.from_string(
            "video/x-raw,framerate=30/1"
            if "fps" in properties
            else "video/x-raw,width=1280,height=720"
        )
        assert camera_caps.get_structure(0).is_equal(expected.get_structure(0))
    finally:
        parsed.set_state(Gst.State.NULL)


@pytest.mark.parametrize("fourcc", [None, "MJPG"])
@pytest.mark.parametrize(
    "fps,rate",
    [
        (23.976, "24000/1001"),
        (29.97, "30000/1001"),
        (59.94, "60000/1001"),
        (30000 / 1001, "30000/1001"),
        (30, "30/1"),
        (29.5, "59/2"),
    ],
)
def test_v4l2_requested_rate_matches_advertised_camera_fraction(fourcc, fps, rate):
    properties = {"fps": fps}
    if fourcc is not None:
        properties["fourcc"] = fourcc

    pipeline = build_gstreamer_pipeline(0, video_source_properties=properties)

    assert f"framerate={rate}" in pipeline


@pytest.mark.parametrize(
    "properties",
    [
        {"fourcc": "ZZZZ"},
        {"fourcc": "mjpg"},
        {"fourcc": "RGB3"},
        {"fourcc": "BGR3"},
        {"fourcc": "MJPG ! fakesink"},
        {"fourcc": "MJ PG"},
        {"fourcc": 'MJ"G'},
        {"fourcc": "MJ'G"},
        {"fourcc": "MJ!G"},
        {"fourcc": "MJ G"},
        {"frame_width": -1},
        {"frame_height": 1080.5},
        {"fps": float("nan")},
        {"fps": 0},
        {"frame_width": "1920 ! fakesink"},
    ],
)
def test_v4l2_invalid_requests_fail_instead_of_silently_negotiating(properties):
    with pytest.raises(ValueError):
        build_gstreamer_pipeline(0, video_source_properties=properties)


def _camera_producer_with_fake_bridge(monkeypatch):
    import streamvision.camera.jetson_producer as module
    import streamvision.camera.jetson_tensor_bridge as bridge

    opened = []

    class CameraPipeline(_NativePipeline):
        def __init__(self, pipeline, **kwargs):
            super().__init__(factories={"nvv4l2decoder"})
            self.pipeline = pipeline
            self.closed = False
            opened.append(self)

        def close(self):
            self.closed = True

        def stats(self):
            if self.closed:
                raise RuntimeError("Jetson tensor pipeline is closed")

            return {"frames": self.grab_calls}

    monkeypatch.setattr(
        module, "probe_gstreamer_elements", lambda *a, **k: (True, "ok")
    )
    monkeypatch.setattr(bridge, "jetson_tensor_bridge_available", lambda: (True, "ok"))
    monkeypatch.setattr(bridge, "NativeJetsonTensorPipeline", CameraPipeline)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    return JetsonVideoFrameProducer(0, tensor_device="cuda:0"), opened


def test_camera_is_opened_only_after_requested_properties_are_applied(monkeypatch):
    producer, opened = _camera_producer_with_fake_bridge(monkeypatch)
    assert not opened
    properties = {"fourcc": "MJPG", "frame_width": 1920, "fps": 30}

    producer.initialize_source_properties(properties)
    producer.discover_source_properties()
    assert producer.grab()
    assert producer.retrieve()[0]
    assert producer.grab()

    assert len(opened) == 1
    assert "image/jpeg,width=1920,framerate=30/1" in opened[0].pipeline
    assert opened[0].grab_calls == 2
    producer.release()
    assert opened[0].closed


@pytest.mark.parametrize("start_capture", [False, True])
def test_camera_stats_are_safe_before_deferred_open_and_after_release(
    monkeypatch, start_capture
):
    producer, opened = _camera_producer_with_fake_bridge(monkeypatch)

    assert producer.tensor_bridge_stats == {}
    assert not opened
    if start_capture:
        assert producer.grab()
        assert producer.tensor_bridge_stats == {"frames": 1}

    producer.release()
    assert producer.tensor_bridge_stats == {}
    assert all(pipeline.closed for pipeline in opened)


def test_direct_camera_grab_without_properties_still_opens_default_pipeline(
    monkeypatch,
):
    producer, opened = _camera_producer_with_fake_bridge(monkeypatch)

    assert producer.grab()
    assert producer.retrieve()[0]
    assert len(opened) == 1
    assert 'v4l2src device="/dev/video0" ! decodebin ! queue' in opened[0].pipeline
    producer.release()


def test_camera_can_be_released_before_deferred_open(monkeypatch):
    producer, opened = _camera_producer_with_fake_bridge(monkeypatch)

    producer.release()

    assert not producer.isOpened()
    assert not producer.grab()
    assert not opened
