import builtins
import sys
from unittest.mock import patch

from streamvision.camera.discoverability import (
    DGPU,
    GSTREAMER_CUDA,
    JETSON,
    ProducerAvailability,
    _resolution_order,
    available_producers,
    build_hw_producer,
    check_gstreamer_cuda,
    check_jetson_gstreamer,
    check_pynvvideocodec,
)


@patch("platform.system", return_value="Linux")
@patch("platform.machine", return_value="aarch64")
def test_jetson_routes_gstreamer_for_streams_and_files(
    machine_mock,
    system_mock,
) -> None:
    # Streams / cameras -> Jetson HW GStreamer (latest-wins slot).
    assert _resolution_order(prefer=None, video="rtsp://cam/stream") == [JETSON]
    # Local files -> the same producer via the bridge's lossless handoff
    # (bridge v6+); cv2 remains only the construction-failure fallback.
    assert _resolution_order(prefer=None, video="sample.mp4") == [JETSON]


@patch("platform.system", return_value="Linux")
@patch("platform.machine", return_value="x86_64")
def test_dgpu_routes_gstreamer_for_streams_and_pynvdec_for_files(
    machine_mock,
    system_mock,
) -> None:
    # Streams -> GStreamer CUDA.
    assert _resolution_order(prefer=None, video="rtsp://cam/stream") == [GSTREAMER_CUDA]
    # Local files -> PyNvVideoCodec (dGPU).
    assert _resolution_order(prefer=None, video="sample.mp4") == [DGPU]


def test_explicit_backend_preference_takes_priority() -> None:
    assert _resolution_order(prefer=GSTREAMER_CUDA) == [
        GSTREAMER_CUDA,
        JETSON,
        DGPU,
    ]


def test_generic_file_decoders_reject_v4l2_device_paths() -> None:
    availability = check_gstreamer_cuda("/dev/video0")

    assert not availability.available


@patch(
    "streamvision.camera.discoverability.check_pynvvideocodec",
    return_value=ProducerAvailability(DGPU, True, "ok"),
)
@patch(
    "streamvision.camera.discoverability.check_jetson_gstreamer",
    return_value=ProducerAvailability(JETSON, True, "ok"),
)
@patch(
    "streamvision.camera.discoverability.check_gstreamer_cuda",
    return_value=ProducerAvailability(GSTREAMER_CUDA, True, "ok"),
)
def test_generic_cuda_backend_remains_available_for_numpy_consumers(
    check_gstreamer_cuda_mock,
    check_jetson_gstreamer_mock,
    check_pynvvideocodec_mock,
) -> None:
    availability = available_producers(video="sample.mp4", require_cuda_tensor=False)

    assert availability[GSTREAMER_CUDA].available
    check_gstreamer_cuda_mock.assert_called_once_with("sample.mp4")
    check_jetson_gstreamer_mock.assert_called_once_with(
        video="sample.mp4", require_cuda_tensor=False
    )
    check_pynvvideocodec_mock.assert_not_called()


@patch("torch.cuda.is_available", return_value=True)
@patch(
    "streamvision.camera.jetson_tensor_bridge.jetson_tensor_bridge_available",
    return_value=(True, "ok"),
)
@patch(
    "streamvision.camera.jetson_producer.probe_gstreamer_elements",
    return_value=(True, "ok"),
)
def test_jetson_numpy_probe_requires_the_native_cuda_bridge(
    probe_gstreamer_elements_mock,
    bridge_available_mock,
    cuda_available_mock,
) -> None:
    availability = check_jetson_gstreamer(video="sample.mp4", require_cuda_tensor=False)

    assert availability.available
    bridge_available_mock.assert_called_once_with()
    cuda_available_mock.assert_called_once_with()
    required_elements = probe_gstreamer_elements_mock.call_args.args[0]
    assert "nvvidconv" in required_elements
    assert "videoconvert" not in required_elements


@patch("platform.system", return_value="Linux")
@patch("platform.machine", return_value="aarch64")
@patch(
    "streamvision.camera.jetson_producer.JetsonVideoFrameProducer",
    side_effect=RuntimeError("preroll failed"),
)
@patch(
    "streamvision.camera.discoverability.available_producers",
    return_value={
        GSTREAMER_CUDA: ProducerAvailability(GSTREAMER_CUDA, False, "unavailable"),
        JETSON: ProducerAvailability(JETSON, True, "ok"),
        DGPU: ProducerAvailability(DGPU, False, "unavailable"),
    },
)
def test_factory_logs_construction_failures_before_falling_back(
    available_producers_mock,
    producer_class_mock,
    machine_mock,
    system_mock,
    streamvision_caplog,
) -> None:
    # given - the probe passes but the producer constructor raises; without a
    # log line the caller-side fallback to cv2 is undiagnosable
    caplog = streamvision_caplog
    with caplog.at_level("WARNING"):
        producer = build_hw_producer("sample.mp4")

    assert producer is None
    assert "Constructing the 'jetson' hardware decoder" in caplog.text
    assert "preroll failed" in caplog.text


@patch("streamvision.camera.gstreamer_cuda_producer.GstreamerCudaVideoFrameProducer")
@patch(
    "streamvision.camera.discoverability.available_producers",
    return_value={
        GSTREAMER_CUDA: ProducerAvailability(GSTREAMER_CUDA, True, "ok"),
        JETSON: ProducerAvailability(JETSON, False, "unavailable"),
        DGPU: ProducerAvailability(DGPU, False, "unavailable"),
    },
)
def test_factory_requests_numpy_from_the_native_generic_cuda_producer(
    available_producers_mock,
    producer_class_mock,
) -> None:
    producer = build_hw_producer(
        "sample.mp4", prefer=GSTREAMER_CUDA, output_tensor=False
    )

    assert producer is producer_class_mock.return_value
    producer_class_mock.assert_called_once_with("sample.mp4", output_tensor=False)
    available_producers_mock.assert_called_once_with(
        video="sample.mp4", require_cuda_tensor=False
    )


def test_check_pynvvideocodec_reports_missing_package_with_install_hint(
    monkeypatch,
) -> None:
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", None)

    availability = check_pynvvideocodec()

    assert availability.available is False
    assert "streamvision[nvdec]" in availability.reason


def test_check_pynvvideocodec_keeps_old_reason_for_non_missing_import_failures(
    monkeypatch,
) -> None:
    real_import = builtins.__import__

    def _raise_for_pynvvideocodec(name, *args, **kwargs):
        if name == "PyNvVideoCodec":
            raise OSError("libnvidia-encode.so.1: cannot open shared object file")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _raise_for_pynvvideocodec)

    availability = check_pynvvideocodec()

    assert availability.reason.startswith("PyNvVideoCodec import failed:")
    assert "streamvision[nvdec]" not in availability.reason


def test_build_hw_producer_returns_none_without_raising_when_pynvvideocodec_is_missing(
    monkeypatch,
) -> None:
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", None)
    monkeypatch.setattr(
        "streamvision.camera.discoverability.check_gstreamer_cuda",
        lambda video=None: ProducerAvailability(GSTREAMER_CUDA, False, "unavailable"),
    )
    monkeypatch.setattr(
        "streamvision.camera.discoverability.check_jetson_gstreamer",
        lambda video=None, require_cuda_tensor=True: ProducerAvailability(
            JETSON, False, "unavailable"
        ),
    )

    producer = build_hw_producer("video.mp4", prefer="dgpu", output_tensor=True)

    assert producer is None
