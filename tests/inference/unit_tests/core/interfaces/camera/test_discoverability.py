from unittest.mock import patch

from inference.core.interfaces.camera.discoverability import (
    DGPU,
    GSTREAMER_CUDA,
    JETSON,
    ProducerAvailability,
    _resolution_order,
    available_producers,
    build_hw_producer,
    check_gstreamer_cuda,
    check_jetson_gstreamer,
)


@patch("platform.system", return_value="Linux")
@patch("platform.machine", return_value="aarch64")
def test_jetson_routes_gstreamer_for_streams_and_cv2_for_files(
    machine_mock,
    system_mock,
) -> None:
    # Streams / cameras -> Jetson HW GStreamer.
    assert _resolution_order(prefer=None, video="rtsp://cam/stream") == [JETSON]
    # Local files -> no HW producer selected (cv2 CPU decode, tensorised lazily).
    assert _resolution_order(prefer=None, video="sample.mp4") == []


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
    "inference.core.interfaces.camera.discoverability.check_pynvvideocodec",
    return_value=ProducerAvailability(DGPU, True, "ok"),
)
@patch(
    "inference.core.interfaces.camera.discoverability.check_jetson_gstreamer",
    return_value=ProducerAvailability(JETSON, True, "ok"),
)
@patch(
    "inference.core.interfaces.camera.discoverability.check_gstreamer_cuda",
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
    "inference.core.interfaces.camera.jetson_tensor_bridge.jetson_tensor_bridge_available",
    return_value=(True, "ok"),
)
@patch(
    "inference.core.interfaces.camera.jetson_producer.probe_gstreamer_elements",
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


@patch(
    "inference.core.interfaces.camera.gstreamer_cuda_producer.GstreamerCudaVideoFrameProducer"
)
@patch(
    "inference.core.interfaces.camera.discoverability.available_producers",
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
