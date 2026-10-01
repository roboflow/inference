import logging
from typing import Any, Dict, List, Tuple

import cv2
import pytest
from pydantic import ValidationError
from streamvision.camera import gstreamer_rtsp_producer, video_source
from streamvision.camera.capture_properties import (
    apply_capture_properties,
    parse_fourcc,
)
from streamvision.stream_manager.manager_app.entities import VideoConfiguration

MJPG = cv2.VideoWriter_fourcc(*"MJPG")


class _RecordingCapture:
    def __init__(self, readback: Dict[int, float] = None) -> None:
        self.set_calls: List[Tuple[int, Any]] = []
        self._readback = readback or {}

    def set(self, property_id: int, value: Any) -> bool:
        self.set_calls.append((property_id, value))
        return True

    def get(self, property_id: int) -> float:
        return self._readback.get(property_id, 0.0)


def test_fourcc_is_applied_first_and_fps_last_regardless_of_input_order() -> None:
    # given - the key order a device received from the platform
    capture = _RecordingCapture()
    properties = {
        "frame_width": 1920,
        "frame_height": 1080,
        "fps": 30,
        "fourcc": MJPG,
    }

    # when
    apply_capture_properties(capture, properties=properties)

    # then
    assert capture.set_calls == [
        (cv2.CAP_PROP_FOURCC, MJPG),
        (cv2.CAP_PROP_FRAME_WIDTH, 1920),
        (cv2.CAP_PROP_FRAME_HEIGHT, 1080),
        (cv2.CAP_PROP_FPS, 30),
    ]


def test_other_properties_keep_their_given_order_between_fourcc_and_fps() -> None:
    # given
    capture = _RecordingCapture()
    properties = {
        "frame_height": 720,
        "fps": 60,
        "brightness": 0.5,
        "fourcc": MJPG,
        "frame_width": 1280,
    }

    # when
    apply_capture_properties(capture, properties=properties)

    # then
    assert capture.set_calls == [
        (cv2.CAP_PROP_FOURCC, MJPG),
        (cv2.CAP_PROP_FRAME_HEIGHT, 720),
        (cv2.CAP_PROP_BRIGHTNESS, 0.5),
        (cv2.CAP_PROP_FRAME_WIDTH, 1280),
        (cv2.CAP_PROP_FPS, 60),
    ]


@pytest.mark.parametrize("fourcc", ["MJPG", " MJPG ", str(MJPG), float(MJPG)])
def test_fourcc_given_as_code_or_number_is_converted_to_int(fourcc: Any) -> None:
    # given
    capture = _RecordingCapture()

    # when
    apply_capture_properties(capture, properties={"fps": 30, "fourcc": fourcc})

    # then
    assert capture.set_calls == [
        (cv2.CAP_PROP_FOURCC, MJPG),
        (cv2.CAP_PROP_FPS, 30),
    ]
    assert isinstance(capture.set_calls[0][1], int)


@pytest.mark.parametrize("fourcc", ["avc1", "Y16 ", "pRAA"])
def test_four_character_codes_are_used_exactly_as_given(fourcc: str) -> None:
    # given - FOURCC codes are case-sensitive and may end in a space
    capture = _RecordingCapture()

    # when
    apply_capture_properties(capture, properties={"fourcc": fourcc})

    # then
    assert capture.set_calls == [
        (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc)),
    ]


class _RejectingCapture(_RecordingCapture):
    def set(self, property_id: int, value: Any) -> bool:
        super().set(property_id, value)
        return False


def test_rejected_property_is_logged(
    streamvision_caplog: pytest.LogCaptureFixture,
) -> None:
    # given
    capture = _RejectingCapture()

    # when
    with streamvision_caplog.at_level(logging.DEBUG):
        apply_capture_properties(capture, properties={"exposure": -6})

    # then
    assert "did not accept property exposure=-6" in streamvision_caplog.text
    assert all(
        record.levelno == logging.DEBUG for record in streamvision_caplog.records
    )


def test_rejected_fourcc_is_logged_as_a_warning(
    streamvision_caplog: pytest.LogCaptureFixture,
) -> None:
    # given
    capture = _RejectingCapture()

    # when
    with streamvision_caplog.at_level(logging.WARNING):
        apply_capture_properties(capture, properties={"fourcc": "MJPG"})

    # then
    assert f"did not accept property fourcc={MJPG!r}" in streamvision_caplog.text


@pytest.mark.parametrize(
    "fourcc",
    [
        "MJPEG",
        "MJ",
        "",
        "\u00b2\u00b2\u00b2\u00b2",
        -1,
        1.5,
        float("nan"),
        float("inf"),
        True,
        None,
    ],
)
def test_invalid_fourcc_is_skipped_with_a_warning(
    fourcc: Any,
    streamvision_caplog: pytest.LogCaptureFixture,
) -> None:
    # given
    capture = _RecordingCapture()

    # when
    with streamvision_caplog.at_level(logging.WARNING):
        apply_capture_properties(
            capture,
            properties={"fourcc": fourcc, "frame_width": 640, "fps": 30},
        )

    # then
    assert capture.set_calls == [
        (cv2.CAP_PROP_FRAME_WIDTH, 640),
        (cv2.CAP_PROP_FPS, 30),
    ]
    assert "Ignoring invalid fourcc" in streamvision_caplog.text


@pytest.mark.parametrize(
    "fourcc",
    [
        "MJPEG",
        "MJ",
        "",
        "\u00b2\u00b2\u00b2\u00b2",
        "\u0661\u0662",
        -1,
        1.5,
        True,
        None,
    ],
)
def test_parse_fourcc_returns_none_for_invalid_values(fourcc: Any) -> None:
    # when / then - non-ASCII digits pass str.isdigit() but must not reach int()
    assert parse_fourcc(fourcc) is None


@pytest.mark.parametrize("properties", [None, {}])
def test_empty_properties_set_nothing(properties: Any) -> None:
    # given
    capture = _RecordingCapture()

    # when
    apply_capture_properties(capture, properties=properties)

    # then
    assert capture.set_calls == []


def test_unknown_property_still_raises() -> None:
    # given
    capture = _RecordingCapture()

    # when / then
    with pytest.raises(AttributeError):
        apply_capture_properties(capture, properties={"not_a_property": 1})


def test_effective_format_is_logged_after_fourcc_and_fps(
    streamvision_caplog: pytest.LogCaptureFixture,
) -> None:
    # given - a driver that kept the clamped frame rate
    capture = _RecordingCapture(
        readback={
            cv2.CAP_PROP_FOURCC: float(MJPG),
            cv2.CAP_PROP_FPS: 5.0,
            cv2.CAP_PROP_FRAME_WIDTH: 1920.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080.0,
        }
    )

    # when
    with streamvision_caplog.at_level(logging.INFO):
        apply_capture_properties(capture, properties={"fourcc": "MJPG", "fps": 30})

    # then
    assert "fourcc='MJPG', fps=5.0, size=1920x1080" in streamvision_caplog.text


@pytest.mark.parametrize(
    "producer_class",
    [
        video_source.CV2VideoFrameProducer,
        gstreamer_rtsp_producer.GStreamerRtspVideoFrameProducer,
    ],
)
def test_producers_apply_properties_in_camera_safe_order(producer_class) -> None:
    # given
    producer = producer_class.__new__(producer_class)
    producer.stream = _RecordingCapture()

    # when
    producer.initialize_source_properties(
        {"frame_height": 1080, "fourcc": "MJPG", "fps": 30, "frame_width": 1920}
    )

    # then
    assert producer.stream.set_calls == [
        (cv2.CAP_PROP_FOURCC, MJPG),
        (cv2.CAP_PROP_FRAME_HEIGHT, 1080),
        (cv2.CAP_PROP_FRAME_WIDTH, 1920),
        (cv2.CAP_PROP_FPS, 30),
    ]


def test_video_configuration_accepts_string_fourcc_and_coerces_numbers() -> None:
    # when
    config = VideoConfiguration(
        type="VideoConfiguration",
        video_reference=0,
        video_source_properties={"fourcc": "MJPG", "fps": "30", "frame_width": 1920},
    )

    # then
    assert config.video_source_properties == {
        "fourcc": "MJPG",
        "fps": 30.0,
        "frame_width": 1920.0,
    }
    assert isinstance(config.video_source_properties["fps"], float)


def test_video_configuration_rejects_non_numeric_value_for_other_properties() -> None:
    # when / then
    with pytest.raises(ValidationError):
        VideoConfiguration(
            type="VideoConfiguration",
            video_reference=0,
            video_source_properties={"fps": "fast"},
        )


@pytest.mark.parametrize(
    "fourcc", ["MJPEG", "MJ", "\u00b2\u00b2\u00b2\u00b2", -1, 1.5, float("nan"), True]
)
def test_video_configuration_rejects_invalid_fourcc(fourcc: Any) -> None:
    # when / then
    with pytest.raises(ValidationError, match="fourcc"):
        VideoConfiguration(
            type="VideoConfiguration",
            video_reference=0,
            video_source_properties={"fourcc": fourcc},
        )


@pytest.mark.parametrize("fourcc", ["avc1", "Y16 ", MJPG, str(MJPG)])
def test_video_configuration_accepts_valid_fourcc(fourcc: Any) -> None:
    # when
    config = VideoConfiguration(
        type="VideoConfiguration",
        video_reference=0,
        video_source_properties={"fourcc": fourcc},
    )

    # then
    assert parse_fourcc(config.video_source_properties["fourcc"]) == parse_fourcc(
        fourcc
    )


@pytest.mark.parametrize("code", ["MJPG", "YUYV", "avc1", "Y16 ", "pRAA"])
def test_fourcc_matches_opencv_packing(code: str) -> None:
    assert parse_fourcc(code) == cv2.VideoWriter_fourcc(*code)
