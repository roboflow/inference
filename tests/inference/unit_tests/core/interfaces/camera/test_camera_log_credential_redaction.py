"""Regression tests: no camera-path log line may leak source credentials.

The camera fleet's RTSP references embed ``user:password@`` inline, and
GStreamer error reprs embed the full launch string carrying the same URL.
These tests drive every producer-selection / fallback log path with a
credentialed reference and assert the password never reaches a log record,
while the sanitized host stays visible so the logs remain useful.

The ``inference`` logger has ``propagate = False`` (see
``inference/core/logger.py``), so pytest's root-attached caplog handler never
sees these records. The handler is attached directly to the ``inference``
logger; the ``discoverability`` module logs via
``logging.getLogger(__name__)``, which is a descendant of ``inference`` and
propagates up to that same handler.
"""

import logging
from contextlib import contextmanager

import pytest

from inference.core.interfaces.camera import discoverability, video_source
from inference.core.interfaces.camera.discoverability import ProducerAvailability
from inference.core.interfaces.camera.source_reference_sanitizer import (
    sanitize_source_reference,
)

SECRET = "secret-password-value"
CREDENTIALED_URL = f"rtsp://user:{SECRET}@192.168.0.99:554/stream"
SANITIZED_URL = sanitize_source_reference(CREDENTIALED_URL)
FAKE_LAUNCH_STRING = (
    f'rtspsrc location="{CREDENTIALED_URL}" protocols=tcp ! '
    "application/x-rtp,media=video ! queue ! rtph264depay"
)


class _DummyProducer:
    def __init__(self, *args, **kwargs):
        pass


@contextmanager
def _capture_camera_logs(caplog):
    inference_logger = logging.getLogger("inference")
    inference_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.DEBUG, logger="inference"):
            yield
    finally:
        inference_logger.removeHandler(caplog.handler)


def _assert_no_secret_and_sanitized_reference_present(caplog) -> None:
    assert caplog.records, "expected at least one captured log record"
    joined = "\n".join(record.getMessage() for record in caplog.records)
    assert SECRET not in joined, "credential leaked into a log record"
    assert "user:" not in joined, "userinfo leaked into a log record"
    assert "192.168.0.99" in joined, (
        "sanitized reference should keep the host visible so the log stays "
        "useful for debugging"
    )


def test_selected_hardware_decoder_log_is_sanitized(monkeypatch, caplog) -> None:
    # given: the tensor path selects a hardware decoder for a credentialed URL
    monkeypatch.setattr(video_source, "ENABLE_TENSOR_DATA_REPRESENTATION", True)
    monkeypatch.setattr(
        discoverability, "build_hw_producer", lambda *args, **kwargs: _DummyProducer()
    )

    # when
    with _capture_camera_logs(caplog):
        producer = video_source._build_default_producer(
            CREDENTIALED_URL, output_tensor=True
        )

    # then
    assert isinstance(producer, _DummyProducer)
    _assert_no_secret_and_sanitized_reference_present(caplog)
    assert any(SANITIZED_URL in record.getMessage() for record in caplog.records)


def test_hardware_decoder_failure_log_redacts_url_in_error_repr(
    monkeypatch, caplog
) -> None:
    # given: decoder construction raises an error embedding the full launch
    # string (the GStreamer failure shape), and the cv2 fallback must not
    # open a network connection from a unit test
    monkeypatch.setattr(video_source, "ENABLE_TENSOR_DATA_REPRESENTATION", True)

    def _raise(*args, **kwargs):
        raise RuntimeError(f"could not link pipeline: {FAKE_LAUNCH_STRING}")

    monkeypatch.setattr(discoverability, "build_hw_producer", _raise)
    monkeypatch.setattr(
        discoverability,
        "available_producers",
        lambda *args, **kwargs: {
            discoverability.JETSON: ProducerAvailability(
                discoverability.JETSON, False, "probe declined"
            )
        },
    )
    monkeypatch.setattr(video_source, "CV2VideoFrameProducer", _DummyProducer)

    # when
    with _capture_camera_logs(caplog):
        producer = video_source._build_default_producer(
            CREDENTIALED_URL, output_tensor=True
        )

    # then
    assert isinstance(producer, _DummyProducer)
    _assert_no_secret_and_sanitized_reference_present(caplog)


def test_no_usable_decoder_log_redacts_url_in_probe_reasons(
    monkeypatch, caplog
) -> None:
    # given: no decoder is usable and a probe reason itself embeds the URL
    monkeypatch.setattr(video_source, "ENABLE_TENSOR_DATA_REPRESENTATION", True)
    monkeypatch.setattr(
        discoverability, "build_hw_producer", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        discoverability,
        "available_producers",
        lambda *args, **kwargs: {
            discoverability.JETSON: ProducerAvailability(
                discoverability.JETSON,
                False,
                f"pipeline probe failed for {CREDENTIALED_URL}",
            )
        },
    )
    monkeypatch.setattr(video_source, "CV2VideoFrameProducer", _DummyProducer)

    # when
    with _capture_camera_logs(caplog):
        producer = video_source._build_default_producer(
            CREDENTIALED_URL, output_tensor=True
        )

    # then
    assert isinstance(producer, _DummyProducer)
    _assert_no_secret_and_sanitized_reference_present(caplog)


def test_legacy_path_debug_log_is_sanitized(monkeypatch, caplog) -> None:
    # given: the flag-off path logs the reference at debug level
    monkeypatch.setattr(video_source, "ENABLE_TENSOR_DATA_REPRESENTATION", False)
    monkeypatch.setattr(
        video_source, "_create_video_frame_producer", lambda video: _DummyProducer()
    )

    # when
    with _capture_camera_logs(caplog):
        producer = video_source._build_default_producer(CREDENTIALED_URL)

    # then
    assert isinstance(producer, _DummyProducer)
    _assert_no_secret_and_sanitized_reference_present(caplog)


def test_discoverability_fallback_loop_log_is_sanitized(monkeypatch, caplog) -> None:
    # given: the jetson candidate passes its probe but its constructor raises
    # an error embedding the credentialed launch string
    from inference.core.interfaces.camera import jetson_producer

    monkeypatch.setattr(
        discoverability,
        "available_producers",
        lambda *args, **kwargs: {
            discoverability.JETSON: ProducerAvailability(
                discoverability.JETSON, True, "ok"
            )
        },
    )
    monkeypatch.setattr(
        discoverability, "_resolution_order", lambda *args, **kwargs: ["jetson"]
    )

    def _raising_producer(*args, **kwargs):
        raise RuntimeError(f"could not construct pipeline: {FAKE_LAUNCH_STRING}")

    monkeypatch.setattr(jetson_producer, "JetsonVideoFrameProducer", _raising_producer)

    # when
    with _capture_camera_logs(caplog):
        result = discoverability.build_hw_producer(CREDENTIALED_URL, output_tensor=True)

    # then: every candidate failed, and the fallback warning leaked nothing
    assert result is None
    _assert_no_secret_and_sanitized_reference_present(caplog)


def test_discoverability_display_reference_passes_ints_through() -> None:
    assert discoverability._display_reference(0) == 0
    assert discoverability._display_reference(None) is None
    assert SECRET not in str(discoverability._display_reference(CREDENTIALED_URL))
