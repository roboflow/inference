from unittest.mock import MagicMock

import pytest

from inference.core.interfaces.camera.source_reference_sanitizer import (
    classify_source_reference,
    sanitize_source_reference,
    sanitize_source_reference_for_log,
)


class TestSanitizeSourceReference:
    def test_strips_credentials_from_rtsp_url(self):
        assert (
            sanitize_source_reference("rtsp://admin:secret@192.168.1.1:554/stream1")
            == "rtsp://192.168.1.1:554/stream1"
        )

    def test_strips_credentials_from_http_url(self):
        assert (
            sanitize_source_reference("http://user:pass@example.com:8080/feed")
            == "http://example.com:8080/feed"
        )

    def test_strips_username_only(self):
        assert (
            sanitize_source_reference("rtsp://admin@10.0.0.1/live")
            == "rtsp://10.0.0.1/live"
        )

    def test_preserves_url_without_credentials(self):
        assert (
            sanitize_source_reference("rtsp://192.168.1.1:554/stream")
            == "rtsp://192.168.1.1:554/stream"
        )

    def test_preserves_device_index(self):
        assert sanitize_source_reference("0") == "0"

    def test_preserves_file_path(self):
        assert sanitize_source_reference("/dev/video0") == "/dev/video0"

    def test_preserves_regular_file_path(self):
        assert (
            sanitize_source_reference("/home/user/video.mp4")
            == "/home/user/video.mp4"
        )

    def test_strips_credentials_and_query_params(self):
        assert (
            sanitize_source_reference(
                "rtsp://user:p%40ss@cam.local:554/ch1?transport=tcp"
            )
            == "rtsp://cam.local:554/ch1"
        )

    def test_strips_query_params_without_credentials(self):
        assert (
            sanitize_source_reference(
                "rtsp://cam.local:554/stream?token=secret123&channel=1"
            )
            == "rtsp://cam.local:554/stream"
        )

    def test_strips_fragment(self):
        assert (
            sanitize_source_reference("http://example.com/feed#section")
            == "http://example.com/feed"
        )


class TestSanitizeSourceReferenceForLog:
    def test_delegates_to_sanitize_source_reference(self):
        ref = "rtsp://user:pass@cam.local/stream?token=secret"
        assert sanitize_source_reference_for_log(ref) == sanitize_source_reference(ref)


class TestClassifySourceReference:
    def test_callable_uses_type_name(self):
        def producer_factory():
            return MagicMock()

        assert classify_source_reference(producer_factory) == "function"

    def test_callable_class_uses_class_name(self):
        class WebRTCVideoFrameProducer:
            def __call__(self):
                return MagicMock()

        assert classify_source_reference(WebRTCVideoFrameProducer()) == (
            "WebRTCVideoFrameProducer"
        )

    def test_integer_device_id(self):
        assert classify_source_reference(0) == "0"

    def test_list_joins_sanitized_elements(self):
        assert classify_source_reference(
            [
                "rtsp://user:pass@cam1.local/stream",
                "rtsp://admin:secret@cam2.local:554/live",
            ]
        ) == "rtsp://cam1.local/stream,rtsp://cam2.local:554/live"

    def test_string_delegates_to_sanitize_source_reference(self):
        ref = "rtsp://user:pass@10.0.0.1/live"
        assert classify_source_reference(ref) == sanitize_source_reference(ref)
