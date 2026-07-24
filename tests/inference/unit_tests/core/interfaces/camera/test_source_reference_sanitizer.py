from inference.core.interfaces.camera.source_reference_sanitizer import (
    sanitize_source_reference,
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
            sanitize_source_reference("/home/user/video.mp4") == "/home/user/video.mp4"
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
