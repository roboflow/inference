from inference.core.interfaces.camera.source_reference_sanitizer import (
    UNPARSEABLE_SOURCE,
    redact_credentials_in_text,
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

    def test_strips_credentials_without_url_scheme(self):
        assert (
            sanitize_source_reference("admin:secret@192.168.1.1:554/stream")
            == "192.168.1.1:554/stream"
        )

    def test_strips_credentials_when_password_contains_at_sign(self):
        assert (
            sanitize_source_reference("rtsp://user:p@ss@host:554/stream")
            == "rtsp://host:554/stream"
        )

    def test_strips_schemeless_credentials_when_password_contains_at_sign(self):
        assert (
            sanitize_source_reference("user:p@ss@host:554/stream")
            == "host:554/stream"
        )

    def test_malformed_port_does_not_raise(self):
        assert (
            sanitize_source_reference("rtsp://user:pass@host:notaport/path")
            == "rtsp://host:notaport/path"
        )

    def test_open_bracket_in_password_does_not_raise(self):
        assert (
            sanitize_source_reference("rtsp://admin:pass[123@cam.local:554/stream")
            == "rtsp://cam.local:554/stream"
        )

    def test_close_bracket_in_password_does_not_raise(self):
        assert (
            sanitize_source_reference("rtsp://admin:pass]123@cam.local:554/stream")
            == "rtsp://cam.local:554/stream"
        )

    def test_password_with_slash_returns_placeholder_not_credentials(self):
        assert (
            sanitize_source_reference("rtsp://user:pa/ss@host:554/stream")
            == UNPARSEABLE_SOURCE
        )

    def test_numeric_password_prefix_returns_placeholder_not_credentials(self):
        assert (
            sanitize_source_reference("rtsp://user:12/34@host:554/stream")
            == UNPARSEABLE_SOURCE
        )

    def test_at_sign_in_path_without_netloc_credentials_returns_placeholder(self):
        assert (
            sanitize_source_reference("rtsp://host:554/stream@2x")
            == UNPARSEABLE_SOURCE
        )

    def test_password_with_question_mark_recovers_host(self):
        assert (
            sanitize_source_reference("rtsp://user:pa?ss@host:554/stream")
            == "rtsp://host:554/stream"
        )

    def test_password_with_hash_recovers_host(self):
        assert (
            sanitize_source_reference("rtsp://user:pa#ss@host:554/stream")
            == "rtsp://host:554/stream"
        )

    def test_embedded_url_after_userinfo_has_credentials_redacted(self):
        assert (
            sanitize_source_reference("foo@rtsp://user:pass@host/path")
            == "foo@rtsp://host/path"
        )

    def test_preserves_filename_with_at_sign(self):
        assert sanitize_source_reference("video@2x.mp4") == "video@2x.mp4"

    def test_preserves_windows_path_with_at_sign(self):
        assert (
            sanitize_source_reference("C:\\videos\\cam@1.mp4") == "C:\\videos\\cam@1.mp4"
        )

    def test_preserves_forward_slash_windows_path_with_at_sign(self):
        assert sanitize_source_reference("C:/cam@1.mp4") == "C:/cam@1.mp4"

    def test_strips_schemeless_username_only_before_host_with_port(self):
        assert (
            sanitize_source_reference("admin@10.0.0.1:554/stream")
            == "10.0.0.1:554/stream"
        )

    def test_lowercases_scheme_and_host_preserving_path_case(self):
        assert (
            sanitize_source_reference("RTSP://User:Pass@CAM.Local:554/Stream")
            == "rtsp://cam.local:554/Stream"
        )

    def test_preserves_ipv6_host_brackets(self):
        assert (
            sanitize_source_reference("rtsp://user:pass@[::1]:554/stream")
            == "rtsp://[::1]:554/stream"
        )


class TestRedactCredentialsInText:
    def test_redacts_url_credentials_embedded_in_stderr_line(self):
        assert redact_credentials_in_text(
            'OpenCV: Couldn\'t read video stream from file "rtsp://user:secret@host:554/stream"'
        ) == ('OpenCV: Couldn\'t read video stream from file "rtsp://host:554/stream"')

    def test_redacts_password_containing_at_sign(self):
        assert (
            redact_credentials_in_text("open rtsp://user:p@ss@host/s failed")
            == "open rtsp://host/s failed"
        )

    def test_leaves_text_without_credentials_unchanged(self):
        assert (
            redact_credentials_in_text("Connection refused: rtsp://host:554/stream")
            == "Connection refused: rtsp://host:554/stream"
        )

    def test_replaces_url_with_slash_password_by_placeholder(self):
        assert (
            redact_credentials_in_text("err: rtsp://user:pa/ss@host:554/stream fail")
            == f"err: {UNPARSEABLE_SOURCE} fail"
        )

    def test_leaves_windows_forward_slash_path_unchanged(self):
        assert redact_credentials_in_text("cannot open C:/cam@1.mp4") == (
            "cannot open C:/cam@1.mp4"
        )
