from inference.core.interfaces.camera.exceptions import SourceConnectionError
from inference.core.interfaces.camera.stream_error_classifier import (
    build_source_connection_error_message,
    classify_stream_error_message,
    extract_stream_open_error,
    wrap_source_connection_error,
)
from inference.core.interfaces.camera.stream_error_codes import StreamErrorCode


def test_classify_auth_failed() -> None:
    assert (
        classify_stream_error_message("401 Unauthorized")
        == StreamErrorCode.STREAM_AUTH_FAILED
    )


def test_classify_tls_certificate() -> None:
    assert (
        classify_stream_error_message("certificate verify failed")
        == StreamErrorCode.STREAM_TLS_CERTIFICATE
    )


def test_classify_tls_handshake() -> None:
    assert (
        classify_stream_error_message("TLS handshake failed")
        == StreamErrorCode.STREAM_TLS_HANDSHAKE
    )


def test_classify_ssl_handshake_not_certificate() -> None:
    assert (
        classify_stream_error_message("SSL handshake failed")
        == StreamErrorCode.STREAM_TLS_HANDSHAKE
    )


def test_classify_timeout() -> None:
    assert (
        classify_stream_error_message("Connection timed out")
        == StreamErrorCode.STREAM_TIMEOUT
    )


def test_classify_not_found() -> None:
    assert (
        classify_stream_error_message("404 stream not found")
        == StreamErrorCode.STREAM_NOT_FOUND
    )


def test_classify_codec_unsupported() -> None:
    assert (
        classify_stream_error_message("unsupported codec for stream")
        == StreamErrorCode.STREAM_CODEC_UNSUPPORTED
    )


def test_classify_default() -> None:
    assert (
        classify_stream_error_message("something else")
        == StreamErrorCode.STREAM_CONNECTION_FAILED
    )


def test_classify_empty_message() -> None:
    assert classify_stream_error_message("") == StreamErrorCode.STREAM_CONNECTION_FAILED
    assert classify_stream_error_message(None) == StreamErrorCode.STREAM_CONNECTION_FAILED


def test_classify_status_code_false_positives() -> None:
    assert (
        classify_stream_error_message(
            "Cannot connect to rtsp://camera401.example/stream"
        )
        == StreamErrorCode.STREAM_CONNECTION_FAILED
    )
    assert (
        classify_stream_error_message("Connection refused on port 1401")
        == StreamErrorCode.STREAM_CONNECTION_FAILED
    )
    assert (
        classify_stream_error_message("rtsp://host:4040/stream")
        == StreamErrorCode.STREAM_CONNECTION_FAILED
    )


def test_classify_explicit_status_phrases() -> None:
    assert (
        classify_stream_error_message("HTTP status 404 for stream")
        == StreamErrorCode.STREAM_NOT_FOUND
    )
    assert (
        classify_stream_error_message("403 forbidden")
        == StreamErrorCode.STREAM_AUTH_FAILED
    )


def test_wrap_source_connection_error_sets_code_and_reference() -> None:
    error = wrap_source_connection_error(
        "401 Unauthorized",
        source_reference="rtsp://camera.example/stream",
    )
    assert isinstance(error, SourceConnectionError)
    assert error.code == StreamErrorCode.STREAM_AUTH_FAILED
    assert error.source_reference == "rtsp://camera.example/stream"
    assert str(error) == "401 Unauthorized"


def test_extract_stream_open_error_prefers_actionable_line() -> None:
    stderr = (
        "[rtsp @ 0x1] method DESCRIBE failed: 401 Unauthorized\n"
        "OpenCV: Couldn't read video stream"
    )
    assert extract_stream_open_error(stderr) == (
        "[rtsp @ 0x1] method DESCRIBE failed: 401 Unauthorized"
    )


def test_build_source_connection_error_message_includes_underlying_error() -> None:
    message = build_source_connection_error_message(
        source_reference="rtsp://camera.example/stream",
        underlying_error="401 Unauthorized",
    )
    assert message.startswith("401 Unauthorized:")
    assert "rtsp://camera.example/stream" in message


def test_build_source_connection_error_message_redacts_credentialed_stderr() -> None:
    message = build_source_connection_error_message(
        source_reference="rtsp://192.168.1.1:554/stream",
        underlying_error=(
            "OpenCV: Couldn't read video stream from file "
            '"rtsp://user:secret@192.168.1.1:554/stream"'
        ),
    )
    assert "secret" not in message
    assert "rtsp://192.168.1.1:554/stream" in message


def test_wrap_source_connection_error_classifies_underlying_ffmpeg_error() -> None:
    error = wrap_source_connection_error(
        build_source_connection_error_message(
            source_reference="rtsp://camera.example/stream",
            underlying_error="TLS handshake failed",
        ),
        source_reference="rtsp://camera.example/stream",
    )
    assert error.code == StreamErrorCode.STREAM_TLS_HANDSHAKE
