import os
import threading

import pytest

from inference.core.interfaces.camera.rtsp_opencv_tls import (
    OPENCV_FFMPEG_CAPTURE_OPTIONS_ENV_VAR,
    SSL_CERT_FILE_ENV_VAR,
    build_opencv_ffmpeg_capture_options,
    merge_opencv_ffmpeg_capture_options,
    opencv_rtsps_tls_env,
)
from inference.core.interfaces.camera.rtsp_tls import (
    GST_SSL_CA_CERTIFICATE_ENV_VAR,
    RTSP_TLS_VALIDATION_FLAGS_ENV_VAR,
    rtsp_tls_validation_flags_gstreamer_suffix,
)


def test_build_options_non_rtsps() -> None:
    assert build_opencv_ffmpeg_capture_options("rtsp://cam/stream") is None


def test_build_options_rtsps_default_strict_verify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(RTSP_TLS_VALIDATION_FLAGS_ENV_VAR, raising=False)
    got = build_opencv_ffmpeg_capture_options("rtsps://cam/stream")
    assert got is not None
    assert "rtsp_transport;tcp" in got
    assert "tls_verify;1" in got


def test_build_options_rtsps_with_ca(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(GST_SSL_CA_CERTIFICATE_ENV_VAR, "/etc/ssl/certs/ca-certificates.crt")
    got = build_opencv_ffmpeg_capture_options("rtsps://cam/stream")
    assert got is not None
    assert "cafile;/etc/ssl/certs/ca-certificates.crt" in got
    assert "rtsp_transport;tcp" in got
    assert "tls_verify;1" in got


def test_build_options_rtsps_with_ssl_cert_file_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(GST_SSL_CA_CERTIFICATE_ENV_VAR, raising=False)
    monkeypatch.setenv(SSL_CERT_FILE_ENV_VAR, "/etc/ssl/cert.pem")
    got = build_opencv_ffmpeg_capture_options("rtsps://cam/stream")
    assert got is not None
    assert "cafile;/etc/ssl/cert.pem" in got
    assert "tls_verify;1" in got


def test_build_options_self_signed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RTSP_TLS_VALIDATION_FLAGS_ENV_VAR, "0")
    got = build_opencv_ffmpeg_capture_options("rtsps://cam/stream")
    assert got is not None
    assert "tls_verify;0" in got


def test_build_options_allow_self_signed_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RTSP_TLS_VALIDATION_FLAGS_ENV_VAR, "126")
    got = build_opencv_ffmpeg_capture_options("rtsps://cam/stream")
    assert got is not None
    assert "tls_verify;0" in got


def test_build_options_strict_verify_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RTSP_TLS_VALIDATION_FLAGS_ENV_VAR, "1")
    got = build_opencv_ffmpeg_capture_options("rtsps://cam/stream")
    assert got is not None
    assert "tls_verify;1" in got


def test_merge_preserves_unrelated_existing_options() -> None:
    existing = "stimeout;5000000|buffer_size;102400"
    got = merge_opencv_ffmpeg_capture_options(
        existing,
        {"rtsp_transport": "tcp", "tls_verify": "1"},
    )
    assert "stimeout;5000000" in got
    assert "buffer_size;102400" in got
    assert "rtsp_transport;tcp" in got
    assert "tls_verify;1" in got


def test_build_options_merges_existing_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(OPENCV_FFMPEG_CAPTURE_OPTIONS_ENV_VAR, "stimeout;5000000")
    got = build_opencv_ffmpeg_capture_options("rtsps://cam/stream")
    assert got is not None
    assert "stimeout;5000000" in got
    assert "rtsp_transport;tcp" in got
    assert "tls_verify;1" in got


def test_rtsp_tls_validation_flags_gstreamer_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(RTSP_TLS_VALIDATION_FLAGS_ENV_VAR, raising=False)
    assert rtsp_tls_validation_flags_gstreamer_suffix() == ""
    monkeypatch.setenv(RTSP_TLS_VALIDATION_FLAGS_ENV_VAR, "126")
    assert rtsp_tls_validation_flags_gstreamer_suffix() == " tls-validation-flags=126"


def test_opencv_rtsps_tls_env_sets_and_restores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(OPENCV_FFMPEG_CAPTURE_OPTIONS_ENV_VAR, raising=False)
    with opencv_rtsps_tls_env("rtsps://cam/stream"):
        assert "rtsp_transport;tcp" in os.environ[OPENCV_FFMPEG_CAPTURE_OPTIONS_ENV_VAR]
        assert "tls_verify;1" in os.environ[OPENCV_FFMPEG_CAPTURE_OPTIONS_ENV_VAR]
    assert OPENCV_FFMPEG_CAPTURE_OPTIONS_ENV_VAR not in os.environ


def test_opencv_rtsps_tls_env_restores_previous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(OPENCV_FFMPEG_CAPTURE_OPTIONS_ENV_VAR, "old;value")
    with opencv_rtsps_tls_env("rtsps://cam/stream"):
        assert "rtsp_transport;tcp" in os.environ[OPENCV_FFMPEG_CAPTURE_OPTIONS_ENV_VAR]
    assert os.environ[OPENCV_FFMPEG_CAPTURE_OPTIONS_ENV_VAR] == "old;value"


def test_opencv_rtsps_tls_env_serializes_concurrent_opens() -> None:
    os.environ.pop(OPENCV_FFMPEG_CAPTURE_OPTIONS_ENV_VAR, None)
    observed: list[str] = []
    barrier = threading.Barrier(2)

    def worker(video: str, flag_value: str) -> None:
        previous_flags = os.environ.get(RTSP_TLS_VALIDATION_FLAGS_ENV_VAR)
        os.environ[RTSP_TLS_VALIDATION_FLAGS_ENV_VAR] = flag_value
        try:
            with opencv_rtsps_tls_env(video):
                barrier.wait()
                observed.append(os.environ[OPENCV_FFMPEG_CAPTURE_OPTIONS_ENV_VAR])
        finally:
            if previous_flags is None:
                os.environ.pop(RTSP_TLS_VALIDATION_FLAGS_ENV_VAR, None)
            else:
                os.environ[RTSP_TLS_VALIDATION_FLAGS_ENV_VAR] = previous_flags

    strict = threading.Thread(target=worker, args=("rtsps://strict/stream", "1"))
    relaxed = threading.Thread(target=worker, args=("rtsps://relaxed/stream", "0"))
    strict.start()
    relaxed.start()
    strict.join()
    relaxed.join()

    assert any("tls_verify;1" in value for value in observed)
    assert any("tls_verify;0" in value for value in observed)
    assert OPENCV_FFMPEG_CAPTURE_OPTIONS_ENV_VAR not in os.environ
