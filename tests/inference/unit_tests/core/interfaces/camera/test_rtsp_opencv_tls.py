import os

import pytest

from inference.core.interfaces.camera.rtsp_opencv_tls import (
    apply_opencv_rtsps_tls_env,
    build_opencv_ffmpeg_capture_options,
)
from inference.core.interfaces.camera.rtsp_tls import (
    GST_SSL_CA_CERTIFICATE_ENV_VAR,
    RTSP_TLS_VALIDATION_FLAGS_ENV_VAR,
)


def test_build_options_non_rtsps() -> None:
    assert build_opencv_ffmpeg_capture_options("rtsp://cam/stream") is None


def test_build_options_rtsps_with_ca(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(GST_SSL_CA_CERTIFICATE_ENV_VAR, "/etc/ssl/certs/ca-certificates.crt")
    got = build_opencv_ffmpeg_capture_options("rtsps://cam/stream")
    assert got is not None
    assert "cafile;/etc/ssl/certs/ca-certificates.crt" in got
    assert "rtsp_transport;tcp" in got


def test_build_options_self_signed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RTSP_TLS_VALIDATION_FLAGS_ENV_VAR, "0")
    got = build_opencv_ffmpeg_capture_options("rtsps://cam/stream")
    assert got is not None
    assert "tls_verify;0" in got


def test_apply_sets_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", raising=False)
    apply_opencv_rtsps_tls_env("rtsps://cam/stream")
    assert "rtsp_transport;tcp" in os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
