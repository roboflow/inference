import os

import pytest

from inference.core.interfaces.camera.rtsp_tls import (
    RTSP_TLS_VALIDATION_FLAGS_ENV_VAR,
    is_rtsps_url,
    is_rtsp_url,
    rtsp_tls_validation_flags_gstreamer_suffix,
)


def test_is_rtsps_url() -> None:
    assert is_rtsps_url("rtsps://cam.example/stream")
    assert not is_rtsps_url("rtsp://cam.example/stream")


def test_is_rtsp_url() -> None:
    assert is_rtsp_url("rtsp://cam.example/stream")
    assert is_rtsp_url("rtsps://cam.example/stream")
    assert not is_rtsp_url("/dev/video0")


def test_tls_suffix_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(RTSP_TLS_VALIDATION_FLAGS_ENV_VAR, raising=False)
    assert rtsp_tls_validation_flags_gstreamer_suffix() == ""


def test_tls_suffix_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RTSP_TLS_VALIDATION_FLAGS_ENV_VAR, "0")
    assert rtsp_tls_validation_flags_gstreamer_suffix() == " tls-validation-flags=0"


def test_tls_suffix_allow_self_signed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RTSP_TLS_VALIDATION_FLAGS_ENV_VAR, "126")
    assert rtsp_tls_validation_flags_gstreamer_suffix() == " tls-validation-flags=126"


def test_tls_suffix_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RTSP_TLS_VALIDATION_FLAGS_ENV_VAR, "nope")
    with pytest.raises(ValueError):
        rtsp_tls_validation_flags_gstreamer_suffix()
