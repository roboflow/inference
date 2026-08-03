import os
import threading
import time

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
    monkeypatch.setenv(
        GST_SSL_CA_CERTIFICATE_ENV_VAR, "/etc/ssl/certs/ca-certificates.crt"
    )
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
<<<<<<< HEAD
    """Concurrent opens must not overlap inside the env context.

    The previous Barrier-based version deadlocked: both workers waited on the
    barrier while holding ``_opencv_rtsps_tls_lock``, so the second thread never
    reached the barrier. That hung CI unit tests at this module until the job
    timeout killed the runner.
    """
    os.environ.pop(OPENCV_FFMPEG_CAPTURE_OPTIONS_ENV_VAR, None)
    previous_flags = os.environ.pop(RTSP_TLS_VALIDATION_FLAGS_ENV_VAR, None)
    counter_lock = threading.Lock()
    inside_count = 0
    max_inside = 0
    entries = 0

    def worker(video: str) -> None:
        nonlocal inside_count, max_inside, entries
        with opencv_rtsps_tls_env(video):
            with counter_lock:
                inside_count += 1
                max_inside = max(max_inside, inside_count)
                entries += 1
            # Hold long enough that a non-serialized second enter would
            # overlap and bump max_inside above 1.
            time.sleep(0.05)
            with counter_lock:
                inside_count -= 1

    try:
        first = threading.Thread(target=worker, args=("rtsps://cam-a/stream",))
        second = threading.Thread(target=worker, args=("rtsps://cam-b/stream",))
        first.start()
        second.start()
        first.join(timeout=5)
        second.join(timeout=5)

        assert not first.is_alive(), "worker deadlocked inside opencv_rtsps_tls_env"
        assert not second.is_alive(), "worker deadlocked inside opencv_rtsps_tls_env"
        assert entries == 2
        assert max_inside == 1, "opencv_rtsps_tls_env must serialize concurrent holders"
        assert OPENCV_FFMPEG_CAPTURE_OPTIONS_ENV_VAR not in os.environ
    finally:
        if previous_flags is None:
            os.environ.pop(RTSP_TLS_VALIDATION_FLAGS_ENV_VAR, None)
        else:
            os.environ[RTSP_TLS_VALIDATION_FLAGS_ENV_VAR] = previous_flags
=======
    # The context manager holds its lock across the yield (set -> open ->
    # restore), so no cross-thread rendezvous may happen INSIDE the `with`
    # block - a barrier there deadlocks. Serialization is asserted instead by
    # checking the two workers were never inside the context simultaneously.
    os.environ.pop(OPENCV_FFMPEG_CAPTURE_OPTIONS_ENV_VAR, None)
    observed: list[str] = []
    inside: list[str] = []
    overlaps: list[int] = []

    def worker(video: str) -> None:
        with opencv_rtsps_tls_env(video):
            inside.append(video)
            overlaps.append(len(inside))
            observed.append(os.environ[OPENCV_FFMPEG_CAPTURE_OPTIONS_ENV_VAR])
            time.sleep(0.05)
            inside.remove(video)

    first = threading.Thread(target=worker, args=("rtsps://first/stream",))
    second = threading.Thread(target=worker, args=("rtsps://second/stream",))
    first.start()
    second.start()
    first.join()
    second.join()

    assert max(overlaps) == 1, "opens must be serialized, never concurrent"
    assert len(observed) == 2
    assert all("rtsp_transport;tcp" in value for value in observed)
    assert OPENCV_FFMPEG_CAPTURE_OPTIONS_ENV_VAR not in os.environ
>>>>>>> main
