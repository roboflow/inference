"""OpenCV/FFmpeg RTSPS TLS option builder (ENT-1544 B3).

Environment variables (also set by rfdm on Roboflow Edge devices):

* ``ROBOFLOW_RTSP_TLS_VALIDATION_FLAGS`` — ``127`` strict (default when unset),
  ``126`` or ``0`` to disable certificate verification (self-signed / custom CA).
* ``GST_SSL_CA_CERTIFICATE`` or ``SSL_CERT_FILE`` — PEM bundle for ``cafile``.

Self-hosted / OSS deployments without rfdm must set these explicitly on the
inference container. Unmanaged ``rtsps://`` sources with self-signed certs need
``ROBOFLOW_RTSP_TLS_VALIDATION_FLAGS=126`` or the open will fail after this wiring.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Dict, Iterator, Optional, Union

from inference.core.interfaces.camera.rtsp_tls import (
    GST_SSL_CA_CERTIFICATE_ENV_VAR,
    RTSP_TLS_VALIDATION_FLAGS_ENV_VAR,
    is_rtsps_url,
)

OPENCV_FFMPEG_CAPTURE_OPTIONS_ENV_VAR = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
SSL_CERT_FILE_ENV_VAR = "SSL_CERT_FILE"

_opencv_rtsps_tls_lock = threading.Lock()


def _parse_opencv_ffmpeg_capture_options(options: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for part in options.split("|"):
        if ";" not in part:
            continue
        key, value = part.split(";", 1)
        parsed[key] = value
    return parsed


def _format_opencv_ffmpeg_capture_options(options: Dict[str, str]) -> str:
    return "|".join(f"{key};{value}" for key, value in options.items())


def _build_rtsps_tls_option_overrides() -> Dict[str, str]:
    overrides: Dict[str, str] = {"rtsp_transport": "tcp"}

    ca_path = os.getenv(GST_SSL_CA_CERTIFICATE_ENV_VAR) or os.getenv(
        SSL_CERT_FILE_ENV_VAR
    )
    if ca_path:
        overrides["cafile"] = ca_path

    raw_flags = os.getenv(RTSP_TLS_VALIDATION_FLAGS_ENV_VAR)
    if raw_flags is not None:
        stripped = str(raw_flags).strip()
        # rfdm maps allow_self_signed to 126 (unknown-CA only). FFmpeg only
        # supports full verify on/off, so treat both 0 and 126 as tls_verify;0.
        if stripped in {"0", "126"}:
            overrides["tls_verify"] = "0"
        else:
            overrides["tls_verify"] = "1"
    else:
        # FFmpeg defaults tls_verify to off; enable verification for strict RTSPS.
        overrides["tls_verify"] = "1"

    return overrides


def merge_opencv_ffmpeg_capture_options(
    existing: Optional[str], rtsps_overrides: Dict[str, str]
) -> str:
    merged = _parse_opencv_ffmpeg_capture_options(existing or "")
    for key, value in rtsps_overrides.items():
        merged[key] = value
    return _format_opencv_ffmpeg_capture_options(merged)


def build_opencv_ffmpeg_capture_options(video: Union[str, int]) -> Optional[str]:
    """Build OPENCV_FFMPEG_CAPTURE_OPTIONS for RTSPS sources."""
    if not is_rtsps_url(video):
        return None

    overrides = _build_rtsps_tls_option_overrides()
    return merge_opencv_ffmpeg_capture_options(
        os.environ.get(OPENCV_FFMPEG_CAPTURE_OPTIONS_ENV_VAR),
        overrides,
    )


@contextmanager
def opencv_rtsps_tls_env(video: Union[str, int]) -> Iterator[None]:
    """Hold OPENCV_FFMPEG_CAPTURE_OPTIONS for one VideoCapture open.

    OpenCV reads this env var at capture-open time. The lock spans set →
    VideoCapture open → restore so concurrent RTSPS sources cannot clobber each
    other's TLS options. Isolation is RTSPS-vs-RTSPS only; concurrent non-RTSPS
    opens do not take the lock and may briefly inherit these options.
    """
    options = build_opencv_ffmpeg_capture_options(video)
    if options is None:
        yield
        return

    with _opencv_rtsps_tls_lock:
        previous = os.environ.get(OPENCV_FFMPEG_CAPTURE_OPTIONS_ENV_VAR)
        os.environ[OPENCV_FFMPEG_CAPTURE_OPTIONS_ENV_VAR] = options
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop(OPENCV_FFMPEG_CAPTURE_OPTIONS_ENV_VAR, None)
            else:
                os.environ[OPENCV_FFMPEG_CAPTURE_OPTIONS_ENV_VAR] = previous
