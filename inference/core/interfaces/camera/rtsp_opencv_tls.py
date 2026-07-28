"""OpenCV/FFmpeg RTSPS TLS option builder (ENT-1544 B2)."""

from __future__ import annotations

import os
from typing import Optional

from inference.core.interfaces.camera.rtsp_tls import (
    GST_SSL_CA_CERTIFICATE_ENV_VAR,
    RTSP_TLS_VALIDATION_FLAGS_ENV_VAR,
    SSL_CERT_FILE_ENV_VAR,
    is_rtsps_url,
)


def build_opencv_ffmpeg_capture_options(video: str) -> Optional[str]:
    """Build OPENCV_FFMPEG_CAPTURE_OPTIONS for RTSPS sources."""
    if not is_rtsps_url(video):
        return None

    parts = ["rtsp_transport;tcp"]
    ca_path = os.getenv(GST_SSL_CA_CERTIFICATE_ENV_VAR) or os.getenv(
        SSL_CERT_FILE_ENV_VAR
    )
    if ca_path:
        parts.append(f"cafile;{ca_path}")

    raw_flags = os.getenv(RTSP_TLS_VALIDATION_FLAGS_ENV_VAR)
    if raw_flags is not None and str(raw_flags).strip() == "0":
        parts.append("tls_verify;0")

    return "|".join(parts) if parts else None


def apply_opencv_rtsps_tls_env(video: str) -> Optional[str]:
    """Set OPENCV_FFMPEG_CAPTURE_OPTIONS when needed; returns previous value."""
    options = build_opencv_ffmpeg_capture_options(video)
    if options is None:
        return os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
    previous = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = options
    return previous
