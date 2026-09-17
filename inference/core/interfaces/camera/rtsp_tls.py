"""RTSPS TLS helpers shared by GStreamer and OpenCV ingest paths (ENT-1544 B1/B2)."""

from __future__ import annotations

import os

RTSP_TLS_VALIDATION_FLAGS_ENV_VAR = "ROBOFLOW_RTSP_TLS_VALIDATION_FLAGS"
GST_SSL_CA_CERTIFICATE_ENV_VAR = "GST_SSL_CA_CERTIFICATE"


def is_rtsps_url(url: str) -> bool:
    return isinstance(url, str) and url.lower().startswith("rtsps://")


def is_rtsp_url(url: str) -> bool:
    return isinstance(url, str) and url.lower().startswith(("rtsp://", "rtsps://"))


def rtsp_tls_validation_flags_gstreamer_suffix() -> str:
    """Return an rtspsrc tls-validation-flags suffix when env requests it.

    ``0`` disables certificate validation for cameras with private/self-signed
    certificates. Keeping this unset avoids weakening RTSPS validation.
    """
    raw = os.getenv(RTSP_TLS_VALIDATION_FLAGS_ENV_VAR)
    if raw is None or not str(raw).strip():
        return ""
    try:
        flags = int(str(raw).strip())
    except ValueError as error:
        raise ValueError(
            f"{RTSP_TLS_VALIDATION_FLAGS_ENV_VAR} must be a non-negative integer"
        ) from error
    if flags < 0:
        raise ValueError(
            f"{RTSP_TLS_VALIDATION_FLAGS_ENV_VAR} must be a non-negative integer"
        )
    return f" tls-validation-flags={flags}"
