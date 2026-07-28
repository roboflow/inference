"""RTSPS TLS helpers shared by GStreamer and OpenCV ingest paths (ENT-1544 B1/B2)."""

from __future__ import annotations

import os

RTSP_TLS_VALIDATION_FLAGS_ENV_VAR = "ROBOFLOW_RTSP_TLS_VALIDATION_FLAGS"
GST_SSL_CA_CERTIFICATE_ENV_VAR = "GST_SSL_CA_CERTIFICATE"
SSL_CERT_FILE_ENV_VAR = "SSL_CERT_FILE"


def is_rtsps_url(url: str) -> bool:
    return isinstance(url, str) and url.lower().startswith("rtsps://")


def rtsp_tls_validation_flags_gstreamer_suffix() -> str:
    raw = os.getenv(RTSP_TLS_VALIDATION_FLAGS_ENV_VAR)
    if raw is None or not str(raw).strip():
        return ""
    flags = int(str(raw).strip())
    if flags < 0:
        raise ValueError(
            f"{RTSP_TLS_VALIDATION_FLAGS_ENV_VAR} must be a non-negative integer"
        )
    return f" tls-validation-flags={flags}"
