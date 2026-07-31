"""Classify producer failures into structured stream error codes."""

from __future__ import annotations

import re

from inference.core.interfaces.camera.exceptions import SourceConnectionError
from inference.core.interfaces.camera.stream_error_codes import StreamErrorCode

_AUTH_STATUS_PATTERN = re.compile(r"\b401\b|\b403\b")
_NOT_FOUND_STATUS_PATTERN = re.compile(r"\b404\b")


def classify_stream_error_message(message: str) -> StreamErrorCode:
    lowered = (message or "").lower()
    if (
        _AUTH_STATUS_PATTERN.search(lowered)
        or "unauthorized" in lowered
        or "authentication" in lowered
        or "forbidden" in lowered
    ):
        return StreamErrorCode.STREAM_AUTH_FAILED
    if (
        ("handshake" in lowered and ("tls" in lowered or "ssl" in lowered))
        or "ssl handshake failed" in lowered
        or "tls handshake failed" in lowered
    ):
        return StreamErrorCode.STREAM_TLS_HANDSHAKE
    if any(
        token in lowered
        for token in ("certificate verify failed", "x509", "tls certificate")
    ):
        return StreamErrorCode.STREAM_TLS_CERTIFICATE
    if "timed out" in lowered or "timeout" in lowered:
        return StreamErrorCode.STREAM_TIMEOUT
    if _NOT_FOUND_STATUS_PATTERN.search(lowered) or "not found" in lowered:
        return StreamErrorCode.STREAM_NOT_FOUND
    if "codec" in lowered and "unsupported" in lowered:
        return StreamErrorCode.STREAM_CODEC_UNSUPPORTED
    return StreamErrorCode.STREAM_CONNECTION_FAILED


def wrap_source_connection_error(
    message: str, source_reference: str = ""
) -> SourceConnectionError:
    code = classify_stream_error_message(message)
    error = SourceConnectionError(message)
    setattr(error, "code", code)
    setattr(error, "source_reference", source_reference)
    return error
