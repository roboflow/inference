"""Classify producer failures into structured stream error codes."""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from typing import Iterator

from inference.core.interfaces.camera.exceptions import SourceConnectionError
from inference.core.interfaces.camera.source_reference_sanitizer import (
    redact_credentials_in_text,
)
from inference.core.interfaces.camera.stream_error_codes import StreamErrorCode

_AUTH_STATUS_PATTERN = re.compile(r"\b401\b|\b403\b")
_NOT_FOUND_STATUS_PATTERN = re.compile(r"\b404\b")
_STREAM_OPEN_ERROR_HINTS = (
    "error",
    "failed",
    "unauthorized",
    "forbidden",
    "not found",
    "timeout",
    "timed out",
    "handshake",
    "certificate",
    "401",
    "403",
    "404",
)


@contextmanager
def capture_process_stderr() -> Iterator[list[str]]:
    """Capture OS stderr (fd 2) for native backends such as FFmpeg/GStreamer."""
    read_fd, write_fd = os.pipe()
    saved_stderr_fd = os.dup(2)
    captured_chunks: list[str] = []
    try:
        os.dup2(write_fd, 2)
        yield captured_chunks
    finally:
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stderr_fd)
        os.close(write_fd)
        captured = os.read(read_fd, 65536)
        os.close(read_fd)
        if captured:
            captured_chunks.append(captured.decode("utf-8", errors="replace"))


def extract_stream_open_error(stderr: str) -> str:
    if not stderr:
        return ""
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    for line in reversed(lines):
        lowered = line.lower()
        if any(hint in lowered for hint in _STREAM_OPEN_ERROR_HINTS):
            return line
    return lines[-1] if lines else ""


def build_source_connection_error_message(
    source_reference: str, underlying_error: str = ""
) -> str:
    summary = f"Cannot connect to video source under reference: {source_reference}"
    detail = redact_credentials_in_text((underlying_error or "").strip())
    if detail:
        return f"{detail}: {summary}"
    return summary


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
