"""Structured v2 error responses."""

from __future__ import annotations

import json
from typing import Optional

from fastapi import Response


def error_response(
    status_code: int,
    error_code: str,
    description: str,
    follow_up: Optional[str] = None,
    help_url: Optional[str] = None,
    headers: Optional[dict] = None,
) -> Response:
    body: dict = {
        "error_code": error_code,
        "description": description,
    }
    if follow_up:
        body["actionable_follow_up"] = follow_up
    if help_url:
        body["help_url"] = help_url
    return Response(
        status_code=status_code,
        content=json.dumps(body).encode(),
        media_type="application/json",
        headers=headers,
    )


class ServerBusyError(RuntimeError):
    """Capacity exhausted (slot pool full, alloc timed out, per-model cap).

    Retryable — mapped to 503 + Retry-After. Raised by type so hard failures
    (e.g. CUDA OOM messages containing the word 'allocate') can never be
    misclassified as busy by substring matching.
    """


class PayloadTooLargeError(ValueError):
    """Input exceeds the SHM slot capacity — mapped to 413."""


class AuthBackendUnavailable(Exception):
    """Roboflow API unreachable during key validation.

    A transport failure is not a key rejection: mapped to 503 (retryable),
    never cached, never surfaced as 403 'Invalid API key'.
    """
