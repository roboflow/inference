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
