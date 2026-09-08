"""Validate the dedicated stream-administration token before starting services."""

import re


def validate_stream_api_key(token: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_-]{32,256}", token):
        raise RuntimeError(
            "ENABLE_STREAM_API requires STREAM_API_KEY to be a URL-safe token of "
            "32-256 characters. Generate a random token with secrets.token_urlsafe(32)."
        )
