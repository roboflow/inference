"""Fetch an image from an HTTP(S) URL with size/time limits."""

from __future__ import annotations

import asyncio
from typing import Optional

import aiohttp
from fastapi import Response

from inference_server.errors import error_response

URL_FETCH_TIMEOUT_S = 10
URL_FETCH_MAX_BYTES = 50 * 1024 * 1024  # 50 MB


async def fetch_image_from_url(
    url: str,
) -> tuple[Optional[bytes], Optional[Response]]:
    """Fetch image bytes from a URL. Returns (bytes, None) or (None, error)."""
    if not url.startswith(("http://", "https://")):
        return None, error_response(
            400, "INVALID_URL", "image URL must start with http:// or https://"
        )
    timeout = aiohttp.ClientTimeout(total=URL_FETCH_TIMEOUT_S)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None, error_response(
                        502,
                        "URL_FETCH_FAILED",
                        f"fetching image URL returned status {resp.status}",
                    )
                content_length = resp.content_length or 0
                if content_length > URL_FETCH_MAX_BYTES:
                    return None, error_response(
                        413,
                        "URL_IMAGE_TOO_LARGE",
                        f"image at URL exceeds {URL_FETCH_MAX_BYTES // (1024*1024)}MB limit",
                    )
                data = await resp.read()
                if len(data) > URL_FETCH_MAX_BYTES:
                    return None, error_response(
                        413,
                        "URL_IMAGE_TOO_LARGE",
                        f"image at URL exceeds {URL_FETCH_MAX_BYTES // (1024*1024)}MB limit",
                    )
                return data, None
    except asyncio.TimeoutError:
        return None, error_response(
            504,
            "URL_FETCH_TIMEOUT",
            f"fetching image URL timed out after {URL_FETCH_TIMEOUT_S}s",
        )
    except aiohttp.ClientError as exc:
        return None, error_response(
            502, "URL_FETCH_FAILED", f"fetching image URL failed: {exc}"
        )
