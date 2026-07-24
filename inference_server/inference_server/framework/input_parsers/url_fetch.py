"""Fetch an image from an HTTP(S) URL with size/time limits."""

from __future__ import annotations

import asyncio
from typing import Optional

import aiohttp
from fastapi import Response

from inference_server import configuration
from inference_server.errors import error_response

URL_FETCH_TIMEOUT_S = 10
URL_FETCH_MAX_BYTES = 50 * 1024 * 1024  # 50 MB
_CHUNK_BYTES = 64 * 1024


async def fetch_images_from_urls(
    urls: list[str],
) -> tuple[Optional[list[bytes]], Optional[Response]]:
    """Fetch all URLs concurrently under a shared aggregate byte budget.

    Returns (images, None) or (None, error).
    """
    if len(urls) > configuration.MAX_IMAGE_URLS:
        return None, error_response(
            400,
            "TOO_MANY_IMAGES",
            f"at most {configuration.MAX_IMAGE_URLS} image URLs per request",
        )
    budget = {"left": configuration.MAX_BODY_BYTES}
    results = await asyncio.gather(
        *(fetch_image_from_url(u, _budget=budget) for u in urls)
    )
    images: list[bytes] = []
    for data, err in results:
        if err is not None:
            return None, err
        images.append(data)
    return images, None


async def fetch_image_from_url(
    url: str,
    _budget: Optional[dict] = None,
) -> tuple[Optional[bytes], Optional[Response]]:
    """Fetch image bytes from a URL. Returns (bytes, None) or (None, error).

    The body is read in chunks and both the per-fetch cap and the shared
    ``_budget`` are enforced while streaming, so a response without
    Content-Length cannot buffer unbounded before the check.
    """
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
                data = bytearray()
                async for chunk in resp.content.iter_chunked(_CHUNK_BYTES):
                    data.extend(chunk)
                    if len(data) > URL_FETCH_MAX_BYTES:
                        return None, error_response(
                            413,
                            "URL_IMAGE_TOO_LARGE",
                            f"image at URL exceeds {URL_FETCH_MAX_BYTES // (1024*1024)}MB limit",
                        )
                    if _budget is not None:
                        _budget["left"] -= len(chunk)
                        if _budget["left"] < 0:
                            return None, error_response(
                                413,
                                "PAYLOAD_TOO_LARGE",
                                "combined size of images at URLs exceeds "
                                f"{configuration.MAX_BODY_BYTES} byte limit",
                            )
                return bytes(data), None
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
