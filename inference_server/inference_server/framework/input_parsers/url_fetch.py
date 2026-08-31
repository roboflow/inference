"""Fetch an image from an HTTP(S) URL with destination, size and time limits."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
from typing import Optional
from urllib.parse import urljoin, urlsplit

import aiohttp
from aiohttp.abc import AbstractResolver
from fastapi import Response

from inference_server import configuration
from inference_server.errors import error_response
from inference_server.framework.input_parsers.image_limits import too_many_images

logger = logging.getLogger(__name__)

URL_FETCH_TIMEOUT_S = 10
URL_FETCH_MAX_BYTES = 50 * 1024 * 1024  # 50 MB
_CHUNK_BYTES = 64 * 1024
_REDIRECT_STATUSES = frozenset((301, 302, 303, 307, 308))


async def fetch_images_from_urls(
    urls: list[str],
) -> tuple[Optional[list[bytes]], Optional[Response]]:
    """Fetch all URLs concurrently under a shared aggregate byte budget.

    Returns (images, None) or (None, error).
    """
    limit_error = too_many_images(len(urls))
    if limit_error is not None:
        return None, limit_error
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


async def resolve_host(host: str) -> list[str]:
    """Resolve a hostname to its literal addresses (monkeypatched in tests)."""
    infos = await asyncio.get_running_loop().getaddrinfo(
        host, None, type=socket.SOCK_STREAM
    )
    return [info[4][0] for info in infos]


def _is_global_address(literal: str) -> bool:
    try:
        address = ipaddress.ip_address(literal)
    except ValueError:
        return False
    if (
        address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    ):
        return False
    # IPv4-mapped / 6to4 / Teredo hide a v4 address inside a v6 one.
    mapped = getattr(address, "ipv4_mapped", None) or getattr(
        address, "sixtofour", None
    )
    if mapped is not None and not _is_global_address(str(mapped)):
        return False
    return bool(address.is_global)


class _PinnedResolver(AbstractResolver):
    """Resolve a host only to the addresses this fetch already validated.

    aiohttp resolves again when it connects, so a hostile DNS server can answer
    with a global address for the check and an internal one for the connection.
    Serving the recorded answer closes that window; a host nothing validated
    fails to resolve.
    """

    def __init__(self, pinned: dict[str, list[str]]) -> None:
        self._pinned = pinned

    async def resolve(
        self, host: str, port: int = 0, family: int = socket.AF_INET
    ) -> list[dict]:
        addresses = self._pinned.get(host.lower())
        if not addresses:
            raise OSError(f"host {host!r} was not validated for this fetch")
        return [
            {
                "hostname": host,
                "host": literal,
                "port": port,
                "family": socket.AF_INET6 if ":" in literal else socket.AF_INET,
                "proto": 0,
                "flags": 0,
            }
            for literal in addresses
        ]

    async def close(self) -> None:
        return None


def _forbidden() -> Response:
    return error_response(403, "URL_DESTINATION_FORBIDDEN", "image URL destination")


async def _ensure_destination_allowed(
    url: str,
) -> tuple[Optional[tuple[str, list[str]]], Optional[Response]]:
    """Validate a URL's scheme and destination.

    Returns ``((host, addresses), None)`` — the addresses the connection must
    be pinned to — or ``(None, error)``.
    """
    if not url.startswith(("http://", "https://")):
        return None, error_response(
            400, "INVALID_URL", "image URL must start with http:// or https://"
        )
    host = (urlsplit(url).hostname or "").lower()
    if not host:
        return None, error_response(400, "INVALID_URL", "image URL has no host")

    blocked = configuration.BLACKLISTED_DESTINATIONS_FOR_URL_INPUT
    if blocked and host in blocked:
        return None, _forbidden()

    allowed = configuration.WHITELISTED_DESTINATIONS_FOR_URL_INPUT
    if allowed is not None and host not in allowed:
        return None, _forbidden()
    # An operator allowlist entry and the non-global opt-in skip the address
    # check — never the pinning.
    skip_address_check = (
        allowed is not None or configuration.ALLOW_URL_TO_NON_GLOBAL_ADDRESSES
    )

    try:
        addresses = await resolve_host(host)
    except (socket.gaierror, OSError) as exc:
        logger.warning("Resolving image URL host failed: %s", exc)
        addresses = []
    if not addresses:
        return None, error_response(
            502, "URL_FETCH_FAILED", "fetching image URL failed"
        )
    if not skip_address_check and not all(_is_global_address(a) for a in addresses):
        # Loopback / RFC1918 / link-local (169.254.169.254) / reserved.
        return None, _forbidden()
    return (host, addresses), None


async def fetch_image_from_url(
    url: str,
    _budget: Optional[dict] = None,
) -> tuple[Optional[bytes], Optional[Response]]:
    """Fetch image bytes from a URL. Returns (bytes, None) or (None, error).

    Redirects are followed manually so every hop passes the same destination
    check; the body is read in chunks and both the per-fetch cap and the shared
    ``_budget`` are enforced while streaming, so a response without
    Content-Length cannot buffer unbounded before the check.

    Every hop's connection is pinned to the addresses its check validated, so
    the connect cannot land somewhere the check never saw.
    """
    timeout = aiohttp.ClientTimeout(total=URL_FETCH_TIMEOUT_S)
    current = url
    pinned: dict[str, list[str]] = {}
    try:
        connector = aiohttp.TCPConnector(
            resolver=_PinnedResolver(pinned), use_dns_cache=False
        )
        async with aiohttp.ClientSession(
            timeout=timeout, connector=connector
        ) as session:
            for _ in range(configuration.MAX_IMAGE_URL_REDIRECTS + 1):
                validated, destination_error = await _ensure_destination_allowed(
                    current
                )
                if destination_error is not None:
                    return None, destination_error
                host, addresses = validated
                pinned[host] = addresses
                async with session.get(current, allow_redirects=False) as resp:
                    if resp.status in _REDIRECT_STATUSES:
                        location = resp.headers.get("location")
                        if not location:
                            return None, error_response(
                                502,
                                "URL_FETCH_FAILED",
                                "fetching image URL failed",
                            )
                        current = urljoin(current, location)
                        continue
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
            return None, error_response(
                502, "URL_FETCH_FAILED", "too many redirects fetching image URL"
            )
    except asyncio.TimeoutError:
        return None, error_response(
            504,
            "URL_FETCH_TIMEOUT",
            f"fetching image URL timed out after {URL_FETCH_TIMEOUT_S}s",
        )
    except aiohttp.ClientError as exc:
        logger.warning("Fetching image URL failed: %s", exc)
        return None, error_response(
            502, "URL_FETCH_FAILED", f"fetching image URL failed"
        )
