"""API key validation against Roboflow API with in-memory TTL cache.

Calls ``GET {API_BASE_URL}/?api_key=...&nocache=true`` to verify the key
and retrieve the workspace ID.

Environment variables::

    API_BASE_URL        Base URL (default: https://api.roboflow.com).
                            Set to https://api.roboflow.one for staging.
    AUTH_CACHE_TTL_S        Cache TTL for successful auth (default: 3600).
    AUTH_CACHE_FAIL_TTL_S   Cache TTL for failed auth (default: 60).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.roboflow.com")
_CACHE_TTL_S = int(os.environ.get("AUTH_CACHE_TTL_S", "3600"))
_CACHE_FAIL_TTL_S = int(os.environ.get("AUTH_CACHE_FAIL_TTL_S", "60"))
_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=10)


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    expires_at: float
    valid: bool
    workspace_id: Optional[str] = None


_cache: dict[str, _CacheEntry] = {}
_session: Optional[aiohttp.ClientSession] = None


def _get_session() -> aiohttp.ClientSession:
    """Reuse a single aiohttp session per worker process.
    Safe: only called from async code on a single event loop thread.
    """
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession(timeout=_REQUEST_TIMEOUT)
    return _session


async def validate_api_key(api_key: str) -> tuple[bool, Optional[str]]:
    """Validate api_key against Roboflow API.

    Returns:
        (True, workspace_id) on success.
        (False, None) on failure (bad key, network error, etc.).

    Results are cached in-memory with TTL.
    """
    now = time.monotonic()

    entry = _cache.get(api_key)
    if entry is not None and entry.expires_at > now:
        return entry.valid, entry.workspace_id

    try:
        session = _get_session()
        async with session.get(
            f"{API_BASE_URL}/",
            params={"api_key": api_key, "nocache": "true"},
        ) as resp:
            if resp.status != 200:
                _cache[api_key] = _CacheEntry(
                    expires_at=now + _CACHE_FAIL_TTL_S,
                    valid=False,
                )
                return False, None

            data = await resp.json()
            workspace_id = data.get("workspace")
            if not workspace_id:
                _cache[api_key] = _CacheEntry(
                    expires_at=now + _CACHE_FAIL_TTL_S,
                    valid=False,
                )
                return False, None

            _cache[api_key] = _CacheEntry(
                expires_at=now + _CACHE_TTL_S,
                valid=True,
                workspace_id=workspace_id,
            )
            return True, workspace_id

    except Exception:
        logger.warning("Auth validation failed (network error)", exc_info=True)
        _cache[api_key] = _CacheEntry(
            expires_at=now + _CACHE_FAIL_TTL_S,
            valid=False,
        )
        return False, None
