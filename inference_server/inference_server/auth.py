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

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp

from inference_server import configuration
from inference_server.errors import AuthBackendUnavailable

logger = logging.getLogger(__name__)


def extract_bearer(value: str) -> str:
    """Case-insensitive Bearer-token extraction (RFC 7235: scheme is
    case-insensitive). Single source for all four call sites."""
    if value[:7].lower() == "bearer ":
        return value[7:].strip()
    return ""


API_BASE_URL = configuration.API_BASE_URL
_CACHE_TTL_S = configuration.AUTH_CACHE_TTL_S
_CACHE_FAIL_TTL_S = configuration.AUTH_CACHE_FAIL_TTL_S
_MAX_CACHE_SIZE = configuration.AUTH_CACHE_MAX_SIZE
_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=10)


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    expires_at: float
    valid: bool
    workspace_id: Optional[str] = None


_cache: dict[str, _CacheEntry] = {}
_inflight: dict[str, asyncio.Task] = {}
_session: Optional[aiohttp.ClientSession] = None


def _key_hash(api_key: str) -> str:
    """SHA-256 of raw key — avoids storing plaintext in memory."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def _get_session() -> aiohttp.ClientSession:
    """Reuse a single aiohttp session per worker process.
    Safe: only called from async code on a single event loop thread.
    """
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession(timeout=_REQUEST_TIMEOUT)
    return _session


def _enforce_cache_limit() -> None:
    """Evict entries when cache exceeds max size to prevent memory DoS.

    Strategy: purge expired entries first; if still over limit, evict the
    oldest-expiring entries (drops short-TTL failures before valid keys).
    Never clears wholesale — a token-spray attack must not flush valid keys.
    """
    if len(_cache) <= _MAX_CACHE_SIZE:
        return
    now = time.monotonic()
    expired_keys = [k for k, v in _cache.items() if v.expires_at <= now]
    for k in expired_keys:
        del _cache[k]
    overflow = len(_cache) - _MAX_CACHE_SIZE
    if overflow > 0:
        oldest = sorted(_cache, key=lambda k: _cache[k].expires_at)[:overflow]
        for k in oldest:
            del _cache[k]


async def validate_api_key(api_key: str) -> tuple[bool, Optional[str]]:
    """Validate api_key against Roboflow API.

    Returns:
        (True, workspace_id) on success.
        (False, None) on rejection (bad key).

    Raises:
        AuthBackendUnavailable: Roboflow API unreachable (transport failure)
            or answering 5xx/429. NOT cached — a brief outage must not
            hard-fail keys for the negative-cache TTL, and must surface as
            503, never 403.

    Results are cached in-memory with TTL. Concurrent misses on the same key
    share a single upstream request (single-flight).
    """
    key_hash = _key_hash(api_key)
    entry = _cache.get(key_hash)
    if entry is not None and entry.expires_at > time.monotonic():
        return entry.valid, entry.workspace_id

    task = _inflight.get(key_hash)
    if task is None:
        task = asyncio.ensure_future(_validate_uncached(api_key, key_hash))
        _inflight[key_hash] = task
        task.add_done_callback(lambda _: _inflight.pop(key_hash, None))
    return await asyncio.shield(task)


async def _validate_uncached(api_key: str, key_hash: str) -> tuple[bool, Optional[str]]:
    try:
        session = _get_session()
        async with session.get(
            f"{API_BASE_URL}/",
            params={"api_key": api_key, "nocache": "true"},
        ) as resp:
            if resp.status >= 500 or resp.status == 429:
                raise AuthBackendUnavailable(
                    f"auth backend returned status {resp.status}"
                )
            if resp.status != 200:
                return _store(key_hash, valid=False)

            data = await resp.json()
            workspace_id = data.get("workspace")
            if not workspace_id:
                return _store(key_hash, valid=False)
            return _store(key_hash, valid=True, workspace_id=workspace_id)

    except AuthBackendUnavailable:
        raise
    except Exception as exc:
        logger.warning("Auth validation failed (network error)", exc_info=True)
        raise AuthBackendUnavailable(str(exc)) from exc


def _store(
    key_hash: str, valid: bool, workspace_id: Optional[str] = None
) -> tuple[bool, Optional[str]]:
    _enforce_cache_limit()
    ttl = _CACHE_TTL_S if valid else _CACHE_FAIL_TTL_S
    _cache[key_hash] = _CacheEntry(
        expires_at=time.monotonic() + ttl,
        valid=valid,
        workspace_id=workspace_id,
    )
    return valid, workspace_id
