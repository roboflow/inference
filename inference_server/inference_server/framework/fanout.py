"""Bounded per-request fan-out.

A batch request produces one gateway call per image, and every one of those
lands on the model manager's worker pool. Awaiting them all at once lets a
single request queue its whole batch ahead of every other client's work.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Optional

from inference_server import configuration


async def gather_bounded(
    *coros: Awaitable[Any], limit: Optional[int] = None
) -> list[Any]:
    """asyncio.gather with at most ``limit`` awaitables running concurrently."""
    if limit is None:
        limit = configuration.MAX_CONCURRENT_IMAGES_PER_REQUEST
    if limit <= 0 or len(coros) <= limit:
        return await asyncio.gather(*coros)

    semaphore = asyncio.Semaphore(limit)

    async def _run(coro: Awaitable[Any]) -> Any:
        async with semaphore:
            return await coro

    return await asyncio.gather(*(_run(coro) for coro in coros))
