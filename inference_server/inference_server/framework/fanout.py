"""Bounded per-request fan-out.

A batch request produces one gateway call per image, and every one of those
lands on the model manager's worker pool. Awaiting them all at once lets a
single request queue its whole batch ahead of every other client's work.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Awaitable, Optional

from inference_server import configuration


async def gather_bounded(
    *coros: Awaitable[Any], limit: Optional[int] = None
) -> list[Any]:
    """asyncio.gather with at most ``limit`` awaitables running concurrently.

    The first failure, or cancellation of the caller, cancels the awaitables
    still pending or running and waits for them to settle before re-raising.
    """
    if not coros:
        return []

    if limit is None:
        limit = configuration.MAX_CONCURRENT_IMAGES_PER_REQUEST
    if limit <= 0 or len(coros) <= limit:
        tasks = [asyncio.ensure_future(coro) for coro in coros]
    else:
        semaphore = asyncio.Semaphore(limit)

        async def _run(coro: Awaitable[Any]) -> Any:
            async with semaphore:
                return await coro

        tasks = [asyncio.ensure_future(_run(coro)) for coro in coros]

    failures: list[BaseException] = []
    first_outcome = asyncio.get_running_loop().create_future()
    remaining = len(tasks)

    def _record_outcome(task: asyncio.Future) -> None:
        nonlocal remaining
        remaining -= 1
        if task.cancelled():
            failure = asyncio.CancelledError()
        else:
            failure = task.exception()
        if failure is not None and not failures:
            failures.append(failure)
        if (failure is not None or remaining == 0) and not first_outcome.done():
            first_outcome.set_result(None)

    for task in tasks:
        task.add_done_callback(_record_outcome)
    try:
        await first_outcome
    finally:
        await _settle(tasks, coros)
    if failures:
        raise failures[0]

    results = [task.result() for task in tasks]

    return results


async def _settle(tasks: list[asyncio.Future], coros: tuple) -> None:
    for task in tasks:
        task.cancel()

    interrupted = False
    try:
        pending = [task for task in tasks if not task.done()]
        while pending:
            try:
                await asyncio.wait(pending)
            except asyncio.CancelledError:
                interrupted = True
            pending = [task for task in tasks if not task.done()]
    finally:
        for coro in coros:
            if (
                inspect.iscoroutine(coro)
                and inspect.getcoroutinestate(coro) == inspect.CORO_CREATED
            ):
                coro.close()

    if interrupted:
        raise asyncio.CancelledError()
