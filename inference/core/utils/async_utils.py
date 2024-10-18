import asyncio
import concurrent.futures
import contextlib
from threading import Lock
from typing import Optional, Union


@contextlib.asynccontextmanager
async def async_lock(
    lock: Union[Lock],
    pool: concurrent.futures.ThreadPoolExecutor,
    loop: Optional[asyncio.AbstractEventLoop] = None,
):
    if not loop:
        loop = asyncio.get_event_loop()
    await loop.run_in_executor(pool, lock.acquire)
    try:
        yield  # the lock is held
    finally:
        lock.release()


async def create_async_queue() -> asyncio.Queue:
    return asyncio.Queue()


class Queue:
    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        self._loop = loop
        if not self._loop:
            self._loop = asyncio.get_running_loop()
        self._queue = asyncio.run_coroutine_threadsafe(
            create_async_queue(), self._loop
        ).result()

    def sync_put_nowait(self, item):
        self._loop.call_soon(self._queue.put_nowait, item)

    def sync_put(self, item):
        asyncio.run_coroutine_threadsafe(self._queue.put(item), self._loop).result()

    def sync_get(self, timeout: float):
        return asyncio.run_coroutine_threadsafe(
            asyncio.wait_for(self._queue.get(), timeout), self._loop
        ).result()

    def sync_empty(self):
        return self._queue.empty()

    def async_put_nowait(self, item):
        self._queue.put_nowait(item)

    async def async_put(self, item):
        await self._queue.put(item)

    async def async_get(self):
        return await self._queue.get()

    async def async_empty(self):
        return self._queue.empty()
