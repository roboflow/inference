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


async def create_async_queue(maxsize: int = 0) -> asyncio.Queue:
    return asyncio.Queue(maxsize=maxsize)


class Queue:
    def __init__(
        self, loop: Optional[asyncio.AbstractEventLoop] = None, maxsize: int = 0
    ):
        self._loop = loop
        if not self._loop:
            self._loop = asyncio.get_running_loop()
        self._queue = asyncio.run_coroutine_threadsafe(
            create_async_queue(maxsize=maxsize), self._loop
        ).result()

    def sync_put_nowait(self, item):
        self._queue.put_nowait(item)

    def sync_put(self, item):
        asyncio.run_coroutine_threadsafe(self._queue.put(item), self._loop).result()

    def sync_get_nowait(self):
        return self._queue.get_nowait()

    def sync_get(self, timeout: Optional[float] = None):
        return asyncio.run_coroutine_threadsafe(
            asyncio.wait_for(self._queue.get(), timeout), self._loop
        ).result()

    def sync_empty(self):
        return self._queue.empty()

    def sync_full(self):
        return self._queue.full()

    async def async_put_nowait(self, item):
        self._queue.put_nowait(item)

    async def async_put(self, item):
        await self._queue.put(item)

    async def async_get_nowait(self):
        return self._queue.get_nowait()

    async def async_get(self, timeout: Optional[float] = None):
        return await asyncio.wait_for(self._queue.get(), timeout)

    async def async_empty(self):
        return self._queue.empty()

    async def async_full(self):
        return self._queue.full()
