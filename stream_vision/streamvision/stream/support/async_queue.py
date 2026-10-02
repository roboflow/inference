"""A queue shared by synchronous threads and an asyncio event loop.

Copy of `inference.core.utils.async_utils.Queue`. Both sides use ONE
`asyncio.Queue` living on one loop, so a blocked consumer on either side is
woken by a producer on the other.
"""

import asyncio
from threading import Thread
from typing import Optional


async def create_async_queue(maxsize: int = 0) -> asyncio.Queue:
    return asyncio.Queue(maxsize=maxsize)


class Queue:
    def __init__(
        self, loop: Optional[asyncio.AbstractEventLoop] = None, maxsize: int = 0
    ):
        current_loop: Optional[asyncio.AbstractEventLoop] = None
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        self._loop = loop

        if (
            loop is None and current_loop is None
        ):  # Running in sync code with no running loop
            self._loop = asyncio.new_event_loop()
            self._thread: Thread = Thread(target=self._loop.run_forever, daemon=True)
            self._thread.start()
            self._queue = asyncio.run_coroutine_threadsafe(
                create_async_queue(maxsize=maxsize), self._loop
            ).result()
        elif (
            loop is not None
            and loop is current_loop
            or loop is None
            and current_loop is not None
        ):
            self._loop = current_loop
            self._queue = asyncio.Queue(maxsize=maxsize)
        else:
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
