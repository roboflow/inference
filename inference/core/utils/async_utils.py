import asyncio
import concurrent.futures
import contextlib
from threading import Lock


@contextlib.asynccontextmanager
async def async_lock(lock: Lock, pool: concurrent.futures.ThreadPoolExecutor):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(pool, lock.acquire)
    try:
        yield  # the lock is held
    finally:
        lock.release()
