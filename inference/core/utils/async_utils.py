import asyncio
import concurrent.futures
import contextlib
from multiprocessing.synchronize import Lock as LockType


@contextlib.asynccontextmanager
async def async_lock(lock: LockType, pool: concurrent.futures.ThreadPoolExecutor):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(pool, lock.acquire)
    try:
        yield  # the lock is held
    finally:
        lock.release()
