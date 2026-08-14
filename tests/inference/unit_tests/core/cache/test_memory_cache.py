import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from inference.core.cache.memory import MemoryCache


def _build_cache() -> MemoryCache:
    """Construct a MemoryCache without starting the background ``_expire`` thread."""
    with patch("inference.core.cache.memory.threading.Thread"):
        return MemoryCache()


def test_acquire_lock_without_expire_keeps_the_lock_cached():
    # given - the default expire=None, as used by the model-loading paths
    cache = _build_cache()

    # when
    lock = cache.acquire_lock("some-key")
    lock.release()

    # then - no expiry is recorded, so the entry is not discarded on the next read
    assert cache.expires.get("some-key") is None
    assert cache.get("some-key") is lock


def test_acquire_lock_without_expire_reuses_one_lock_across_calls():
    # given
    cache = _build_cache()

    # when
    first = cache.acquire_lock("some-key")
    first.release()
    second = cache.acquire_lock("some-key")
    second.release()

    # then - the whole point of a keyed lock: the same key means the same lock
    assert first is second


def test_acquire_lock_with_expire_still_records_the_expiry():
    # given
    cache = _build_cache()

    # when
    lock = cache.acquire_lock("some-key", expire=60)
    lock.release()

    # then - an explicit expiry is honoured, and is in the future
    assert cache.expires["some-key"] > time.time()


def test_concurrent_first_acquisition_shares_one_lock():
    # given - a barrier that holds both callers past their first read, so the
    # two would build separate locks if the lookup-and-create were not guarded.
    #
    # Once it is guarded the second caller cannot reach the barrier while the
    # first holds the guard, so the barrier is expected to time out rather than
    # trip. That timeout is the whole wait in this test, hence a short one.
    cache = _build_cache()
    barrier = threading.Barrier(2)
    barrier_timeout = 0.5
    original_get = cache.get

    def get_with_scheduling_gap(key):
        value = original_get(key)
        try:
            barrier.wait(timeout=barrier_timeout)
        except threading.BrokenBarrierError:
            pass
        return value

    cache.get = get_with_scheduling_gap
    acquired = []

    def worker():
        lock = cache.acquire_lock("same-key")
        acquired.append(lock)
        lock.release()

    # when
    with ThreadPoolExecutor(max_workers=2) as executor:
        for future in [executor.submit(worker), executor.submit(worker)]:
            future.result()

    # then - one lock object, so the protected section is actually serialised
    assert len(acquired) == 2
    assert len({id(lock) for lock in acquired}) == 1


def test_acquire_lock_serialises_the_protected_section():
    # given - the behaviour the lock exists to provide, stated directly
    cache = _build_cache()
    concurrent = 0
    peak = 0
    peak_guard = threading.Lock()

    def worker():
        nonlocal concurrent, peak
        lock = cache.acquire_lock("model-load")
        try:
            with peak_guard:
                concurrent += 1
                peak = max(peak, concurrent)
            time.sleep(0.02)
            with peak_guard:
                concurrent -= 1
        finally:
            lock.release()

    # when
    with ThreadPoolExecutor(max_workers=4) as executor:
        for future in [executor.submit(worker) for _ in range(4)]:
            future.result()

    # then - never two callers inside at once
    assert peak == 1
