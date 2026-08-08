import threading
from concurrent.futures import ThreadPoolExecutor

from inference.core.cache.memory import MemoryCache


def test_acquire_lock_serializes_concurrent_initialization(monkeypatch) -> None:
    cache = MemoryCache()
    initial_reads_complete = threading.Barrier(2)
    get_calls_lock = threading.Lock()
    get_calls = 0
    original_get = cache.get

    def get_with_scheduling_gap(key: str):
        nonlocal get_calls
        value = original_get(key)
        with get_calls_lock:
            get_calls += 1
            is_initial_read = get_calls <= 2
        if is_initial_read:
            initial_reads_complete.wait(timeout=5.0)
        return value

    monkeypatch.setattr(cache, "get", get_with_scheduling_gap)

    def acquire_shared_lock():
        lock = cache.acquire_lock("shared-key", expire=1.0)
        lock.release()
        return lock

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(acquire_shared_lock) for _ in range(2)]
        locks = [future.result(timeout=10.0) for future in futures]

    assert locks[0] is locks[1]
