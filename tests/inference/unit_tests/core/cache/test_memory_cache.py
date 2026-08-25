import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from inference.core.cache.memory import MemoryCache


def _build_cache() -> MemoryCache:
    """Construct a MemoryCache without starting the background ``_expire`` thread."""
    with patch("inference.core.cache.memory.threading.Thread"):
        return MemoryCache()


class _StopExpirationLoop(Exception):
    pass


class _ExpirySnapshot(dict):
    def __init__(self, values, after_snapshot):
        super().__init__(values)
        self._after_snapshot = after_snapshot
        self._copies = 0

    def copy(self):
        self._copies += 1
        if self._copies > 1:
            raise _StopExpirationLoop
        snapshot = dict(self)
        self._after_snapshot()
        return snapshot


class _ObservedLock:
    def __init__(self, waiting_event=None, retry_event=None):
        self._lock = threading.Lock()
        self._waiting_event = waiting_event
        self._retry_event = retry_event
        self._acquire_calls = 0

    def acquire(self, *args, **kwargs):
        self._acquire_calls += 1
        if self._waiting_event is not None:
            self._waiting_event.set()
        if self._retry_event is not None and self._acquire_calls > 1:
            self._retry_event.set()
        return self._lock.acquire(*args, **kwargs)

    def release(self):
        self._lock.release()


class _RecordingLock:
    def __init__(self, on_acquire=None):
        self._on_acquire = on_acquire
        self.timeouts = []
        self.released = False

    def acquire(self, *args, **kwargs):
        self.timeouts.append(kwargs.get("timeout"))
        if self._on_acquire is not None:
            self._on_acquire()
        return True

    def release(self):
        self.released = True


def test_acquire_lock_without_expire_keeps_the_lock_cached():
    # given - the default expire=None contract
    cache = _build_cache()

    # when
    lock = cache.acquire_lock("some-key")
    lock.release()

    # then - no expiry is recorded, so the entry is not discarded on the next read
    assert cache.expires.get("some-key") is None
    assert cache.get("some-key") is lock


def test_acquire_lock_blocking_modes_reuse_one_lock_across_calls():
    # given - None and -1 both mean block forever, without a cache TTL
    cache = _build_cache()

    # when
    first = cache.acquire_lock("some-key", expire=-1)
    first.release()
    second = cache.acquire_lock("some-key")
    second.release()

    # then
    assert first is second
    assert "some-key" not in cache.expires


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

    class SchedulingGapDict(dict):
        def get(self, key, default=None):
            value = super().get(key, default)
            try:
                barrier.wait(timeout=barrier_timeout)
            except threading.BrokenBarrierError:
                pass
            return value

    cache.cache = SchedulingGapDict()
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


def test_acquire_lock_retries_when_waited_lock_has_expired():
    # given - a waiter has already selected L1 when its TTL expires and another
    # caller replaces it with L2. The waiter timeout starts after the L1 TTL, so
    # it can still acquire L1 before its own timeout expires.
    cache = _build_cache()
    waiter_started = threading.Event()
    waiter_retried_current_lock = threading.Event()
    stale_lock = _ObservedLock(waiting_event=waiter_started)
    replacement_lock = _ObservedLock(retry_event=waiter_retried_current_lock)
    stale_lock._lock.acquire()
    cache.cache["model-load"] = stale_lock
    cache.expires["model-load"] = time.time() + 60

    with ThreadPoolExecutor(max_workers=1) as executor:
        waiter = executor.submit(cache.acquire_lock, "model-load", 60)
        assert waiter_started.wait(timeout=1)

        cache.expires["model-load"] = time.time() - 1
        with patch("inference.core.cache.memory.Lock", return_value=replacement_lock):
            current_lock = cache.acquire_lock("model-load", expire=60)

        stale_lock._lock.release()
        waiter_retried = waiter_retried_current_lock.wait(timeout=1)
        waiter_was_blocked_by_current_lock = not waiter.done()
        current_lock.release()
        returned_lock = waiter.result(timeout=1)
        returned_lock.release()

    # then - the waiter rejects L1 and waits for the current L2 generation
    assert waiter_retried
    assert waiter_was_blocked_by_current_lock
    assert returned_lock is replacement_lock
    assert cache.get("model-load") is replacement_lock


def test_acquire_lock_retries_with_remaining_timeout_budget():
    # given - acquiring the first candidate consumes part of the original
    # timeout before the cache rotates to a new lock generation
    cache = _build_cache()
    current_lock = _RecordingLock()

    def rotate_lock():
        cache.set("some-key", current_lock, expire=5)

    stale_lock = _RecordingLock(on_acquire=rotate_lock)
    cache.set("some-key", stale_lock, expire=5)

    # when
    with patch(
        "inference.core.cache.memory.time.monotonic",
        side_effect=[100.0, 101.0, 104.0],
    ):
        returned_lock = cache.acquire_lock("some-key", expire=5)

    # then - retry uses what remains of the first five-second budget
    assert returned_lock is current_lock
    assert stale_lock.released
    assert stale_lock.timeouts == [4.0]
    assert current_lock.timeouts == [1.0]
    returned_lock.release()


def test_acquire_lock_stops_when_timeout_budget_is_exhausted():
    # given - the first candidate becomes stale after consuming the complete
    # timeout budget, while the replacement lock itself would be immediately free
    cache = _build_cache()
    current_lock = _RecordingLock()

    def rotate_lock():
        cache.set("some-key", current_lock, expire=5)

    stale_lock = _RecordingLock(on_acquire=rotate_lock)
    cache.set("some-key", stale_lock, expire=5)

    # when / then - no fresh five-second wait and no post-deadline acquisition
    with patch(
        "inference.core.cache.memory.time.monotonic",
        side_effect=[100.0, 101.0, 106.0],
    ):
        with pytest.raises(TimeoutError):
            cache.acquire_lock("some-key", expire=5)

    assert stale_lock.released
    assert stale_lock.timeouts == [4.0]
    assert current_lock.timeouts == []


def test_waiting_for_one_key_does_not_block_another_key():
    # given
    cache = _build_cache()
    first_key_waiting = threading.Event()
    second_key_acquired = threading.Event()
    first_lock = _ObservedLock(waiting_event=first_key_waiting)
    first_lock._lock.acquire()
    cache.cache["first"] = first_lock

    def acquire_second_key():
        lock = cache.acquire_lock("second")
        second_key_acquired.set()
        lock.release()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_waiter = executor.submit(cache.acquire_lock, "first")
        assert first_key_waiting.wait(timeout=1)
        second_waiter = executor.submit(acquire_second_key)
        second_completed_while_first_waited = second_key_acquired.wait(timeout=1)
        first_lock._lock.release()
        returned_first_lock = first_waiter.result(timeout=1)
        returned_first_lock.release()
        second_waiter.result(timeout=1)

    # then - the shared state guard is not held during per-key lock acquisition
    assert second_completed_while_first_waited


def test_acquire_lock_with_zero_expire_is_non_blocking_and_has_no_ttl():
    # given
    cache = _build_cache()

    # when
    lock = cache.acquire_lock("some-key", expire=0)

    # then
    assert "some-key" not in cache.expires
    with pytest.raises(TimeoutError):
        cache.acquire_lock("some-key", expire=0)
    lock.release()


def test_set_without_expire_clears_previous_ttl():
    # given
    cache = _build_cache()
    cache.set("some-key", "temporary", expire=60)

    # when
    cache.set("some-key", "permanent")

    # then - match Redis SET semantics rather than retaining an old deadline
    assert cache.get("some-key") == "permanent"
    assert "some-key" not in cache.expires


def test_expiration_worker_preserves_entry_refreshed_after_snapshot():
    # given - the worker observes an expired deadline, then set() refreshes the
    # same key before deletion begins
    cache = _build_cache()
    cache.cache["some-key"] = "old"

    def refresh_after_snapshot():
        cache.set("some-key", "refreshed", expire=60)

    cache.expires = _ExpirySnapshot(
        {"some-key": time.time() - 1},
        after_snapshot=refresh_after_snapshot,
    )

    # when - stop deterministically at the start of the second worker iteration
    with patch("inference.core.cache.memory.MEMORY_CACHE_EXPIRE_INTERVAL", 0):
        with pytest.raises(_StopExpirationLoop):
            cache._expire()

    # then
    assert cache.get("some-key") == "refreshed"
    assert cache.expires["some-key"] > time.time()


def test_expiration_worker_tolerates_entry_deleted_after_snapshot():
    # given - get() or another worker removes the candidate after the worker's
    # snapshot but before its deletion step
    cache = _build_cache()
    cache.cache["some-key"] = "value"

    def delete_after_snapshot():
        cache.cache.pop("some-key", None)
        cache.expires.pop("some-key", None)

    cache.expires = _ExpirySnapshot(
        {"some-key": time.time() - 1},
        after_snapshot=delete_after_snapshot,
    )

    # when / then - the first iteration must not die with KeyError
    with patch("inference.core.cache.memory.MEMORY_CACHE_EXPIRE_INTERVAL", 0):
        with pytest.raises(_StopExpirationLoop):
            cache._expire()

    assert cache.get("some-key") is None
