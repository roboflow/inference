"""Worker pool without a queue, quiescence, shutdown and run-owned threads."""

import threading
import time
from typing import Callable, List

import pytest
from roboflow_workflows.execution_engine.v2.errors import ContractError
from roboflow_workflows.execution_engine.v2.pipelining.stages import PipelineCounters
from roboflow_workflows.execution_engine.v2.pipelining.workers import (
    OwnedThreads,
    WorkerPool,
)

TIMEOUT = 5.0


def blocking_work(started: threading.Event, release: threading.Event) -> Callable:
    def work() -> None:
        started.set()
        release.wait(TIMEOUT)

    return work


def test_busy_pool_refuses_work_instead_of_queueing() -> None:
    counters = PipelineCounters()
    pool = WorkerPool(2, name="test", owner=OwnedThreads(), counters=counters)
    release = threading.Event()
    starts = [threading.Event(), threading.Event()]

    for started in starts:
        assert pool.try_submit(blocking_work(started, release))
    for started in starts:
        assert started.wait(TIMEOUT)

    assert not pool.try_submit(lambda: None)
    assert not pool.submit(lambda: None, timeout=0.01)
    assert not pool.wait_free(timeout=0.01)
    assert counters.current("executing") == 2

    release.set()
    assert pool.wait_idle(TIMEOUT)
    assert pool.wait_free(timeout=0)
    assert counters.snapshot()["peak"]["executing"] == 2
    pool.shutdown()


def test_on_free_wakes_a_waiting_submitter() -> None:
    freed = threading.Event()
    pool = WorkerPool(1, name="test", owner=OwnedThreads(), on_free=freed.set)
    release, started = threading.Event(), threading.Event()
    assert pool.try_submit(blocking_work(started, release))
    assert started.wait(TIMEOUT)

    ran = threading.Event()
    submitter = threading.Thread(target=lambda: pool.submit(ran.set))
    submitter.start()
    release.set()
    submitter.join(TIMEOUT)

    assert ran.wait(TIMEOUT)
    assert freed.is_set()
    pool.shutdown()


def test_close_refuses_new_work_and_wakes_waiters() -> None:
    pool = WorkerPool(1, name="test", owner=OwnedThreads())
    release, started = threading.Event(), threading.Event()
    assert pool.try_submit(blocking_work(started, release))
    assert started.wait(TIMEOUT)
    results: List[bool] = []
    waiter = threading.Thread(target=lambda: results.append(pool.wait_free()))
    waiter.start()

    pool.close()
    waiter.join(TIMEOUT)

    assert results == [False]
    assert not pool.try_submit(lambda: None)
    release.set()
    pool.shutdown()
    assert not any(thread.is_alive() for thread in pool._threads)


def test_shutdown_lets_running_work_finish_and_joins() -> None:
    pool = WorkerPool(2, name="test", owner=OwnedThreads())
    finished: List[str] = []
    started, release = threading.Event(), threading.Event()

    def work() -> None:
        started.set()
        release.wait(TIMEOUT)
        finished.append("done")

    assert pool.try_submit(work)
    assert started.wait(TIMEOUT)
    stopper = threading.Thread(target=pool.shutdown)
    stopper.start()
    release.set()
    stopper.join(TIMEOUT)

    assert finished == ["done"]
    assert not any(thread.is_alive() for thread in pool._threads)


def test_workers_are_owned_and_cannot_wait_for_their_own_pool() -> None:
    owned = OwnedThreads()
    pool = WorkerPool(1, name="test", owner=owned)
    seen: List[object] = []

    def work() -> None:
        seen.append(owned.owns_current())
        for action in (pool.wait_idle, pool.shutdown):
            try:
                action()
            except ContractError as error:
                seen.append(str(error))
        try:
            owned.reject_wait("ActiveRun.wait()")
        except ContractError as error:
            seen.append(str(error))

    assert pool.try_submit(work)
    assert pool.wait_idle(TIMEOUT)

    assert seen[0] is True
    assert "own workers" in seen[1] and "own workers" in seen[2]
    assert "ActiveRun.wait()" in seen[3]
    assert not owned.owns_current()
    owned.reject_wait("ActiveRun.wait()")
    pool.shutdown()


def test_unexpected_work_error_is_kept_and_the_worker_survives() -> None:
    pool = WorkerPool(1, name="test", owner=OwnedThreads())

    def broken() -> None:
        raise ValueError("driver bug")

    assert pool.try_submit(broken)
    assert pool.wait_idle(TIMEOUT)
    assert isinstance(pool.unexpected, ValueError)

    ran = threading.Event()
    assert pool.submit(ran.set, timeout=TIMEOUT)
    assert ran.wait(TIMEOUT)
    pool.shutdown()


def test_invalid_size_is_rejected() -> None:
    for size in (0, True, 1.5):
        with pytest.raises(ContractError, match="size"):
            WorkerPool(size, name="test", owner=OwnedThreads())


def test_executing_never_exceeds_the_pool_size_under_contention() -> None:
    counters = PipelineCounters()
    pool = WorkerPool(3, name="test", owner=OwnedThreads(), counters=counters)
    accepted = 0
    for _ in range(200):
        if pool.submit(lambda: time.sleep(0), timeout=TIMEOUT):
            accepted += 1
    assert pool.wait_idle(TIMEOUT)

    assert accepted == 200
    assert counters.peak("executing") <= 3
    pool.shutdown()


def refuse_second_worker(monkeypatch, prefix: str) -> List[threading.Thread]:
    """Make ``Thread.start`` fail for worker 1 of pools named ``prefix``."""
    real_start = threading.Thread.start
    started: List[threading.Thread] = []

    def start(thread: threading.Thread) -> None:
        if thread.name.startswith(prefix) and thread.name.endswith("-1"):
            raise RuntimeError("injected second-worker launch failure")
        real_start(thread)
        if thread.name.startswith(prefix):
            started.append(thread)

    monkeypatch.setattr(threading.Thread, "start", start)

    return started


def live_threads(prefix: str) -> List[str]:
    names = [
        thread.name
        for thread in threading.enumerate()
        if thread.name.startswith(prefix)
    ]

    return names


def test_a_failed_worker_launch_stops_and_joins_the_started_workers(
    monkeypatch,
) -> None:
    started = refuse_second_worker(monkeypatch, "launch-test")

    with pytest.raises(RuntimeError, match="injected second-worker"):
        WorkerPool(3, name="launch-test", owner=OwnedThreads())

    # Worker 0 started; it is joined before the constructor raises.
    assert [thread.name for thread in started] == ["launch-test-0"]
    assert live_threads("launch-test") == []
