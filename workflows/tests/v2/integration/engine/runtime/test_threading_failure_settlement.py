"""A broken failure reporter cannot retain admitted capacity or in-flight work."""

import threading
from functools import partial
from types import SimpleNamespace

from roboflow_workflows.v2.engine.active.pipeline import PipelinedDriver
from roboflow_workflows.v2.engine.controls import InFlightWork
from roboflow_workflows.v2.engine.pipelining.stages import PipelineCounters
from roboflow_workflows.v2.engine.pipelining.workers import (
    OwnedThreads,
    WorkerPool,
)


def test_failure_reporter_raising_still_releases_the_unit_and_worker() -> None:
    driver = PipelinedDriver.__new__(PipelinedDriver)
    driver._condition = threading.Condition(threading.RLock())
    driver._in_flight = InFlightWork()
    driver._in_flight.enter(7)
    driver._running = 1
    slot = SimpleNamespace(in_flight=1, admission=threading.Semaphore(0))
    ingress = SimpleNamespace(slot=slot, latest=False)
    reports = []
    successor_selections = []
    original_error = ValueError("unit failed")
    reporting_error = RuntimeError("failure reporter failed")

    def broken_unit():
        raise original_error

    def broken_report(failure):
        reports.append(failure)
        raise reporting_error

    def select_successor():
        successor_selections.append(True)
        return None

    driver._run = SimpleNamespace(_fail=broken_report)
    driver._take_work = select_successor
    counters = PipelineCounters()
    pool = WorkerPool(
        1, name="failure-settlement-test", owner=OwnedThreads(), counters=counters
    )
    try:
        work = partial(
            driver._run_unit,
            broken_unit,
            controls=SimpleNamespace(version=7),
            ingress=ingress,
        )
        assert pool.try_submit(work)
        assert pool.wait_idle(timeout=5)
        assert pool.unexpected is reporting_error
        assert len(reports) == 1
        assert reports[0].__cause__ is original_error
        assert successor_selections == []
        assert driver._running == 0
        assert driver._in_flight.describe() == {}
        assert slot.in_flight == 0
        assert slot.admission.acquire(blocking=False)
        assert not slot.admission.acquire(blocking=False)
        assert counters.current("executing") == 0
        assert counters.peak("executing") == 1
        ran_after_failure = threading.Event()
        assert pool.try_submit(ran_after_failure.set)
        assert ran_after_failure.wait(timeout=5)
        assert pool.wait_idle(timeout=5)
    finally:
        pool.shutdown()
    assert not any(thread.is_alive() for thread in pool.threads)
