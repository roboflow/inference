import sys
import threading
import time
from pathlib import Path

PROCESSOR_DIR = (
    Path(__file__).resolve().parents[3] / "development" / "video_poc" / "processor"
)
sys.path.insert(0, str(PROCESSOR_DIR))

from run_lifecycle import finish_run_once  # noqa: E402


class FakeDomains:
    def __init__(self):
        self.released = []
        self.lock = threading.Lock()

    def release_job(self, job_id):
        with self.lock:
            self.released.append(job_id)


class FakeRun:
    def __init__(self, job_id, stop=None):
        self.job_id = job_id
        self._stop = stop or (lambda: None)
        self.outcomes = []
        self.lock = threading.Lock()
        self.cancelling = False
        self.stopped = threading.Event()

    def _record_outcome(self, outcome):
        with self.lock:
            self.outcomes.append(outcome)

    def stop(self):
        try:
            self._stop()
        finally:
            self.stopped.set()


class FakeWorker:
    def __init__(self, runs):
        self.runs = {run.job_id: run for run in runs}
        self.runs_lock = threading.Lock()
        self.execution_domains = FakeDomains()
        self.retire_calls = 0
        self.retire_lock = threading.Lock()

    def maybe_retire(self):
        with self.retire_lock:
            self.retire_calls += 1


def finish(worker, run, failures, outcome="cancelled", timeout=1.0):
    return finish_run_once(
        worker,
        run,
        outcome=outcome,
        stop_timeout_s=timeout,
        on_stop_failure=lambda failed_run, reason: failures.append(
            (failed_run.job_id, reason)
        ),
    )


def wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not reached")


def test_concurrent_cancels_release_each_job_without_stale_accounting():
    entered = threading.Barrier(3)
    release = threading.Event()

    def delayed_stop():
        entered.wait()
        release.wait()

    runs = [FakeRun("job-a", delayed_stop), FakeRun("job-b", delayed_stop)]
    worker = FakeWorker(runs)
    failures = []
    for run in runs:
        assert finish(worker, run, failures) is True
    entered.wait()
    assert set(worker.runs) == {"job-a", "job-b"}
    release.set()
    wait_until(lambda: worker.runs == {})

    assert worker.runs == {}
    assert sorted(worker.execution_domains.released) == ["job-a", "job-b"]
    assert worker.retire_calls == 2
    assert failures == []


def test_duplicate_cancel_has_one_stop_release_and_retirement():
    stop_entered = threading.Event()
    release = threading.Event()
    stop_calls = []

    def delayed_stop():
        stop_calls.append(True)
        stop_entered.set()
        release.wait()

    run = FakeRun("job-a", delayed_stop)
    worker = FakeWorker([run])
    failures = []
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(finish(worker, run, failures)))
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    assert stop_entered.wait(timeout=1)
    release.set()
    for thread in threads:
        thread.join(timeout=2)
    wait_until(lambda: worker.runs == {})

    assert results.count(True) == 1
    assert results.count(False) == 3
    assert len(stop_calls) == 1
    assert worker.execution_domains.released == ["job-a"]
    assert worker.retire_calls == 1
    assert run.outcomes == ["cancelled"]


def test_cancel_teardown_does_not_block_sibling_heartbeat_loop():
    stop_entered = threading.Event()
    release = threading.Event()

    def delayed_stop():
        stop_entered.set()
        release.wait()

    run = FakeRun("job-a", delayed_stop)
    sibling = FakeRun("job-b")
    worker = FakeWorker([run, sibling])
    failures = []

    started = time.monotonic()
    assert finish(worker, run, failures) is True
    elapsed = time.monotonic() - started

    assert stop_entered.wait(timeout=1)
    assert elapsed < 0.1
    assert run.cancelling is True
    assert worker.runs == {"job-a": run, "job-b": sibling}
    release.set()
    wait_until(lambda: set(worker.runs) == {"job-b"})
    assert worker.execution_domains.released == ["job-a"]


def test_stop_timeout_contains_worker_without_dropping_live_run():
    release = threading.Event()
    run = FakeRun("job-a", release.wait)
    worker = FakeWorker([run])
    failures = []

    assert finish(worker, run, failures, timeout=0.01) is True
    wait_until(lambda: bool(failures))

    assert failures == [("job-a", "timeout")]
    assert worker.runs == {"job-a": run}
    assert worker.execution_domains.released == []
    assert worker.retire_calls == 0
    release.set()


def test_stop_exception_contains_worker_without_dropping_live_run():
    def fail_stop():
        raise RuntimeError("cleanup failed")

    run = FakeRun("job-a", fail_stop)
    worker = FakeWorker([run])
    failures = []

    assert finish(worker, run, failures) is True
    wait_until(lambda: bool(failures))

    assert failures == [("job-a", "exception: RuntimeError")]
    assert worker.runs == {"job-a": run}
    assert worker.execution_domains.released == []
    assert worker.retire_calls == 0


def test_old_run_completion_cannot_remove_new_same_id_owner():
    old = FakeRun("job-a")
    new = FakeRun("job-a")
    worker = FakeWorker([new])
    failures = []

    assert finish(worker, old, failures) is True
    assert old.stopped.wait(timeout=1)

    assert worker.runs == {"job-a": new}
    assert worker.execution_domains.released == []
    assert worker.retire_calls == 0
    assert failures == []
