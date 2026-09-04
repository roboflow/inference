from __future__ import annotations

import time
import threading

import pytest

from development.video_poc.processor.execution_domains import (
    ExecutionDomainMode,
    WorkspaceProbeExecutionDomains,
    build_execution_domains,
    wait_for_threads,
)


def _wait_for_failure(manager, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        failures = manager.poll_failures()
        if failures:
            return failures
        time.sleep(0.02)
    raise AssertionError("execution-domain failure was not observed")


def test_default_mode_preserves_in_process_behavior():
    manager = build_execution_domains(None)
    manager.start_job("job-a", None)
    assert manager.mode is ExecutionDomainMode.IN_PROCESS
    assert manager.poll_failures() == []
    assert manager.snapshot() == {
        "mode": "in_process",
        "experimental": False,
        "activeDomains": 0,
    }


def test_workspace_probe_groups_jobs_and_keeps_diagnostics_credential_free():
    secret = "api-key-must-never-appear"
    with WorkspaceProbeExecutionDomains() as manager:
        manager.start_job("job-a", "workspace-a")
        manager.start_job("job-b", "workspace-a")
        manager.start_job("job-c", "workspace-b")
        assert manager.snapshot()["activeDomains"] == 2

        manager.crash_workspace_for_test("workspace-a", exit_code=73)
        failures = _wait_for_failure(manager)

        assert len(failures) == 1
        assert failures[0].job_ids == ("job-a", "job-b")
        assert failures[0].exit_code == 73
        assert "workspace-a" not in failures[0].domain_id
        assert "workspace-a" not in failures[0].diagnostic
        assert secret not in repr(failures[0])
        assert manager.snapshot()["activeDomains"] == 1


def test_releasing_last_job_reaps_only_its_workspace_probe():
    with WorkspaceProbeExecutionDomains() as manager:
        manager.start_job("job-a", "workspace-a")
        manager.start_job("job-b", "workspace-a")
        manager.start_job("job-c", "workspace-b")
        manager.release_job("job-a")
        assert manager.snapshot()["activeDomains"] == 2
        manager.release_job("job-b")
        assert manager.snapshot()["activeDomains"] == 1
        manager.release_job("job-c")
        assert manager.snapshot()["activeDomains"] == 0


def test_workspace_probe_rejects_missing_ownership_metadata():
    with WorkspaceProbeExecutionDomains() as manager:
        with pytest.raises(ValueError, match="workspace id"):
            manager.start_job("job-a", None)


def test_invalid_mode_fails_closed():
    with pytest.raises(ValueError, match="PROCESSOR_EXECUTION_DOMAIN_MODE"):
        build_execution_domains("workspace")


def test_shutdown_and_last_job_release_serialize_handle_teardown():
    manager = WorkspaceProbeExecutionDomains()
    manager.start_job("job-a", "workspace-a")
    barrier = threading.Barrier(3)
    errors = []

    def invoke(action):
        barrier.wait()
        try:
            action()
        except Exception as error:  # pragma: no cover - assertion captures it
            errors.append(error)

    shutdown = threading.Thread(target=invoke, args=(manager.shutdown,))
    release = threading.Thread(
        target=invoke, args=(lambda: manager.release_job("job-a"),)
    )
    shutdown.start()
    release.start()
    barrier.wait()
    shutdown.join(timeout=10)
    release.join(timeout=10)

    assert not shutdown.is_alive()
    assert not release.is_alive()
    assert errors == []
    assert manager.snapshot()["activeDomains"] == 0


def test_thread_wait_uses_one_bounded_deadline():
    release = threading.Event()
    threads = [threading.Thread(target=release.wait) for _ in range(3)]
    for thread in threads:
        thread.start()

    started = time.monotonic()
    assert wait_for_threads(threads, timeout=0.05) is False
    elapsed = time.monotonic() - started
    assert elapsed < 0.2

    release.set()
    assert wait_for_threads(threads, timeout=1.0) is True
