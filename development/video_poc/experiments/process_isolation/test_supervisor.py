from __future__ import annotations

import time

import pytest

from development.video_poc.experiments.process_isolation.supervisor import (
    EventType,
    IsolationMode,
    JobDescriptor,
    ProcessSupervisor,
    isolation_key,
)


def _job(job_id: str, workspace_id: str) -> JobDescriptor:
    return JobDescriptor(job_id, workspace_id, {"apiKey": "must-not-be-in-domain-key"})


def _collect_until(supervisor, predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    events = []
    while time.monotonic() < deadline:
        events.extend(supervisor.events(timeout=0.05))
        if predicate(events):
            return events
    raise AssertionError(f"condition not reached; events={events!r}")


def test_workspace_mode_groups_only_same_workspace():
    with ProcessSupervisor(IsolationMode.WORKSPACE) as supervisor:
        first = supervisor.start(_job("a", "workspace-1"))
        second = supervisor.start(_job("b", "workspace-1"))
        third = supervisor.start(_job("c", "workspace-2"))
        assert first == second
        assert third != first
        assert set(supervisor.active_domains()) == {first, third}


def test_job_mode_uses_one_process_per_job():
    with ProcessSupervisor(IsolationMode.JOB) as supervisor:
        first = supervisor.start(_job("a", "workspace-1"))
        second = supervisor.start(_job("b", "workspace-1"))
        assert first != second


def test_hard_crash_is_contained_to_one_workspace():
    with ProcessSupervisor(IsolationMode.WORKSPACE) as supervisor:
        first = supervisor.start(_job("a", "workspace-1"))
        supervisor.start(_job("b", "workspace-1"))
        sibling = supervisor.start(_job("c", "workspace-2"))
        _collect_until(
            supervisor,
            lambda events: sum(e.event_type is EventType.STARTED for e in events) == 3,
        )

        supervisor.crash_for_test(first)
        events = _collect_until(
            supervisor,
            lambda items: {e.job_id for e in items if e.event_type is EventType.FAILED}
            == {"a", "b"},
        )

        failed = {e.job_id for e in events if e.event_type is EventType.FAILED}
        assert failed == {"a", "b"}
        assert sibling in supervisor.active_domains()
        assert supervisor.active_domains()[sibling] == {"c"}


def test_domain_key_never_contains_payload_credentials():
    key = isolation_key(_job("job", "workspace"), IsolationMode.WORKSPACE)
    assert key == "workspace:workspace"
    assert "must-not-be-in-domain-key" not in key


def test_workspace_mode_rejects_missing_workspace():
    with pytest.raises(ValueError, match="workspace_id"):
        isolation_key(_job("job", ""), IsolationMode.WORKSPACE)
