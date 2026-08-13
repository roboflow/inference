import asyncio
import threading

import pytest

from workspace_model_managers import _ManagerDomain, WorkspaceModelManagerDomains


class FakeAdapter:
    def __init__(self, backend):
        self.backend = backend
        self.started = False
        self.stopped = False
        self.loop = None

    async def start(self):
        self.started = True
        self.loop = asyncio.get_running_loop()

    async def shutdown(self):
        assert asyncio.get_running_loop() is self.loop
        self.stopped = True


def test_legacy_mode_is_noop():
    domains = WorkspaceModelManagerDomains()
    assert domains.get(None) is None
    assert domains.snapshot() == {
        "mode": "legacy",
        "experimental": False,
        "activeDomains": 0,
        "crossWorkspaceModelSharing": False,
    }


def test_same_workspace_shares_adapter_and_other_workspace_is_isolated():
    created = []

    def factory(backend):
        adapter = FakeAdapter(backend)
        created.append(adapter)
        return adapter

    domains = WorkspaceModelManagerDomains(
        "mmp-bundled-subprocess", adapter_factory=factory
    )
    first = domains.get("workspace-a")
    assert domains.get("workspace-a") is first
    second = domains.get("workspace-b")

    assert first is not second
    assert all(adapter.backend == "subprocess" for adapter in created)
    assert all(adapter.started for adapter in created)
    assert domains.snapshot()["activeDomains"] == 2
    assert "workspace-a" not in str(domains.snapshot())

    domains.shutdown()
    assert all(adapter.stopped for adapter in created)


def test_direct_mode_is_an_explicit_control_variant():
    created = []
    domains = WorkspaceModelManagerDomains(
        "mmp-bundled-direct",
        adapter_factory=lambda backend: created.append(FakeAdapter(backend))
        or created[-1],
    )
    assert domains.get("workspace-a").backend == "direct"
    domains.shutdown()


def test_experimental_mode_requires_workspace_and_rejects_after_shutdown():
    domains = WorkspaceModelManagerDomains(
        "mmp-bundled-subprocess", adapter_factory=FakeAdapter
    )
    with pytest.raises(ValueError, match="workspace identity"):
        domains.get(None)
    domains.shutdown()
    with pytest.raises(RuntimeError, match="shutting down"):
        domains.get("workspace-a")


def test_unknown_mode_fails_closed():
    with pytest.raises(ValueError, match="PROCESSOR_MODEL_MANAGER_MODE"):
        WorkspaceModelManagerDomains("shared-global")


def test_failed_start_does_not_leave_manager_thread_running():
    blocker = threading.Event()

    class BlockedAdapter(FakeAdapter):
        async def start(self):
            while not blocker.is_set():
                await asyncio.sleep(0.01)

    before = {
        thread.ident
        for thread in threading.enumerate()
        if thread.name == "video-workspace-model-manager"
    }
    with pytest.raises(RuntimeError, match="failed to start"):
        _ManagerDomain.start(BlockedAdapter("subprocess"), timeout_s=0.05)
    after = {
        thread.ident
        for thread in threading.enumerate()
        if thread.name == "video-workspace-model-manager"
    }
    assert after == before
