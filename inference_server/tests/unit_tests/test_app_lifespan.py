from __future__ import annotations

import asyncio
import os
import runpy

import pytest


class TestWatchdogWiring:
    @pytest.mark.asyncio
    async def test_lifespan_starts_and_stops_watchdogs(self, monkeypatch):
        import inference_server.app as app_mod

        monkeypatch.delenv("INFERENCE_PRELOAD_MODELS", raising=False)

        class _Daemon:
            stopped = False

            def stop(self, timeout=None):
                self.stopped = True

        daemon = _Daemon()
        monkeypatch.setattr(
            "inference_model_manager.watchdogs.start_enabled_watchdogs",
            lambda: [daemon],
        )

        class _StubProxy:
            async def start(self):
                pass

            async def shutdown(self):
                pass

        monkeypatch.setattr(
            "inference_server.gateway_resolver.resolve_gateway",
            lambda: _StubProxy(),
        )

        async with app_mod._lifespan(app_mod.app):
            pass

        assert daemon.stopped is True

    @pytest.mark.asyncio
    async def test_lifespan_stops_watchdogs_even_if_shutdown_raises(self, monkeypatch):
        import inference_server.app as app_mod

        monkeypatch.delenv("INFERENCE_PRELOAD_MODELS", raising=False)

        class _Daemon:
            stopped = False

            def stop(self, timeout=None):
                self.stopped = True

        daemon = _Daemon()
        monkeypatch.setattr(
            "inference_model_manager.watchdogs.start_enabled_watchdogs",
            lambda: [daemon],
        )

        class _StubProxy:
            async def start(self):
                pass

            async def shutdown(self):
                raise RuntimeError("shutdown boom")

        monkeypatch.setattr(
            "inference_server.gateway_resolver.resolve_gateway",
            lambda: _StubProxy(),
        )

        with pytest.raises(RuntimeError, match="shutdown boom"):
            async with app_mod._lifespan(app_mod.app):
                pass

        assert daemon.stopped is True


class TestPreloadTaskLifecycle:
    @pytest.mark.asyncio
    async def test_lifespan_awaits_preload_task_before_shutdown(self, monkeypatch):
        import inference_server.app as app_mod

        monkeypatch.setenv("INFERENCE_PRELOAD_MODELS", "m1")
        monkeypatch.setattr(app_mod._cfg, "PRELOAD_API_KEY", "")
        monkeypatch.setattr(
            "inference_model_manager.watchdogs.start_enabled_watchdogs",
            lambda: [],
        )

        calls = []
        events = []

        class _StubProxy:
            async def start(self):
                pass

            async def load(self, mid, api_key="", timeout_s=None, pinned=True):
                calls.append((mid, api_key))
                try:
                    await asyncio.sleep(5)
                except asyncio.CancelledError:
                    events.append("load_cancelled")
                    raise

            async def shutdown(self):
                events.append("shutdown")

        monkeypatch.setattr(
            "inference_server.gateway_resolver.resolve_gateway",
            lambda: _StubProxy(),
        )

        async with app_mod._lifespan(app_mod.app):
            await asyncio.sleep(0.01)

        assert calls == [("m1", None)]
        assert events == ["load_cancelled", "shutdown"]

    @pytest.mark.asyncio
    async def test_lifespan_forwards_preload_api_key_when_set(self, monkeypatch):
        import inference_server.app as app_mod

        monkeypatch.setenv("INFERENCE_PRELOAD_MODELS", "m1")
        monkeypatch.setattr(app_mod._cfg, "PRELOAD_API_KEY", "secret")
        monkeypatch.setattr(
            "inference_model_manager.watchdogs.start_enabled_watchdogs",
            lambda: [],
        )

        calls = []

        class _StubProxy:
            async def start(self):
                pass

            async def load(self, mid, api_key="", timeout_s=None, pinned=True):
                calls.append((mid, api_key))
                return ("ok",)

            async def shutdown(self):
                pass

        monkeypatch.setattr(
            "inference_server.gateway_resolver.resolve_gateway",
            lambda: _StubProxy(),
        )

        async with app_mod._lifespan(app_mod.app):
            await asyncio.sleep(0.01)

        assert calls == [("m1", "secret")]


_STARTUP_ENV = (
    "INFERENCE_PRELOAD_MODELS",
    "PRELOAD_MODELS",
    "PINNED_MODELS",
    "PRELOAD_HF_IDS",
    "PRELOAD_API_KEY",
    "ROBOFLOW_API_KEY",
    "API_KEY",
    "MAX_ACTIVE_MODELS",
    "INFERENCE_MAX_ACTIVE_MODELS",
)


@pytest.fixture
def startup_env(monkeypatch):
    for name in _STARTUP_ENV:
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)
    monkeypatch.setattr(
        "inference_model_manager.watchdogs.start_enabled_watchdogs", lambda: []
    )
    return monkeypatch


class _RecordingProxy:
    def __init__(self, load_delay_s=0.0):
        self.loads = []
        self.in_flight = 0
        self.max_in_flight = 0
        self.load_delay_s = load_delay_s

    async def start(self):
        pass

    async def shutdown(self):
        pass

    async def load(self, mid, api_key="", timeout_s=None, pinned=True):
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.load_delay_s)
            self.loads.append((mid, api_key, pinned))
        finally:
            self.in_flight -= 1
        return ("ok",)


async def _run_startup(monkeypatch, proxy, until=lambda: True):
    import inference_server.app as app_mod

    monkeypatch.setattr(
        "inference_server.gateway_resolver.resolve_gateway", lambda: proxy
    )
    async with app_mod._lifespan(app_mod.app):
        for _ in range(500):
            if app_mod.app.state.preload_finished and until():
                break
            await asyncio.sleep(0.01)
        finished = app_mod.app.state.preload_finished

    return finished


def _run_configuration():
    import inference_server.configuration as configuration

    namespace = runpy.run_path(configuration.__file__)

    return namespace


class TestStartupPreloadEnv:
    @pytest.mark.asyncio
    async def test_legacy_preload_models_load_unpinned_by_registry_id(
        self, startup_env
    ):
        import inference_server.app as app_mod
        from inference_server.legacy_env import apply_legacy_env

        startup_env.setenv("PRELOAD_MODELS", "yolov8n-640")
        apply_legacy_env()
        startup_env.setattr(app_mod._cfg, "PRELOAD_API_KEY", "pk")
        proxy = _RecordingProxy()

        finished = await _run_startup(startup_env, proxy)

        assert finished is True
        assert proxy.loads == [("coco/3", "pk", False)]

    @pytest.mark.asyncio
    async def test_pinned_models_load_pinned(self, startup_env):
        import inference_server.app as app_mod

        startup_env.setenv("PINNED_MODELS", "ds/1")
        startup_env.setattr(app_mod._cfg, "PRELOAD_API_KEY", "pk")
        proxy = _RecordingProxy()

        finished = await _run_startup(startup_env, proxy)

        assert finished is True
        assert proxy.loads == [("ds/1", "pk", True)]

    @pytest.mark.asyncio
    async def test_model_in_both_lists_loads_once_pinned(self, startup_env):
        import inference_server.app as app_mod

        startup_env.setenv("INFERENCE_PRELOAD_MODELS", "ds/1,ds/2")
        startup_env.setenv("PINNED_MODELS", "ds/1")
        startup_env.setattr(app_mod._cfg, "PRELOAD_API_KEY", "pk")
        proxy = _RecordingProxy()

        await _run_startup(startup_env, proxy)

        assert sorted(proxy.loads) == [("ds/1", "pk", True), ("ds/2", "pk", False)]

    @pytest.mark.asyncio
    async def test_api_key_suffix_becomes_per_id_key(self, startup_env):
        import inference_server.app as app_mod

        startup_env.setenv("INFERENCE_PRELOAD_MODELS", "ds/1:k1,ds/2")
        startup_env.setenv("PINNED_MODELS", "ds/3:k3")
        startup_env.setattr(app_mod._cfg, "PRELOAD_API_KEY", "pk")
        proxy = _RecordingProxy()

        await _run_startup(startup_env, proxy)

        assert sorted(proxy.loads) == [
            ("ds/1", "k1", False),
            ("ds/2", "pk", False),
            ("ds/3", "k3", True),
        ]

    @pytest.mark.asyncio
    async def test_startup_loads_run_two_at_a_time(self, startup_env):
        startup_env.setenv("INFERENCE_PRELOAD_MODELS", "ds/1,ds/2,ds/3,ds/4,ds/5")
        proxy = _RecordingProxy(load_delay_s=0.02)

        finished = await _run_startup(startup_env, proxy)

        assert finished is True
        assert len(proxy.loads) == 5
        assert proxy.max_in_flight == 2

    @pytest.mark.asyncio
    async def test_preload_hf_ids_load_unpinned_owlv2_off_readiness(self, startup_env):
        import inference_server.app as app_mod

        class _Proxy(_RecordingProxy):
            async def load(self, mid, api_key="", timeout_s=None, pinned=True):
                self.loads.append((mid, api_key, pinned))
                if mid == "owlv2/bad-owl":
                    raise RuntimeError("boom")
                return ("ok",)

        startup_env.setenv(
            "PRELOAD_HF_IDS", "google/bad-owl, google/owlv2-base-patch16-ensemble"
        )
        startup_env.setattr(app_mod._cfg, "PRELOAD_API_KEY", "pk")
        proxy = _Proxy()

        finished = await _run_startup(
            startup_env, proxy, until=lambda: len(proxy.loads) == 2
        )

        assert finished is True
        assert proxy.loads == [
            ("owlv2/bad-owl", "pk", False),
            ("owlv2/owlv2-base-patch16-ensemble", "pk", False),
        ]

    def test_readiness_ok_while_hf_load_pending(self, startup_env):
        import threading

        from fastapi.testclient import TestClient

        import inference_server.app as app_mod

        started = threading.Event()
        release = threading.Event()

        class _PendingProxy(_RecordingProxy):
            async def load(self, mid, api_key="", timeout_s=None, pinned=True):
                self.loads.append((mid, api_key, pinned))
                started.set()
                while not release.is_set():
                    await asyncio.sleep(0.01)
                return ("ok",)

            async def stats(self):
                return {"models": {}}

        startup_env.setenv("PRELOAD_HF_IDS", "google/owlv2-base-patch16-ensemble")
        startup_env.setattr(app_mod._cfg, "PRELOAD_API_KEY", "pk")
        proxy = _PendingProxy()
        startup_env.setattr(
            "inference_server.gateway_resolver.resolve_gateway", lambda: proxy
        )

        with TestClient(app_mod.app) as client:
            try:
                assert started.wait(timeout=5)

                readiness = client.get("/readiness")
                v2_ready = client.get("/v2/server/ready")

                assert readiness.status_code == 200
                assert readiness.json() == {"status": "ready"}
                assert v2_ready.status_code == 200
                assert v2_ready.json() == {"ready": True}
                assert proxy.loads == [
                    ("owlv2/owlv2-base-patch16-ensemble", "pk", False)
                ]
            finally:
                release.set()

    def test_preload_api_key_falls_back_to_roboflow_api_key(self, startup_env):
        startup_env.setenv("ROBOFLOW_API_KEY", "rk")

        namespace = _run_configuration()

        assert namespace["PRELOAD_API_KEY"] == "rk"

    def test_preload_api_key_falls_back_to_legacy_api_key(self, startup_env):
        from inference_server.legacy_env import apply_legacy_env

        startup_env.setenv("API_KEY", "ak")
        apply_legacy_env()

        namespace = _run_configuration()

        assert namespace["PRELOAD_API_KEY"] == "ak"

    def test_explicit_preload_api_key_wins(self, startup_env):
        startup_env.setenv("ROBOFLOW_API_KEY", "rk")
        startup_env.setenv("PRELOAD_API_KEY", "pk")

        namespace = _run_configuration()

        assert namespace["PRELOAD_API_KEY"] == "pk"

    def test_max_active_models_reaches_manager_env(self, startup_env):
        from inference_server.legacy_env import apply_legacy_env

        startup_env.setenv("MAX_ACTIVE_MODELS", "3")
        apply_legacy_env()

        assert os.environ["INFERENCE_MAX_ACTIVE_MODELS"] == "3"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result,logged",
    [
        (
            ("error", 5, {"error_type": "RetryError", "message": "DETAIL-TEXT"}),
            "('error', 5)",
        ),
        (("error", 5), "('error', 5)"),
        (None, "None"),
    ],
)
async def test_failed_preload_log_leaves_out_the_failure_description(
    caplog, result, logged
):
    import logging
    from types import SimpleNamespace

    from inference_server.app import _preload_models

    class _Proxy:
        async def load(self, mid, api_key="", timeout_s=None, pinned=True):
            return result

    state = SimpleNamespace(preload_finished=False)
    with caplog.at_level(logging.DEBUG, logger="inference_server.app"):
        await _preload_models(state, _Proxy(), [("ds/1", "k")], [])

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "inference_server.app"
    ]
    assert messages == [f"Preload of 'ds/1' failed: {logged}"]
    assert "DETAIL-TEXT" not in caplog.text
    assert state.preload_finished is True
