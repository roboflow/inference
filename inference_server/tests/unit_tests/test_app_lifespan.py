from __future__ import annotations

import asyncio
import logging
import os
import runpy

import pytest

from inference_server.app import _start_usage_collector as start_usage_collector


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


class _WatchdogSpy:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self, timeout=None):
        self.stopped = True


class _IdleProxy:
    async def start(self):
        pass

    async def shutdown(self):
        pass


class TestOfflineWatchdogWiring:
    @pytest.fixture
    def watchdog_env(self, monkeypatch):
        import inference_server.app as app_mod
        from inference_model_manager import configuration as manager_cfg
        from inference_models import configuration as models_cfg

        monkeypatch.delenv("INFERENCE_PRELOAD_MODELS", raising=False)
        monkeypatch.setattr(
            "inference_server.gateway_resolver.resolve_gateway",
            lambda: _IdleProxy(),
        )
        monkeypatch.setattr(app_mod._cfg, "PRELOAD_API_KEY", "")
        monkeypatch.setattr(manager_cfg, "MAX_INFERENCE_MODELS_CACHE_SIZE_MB", 100)
        monkeypatch.setattr(
            manager_cfg, "ENABLE_CUDA_MEMORY_RECLAMATION_WATCHDOG", False
        )
        cache_spy = []
        cuda_spy = []

        class _CacheSpy(_WatchdogSpy):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                cache_spy.append(self)

        class _CudaSpy(_WatchdogSpy):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                cuda_spy.append(self)

        monkeypatch.setattr(
            "inference_model_manager.watchdogs.InferenceModelsCacheWatchdog",
            _CacheSpy,
        )
        monkeypatch.setattr(
            "inference_model_manager.watchdogs.CudaMemoryReclamationWatchdog",
            _CudaSpy,
        )

        return monkeypatch, manager_cfg, cache_spy, cuda_spy, models_cfg

    @pytest.mark.asyncio
    async def test_offline_never_starts_the_cache_watchdog(self, watchdog_env):
        import inference_server.app as app_mod

        monkeypatch, _, cache_spy, _, models_cfg = watchdog_env
        monkeypatch.setattr(models_cfg, "OFFLINE_MODE", True)

        async with app_mod._lifespan(app_mod.app):
            pass

        assert cache_spy == []

    @pytest.mark.asyncio
    async def test_offline_still_runs_the_cuda_watchdog(self, watchdog_env):
        import inference_server.app as app_mod

        monkeypatch, manager_cfg, cache_spy, cuda_spy, models_cfg = watchdog_env
        monkeypatch.setattr(models_cfg, "OFFLINE_MODE", True)
        monkeypatch.setattr(
            manager_cfg, "ENABLE_CUDA_MEMORY_RECLAMATION_WATCHDOG", True
        )

        async with app_mod._lifespan(app_mod.app):
            assert len(cuda_spy) == 1
            assert cuda_spy[0].started is True
            assert cuda_spy[0].stopped is False

        assert cuda_spy[0].stopped is True
        assert cache_spy == []

    @pytest.mark.asyncio
    async def test_model_layer_offline_wins_over_server_setting(self, watchdog_env):
        import inference_server.app as app_mod

        monkeypatch, _, cache_spy, _, models_cfg = watchdog_env
        monkeypatch.setattr(app_mod._cfg, "OFFLINE_MODE", False)
        monkeypatch.setattr(models_cfg, "OFFLINE_MODE", True)

        async with app_mod._lifespan(app_mod.app):
            pass

        assert cache_spy == []

    @pytest.mark.asyncio
    async def test_online_starts_enabled_watchdogs_once(self, watchdog_env):
        import inference_server.app as app_mod

        monkeypatch, _, _, _, models_cfg = watchdog_env
        monkeypatch.setattr(models_cfg, "OFFLINE_MODE", False)
        calls = []
        monkeypatch.setattr(
            "inference_model_manager.watchdogs.start_enabled_watchdogs",
            lambda: calls.append(1) or [],
        )

        async with app_mod._lifespan(app_mod.app):
            pass

        assert calls == [1]


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


class _UsageCollectorSpy:
    def __init__(self):
        self.started = False
        self.flushed = False
        self.stopped = False
        self.order = []

    def start(self):
        self.started = True
        self.order.append("start")

    def flush(self):
        self.flushed = True
        self.order.append("flush")

    def stop(self, timeout=None):
        self.stopped = True
        self.order.append("stop")
        return True


class TestUsageCollectorWiring:
    @pytest.fixture
    def usage_env(self, monkeypatch):
        import inference_server.app as app_mod

        monkeypatch.delenv("INFERENCE_PRELOAD_MODELS", raising=False)
        monkeypatch.setattr(
            "inference_model_manager.watchdogs.start_enabled_watchdogs", lambda: []
        )
        monkeypatch.setattr(
            "inference_server.gateway_resolver.resolve_gateway", lambda: _IdleProxy()
        )

        return app_mod

    def test_collector_is_started_when_online(self, monkeypatch):
        import inference_server.app as app_mod

        spy = _UsageCollectorSpy()
        monkeypatch.setattr(app_mod, "UsageCollector", lambda: spy)
        monkeypatch.setattr(app_mod._cfg, "LEGACY_OFFLINE_MODE", False)

        assert start_usage_collector() is spy
        assert spy.started is True

    def test_collector_is_not_started_in_offline_mode(self, monkeypatch):
        import inference_server.app as app_mod

        monkeypatch.setattr(
            app_mod, "UsageCollector", lambda: pytest.fail("constructed offline")
        )
        monkeypatch.setattr(app_mod._cfg, "LEGACY_OFFLINE_MODE", True)

        assert start_usage_collector() is None

    @pytest.mark.asyncio
    async def test_lifespan_exposes_the_collector_and_flushes_then_stops_it(
        self, usage_env, monkeypatch
    ):
        app_mod = usage_env
        spy = _UsageCollectorSpy()
        monkeypatch.setattr(app_mod, "_start_usage_collector", lambda: spy)

        async with app_mod._lifespan(app_mod.app):
            assert app_mod.app.state.usage_collector is spy
            assert spy.flushed is False and spy.stopped is False

        assert spy.order == ["flush", "stop"]
        assert app_mod.app.state.usage_collector is None

    @pytest.mark.asyncio
    async def test_lifespan_without_a_collector_leaves_state_empty(
        self, usage_env, monkeypatch
    ):
        app_mod = usage_env
        monkeypatch.setattr(app_mod, "_start_usage_collector", lambda: None)

        async with app_mod._lifespan(app_mod.app):
            assert app_mod.app.state.usage_collector is None

        assert app_mod.app.state.usage_collector is None


class _FakeManagerProcess:
    def __init__(self, *, alive=True, stuck=False, exitcode=None):
        self.alive = alive
        self.stuck = stuck
        self.exitcode = exitcode
        self.events = []

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.events.append("terminate")
        if not self.stuck:
            self.alive = False

    def kill(self):
        self.events.append("kill")
        self.alive = False

    def join(self, timeout=None):
        self.events.append(("join", timeout))


class _OrderedProxy(_IdleProxy):
    def __init__(self, events):
        self.events = events

    async def shutdown(self):
        self.events.append("gateway shutdown")


class TestStreamManagerWiring:
    @pytest.fixture
    def stream_env(self, monkeypatch):
        from streamvision.stream.configuration import reset_configuration

        import inference_server.app as app_mod

        reset_configuration()
        monkeypatch.delenv("INFERENCE_PRELOAD_MODELS", raising=False)
        monkeypatch.setattr(
            "inference_model_manager.watchdogs.start_enabled_watchdogs", lambda: []
        )
        monkeypatch.setattr(
            "inference_server.gateway_resolver.resolve_gateway", lambda: _IdleProxy()
        )
        monkeypatch.setattr(app_mod._cfg, "ENABLE_STREAM_API", True)
        monkeypatch.setattr(app_mod._cfg, "STREAM_API_PRELOADED_PROCESSES", 2)
        monkeypatch.setattr(app_mod._cfg, "STREAM_MANAGER_HOST", "10.0.0.5")
        monkeypatch.setattr(app_mod._cfg, "STREAM_MANAGER_PORT", 7171)
        monkeypatch.setattr(app_mod._cfg, "STREAM_MANAGER_OPERATIONS_TIMEOUT", 2.5)
        launches = []

        def _launch(streams_configuration, **kwargs):
            launches.append((streams_configuration, kwargs))
            return launches_process[0]

        launches_process = [_FakeManagerProcess()]
        monkeypatch.setattr(app_mod, "_start_stream_manager", _launch)
        yield app_mod, launches, launches_process
        reset_configuration()

    @pytest.mark.asyncio
    async def test_manager_starts_with_the_server_host_and_stops_on_exit(
        self, stream_env
    ):
        from inference_server.streams.configuration import (
            build_streams_configuration,
        )
        from inference_server.streams.host import SERVER_PIPELINE_HOST_DESCRIPTOR

        app_mod, launches, (process,) = stream_env

        async with app_mod._lifespan(app_mod.app):
            from streamvision.stream_manager.api.stream_manager_client import (
                StreamManagerClient,
            )

            assert launches == [
                (
                    build_streams_configuration(),
                    {
                        "host_descriptor": SERVER_PIPELINE_HOST_DESCRIPTOR,
                        "expected_warmed_up_pipelines": 2,
                    },
                )
            ]
            assert launches[0][0].stream_manager_host == "10.0.0.5"
            assert app_mod.app.state.stream_manager_process is process
            client = app_mod.app.state.stream_manager_client
            assert isinstance(client, StreamManagerClient)
            assert client._host == "10.0.0.5"
            assert client._port == 7171
            assert client._operations_timeout == 2.5
            assert process.events == []

        assert process.events == [
            "terminate",
            ("join", app_mod.STREAM_MANAGER_STOP_TIMEOUT_S),
        ]
        assert app_mod.app.state.stream_manager_client is None
        assert app_mod.app.state.stream_manager_process is None

    @pytest.mark.asyncio
    async def test_manager_is_stopped_before_the_gateway(self, stream_env, monkeypatch):
        app_mod, _, (process,) = stream_env
        events = []
        process.events = events
        proxy = _OrderedProxy(events)
        monkeypatch.setattr(
            "inference_server.gateway_resolver.resolve_gateway", lambda: proxy
        )

        async with app_mod._lifespan(app_mod.app):
            pass

        assert events == [
            "terminate",
            ("join", app_mod.STREAM_MANAGER_STOP_TIMEOUT_S),
            "gateway shutdown",
        ]

    @pytest.mark.asyncio
    async def test_manager_surviving_terminate_is_killed(self, stream_env, caplog):
        app_mod, _, launches_process = stream_env
        process = _FakeManagerProcess(stuck=True)
        launches_process[0] = process

        with caplog.at_level(logging.WARNING, logger="inference_server.app"):
            async with app_mod._lifespan(app_mod.app):
                pass

        assert process.events == [
            "terminate",
            ("join", app_mod.STREAM_MANAGER_STOP_TIMEOUT_S),
            "kill",
            ("join", app_mod.STREAM_MANAGER_STOP_TIMEOUT_S),
        ]
        assert process.alive is False
        assert any("did not stop" in record.getMessage() for record in caplog.records)

    @pytest.mark.asyncio
    async def test_manager_dying_early_is_logged(self, stream_env, caplog, monkeypatch):
        app_mod, _, launches_process = stream_env
        process = _FakeManagerProcess(alive=False, exitcode=1)
        launches_process[0] = process
        monkeypatch.setattr(app_mod, "STREAM_MANAGER_STARTUP_GRACE_S", 0.0)

        with caplog.at_level(logging.WARNING, logger="inference_server.app"):
            async with app_mod._lifespan(app_mod.app):
                await asyncio.sleep(0.05)

        messages = [record.getMessage() for record in caplog.records]
        assert any("exited" in message and "1" in message for message in messages)
        assert process.events == [
            "terminate",
            ("join", app_mod.STREAM_MANAGER_STOP_TIMEOUT_S),
        ]

    @pytest.mark.asyncio
    async def test_early_exit_check_is_cancelled_by_shutdown(
        self, stream_env, caplog, monkeypatch
    ):
        app_mod, _, _ = stream_env
        monkeypatch.setattr(app_mod, "STREAM_MANAGER_STARTUP_GRACE_S", 0.02)

        with caplog.at_level(logging.WARNING, logger="inference_server.app"):
            async with app_mod._lifespan(app_mod.app):
                pass
            await asyncio.sleep(0.05)

        assert not any("exited" in record.getMessage() for record in caplog.records)

    @pytest.mark.asyncio
    async def test_flag_off_starts_nothing(self, stream_env, monkeypatch):
        app_mod, launches, _ = stream_env
        monkeypatch.setattr(app_mod._cfg, "ENABLE_STREAM_API", False)

        async with app_mod._lifespan(app_mod.app):
            assert launches == []
            assert getattr(app_mod.app.state, "stream_manager_client", None) is None
            assert getattr(app_mod.app.state, "stream_manager_process", None) is None

        assert launches == []


class TestVllmRequestIdProvider:
    @pytest.fixture
    def lifespan_env(self, monkeypatch):
        from inference_models.models.vllm_proxy import vllm_client

        monkeypatch.delenv("INFERENCE_PRELOAD_MODELS", raising=False)
        monkeypatch.setattr(
            "inference_model_manager.watchdogs.start_enabled_watchdogs",
            lambda: [],
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
        monkeypatch.setattr(vllm_client, "_REQUEST_ID_PROVIDER", None)
        return vllm_client

    @pytest.mark.asyncio
    async def test_provider_returns_the_current_correlation_id(self, lifespan_env):
        import inference_server.app as app_mod
        from inference_server.middlewares.correlation_id import correlation_id

        async with app_mod._lifespan(app_mod.app):
            assert lifespan_env.get_request_id() is None
            token = correlation_id.set("abc123")
            try:
                assert lifespan_env.get_request_id() == "abc123"
            finally:
                correlation_id.reset(token)
            assert lifespan_env.get_request_id() is None

    @pytest.mark.asyncio
    async def test_provider_is_not_installed_before_startup(self, lifespan_env):
        from inference_server.middlewares.correlation_id import correlation_id

        token = correlation_id.set("abc123")
        try:
            assert lifespan_env.get_request_id() is None
        finally:
            correlation_id.reset(token)
