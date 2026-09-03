from __future__ import annotations

import asyncio

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

            async def load(self, mid, api_key="", timeout_s=None):
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

            async def load(self, mid, api_key="", timeout_s=None):
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
