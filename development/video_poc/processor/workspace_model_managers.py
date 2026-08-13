"""Workspace-scoped model-manager domains for the video worker experiment.

The default mode returns ``None`` and preserves the current InferencePipeline
model manager. Experimental bundled modes create one new ModelManagerAdapter
per workspace. Jobs in one workspace share models and batching, while jobs in
different workspaces never share a model subprocess or model-manager cache.
"""

import asyncio
import threading
from dataclasses import dataclass
from typing import Callable, Optional


LEGACY_MODE = "legacy"
BUNDLED_SUBPROCESS_MODE = "mmp-bundled-subprocess"
BUNDLED_DIRECT_MODE = "mmp-bundled-direct"
ALLOWED_MODES = {LEGACY_MODE, BUNDLED_SUBPROCESS_MODE, BUNDLED_DIRECT_MODE}


def _new_adapter(backend):
    from inference.core.cache import cache
    from inference.core.env import MAX_ACTIVE_MODELS
    from inference.core.managers import mmp_adapter
    from inference.core.managers.active_learning import (
        BackgroundTaskActiveLearningManager,
    )
    from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
    from inference.core.registries.roboflow import RoboflowModelRegistry
    from inference.models.utils import ROBOFLOW_MODEL_TYPES

    if mmp_adapter.LEGACY_MMP_ADAPTER_MODE != "bundled":
        raise RuntimeError(
            "experimental video model managers require "
            "LEGACY_MMP_ADAPTER_MODE=bundled"
        )
    if mmp_adapter.LEGACY_MMP_ADAPTER_BUNDLED_BACKEND != backend:
        raise RuntimeError(
            "LEGACY_MMP_ADAPTER_BUNDLED_BACKEND does not match "
            "PROCESSOR_MODEL_MANAGER_MODE"
        )
    registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
    legacy = BackgroundTaskActiveLearningManager(model_registry=registry, cache=cache)
    legacy = WithFixedSizeCache(legacy, max_size=MAX_ACTIVE_MODELS)
    return mmp_adapter.ModelManagerAdapter(legacy_stack=legacy)


@dataclass
class _ManagerDomain:
    adapter: object
    loop: asyncio.AbstractEventLoop
    thread: threading.Thread

    @classmethod
    def start(cls, adapter, timeout_s=30.0):
        loop = asyncio.new_event_loop()
        ready = threading.Event()

        def run():
            asyncio.set_event_loop(loop)
            ready.set()
            loop.run_forever()
            loop.close()

        thread = threading.Thread(
            target=run,
            name="video-workspace-model-manager",
            daemon=True,
        )
        thread.start()
        if not ready.wait(timeout_s):
            raise RuntimeError("workspace model manager startup timed out")
        future = asyncio.run_coroutine_threadsafe(adapter.start(), loop)
        try:
            future.result(timeout=timeout_s)
        except BaseException as exc:
            future.cancel()
            # Let the loop deliver cancellation before closing it; otherwise
            # asyncio reports a destroyed pending startup task and may retain
            # resources allocated before the await that was interrupted.
            try:
                asyncio.run_coroutine_threadsafe(
                    asyncio.sleep(0), loop
                ).result(timeout=min(timeout_s, 1.0))
            except Exception:
                pass
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout_s)
            raise RuntimeError("workspace model manager failed to start") from exc
        return cls(adapter=adapter, loop=loop, thread=thread)

    def shutdown(self, timeout_s=30.0):
        if not self.thread.is_alive():
            return
        future = asyncio.run_coroutine_threadsafe(self.adapter.shutdown(), self.loop)
        try:
            future.result(timeout=timeout_s)
        finally:
            if not future.done():
                future.cancel()
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout_s)
        if self.thread.is_alive():
            raise RuntimeError("workspace model manager shutdown timed out")


class WorkspaceModelManagerDomains:
    """Lazily creates one adapter/model process set per workspace."""

    def __init__(
        self,
        mode=LEGACY_MODE,
        adapter_factory: Optional[Callable[[str], object]] = None,
    ):
        self.mode = str(mode or LEGACY_MODE).strip().lower()
        if self.mode not in ALLOWED_MODES:
            raise ValueError(
                "PROCESSOR_MODEL_MANAGER_MODE must be one of: "
                + ", ".join(sorted(ALLOWED_MODES))
            )
        self.experimental = self.mode != LEGACY_MODE
        self._backend = (
            "subprocess"
            if self.mode == BUNDLED_SUBPROCESS_MODE
            else "direct"
        )
        self._adapter_factory = adapter_factory or _new_adapter
        self._domains = {}
        self._lock = threading.Lock()
        self._shutting_down = False

    def get(self, workspace):
        if not self.experimental:
            return None
        workspace = str(workspace or "").strip()
        if not workspace:
            raise ValueError("MMP video jobs require a workspace identity")
        with self._lock:
            if self._shutting_down:
                raise RuntimeError("workspace model managers are shutting down")
            domain = self._domains.get(workspace)
            if domain is None:
                adapter = self._adapter_factory(self._backend)
                domain = _ManagerDomain.start(adapter)
                self._domains[workspace] = domain
            return domain.adapter

    def snapshot(self):
        with self._lock:
            count = len(self._domains)
        return {
            "mode": self.mode,
            "experimental": self.experimental,
            "activeDomains": count,
            "crossWorkspaceModelSharing": False,
        }

    def shutdown(self):
        with self._lock:
            self._shutting_down = True
            domains, self._domains = list(self._domains.values()), {}
        errors = []
        for domain in domains:
            try:
                domain.shutdown()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError("one or more workspace model managers failed shutdown")
