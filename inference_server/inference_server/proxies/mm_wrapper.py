"""MMWrapper — ModelManagerProxy impl that wraps a ModelManager in-process.

Used in `bundled` deployment mode. No SHM, no ZMQ. Calls
`inference_model_manager.ModelManager` methods directly.

Sync ModelManager methods (load / unload / stats) are wrapped via
`run_in_executor` so the proxy interface stays async on both impls
(MMWrapper and MMPClient).
"""

from __future__ import annotations

import asyncio
import logging
import pickle
from typing import Any, Optional

from fastapi import Request

from inference_model_manager.model_manager import ModelManager

logger = logging.getLogger(__name__)


class MMWrapper:
    """ModelManagerProxy impl over an in-process ModelManager.

    Lifecycle:
        wrapper = MMWrapper(manager)
        ...                          # use as ModelManagerProxy
        await wrapper.shutdown()     # delegates to manager.shutdown()
    """

    def __init__(self, manager: ModelManager) -> None:
        self.manager = manager

    # ------------------------------------------------------------------
    # Lifecycle (lifespan)
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """No-op — manager is constructed externally and passed in."""

    async def shutdown(self) -> None:
        """Shut down the underlying ModelManager."""
        await asyncio.get_running_loop().run_in_executor(None, self.manager.shutdown)

    # ------------------------------------------------------------------
    # ModelManagerProxy interface
    # ------------------------------------------------------------------

    async def ensure_loaded(
        self,
        model_id: str,
        instance: str = "",
        api_key: str = "",
        device: str = "",
    ) -> tuple:
        if model_id in self.manager.loaded_models():
            return ("model_ready",)
        try:
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self.manager.load(model_id, api_key, device=device or None),
            )
        except Exception:
            logger.warning("MMWrapper.ensure_loaded: load failed", exc_info=True)
            return ("error", 1)
        return ("model_ready",)

    async def load(self, model_id: str, api_key: str = "") -> tuple:
        try:
            await asyncio.get_running_loop().run_in_executor(
                None, lambda: self.manager.load(model_id, api_key)
            )
        except Exception:
            logger.warning("MMWrapper.load failed", exc_info=True)
            return ("error", 1)
        return ("ok",)

    async def unload(self, model_id: str) -> tuple:
        try:
            await asyncio.get_running_loop().run_in_executor(
                None, lambda: self.manager.unload(model_id)
            )
        except KeyError:
            return ("error", 2)
        except Exception:
            logger.warning("MMWrapper.unload failed", exc_info=True)
            return ("error", 1)
        return ("ok",)

    async def infer(
        self,
        *,
        model_id: str,
        image: bytes,
        task: Optional[str] = None,
        instance: str = "",
        params: Optional[dict] = None,
        request: Optional[Request] = None,
        raw_pickle: bool = False,
    ) -> Any:
        # `instance` and `request` are ignored in-process: no multi-instance
        # routing (single ModelManager), no client-disconnect race
        # (process_async runs in executor; cancellation propagates via task).
        call_kwargs = dict(params) if params else {}
        call_kwargs["images"] = image
        prediction = await self.manager.process_async(
            model_id, task=task, **call_kwargs
        )
        if raw_pickle:
            return pickle.dumps(prediction)
        return prediction

    async def stats(self) -> dict:
        return await asyncio.get_running_loop().run_in_executor(
            None, self.manager.stats
        )

    async def interface(self, model_id: str) -> dict:
        stats = await self.stats()
        models = stats.get("mmp_models") or stats.get("models") or {}
        info = models.get(model_id)
        if info is None:
            raise RuntimeError(f"model '{model_id}' is not loaded")
        return {"model_id": model_id, "tasks": info.get("tasks", {})}
