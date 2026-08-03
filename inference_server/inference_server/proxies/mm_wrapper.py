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

from inference_model_manager.backends.utils.shm_pool import INPUT_ERROR_PREFIX
from inference_model_manager.model_manager import ModelManager
from inference_server import configuration
from inference_server.errors import PayloadTooLargeError, ServerBusyError

logger = logging.getLogger(__name__)

# Lifecycle error codes mirroring inference_model_manager.model_manager_process
# — the same tuples MMPClient callers translate.
_ERR_LOAD_FAILED = 5
_ERR_NOT_LOADED = 6

# MMPClient.load() waits 30s when no explicit timeout is given (its
# _lifecycle_req default); bundled load() mirrors that.
_LOAD_DEFAULT_TIMEOUT_S = 30.0


def _translate_manager_infer_error(exc: Exception) -> Exception:
    """Map manager/backend failures onto the MMPClient exception surface."""
    if type(exc).__name__ == "ModelInputError":
        return ValueError(str(exc))
    if isinstance(exc, RuntimeError) and str(exc).startswith(INPUT_ERROR_PREFIX):
        return ValueError(str(exc)[len(INPUT_ERROR_PREFIX) :])
    if isinstance(exc, ValueError) and "slot capacity" in str(exc):
        return PayloadTooLargeError(str(exc))
    if isinstance(exc, TimeoutError) and "No free SHM slots" in str(exc):
        return ServerBusyError(str(exc))
    return exc


class MMWrapper:
    """ModelManagerProxy impl over an in-process ModelManager.

    Lifecycle:
        wrapper = MMWrapper(manager)
        ...                          # use as ModelManagerProxy
        await wrapper.shutdown()     # delegates to manager.shutdown()
    """

    def __init__(
        self,
        manager: ModelManager,
        *,
        backend: Optional[str] = None,
        load_wait_s: Optional[float] = None,
        infer_timeout_s: Optional[float] = None,
    ) -> None:
        self.manager = manager
        # None keeps the manager's own default backend; "subprocess" gives the
        # worker-side decode + result marshalling the MMP wire contract has.
        self._backend_choice = backend
        self.load_wait_s = (
            load_wait_s if load_wait_s is not None else configuration.LOAD_WAIT_S
        )
        self.infer_timeout_s = (
            infer_timeout_s
            if infer_timeout_s is not None
            else configuration.INFER_TIMEOUT_S
        )
        self.n_slots = getattr(manager, "n_slots", None) or configuration.SERVER_N_SLOTS
        # Per-model load dedup — two concurrent first requests must not both
        # call manager.load (2x VRAM / load time).
        self._load_locks: dict[str, asyncio.Lock] = {}
        # In-flight executor loads shared between load() and ensure_loaded()
        # so a timed-out load() keeps loading and later calls await the same
        # future instead of double-loading.
        self._pending_loads: dict[str, asyncio.Future] = {}

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

    def _load_sync(
        self, model_id: str, api_key: str, device: Optional[str] = None
    ) -> None:
        kwargs: dict = {}
        if self._backend_choice is not None:
            kwargs["backend"] = self._backend_choice
            # The MMP supervisor forwards these to its manager; an explicit
            # backend choice opts into the same configuration.
            kwargs["decoder"] = configuration.SERVER_DECODER
            kwargs["batch_max_size"] = configuration.SERVER_BATCH_MAX_SIZE
            kwargs["batch_max_delay_ms"] = configuration.SERVER_BATCH_MAX_WAIT_MS
        if device:
            kwargs["device"] = device
        self.manager.load(model_id, api_key, **kwargs)

    async def _acquire_load_future(
        self, model_id: str, api_key: str, device: Optional[str] = None
    ) -> Optional[asyncio.Future]:
        """None when the model is present and healthy; else the shared load.

        Everything happens under the per-model lock: presence check, dead
        backend replacement (the MMP watches worker death and reloads —
        without this, a crashed worker keeps answering model_ready and
        poisons the route), and load-future creation. Serializing the
        unload here means a stale unload can never remove a freshly loaded
        replacement.
        """
        lock = self._load_locks.setdefault(model_id, asyncio.Lock())
        async with lock:
            future = self._pending_loads.get(model_id)
            if future is not None:
                return future
            drop_dead = False
            if model_id in self.manager:
                is_healthy = getattr(self.manager, "is_healthy", None)
                if is_healthy is None or is_healthy(model_id):
                    return None
                logger.warning(
                    "MMWrapper: backend for '%s' is dead — reloading", model_id
                )
                drop_dead = True

            def _reload() -> None:
                if drop_dead:
                    try:
                        self.manager.unload(model_id)
                    except Exception:
                        logger.warning(
                            "MMWrapper: dead-backend unload failed", exc_info=True
                        )
                self._load_sync(model_id, api_key, device)

            # Unload+load run as ONE executor job registered before the lock
            # releases: no await window a cancelled caller could exploit, and
            # followers join the same future covering the whole replacement.
            future = asyncio.get_running_loop().run_in_executor(None, _reload)
            self._pending_loads[model_id] = future
            future.add_done_callback(lambda _f: self._pending_loads.pop(model_id, None))
        return future

    async def _await_load(
        self,
        future: asyncio.Future,
        timeout_s: Optional[float],
        deadline_result: Optional[tuple],
    ) -> Optional[tuple]:
        """Await a shared load; None on success, a lifecycle tuple otherwise.

        deadline_result is returned when OUR deadline fires; None re-raises
        the deadline as asyncio.TimeoutError (MMPClient.load(timeout_s)
        semantics). shield: a cancelled caller must not abort a load other
        requests (and the MMP contract) expect to keep running.
        """
        try:
            if timeout_s is not None:
                await asyncio.wait_for(asyncio.shield(future), timeout=timeout_s)
            else:
                await asyncio.shield(future)
        except asyncio.CancelledError:
            raise
        except (asyncio.TimeoutError, TimeoutError):
            if future.done() and future.exception() is None:
                # The load completed on the deadline boundary — success.
                return None
            if not future.done():
                # Our deadline; the executor load keeps running and callers
                # poll again.
                if deadline_result is None:
                    raise
                return deadline_result
            # A TimeoutError from inside the load itself (worker start
            # timeout etc.) is a load failure, not our deadline.
            logger.warning("MMWrapper: load failed", exc_info=future.exception())
            return ("error", _ERR_LOAD_FAILED)
        except Exception:
            logger.warning("MMWrapper: load failed", exc_info=True)
            return ("error", _ERR_LOAD_FAILED)
        return None

    async def ensure_loaded(
        self,
        model_id: str,
        instance: str = "",
        api_key: str = "",
        device: str = "",
    ) -> tuple:
        future = await self._acquire_load_future(model_id, api_key, device or None)
        if future is None:
            return ("model_ready",)
        failure = await self._await_load(
            future,
            self.load_wait_s,
            deadline_result=("load_timeout", int(self.load_wait_s)),
        )
        if failure is not None:
            return failure
        return ("model_ready",)

    async def load(
        self, model_id: str, api_key: str = "", timeout_s: Optional[float] = None
    ) -> tuple:
        future = await self._acquire_load_future(model_id, api_key)
        if future is None:
            return ("ok",)
        effective_timeout = (
            timeout_s if timeout_s is not None else _LOAD_DEFAULT_TIMEOUT_S
        )
        failure = await self._await_load(
            future, effective_timeout, deadline_result=None
        )
        if failure is not None:
            return failure
        return ("ok",)

    async def unload(self, model_id: str) -> tuple:
        try:
            await asyncio.get_running_loop().run_in_executor(
                None, lambda: self.manager.unload(model_id)
            )
        except KeyError:
            return ("error", _ERR_NOT_LOADED)
        except Exception:
            logger.warning("MMWrapper.unload failed", exc_info=True)
            return ("error", _ERR_LOAD_FAILED)
        return ("ok",)

    async def infer(
        self,
        *,
        model_id: str,
        image: Optional[bytes] = None,
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
        # Empty payload = params-only request; the model resolves inputs from
        # params (mirrors the MMP worker contract for zero-byte slots).
        if image is not None:
            call_kwargs["images"] = image if image else None

        async def _run() -> Any:
            try:
                # serialize=False: L1 output serializers expect the RAW
                # prediction — the MMP wire carries raw pickles, so bundled
                # mode must match.
                return await self.manager.process_async(
                    model_id, task=task, serialize=False, **call_kwargs
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                translated = _translate_manager_infer_error(exc)
                if translated is exc:
                    raise
                raise translated from exc

        prediction = await asyncio.wait_for(_run(), timeout=self.infer_timeout_s)
        if raw_pickle:
            return pickle.dumps(prediction)
        return prediction

    async def stats(self) -> dict:
        stats = await asyncio.get_running_loop().run_in_executor(
            None, self.manager.stats
        )
        # MMPClient consumers read per-model entries from an id-keyed
        # "mmp_models" dict; the manager reports a list. Serve both shapes.
        stats["mmp_models"] = {
            entry["model_id"]: entry
            for entry in stats.get("models") or []
            if entry.get("model_id")
        }
        return stats

    async def interface(self, model_id: str) -> dict:
        stats = await self.stats()
        info = stats["mmp_models"].get(model_id)
        if info is None:
            raise RuntimeError(f"model '{model_id}' is not loaded")
        return {"model_id": model_id, "tasks": info.get("tasks", {})}
