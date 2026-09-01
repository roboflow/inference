"""ModelManagerGateway — wraps a ModelManager in-process.

The built-in "direct" gateway resolved by ``gateway_resolver``. No SHM, no
ZMQ. Calls `inference_model_manager.ModelManager` methods directly.

Sync ModelManager methods (load / unload / stats) are wrapped via
`run_in_executor` so the gateway interface stays async on both impls
(ModelManagerGateway and MMPGateway).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from fastapi import Request

from inference_model_manager.errors import INPUT_ERROR_PREFIX
from inference_model_manager.model_manager import ModelManager
from inference_models.utils.performance import performance_profiler
from inference_server import configuration
from inference_server.errors import PayloadTooLargeError, ServerBusyError

logger = logging.getLogger(__name__)

# Lifecycle error codes mirroring the MMP wire protocol's lifecycle codes
# — the same tuples MMPGateway callers translate.
_ERR_LOAD_FAILED = 5
_ERR_NOT_LOADED = 6

# MMPGateway.load() waits 30s when no explicit timeout is given (its
# _lifecycle_req default); direct load() mirrors that.
_LOAD_DEFAULT_TIMEOUT_S = 30.0

# Backend.inflight_begin / ModelManager.submit refuse a drained backend with
# this phrase.
_NOT_ACCEPTING_MARKER = "not accepting requests"


def routing_key(model_id: str, instance: str = "") -> str:
    """Key a model instance is registered and routed under.

    Matches the MMP wire format exactly: the bare ``model_id`` when no instance
    is requested, ``model_id:instance`` otherwise.
    """
    return f"{model_id}:{instance}" if instance else model_id


def routed_model_id(key: str) -> str:
    """Weights identifier behind a routing key (drops the ``:instance`` suffix)."""
    return key.rsplit(":", 1)[0]


def _translate_manager_infer_error(exc: Exception) -> Exception:
    """Map manager/backend failures onto the MMPGateway exception surface."""
    if type(exc).__name__ == "ModelInputError":
        return ValueError(str(exc))
    if isinstance(exc, RuntimeError) and str(exc).startswith(INPUT_ERROR_PREFIX):
        return ValueError(str(exc)[len(INPUT_ERROR_PREFIX) :])
    if isinstance(exc, ValueError) and "slot capacity" in str(exc):
        return PayloadTooLargeError(str(exc))
    if isinstance(exc, TimeoutError) and "No free SHM slots" in str(exc):
        return ServerBusyError(str(exc))
    return exc


def _model_route_lost(exc: Exception) -> bool:
    """Whether the model vanished under a request that had passed ensure_loaded.

    Inference takes its in-flight lease inside process(), so capacity churn
    between ensure_loaded() and that lease shows up either as a missing route
    (KeyError) or as a backend already drained out of service.
    """
    if isinstance(exc, KeyError):
        return True
    return isinstance(exc, RuntimeError) and _NOT_ACCEPTING_MARKER in str(exc)


def _try_pin(pin: Any, model_id: str) -> bool:
    """False when the model was gone before the pin landed."""
    try:
        pin(model_id)
        return True
    except KeyError:
        return False


class ModelManagerGateway:
    """Gateway duck-surface implementation over an in-process ModelManager.

    Lifecycle:
        gateway = ModelManagerGateway()
        ...                            # use as the resolved gateway
        await gateway.shutdown()       # delegates to manager.shutdown()
    """

    def __init__(
        self,
        manager: Optional[ModelManager] = None,
        *,
        load_kwargs: Optional[dict] = None,
        load_wait_s: Optional[float] = None,
        infer_timeout_s: Optional[float] = None,
    ) -> None:
        self.manager = manager if manager is not None else ModelManager()
        self.load_kwargs = dict(load_kwargs) if load_kwargs else {}
        self.load_wait_s = (
            load_wait_s if load_wait_s is not None else configuration.LOAD_WAIT_S
        )
        self.infer_timeout_s = (
            infer_timeout_s
            if infer_timeout_s is not None
            else configuration.INFER_TIMEOUT_S
        )
        self.n_slots = getattr(self.manager, "n_slots", None)
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
        """Shut down the underlying ModelManager.

        Runs off the manager's own pool on purpose: manager.shutdown() joins
        that pool, which would deadlock if submitted to it.
        """
        await asyncio.get_running_loop().run_in_executor(None, self.manager.shutdown)

    # ------------------------------------------------------------------
    # Gateway duck surface
    # ------------------------------------------------------------------

    @property
    def _model_executor(self):
        """The manager's own worker pool. Model work must not escape onto the
        loop's default executor, which is unrelated to the manager's limits."""
        return getattr(self.manager, "executor", None)

    def _load_sync(
        self,
        key: str,
        api_key: str,
        device: Optional[str] = None,
        pinned: bool = False,
    ) -> None:
        kwargs: dict = dict(self.load_kwargs)
        if device:
            kwargs["device"] = device
        if pinned:
            # The manager pins under the same lock acquisition that registers
            # the backend — no window in which a concurrent load can evict a
            # model this call is about to protect.
            kwargs["pinned"] = True
        model_id = routed_model_id(key)
        if model_id != key:
            # Multi-instance routing key: register under the key, fetch the
            # weights named by the bare model id (mirrors the MMP).
            kwargs["model_id_or_path"] = model_id
        self.manager.load(key, api_key, **kwargs)

    async def _acquire_load_future(
        self,
        key: str,
        api_key: str,
        device: Optional[str] = None,
        pinned: bool = False,
    ) -> Optional[asyncio.Future]:
        """None when the model is present and healthy; else the shared load.

        Everything happens under the per-model lock: presence check, dead
        backend replacement (the MMP watches worker death and reloads —
        without this, a crashed worker keeps answering model_ready and
        poisons the route), and load-future creation. Serializing the
        unload here means a stale unload can never remove a freshly loaded
        replacement.
        """
        lock = self._load_locks.setdefault(key, asyncio.Lock())
        if not performance_profiler.enabled:
            async with lock:
                return self._acquire_load_future_locked(key, api_key, device, pinned)

        lock_started = performance_profiler.start()
        await lock.acquire()
        performance_profiler.stop("wrapper.ensure.lock", lock_started)
        check_started = performance_profiler.start()
        try:
            return self._acquire_load_future_locked(key, api_key, device, pinned)
        finally:
            lock.release()
            performance_profiler.stop("wrapper.ensure.check", check_started)

    def _acquire_load_future_locked(
        self, key: str, api_key: str, device: Optional[str], pinned: bool = False
    ) -> Optional[asyncio.Future]:
        future = self._pending_loads.get(key)
        if future is not None:
            return future
        drop_dead = False
        if key in self.manager:
            is_healthy = getattr(self.manager, "is_healthy", None)
            if is_healthy is None or is_healthy(key):
                return None
            logger.warning(
                "ModelManagerGateway: backend for '%s' is dead — reloading", key
            )
            drop_dead = True

        def _reload() -> None:
            if drop_dead:
                try:
                    self.manager.unload(key)
                except Exception:
                    logger.warning(
                        "ModelManagerGateway: dead-backend unload failed",
                        exc_info=True,
                    )
            self._load_sync(key, api_key, device, pinned)

        # Unload+load run as ONE executor job registered before the lock
        # releases: no await window a cancelled caller could exploit, and
        # followers join the same future covering the whole replacement.
        future = asyncio.get_running_loop().run_in_executor(
            self._model_executor, _reload
        )
        self._pending_loads[key] = future
        future.add_done_callback(lambda _f: self._pending_loads.pop(key, None))
        return future

    async def _await_load(
        self,
        future: asyncio.Future,
        timeout_s: Optional[float],
        deadline_result: Optional[tuple],
    ) -> Optional[tuple]:
        """Await a shared load; None on success, a lifecycle tuple otherwise.

        deadline_result is returned when OUR deadline fires; None re-raises
        the deadline as asyncio.TimeoutError (MMPGateway.load(timeout_s)
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
            logger.warning(
                "ModelManagerGateway: load failed", exc_info=future.exception()
            )
            return ("error", _ERR_LOAD_FAILED)
        except Exception:
            logger.warning("ModelManagerGateway: load failed", exc_info=True)
            return ("error", _ERR_LOAD_FAILED)
        return None

    async def ensure_loaded(
        self,
        model_id: str,
        instance: str = "",
        api_key: str = "",
        device: str = "",
    ) -> tuple:
        started = performance_profiler.start()
        try:
            future = await self._acquire_load_future(
                routing_key(model_id, instance), api_key, device or None
            )
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
        finally:
            performance_profiler.stop("wrapper.ensure.total", started)

    async def _pinned_load(
        self, model_id: str, api_key: str, timeout_s: float
    ) -> Optional[tuple]:
        """Run or join a pinned load; None once the model is loaded."""
        future = await self._acquire_load_future(model_id, api_key, pinned=True)
        if future is None:
            return None
        return await self._await_load(future, timeout_s, deadline_result=None)

    async def load(
        self, model_id: str, api_key: str = "", timeout_s: Optional[float] = None
    ) -> tuple:
        effective_timeout = (
            timeout_s if timeout_s is not None else _LOAD_DEFAULT_TIMEOUT_S
        )
        failure = await self._pinned_load(model_id, api_key, effective_timeout)
        if failure is not None:
            return failure
        pin = getattr(self.manager, "pin", None)
        if pin is None:
            return ("ok",)
        if _try_pin(pin, model_id):
            return ("ok",)
        # Already loaded, or joined an unpinned in-flight load, and the model
        # was evicted before the pin landed. Reload once — reporting ok for an
        # absent model leaves /ready failing forever.
        logger.warning(
            "ModelManagerGateway: '%s' was evicted before it could be pinned "
            "— reloading once",
            model_id,
        )
        failure = await self._pinned_load(model_id, api_key, effective_timeout)
        if failure is not None:
            return failure
        if _try_pin(pin, model_id):
            return ("ok",)
        return ("error", _ERR_LOAD_FAILED)

    async def unload(self, model_id: str) -> tuple:
        try:
            await asyncio.get_running_loop().run_in_executor(
                self._model_executor, lambda: self.manager.unload(model_id)
            )
        except KeyError:
            return ("error", _ERR_NOT_LOADED)
        except Exception:
            logger.warning("ModelManagerGateway.unload failed", exc_info=True)
            return ("error", _ERR_LOAD_FAILED)
        return ("ok",)

    async def infer(
        self,
        *,
        model_id: str,
        image: Any = None,
        task: Optional[str] = None,
        instance: str = "",
        params: Optional[dict] = None,
        request: Optional[Request] = None,
    ) -> Any:
        started = performance_profiler.start()
        try:
            return await self._infer(
                model_id=model_id,
                image=image,
                task=task,
                instance=instance,
                params=params,
                request=request,
            )
        finally:
            performance_profiler.stop("wrapper.infer.total", started)

    async def _infer(
        self,
        *,
        model_id: str,
        image: Any = None,
        task: Optional[str] = None,
        instance: str = "",
        params: Optional[dict] = None,
        request: Optional[Request] = None,
    ) -> Any:
        # `request` is ignored in-process: no client-disconnect race
        # (process_async runs in executor; cancellation propagates via task).
        key = routing_key(model_id, instance)
        call_kwargs = dict(params) if params else {}
        # Empty payload = params-only request; the model resolves inputs from
        # params (mirrors the MMP worker contract for zero-byte slots).
        if image is not None:
            call_kwargs["images"] = (
                None
                if isinstance(image, (bytes, bytearray, memoryview)) and not image
                else image
            )

        async def _process() -> Any:
            # serialize=False: L1 output serializers expect the RAW
            # prediction — the MMP wire carries raw pickles, so the
            # direct gateway must match. wire_marshalling: direct
            # backends get the worker-equivalent decode/rle/numpy/unwrap
            # handling (no-op on subprocess backends — the worker does it).
            return await self.manager.process_async(
                key,
                task=task,
                serialize=False,
                wire_marshalling=True,
                **call_kwargs,
            )

        async def _run() -> Any:
            try:
                try:
                    return await _process()
                except (KeyError, RuntimeError) as exc:
                    if not _model_route_lost(exc):
                        raise
                    # Capacity churn took the model between ensure_loaded()
                    # and the in-flight lease — reload and replay once rather
                    # than failing a request that was admitted.
                    logger.warning(
                        "ModelManagerGateway: '%s' went away mid-request "
                        "— reloading and retrying once",
                        key,
                    )
                    await self.ensure_loaded(model_id, instance)
                    return await _process()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                translated = _translate_manager_infer_error(exc)
                if translated is exc:
                    raise
                raise translated from exc

        return await asyncio.wait_for(_run(), timeout=self.infer_timeout_s)

    async def stats(self) -> dict:
        stats = await asyncio.get_running_loop().run_in_executor(
            None, self.manager.stats
        )
        # Routers consume the neutral "models" id-keyed dict; the manager
        # reports a list — rekey it.
        stats["models"] = {
            entry["model_id"]: entry
            for entry in stats.get("models") or []
            if entry.get("model_id")
        }
        if self.n_slots is not None:
            stats["n_slots"] = self.n_slots
        return stats

    async def interface(self, model_id: str) -> dict:
        stats = await self.stats()
        info = stats["models"].get(model_id)
        if info is None:
            raise RuntimeError(f"model '{model_id}' is not loaded")
        return {"model_id": model_id, "tasks": info.get("tasks", {})}
