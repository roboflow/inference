from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Literal, Optional

from inference_model_manager import configuration as cfg
from inference_model_manager.backends.base import Backend
from inference_model_manager.dispatch import _get_registry, invoke_task, resolve_task

logger = logging.getLogger(__name__)

# Import lazily to avoid circular deps
_to_bytes = None


class _SlotFuture(Future):
    """Future bound to a SHM slot. Cancellation is refused: cancelling would
    fire the free-slot done-callback while the worker still holds the slot,
    letting it be re-allocated and corrupted. The future resolves when the
    worker's result arrives or worker-death cleanup rejects it."""

    def cancel(self) -> bool:
        return False


def _get_to_bytes():
    global _to_bytes
    if _to_bytes is None:
        from inference_model_manager.backends.subproc import _to_bytes as _tb

        _to_bytes = _tb
    return _to_bytes


class ModelManager:
    """Unified model management layer.

    Owns a collection of Backend instances, routes inference calls by model_id,
    and exposes lifecycle + observability for all loaded models.

    Two modes of operation:
      - **In-process** (default): used directly by HTTP server, Workflows, or
        InferencePipeline. No network layer — just call methods.
      - **Fleet mode**: wrapped by a ModelManagerService that adds ZMQ socket for
        receiving requests from the orchestrator, heartbeat reporting, and
        registration. ModelManager itself has zero network awareness.

    In-process usage::

        manager = ModelManager()
        manager.load("yolov8n-640", api_key=key, backend="direct")
        result = manager.process("yolov8n-640", images=image)

    Fleet usage::

        manager = ModelManager()
        service = ModelManagerService(manager, orchestrator_address="tcp://10.0.0.1:5555")
        service.start()  # blocks, listens for ZMQ commands

    Thread safety: all public methods are safe to call from multiple threads.
    The manager serializes load/unload operations internally. Inference calls
    are concurrent (bounded by the GPU execution semaphore).
    """

    def __init__(
        self,
        *,
        max_pinned_memory_mb: int = 0,
        n_slots: int = 32,
        input_mb: float = 20.0,
    ) -> None:
        """
        Args:
            max_pinned_memory_mb: Reserved for future use.
            n_slots: SHM slots for subprocess backends (shared pool).
            input_mb: MB per slot data area.
        """
        self._max_pinned_memory_bytes = max_pinned_memory_mb * 1024 * 1024
        self._n_slots = n_slots
        self._input_mb = input_mb

        self._backends: Dict[str, Backend] = {}
        self._lifecycle_lock = threading.Lock()
        # model_ids reserved by an in-progress load (built outside the lock)
        self._loading_ids: set[str] = set()

        # Shared-base owners by base_key (NOT in _backends — referenced by head views).
        # The sentinel + condition serialize first-creation so concurrent first heads
        # for one base_key spawn a single base worker, not two.
        self._shared_workers: Dict[str, Any] = {}
        self._shared_loading_keys: set[str] = set()
        self._shared_base_preloads: Dict[str, tuple[str, Any]] = {}
        self._shared_cv = threading.Condition(self._lifecycle_lock)
        # Optional hook (set by MMP) invoked on shared-base worker death, after the
        # cache entry is dropped, so the wrapper can clean up its hosted heads.
        self._shared_death_hook: Optional[Callable[[str], None]] = None

        # Shared SHM pool for subprocess backends — created lazily
        self._pool: Optional[Any] = None  # SHMPool, created on first subprocess load

        # Shared thread pool for DirectBackends and infer_async
        self._executor = ThreadPoolExecutor(
            max_workers=cfg.INFERENCE_DIRECT_MAX_WORKERS,
            thread_name_prefix="mm-worker",
        )

        # Pinned memory tracking: model_id → bytes used
        self._pinned_bytes: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(
        self,
        model_id: str,
        api_key: str,
        *,
        model_id_or_path: Optional[str] = None,
        backend: Literal["direct", "subprocess"] = "direct",
        device: Optional[str] = None,
        use_gpu: Optional[bool] = None,
        use_cuda_ipc: Optional[bool] = None,
        batch_max_size: int = 0,
        batch_max_delay_ms: float = 10.0,
        warmup_iters: int = 0,
        **kwargs,
    ) -> None:
        """Load a model and create its backend.

        Blocks until the model is loaded, warmed up, and ready to serve.

        Args:
            model_id: Unique key for routing (``process(model_id, ...)``,
                ``submit(model_id, ...)``). Also used as the model to load
                unless ``model_id_or_path`` is set.
            model_id_or_path: What to pass to ``AutoModel.from_pretrained``.
                Defaults to ``model_id``. Set this to load multiple
                instances of the same model under different routing keys::

                    mm.load("yolov8n-0", key, model_id_or_path="yolov8n-640")
                    mm.load("yolov8n-1", key, model_id_or_path="yolov8n-640")
            device: Device to run the model on (e.g. ``"cpu"``,
                ``"cuda:0"``, ``"cuda:1"``). For multi-GPU pods, this
                selects which GPU the model runs on. ``None`` uses the
                default (CUDA if available for DirectBackend, ``"cuda:0"``
                for SubprocessBackend).

        Raises:
            ValueError: If model_id is already loaded.
            RuntimeError: If loading fails (bad weights, OOM, download error).
        """
        load_target = model_id_or_path or model_id

        # Reserve under the lock; build OUTSIDE it. Holding the lock across
        # _create_backend (weight download + load, seconds to minutes) blocked
        # unload/stats/list_models and every other load for the duration.
        with self._lifecycle_lock:
            if model_id in self._backends or model_id in self._loading_ids:
                raise ValueError(f"Model '{model_id}' is already loaded")
            self._loading_ids.add(model_id)

        logger.info(
            "Loading '%s' (model=%s) with backend=%s, device=%s, batch_max_size=%d",
            model_id,
            load_target,
            backend,
            device,
            batch_max_size,
        )

        try:
            b = self._create_backend(
                model_id=load_target,
                api_key=api_key,
                backend=backend,
                device=device,
                use_gpu=use_gpu,
                use_cuda_ipc=use_cuda_ipc,
                batch_max_size=batch_max_size,
                batch_max_delay_ms=batch_max_delay_ms,
                **kwargs,
            )
            # Register model class in registry for task dispatch + serialization.
            from inference_model_manager.registry_defaults import (
                lazy_register,
                lazy_register_by_names,
            )

            if hasattr(b, "model") and b.model is not None:
                # DirectBackend — model instance available in-process.
                lazy_register(type(b.model))
            elif hasattr(b, "_model_mro_names") and b._model_mro_names:
                # SubprocessBackend — worker sent MRO class names in READY.
                lazy_register_by_names(b._model_mro_names)
            with self._lifecycle_lock:
                self._backends[model_id] = b
        finally:
            self._loading_ids.discard(model_id)

        # Warmup outside the lock — model is registered, other models can load
        if warmup_iters > 0:
            self._warmup(model_id, warmup_iters)

        logger.info(
            "Model '%s' loaded (state=%s, device=%s)",
            model_id,
            b.state,
            b.device,
        )

    def load_shared_head(
        self,
        head_id: str,
        api_key: str,
        resolution: Any,
        *,
        model_id_or_path: Optional[str] = None,
        device: Optional[str] = None,
        batch_max_size: int = 0,
        batch_max_delay_ms: float = 10.0,
        decoder: str = "imagecodecs",
    ) -> None:
        """Load ``head_id`` as a head sharing the base worker keyed by
        ``resolution.base_key``. Reuses the owner if present, else creates it once.

        On head-load failure this raises (the caller must NOT silently fall back to a
        normal subproc load — that would spawn a duplicate base). A failed first head
        tears the just-created base worker back down so it does not orphan.
        """
        base_key = resolution.base_key
        with self._lifecycle_lock:
            if head_id in self._backends or head_id in self._loading_ids:
                raise ValueError(f"Model '{head_id}' is already loaded")
            self._loading_ids.add(head_id)
        try:
            # The owner is returned with a load reservation already held (begin_load),
            # so it cannot reap itself between here and load_head even if a concurrent
            # head load for the same base fails. Released in end_load below.
            owner = self._get_or_create_owner(
                base_key,
                resolution,
                api_key,
                device=device,
                batch_max_size=batch_max_size,
                batch_max_delay_ms=batch_max_delay_ms,
                decoder=decoder,
            )
            try:
                metadata = owner.load_head(head_id, api_key, model_id_or_path)

                from inference_model_manager.registry_defaults import (
                    lazy_register_by_names,
                )

                if metadata.model_mro_names:
                    lazy_register_by_names(metadata.model_mro_names)
                from inference_model_manager.backends.shared_base import (
                    SharedHeadBackend,
                )

                view = SharedHeadBackend(owner, head_id, metadata)
                with self._lifecycle_lock:
                    self._backends[head_id] = view
            finally:
                # Releases the reservation; reaps the worker if this left it empty.
                owner.end_load()
        finally:
            self._loading_ids.discard(head_id)

    def load_shared_base(
        self,
        model_id: str,
        api_key: str,
        resolution: Any,
        *,
        device: Optional[str] = None,
        batch_max_size: int = 0,
        batch_max_delay_ms: float = 10.0,
        decoder: str = "imagecodecs",
    ) -> None:
        """Preload and retain a shared-base owner without loading a head."""
        base_key = resolution.base_key
        with self._lifecycle_lock:
            if (
                model_id in self._backends
                or model_id in self._loading_ids
                or model_id in self._shared_base_preloads
            ):
                raise ValueError(f"Model '{model_id}' is already loaded")
            self._loading_ids.add(model_id)
        try:
            owner = self._get_or_create_owner(
                base_key,
                resolution,
                api_key,
                device=device,
                batch_max_size=batch_max_size,
                batch_max_delay_ms=batch_max_delay_ms,
                decoder=decoder,
            )
            with self._lifecycle_lock:
                self._shared_base_preloads[model_id] = (base_key, owner)
        finally:
            self._loading_ids.discard(model_id)

    def unload_shared_base(self, model_id: str) -> None:
        """Release a retained base-only preload."""
        with self._lifecycle_lock:
            preload = self._shared_base_preloads.pop(model_id, None)
        if preload is None:
            return
        _base_key, owner = preload
        try:
            owner.end_load()
        except Exception:
            logger.warning(
                "Error releasing shared base preload '%s'", model_id, exc_info=True
            )

    def shared_base_preloads(self) -> Dict[str, str]:
        with self._lifecycle_lock:
            return {
                model_id: base_key
                for model_id, (base_key, _) in self._shared_base_preloads.items()
            }

    def _get_or_create_owner(
        self,
        base_key: str,
        resolution: Any,
        api_key: str,
        *,
        device: Optional[str],
        batch_max_size: int,
        batch_max_delay_ms: float,
        decoder: str,
    ) -> Any:
        with self._shared_cv:
            while True:
                owner = self._shared_workers.get(base_key)
                # begin_load atomically reserves against a concurrent retire; if it
                # loses (owner retired under us), fall through to create a fresh one.
                if owner is not None and owner.begin_load():
                    return owner
                if base_key not in self._shared_loading_keys:
                    self._shared_loading_keys.add(base_key)
                    break
                self._shared_cv.wait()  # another thread is creating it
        try:
            owner = self._make_shared_owner(
                base_key,
                resolution,
                api_key,
                device=device,
                batch_max_size=batch_max_size,
                batch_max_delay_ms=batch_max_delay_ms,
                decoder=decoder,
            )
            # Reserve before publishing. begin_load can still fail if the brand-new
            # worker died between __init__ and here — tear it down rather than publish
            # a dead owner.
            if not owner.begin_load():
                owner.unload()
                raise RuntimeError(
                    f"shared-base worker for {base_key!r} died during startup"
                )
        except Exception:
            with self._shared_cv:
                self._shared_loading_keys.discard(base_key)
                self._shared_cv.notify_all()
            raise
        with self._shared_cv:
            self._shared_workers[base_key] = owner
            self._shared_loading_keys.discard(base_key)
            self._shared_cv.notify_all()
        return owner

    def _make_shared_owner(
        self,
        base_key: str,
        resolution: Any,
        api_key: str,
        *,
        device: Optional[str],
        batch_max_size: int,
        batch_max_delay_ms: float,
        decoder: str,
    ) -> Any:
        from inference_model_manager.backends.shared_base import (
            SharedBaseSubprocessBackend,
        )

        pool = self._ensure_pool()
        return SharedBaseSubprocessBackend(
            base_key,
            resolution,
            api_key,
            shm_pool_name=pool.name,
            n_slots=self._n_slots,
            input_mb=self._input_mb,
            device=device,
            batch_max_size=batch_max_size,
            batch_max_delay_ms=batch_max_delay_ms,
            decoder=decoder,
            on_shared_worker_death=self._on_shared_worker_death,
            on_empty=self._retire_shared_owner,
        )

    def shared_owners(self) -> Dict[str, Any]:
        """Snapshot of live shared-base owners by base_key (for VRAM/pid attribution)."""
        with self._lifecycle_lock:
            return dict(self._shared_workers)

    def has_shared_base(self, base_key: str) -> bool:
        """True if a LIVE shared-base owner exists for ``base_key`` — same liveness as
        the load reservation (rejects a dead-but-still-cached owner), so callers don't
        skip admission for a base that is about to be respawned."""
        with self._lifecycle_lock:
            owner = self._shared_workers.get(base_key)
            return owner is not None and owner.alive

    def _on_shared_worker_death(self, base_key: str) -> None:
        with self._lifecycle_lock:
            self._shared_workers.pop(base_key, None)
            for model_id, (preload_base_key, _owner) in list(
                self._shared_base_preloads.items()
            ):
                if preload_base_key == base_key:
                    del self._shared_base_preloads[model_id]
        if self._shared_death_hook is not None:
            try:
                self._shared_death_hook(base_key)
            except Exception:
                logger.exception("shared-base death hook raised for '%s'", base_key)

    def _retire_shared_owner(self, base_key: str, owner: Any) -> None:
        # Drop the cache entry only if it is still this owner — a freshly created
        # replacement for the same base_key must survive.
        with self._lifecycle_lock:
            if self._shared_workers.get(base_key) is owner:
                del self._shared_workers[base_key]

    def _ensure_pool(self) -> Any:
        """Lazily create shared SHM pool on first subprocess backend load."""
        if self._pool is None:
            from inference_model_manager.backends.utils.shm_pool import SHMPool

            self._pool = SHMPool.create(self._n_slots, self._input_mb)
            logger.info(
                "ModelManager: SHM pool created  name=%s  slots=%d  data=%.0fMB",
                self._pool.name,
                self._n_slots,
                self._input_mb,
            )
        return self._pool

    def _create_backend(
        self,
        model_id: str,
        api_key: str,
        backend: str,
        **kwargs,
    ) -> Backend:
        if backend == "direct":
            from inference_model_manager.backends.direct import DirectBackend

            # DirectBackend uses shared executor for submit() when not batching
            return DirectBackend(
                model_id,
                api_key,
                executor=self._executor,
                **kwargs,
            )
        elif backend == "subprocess":
            from inference_model_manager.backends.subproc import SubprocessBackend

            pool = self._ensure_pool()
            return SubprocessBackend(
                model_id,
                api_key,
                shm_pool_name=pool.name,
                n_slots=self._n_slots,
                input_mb=self._input_mb,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unknown backend '{backend}'. Choose 'direct' or 'subprocess'."
            )

    def _warmup(self, model_id: str, iters: int) -> None:
        """Run synthetic inferences to warm up the model."""
        logger.info("Warming up '%s' with %d iterations", model_id, iters)
        try:
            import numpy as np

            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            for i in range(iters):
                self.process(model_id, images=dummy)
            logger.info("Warmup complete for '%s'", model_id)
        except Exception:
            logger.error(
                "Warmup failed for '%s' — model is loaded but may not perform optimally",
                model_id,
                exc_info=True,
            )

    def unload(
        self, model_id: str, *, drain: bool = False, drain_timeout_s: float = 30.0
    ) -> None:
        """Unload a model, releasing all GPU and CPU resources.

        Args:
            drain: If True, wait for in-flight requests to complete before
                killing the backend. If False (default), cancel immediately.
            drain_timeout_s: Max seconds to wait when draining.
        """
        with self._lifecycle_lock:
            backend = self._backends.pop(model_id, None)
            if backend is None:
                raise KeyError(f"Model '{model_id}' is not loaded")
            self._pinned_bytes.pop(model_id, None)

        if drain:
            logger.info(
                "Draining and unloading model '%s' (timeout=%.1fs)",
                model_id,
                drain_timeout_s,
            )
            backend.drain_and_unload(timeout_s=drain_timeout_s)
        else:
            logger.info("Unloading model '%s'", model_id)
            backend.unload()

    # ------------------------------------------------------------------
    # Processing — unified task dispatch
    # ------------------------------------------------------------------

    def process(
        self,
        model_id: str,
        task: Optional[str] = None,
        *,
        serialize: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Process a task on a loaded model. Blocks until result is ready.

        Uses the model registry to resolve ``task`` to the correct method.
        If ``task`` is None, the default task is used.

        ``serialize=False`` returns the raw prediction object without the
        registry-typed envelope — used by proxies whose callers serialize at
        the HTTP layer (keeps bundled and MMP modes on one contract).

        For direct backends, calls the model method in-process.
        For subprocess backends, submits via SHM and waits for result.

        Args:
            model_id: Loaded model key.
            task: Task name (e.g. ``"infer"``, ``"embed_text"``, ``"caption"``).
                None → default task for this model.
            **kwargs: Passed to the model method (images, texts, classes, prompt, etc.).

        Returns:
            Whatever the model method returns.

        Raises:
            KeyError: If model_id is not loaded.
            ValueError: If task is not supported by the model.
        """
        backend = self._get_backend(model_id)

        if hasattr(backend, "submit_slot"):
            raw_input = kwargs.pop("images", None)
            result = self.submit(
                model_id, task=task, raw_input=raw_input, **kwargs
            ).result(timeout=cfg.INFERENCE_PROCESS_TIMEOUT_S)
            if not serialize:
                return result
            # Serialize subprocess result through registry (model lives in worker,
            # parent only has MRO class name strings from READY pipe).
            mro_names = getattr(backend, "_model_mro_names", [])
            if mro_names:
                reg = _get_registry()
                task_name = (
                    task or reg.get_default_task_by_mro_names(mro_names) or "infer"
                )
                entry = reg.get_entry_by_mro_names(mro_names, task_name)
                if entry is not None:
                    return entry.serializer(result, backend)
            return result

        # Resolve task (validates it exists, raises ValueError if not)
        task_name, _entry = resolve_task(backend.model, task)

        # Validate kwargs through registry (if entry exists)
        kwargs = _get_registry().validate(backend.model, task_name, kwargs)

        t0 = time.monotonic()
        _begin = getattr(backend, "inflight_begin", None)
        if _begin is not None:
            _begin()
        try:
            result = invoke_task(backend.model, task=task, **kwargs)
        except Exception:
            backend.record_inference(t0, error=True)
            raise
        finally:
            _end = getattr(backend, "inflight_end", None)
            if _end is not None:
                _end()
        backend.record_inference(t0, error=False)

        if not serialize:
            return result

        # Serialize through registry (if entry exists)
        typed = _get_registry().serialize(backend.model, task_name, result)
        return typed if typed is not None else result

    async def process_async(
        self,
        model_id: str,
        task: Optional[str] = None,
        *,
        serialize: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Process a task asynchronously.

        Same as ``process()`` but non-blocking in an async context.

        Raises:
            KeyError: If model_id is not loaded.
            ValueError: If task is not supported by the model.
        """
        return await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: self.process(model_id, task=task, serialize=serialize, **kwargs),
        )

    def submit(
        self,
        model_id: str,
        *,
        task: Optional[str] = None,
        raw_input: Any = None,
        **kwargs,
    ) -> Future:
        """Submit for processing. Returns a Future immediately.

        For subprocess backends, allocates a SHM slot, writes input,
        and signals the worker with task + params.
        For direct backends, runs in thread pool via task dispatch.

        Args:
            model_id: Loaded model key.
            task: Task name. None → default.
            raw_input: Image bytes / numpy array for SHM path.
                For direct backend, pass images in kwargs instead.
            **kwargs: Additional params (forwarded to worker as params JSON,
                or passed directly to model method for direct backend).

        Raises:
            KeyError: If model_id is not loaded.
        """
        backend = self._get_backend(model_id)

        # Subprocess backend: we manage the pool
        if hasattr(backend, "submit_slot"):
            # Accept images via raw_input param or images kwarg
            if raw_input is None:
                raw_input = kwargs.pop("images", None)
            to_bytes = _get_to_bytes()
            input_bytes = to_bytes(raw_input) if raw_input is not None else b""
            if input_bytes and len(input_bytes) > self._pool.data_slot_bytes:
                raise ValueError(
                    f"Input {len(input_bytes)} B > slot capacity "
                    f"{self._pool.data_slot_bytes} B — increase input_mb"
                )

            # Validate through the registry BEFORE consuming a slot — same
            # contract as the direct path, which validates in process().
            mro_names = getattr(backend, "_model_mro_names", [])
            if mro_names:
                reg = _get_registry()
                task_name = task or reg.get_default_task_by_mro_names(mro_names)
                if task_name:
                    entry = reg.get_entry_by_mro_names(mro_names, task_name)
                    if entry is not None:
                        vkwargs = dict(kwargs)
                        if raw_input is not None:
                            vkwargs["images"] = raw_input
                        entry.validator(vkwargs)

            # Pack task + kwargs into params_bytes for worker — BEFORE the
            # slot alloc so an encoding failure cannot leak a slot.
            params = dict(kwargs)
            if task is not None:
                params["task"] = task
            try:
                params_bytes = json.dumps(params).encode()
            except TypeError as exc:
                # default=str silently sent numpy arrays as their repr — an
                # explicit error beats garbage params at the worker.
                raise ValueError(
                    "params not JSON-serializable for subprocess backend "
                    f"(keys: {sorted(params)}): {exc}"
                ) from exc

            req_id = uuid.uuid4().int & 0xFFFF_FFFF_FFFF_FFFF
            slot_id = self._pool.alloc_slot()
            self._pool.mark_allocated(slot_id, req_id)
            if input_bytes:
                self._pool.data_memoryview(slot_id)[: len(input_bytes)] = input_bytes
            self._pool.mark_written(slot_id, len(input_bytes))

            future: Future = _SlotFuture()

            def _on_done(f: Future) -> None:
                try:
                    self._pool.free_slot(slot_id, request_id=req_id)
                except Exception:
                    pass

            future.add_done_callback(_on_done)
            try:
                backend.submit_slot(slot_id, req_id, future, params_bytes)
            except Exception as exc:
                # Dead recv thread etc. — fail the future; the done-callback
                # frees the slot, so nothing leaks.
                if not future.done():
                    future.set_exception(exc)
            return future

        # Direct backend: validate sync, run in thread pool, record stats.
        if not backend.is_accepting:
            raise RuntimeError(
                f"Backend '{model_id}' not accepting requests (state={backend.state})"
            )
        task_name, _ = resolve_task(backend.model, task)
        kwargs = _get_registry().validate(backend.model, task_name, kwargs)

        def _run():
            t0 = time.monotonic()
            _begin = getattr(backend, "inflight_begin", None)
            if _begin is not None:
                _begin()
            try:
                result = invoke_task(backend.model, task=task, **kwargs)
            except Exception:
                backend.record_inference(t0, error=True)
                raise
            finally:
                _end = getattr(backend, "inflight_end", None)
                if _end is not None:
                    _end()
            backend.record_inference(t0, error=False)
            return result

        return self._executor.submit(_run)

    def get_supported_tasks(self, model_id: str) -> Dict[str, Any]:
        """Return supported tasks for a loaded model.

        Works for both DirectBackend (has model instance) and
        SubprocessBackend (has MRO class names from worker READY pipe).

        Raises:
            KeyError: If model_id is not loaded.
        """
        backend = self._get_backend(model_id)
        mro_names = getattr(backend, "_model_mro_names", None)
        if mro_names:
            from inference_model_manager.dispatch import list_tasks_by_mro_names

            return list_tasks_by_mro_names(mro_names)
        from inference_model_manager.dispatch import list_tasks

        return list_tasks(backend.model)

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Snapshot of the full manager state.

        Non-blocking — never contends with inference.
        """
        gpu_info = self._gpu_memory_info()
        models = []

        with self._lifecycle_lock:
            backends_snapshot = list(self._backends.items())

        for model_id, backend in backends_snapshot:
            s = backend.stats()
            s["model_id"] = model_id
            models.append(s)

        return {
            "gpus": gpu_info,
            "models_loaded": self.loaded_models,
            "models": models,
        }

    def model_stats(self, model_id: str) -> Dict[str, Any]:
        """Stats for a single model. Forces worker stats refresh for subprocess backends."""
        backend = self._get_backend(model_id)
        if hasattr(backend, "refresh_worker_stats"):
            backend.refresh_worker_stats(timeout_s=1.0)
        s = backend.stats()
        s["model_id"] = model_id
        return s

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_ready(self, model_id: str) -> bool:
        """Whether model_id is loaded and accepting requests."""
        backend = self._backends.get(model_id)
        return backend is not None and backend.is_accepting

    def health(self, model_id: str) -> str:
        """Health status for a model.

        Returns one of: 'not_loaded', 'loading', 'loaded', 'draining',
        'unhealthy'.
        """
        backend = self._backends.get(model_id)
        if backend is None:
            return "not_loaded"
        return backend.state

    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models with state, device, queue depth, health."""
        with self._lifecycle_lock:
            backends_snapshot = list(self._backends.items())

        result = []
        for model_id, backend in backends_snapshot:
            result.append(
                {
                    "model_id": model_id,
                    "state": backend.state,
                    "device": backend.device,
                    "is_accepting": backend.is_accepting,
                    "queue_depth": backend.queue_depth,
                    "worker_pid": backend.worker_pid,
                }
            )
        return result

    def get_backend(self, model_id: str) -> Optional[Backend]:
        """Return Backend for model_id, or None if not loaded."""
        return self._backends.get(model_id)

    def _get_backend(self, model_id: str) -> Backend:
        try:
            return self._backends[model_id]
        except KeyError:
            raise KeyError(f"Model '{model_id}' is not loaded") from None

    def __contains__(self, model_id: str) -> bool:
        return model_id in self._backends

    def __len__(self) -> int:
        return len(self._backends)

    @property
    def loaded_models(self) -> List[str]:
        return [mid for mid, b in self._backends.items() if b.state == "loaded"]

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Unload all models and shut down the executor.

        Call this when the process is exiting.
        """
        with self._lifecycle_lock:
            model_ids = list(self._backends.keys())

        for model_id in model_ids:
            try:
                self.unload(model_id)
            except Exception:
                logger.warning(
                    "Error unloading '%s' during shutdown", model_id, exc_info=True
                )

        for model_id in list(self.shared_base_preloads()):
            try:
                self.unload_shared_base(model_id)
            except Exception:
                logger.warning(
                    "Error unloading shared base '%s' during shutdown",
                    model_id,
                    exc_info=True,
                )

        self._executor.shutdown(wait=True, cancel_futures=True)

        if self._pool is not None:
            self._pool.close()
            self._pool = None

        logger.info("ModelManager shut down")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _gpu_memory_info() -> List[Dict[str, Any]]:
        """Get per-GPU memory usage. Returns empty list if CUDA unavailable."""
        try:
            import torch

            if torch.cuda.is_available():
                gpus = []
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    gpus.append(
                        {
                            "device": f"cuda:{i}",
                            "name": props.name,
                            "total_mb": round(props.total_mem / 1024 / 1024, 1),
                            "allocated_mb": round(
                                torch.cuda.memory_allocated(i) / 1024 / 1024,
                                1,
                            ),
                            "reserved_mb": round(
                                torch.cuda.memory_reserved(i) / 1024 / 1024,
                                1,
                            ),
                        }
                    )
                return gpus
        except Exception:
            pass
        return []
