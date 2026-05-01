from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Literal, Optional

from inference_model_manager.backends.base import Backend
from inference_model_manager.dispatch import _get_registry, invoke_task, resolve_task

logger = logging.getLogger(__name__)

# Import lazily to avoid circular deps
_to_bytes = None


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

        # Shared SHM pool for subprocess backends — created lazily
        self._pool: Optional[Any] = None  # SHMPool, created on first subprocess load

        # Shared thread pool for DirectBackends and infer_async
        self._executor = ThreadPoolExecutor(
            max_workers=8,
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

        with self._lifecycle_lock:
            if model_id in self._backends:
                raise ValueError(f"Model '{model_id}' is already loaded")

            logger.info(
                "Loading '%s' (model=%s) with backend=%s, device=%s, batch_max_size=%d",
                model_id,
                load_target,
                backend,
                device,
                batch_max_size,
            )

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
            self._backends[model_id] = b

        # Warmup outside the lock — model is registered, other models can load
        if warmup_iters > 0:
            self._warmup(model_id, warmup_iters)

        logger.info(
            "Model '%s' loaded (state=%s, device=%s)",
            model_id,
            b.state,
            b.device,
        )

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

    def process(self, model_id: str, task: Optional[str] = None, **kwargs: Any) -> Any:
        """Process a task on a loaded model. Blocks until result is ready.

        Uses the model registry to resolve ``task`` to the correct method.
        If ``task`` is None, the default task is used.

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
            ).result()
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
        try:
            result = invoke_task(backend.model, task=task, **kwargs)
        except Exception:
            backend.record_inference(t0, error=True)
            raise
        backend.record_inference(t0, error=False)

        # Serialize through registry (if entry exists)
        typed = _get_registry().serialize(backend.model, task_name, result)
        return typed if typed is not None else result

    async def process_async(
        self, model_id: str, task: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Process a task asynchronously.

        Same as ``process()`` but non-blocking in an async context.

        Raises:
            KeyError: If model_id is not loaded.
            ValueError: If task is not supported by the model.
        """
        backend = self._get_backend(model_id)

        if hasattr(backend, "submit_slot"):
            loop = asyncio.get_running_loop()
            raw_input = kwargs.pop("images", None)
            return await loop.run_in_executor(
                None,
                lambda: self.submit(
                    model_id, task=task, raw_input=raw_input, **kwargs
                ).result(),
            )

        return await asyncio.get_running_loop().run_in_executor(
            None, lambda: self.process(model_id, task=task, **kwargs)
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

            req_id = uuid.uuid4().int & 0xFFFF_FFFF_FFFF_FFFF
            slot_id = self._pool.alloc_slot()
            self._pool.mark_allocated(slot_id, req_id)
            if input_bytes:
                self._pool.data_memoryview(slot_id)[: len(input_bytes)] = input_bytes
            self._pool.mark_written(slot_id, len(input_bytes))

            # Pack task + kwargs into params_bytes for worker
            params = dict(kwargs)
            if task is not None:
                params["task"] = task
            params_bytes = json.dumps(params, default=str).encode()

            future: Future = Future()

            def _on_done(f: Future) -> None:
                try:
                    self._pool.free_slot(slot_id)
                except Exception:
                    pass

            future.add_done_callback(_on_done)
            backend.submit_slot(slot_id, req_id, future, params_bytes)
            return future

        # Direct backend: run in thread pool via task dispatch
        future = self._executor.submit(invoke_task, backend.model, task=task, **kwargs)
        return future

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
