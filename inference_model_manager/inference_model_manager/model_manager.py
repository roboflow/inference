from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from inference_model_manager import configuration as cfg
from inference_model_manager.backends.base import Backend
from inference_model_manager.dispatch import _get_registry, invoke_task, resolve_task
from inference_model_manager.marshalling import (
    model_supports_rle,
    tensors_to_numpy,
)
from inference_models.utils.performance import performance_profiler

logger = logging.getLogger(__name__)


BACKEND_FACTORIES: Dict[str, Callable[..., "Backend"]] = {}
_ENTRY_POINT_BACKENDS_LOADED = False


def register_backend_factory(name: str, factory: Callable[..., "Backend"]) -> None:
    BACKEND_FACTORIES[name] = factory


def _direct_backend_factory(model_id, api_key, *, manager, **kwargs):
    from inference_model_manager.backends.direct import DirectBackend

    return DirectBackend(model_id, api_key, executor=manager.executor, **kwargs)


register_backend_factory("direct", _direct_backend_factory)


_ENTRY_POINT_BACKENDS_LOCK = threading.Lock()


def _load_entry_point_backends() -> None:
    global _ENTRY_POINT_BACKENDS_LOADED
    if _ENTRY_POINT_BACKENDS_LOADED:
        return
    with _ENTRY_POINT_BACKENDS_LOCK:
        if _ENTRY_POINT_BACKENDS_LOADED:
            return
        import importlib.metadata as md

        for ep in md.entry_points(group="inference_model_manager.backends"):
            if ep.name not in BACKEND_FACTORIES:
                register_backend_factory(ep.name, ep.load())
        _ENTRY_POINT_BACKENDS_LOADED = True


def _reset_entry_point_backends_for_tests() -> None:
    global _ENTRY_POINT_BACKENDS_LOADED
    with _ENTRY_POINT_BACKENDS_LOCK:
        _ENTRY_POINT_BACKENDS_LOADED = False


class ModelManager:
    """Unified model management layer.

    Owns a collection of Backend instances, routes inference calls by model_id,
    and exposes lifecycle + observability for all loaded models.

    Backends fall into two kinds:
      - **Direct**: the model instance lives in-process; ModelManager dispatches
        tasks against it through the task registry.
      - **submit_request**: the model does not live in-process; ModelManager
        routes ``process()``/``submit()`` through the backend's
        ``submit_request()`` method instead. Community/plugin backends
        registered via the ``inference_model_manager.backends`` entry point
        can implement either kind.

    Usage::

        manager = ModelManager()
        manager.load("yolov8n-640", api_key=key, backend="direct")
        result = manager.process("yolov8n-640", images=image)

    Thread safety: all public methods are safe to call from multiple threads.
    The manager serializes load/unload operations internally. Inference calls
    are concurrent (bounded by the GPU execution semaphore).
    """

    def __init__(self) -> None:
        self._backends: Dict[str, Backend] = {}
        self._lifecycle_lock = threading.Lock()
        # model_ids reserved by an in-progress load (built outside the lock)
        self._loading_ids: set[str] = set()
        # Set by shutdown() before draining — closes admission so nothing new
        # is queued while in-flight work finishes.
        self._closed = False

        # Shared thread pool for DirectBackends and infer_async
        self._executor = ThreadPoolExecutor(
            max_workers=cfg.INFERENCE_DIRECT_MAX_WORKERS,
            thread_name_prefix="mm-worker",
        )
        performance_profiler.set_metadata(
            "manager.direct_max_workers", cfg.INFERENCE_DIRECT_MAX_WORKERS
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(
        self,
        model_id: str,
        api_key: str,
        *,
        model_id_or_path: Optional[str] = None,
        backend: str = "direct",
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
                backend's own default (CUDA if available for DirectBackend).

        Raises:
            ValueError: If model_id is already loaded.
            RuntimeError: If loading fails (bad weights, OOM, download error).
        """
        self._check_open()
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
                # Out-of-process backend — reports MRO class names instead of a model instance.
                lazy_register_by_names(b._model_mro_names)
            with self._lifecycle_lock:
                self._backends[model_id] = b
            performance_profiler.set_metadata("manager.model_id", model_id)
            performance_profiler.set_metadata("manager.backend", backend)
            performance_profiler.set_metadata("manager.device", b.device)
            model = getattr(b, "model", None)
            if model is not None:
                performance_profiler.set_metadata(
                    "manager.model_class", type(model).__name__
                )
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

    @property
    def executor(self) -> ThreadPoolExecutor:
        return self._executor

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("ModelManager is shutting down")

    def _create_backend(
        self,
        model_id: str,
        api_key: str,
        backend: str,
        **kwargs,
    ) -> Backend:
        factory = BACKEND_FACTORIES.get(backend)
        if factory is None:
            _load_entry_point_backends()
            factory = BACKEND_FACTORIES.get(backend)
        if factory is None:
            raise ValueError(
                f"Unknown backend '{backend}'. Known: {sorted(BACKEND_FACTORIES)}"
            )
        return factory(model_id, api_key, manager=self, **kwargs)

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

    def _wire_marshal_inputs(self, backend: Any, kwargs: dict) -> tuple:
        """Direct-backend half of the subprocess worker's input handling:
        decode encoded image bytes and inject mask_format=rle for models
        that support it. Returns (kwargs, n_images) for result mapping."""
        images = kwargs.get("images")
        n_images = len(images) if isinstance(images, list) else 1
        if images is None:
            # The worker invokes params-only requests WITHOUT an images kwarg
            # (zero-byte slot); mirror that exactly.
            kwargs.pop("images", None)
        else:
            decode = getattr(backend, "_decode_input", None)
            if decode is not None:
                if isinstance(images, list):
                    kwargs["images"] = [decode(image) for image in images]
                else:
                    kwargs["images"] = decode(images)
        if model_supports_rle(backend.model) and "mask_format" not in kwargs:
            kwargs["mask_format"] = "rle"
        return kwargs, n_images

    @staticmethod
    def _wire_marshal_result(
        raw_out: Any, n_images: int, retry_single: Optional[Callable] = None
    ) -> Any:
        """Direct-backend half of the worker's result handling: the same
        per-image mapping as the worker's sub_results block (including the
        per-image retry when a batched call returns a mismatched shape),
        then tensors -> CPU numpy."""
        if isinstance(raw_out, list) and (n_images == 1 or len(raw_out) == n_images):
            results = raw_out
        else:
            shape = getattr(raw_out, "shape", None)
            if shape and n_images > 1 and shape[0] == n_images:
                results = [raw_out[i : i + 1] for i in range(n_images)]
            elif n_images == 1:
                results = [raw_out]
            elif retry_single is not None:
                results = []
                for index in range(n_images):
                    single_out = retry_single(index)
                    if isinstance(single_out, list):
                        single_out = single_out[0] if single_out else None
                    results.append(single_out)
            else:
                results = [raw_out]
        results = [tensors_to_numpy(result) for result in results]
        return results[0] if n_images == 1 else results

    def process(
        self,
        model_id: str,
        task: Optional[str] = None,
        *,
        serialize: bool = True,
        wire_marshalling: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Process a task on a loaded model. Blocks until result is ready.

        Uses the model registry to resolve ``task`` to the correct method.
        If ``task`` is None, the default task is used.

        ``serialize=False`` returns the raw prediction object without the
        registry-typed envelope — used by proxies whose callers serialize at
        the HTTP layer (keeps bundled and MMP modes on one contract).

        For direct backends, calls the model method in-process.
        For backends implementing ``submit_request`` (community/plugin
        backends), submits and waits for the result.

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
        self._check_open()
        backend = self._get_backend(model_id)

        if hasattr(backend, "submit_request"):
            raw_input = kwargs.pop("images", None)
            result = self.submit(
                model_id, task=task, raw_input=raw_input, **kwargs
            ).result(timeout=cfg.INFERENCE_PROCESS_TIMEOUT_S)
            if not serialize:
                return result
            # Serialize the result through the registry using the MRO class
            # names the backend reports (the model may live out-of-process,
            # so the manager doesn't import the real class).
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

        task_setup_ns = 0
        task_setup_started = performance_profiler.start()
        try:
            # Resolve task (validates it exists, raises ValueError if not)
            task_name, _entry = resolve_task(backend.model, task)
        finally:
            if task_setup_started is not None:
                task_setup_ns += time.perf_counter_ns() - task_setup_started

        n_images = 1
        if wire_marshalling:
            decode_started = performance_profiler.start()
            performance_profiler.increment("manager.input_decode.calls")
            try:
                kwargs, n_images = self._wire_marshal_inputs(backend, kwargs)
            finally:
                performance_profiler.stop("manager.input_decode", decode_started)

        task_setup_started = performance_profiler.start()
        try:
            # Validate kwargs through registry (if entry exists)
            kwargs = _get_registry().validate(backend.model, task_name, kwargs)
        finally:
            if task_setup_started is not None:
                task_setup_ns += time.perf_counter_ns() - task_setup_started
                performance_profiler.record(
                    "manager.task_setup", task_setup_ns / 1_000_000, "ms"
                )

        t0 = time.monotonic()
        _begin = getattr(backend, "inflight_begin", None)
        if _begin is not None:
            inflight_started = performance_profiler.start()
            try:
                _begin()
            finally:
                performance_profiler.stop("manager.inflight_wait", inflight_started)
        try:
            invoke_started = performance_profiler.start()
            performance_profiler.increment("manager.model_invoke.calls")
            try:
                result = invoke_task(backend.model, task=task, **kwargs)
            finally:
                performance_profiler.stop("manager.model_invoke", invoke_started)
            if wire_marshalling:
                # Inside the inflight/accounting window: per-image retries are
                # inference too — unload drains must wait for them and their
                # failures must count as errors.
                images = kwargs.get("images")
                retry_invoke_ns = 0

                def _retry_single(index: int) -> Any:
                    nonlocal retry_invoke_ns
                    single_kwargs = dict(kwargs)
                    single_kwargs["images"] = images[index]
                    retry_started = performance_profiler.start()
                    performance_profiler.increment("manager.model_invoke.calls")
                    try:
                        return invoke_task(backend.model, task=task, **single_kwargs)
                    finally:
                        if retry_started is not None:
                            retry_ended = time.perf_counter_ns()
                            retry_invoke_ns += retry_ended - retry_started
                            performance_profiler.stop(
                                "manager.model_invoke", retry_started, retry_ended
                            )

                marshal_started = performance_profiler.start()
                performance_profiler.increment("manager.result_marshal.calls")
                try:
                    result = self._wire_marshal_result(
                        result,
                        n_images,
                        retry_single=(
                            _retry_single if isinstance(images, list) else None
                        ),
                    )
                finally:
                    if marshal_started is not None:
                        marshal_ended = time.perf_counter_ns()
                        performance_profiler.record(
                            "manager.result_marshal",
                            max(0, marshal_ended - marshal_started - retry_invoke_ns)
                            / 1_000_000,
                            "ms",
                        )
            if serialize:
                # Inside the in-flight lease: serialization still reads
                # backend.model, which an unload would drop underneath it.
                typed = _get_registry().serialize(backend.model, task_name, result)
                if typed is not None:
                    result = typed
        except Exception:
            backend.record_inference(t0, error=True)
            raise
        finally:
            _end = getattr(backend, "inflight_end", None)
            if _end is not None:
                _end()
        backend.record_inference(t0, error=False)
        return result

    async def process_async(
        self,
        model_id: str,
        task: Optional[str] = None,
        *,
        serialize: bool = True,
        wire_marshalling: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Process a task asynchronously.

        Same as ``process()`` but non-blocking in an async context.

        Raises:
            KeyError: If model_id is not loaded.
            ValueError: If task is not supported by the model.
        """
        self._check_open()
        loop = asyncio.get_running_loop()
        if not performance_profiler.enabled:
            return await loop.run_in_executor(
                self._executor,
                lambda: self.process(
                    model_id,
                    task=task,
                    serialize=serialize,
                    wire_marshalling=wire_marshalling,
                    **kwargs,
                ),
            )

        queue_started = performance_profiler.start()
        return_started = [None]

        def _run() -> Any:
            performance_profiler.stop("manager.executor.queue", queue_started)
            process_started = performance_profiler.start()
            try:
                return self.process(
                    model_id,
                    task=task,
                    serialize=serialize,
                    wire_marshalling=wire_marshalling,
                    **kwargs,
                )
            finally:
                process_ended = time.perf_counter_ns()
                performance_profiler.stop(
                    "manager.process.total", process_started, process_ended
                )
                return_started[0] = process_ended

        try:
            result = await loop.run_in_executor(self._executor, _run)
        except asyncio.CancelledError:
            performance_profiler.increment("manager.executor.cancelled")
            raise
        except Exception:
            executor_ended = time.perf_counter_ns()
            performance_profiler.stop(
                "manager.return", return_started[0], executor_ended
            )
            performance_profiler.stop(
                "manager.executor.total", queue_started, executor_ended
            )
            performance_profiler.increment("manager.executor.errors")
            raise
        else:
            executor_ended = time.perf_counter_ns()
            performance_profiler.stop(
                "manager.return", return_started[0], executor_ended
            )
            performance_profiler.stop(
                "manager.executor.total", queue_started, executor_ended
            )
            return result

    def submit(
        self,
        model_id: str,
        *,
        task: Optional[str] = None,
        raw_input: Any = None,
        **kwargs,
    ) -> Future:
        """Submit for processing. Returns a Future immediately.

        For backends implementing ``submit_request`` (community/plugin
        backends), forwards task + params and returns their Future.
        For direct backends, runs in thread pool via task dispatch.

        Args:
            model_id: Loaded model key.
            task: Task name. None → default.
            raw_input: Passed to ``submit_request`` backends; direct backends
                take images via kwargs instead.
            **kwargs: Additional params (forwarded to the backend's
                submit_request, or passed directly to model method for
                direct backend).

        Raises:
            KeyError: If model_id is not loaded.
        """
        self._check_open()
        backend = self._get_backend(model_id)

        if hasattr(backend, "submit_request"):
            if raw_input is None:
                raw_input = kwargs.pop("images", None)
            validate = None
            mro_names = getattr(backend, "_model_mro_names", [])
            if mro_names:
                reg = _get_registry()
                task_name = task or reg.get_default_task_by_mro_names(mro_names)
                if task_name:
                    entry = reg.get_entry_by_mro_names(mro_names, task_name)
                    if entry is not None:
                        validate = entry.validator
            return backend.submit_request(
                task=task, raw_input=raw_input, validate=validate, **kwargs
            )

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

        Works for both DirectBackend (has model instance) and backends
        reporting MRO class names instead (community/plugin backends).

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
            # Route-resolution fields the MMP overlays on its stats; kept here
            # too so in-process consumers (ModelManagerGateway) see one shape.
            try:
                mro_names = getattr(backend, "_model_mro_names", None)
                if not mro_names:
                    model = getattr(backend, "model", None)
                    if model is not None:
                        mro_names = [cls.__name__ for cls in type(model).__mro__]
            except Exception:
                mro_names = None
            s["model_mro_names"] = mro_names
            try:
                s["class_names"] = backend.class_names
            except Exception:
                s["class_names"] = None
            try:
                s["key_points_classes"] = getattr(backend, "key_points_classes", None)
            except Exception:
                s["key_points_classes"] = None
            try:
                s["tasks"] = self.get_supported_tasks(model_id)
            except Exception:
                s["tasks"] = {}
            models.append(s)

        return {
            "gpus": gpu_info,
            "models_loaded": self.loaded_models,
            "models": models,
        }

    def model_stats(self, model_id: str) -> Dict[str, Any]:
        """Stats for a single model. Refreshes worker stats for backends that support it."""
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

    def is_healthy(self, model_id: str) -> bool:
        backend = self._backends.get(model_id)
        if backend is None:
            return False
        try:
            return bool(backend.is_healthy)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Close admission, drain in-flight work, then unload all models.

        Call this when the process is exiting. Unloading before the drain tore
        models out from under inference still running in the executor.
        """
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)

        with self._lifecycle_lock:
            model_ids = list(self._backends.keys())

        for model_id in model_ids:
            try:
                self.unload(model_id)
            except Exception:
                logger.warning(
                    "Error unloading '%s' during shutdown", model_id, exc_info=True
                )

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
