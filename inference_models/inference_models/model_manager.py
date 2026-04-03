from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Literal, Optional

from inference_models.backends.base import Backend

logger = logging.getLogger(__name__)


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
        result = manager.infer_sync("yolov8n-640", image)

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
        gpu_exec_slots: int = 4,
        max_pinned_memory_mb: int = 0,
    ) -> None:
        """
        Args:
            gpu_exec_slots: Total GPU execution slots for this pod. Controls
                how many concurrent forward() calls can run across all backends.
                Each backend consumes a configurable number of slots.
            max_pinned_memory_mb: Maximum CPU pinned memory for sleeping models.
                0 = no sleeping tier (models go straight from loaded to unloaded).
        """
        self._gpu_exec_slots_total = gpu_exec_slots
        self._max_pinned_memory_bytes = max_pinned_memory_mb * 1024 * 1024

        self._backends: Dict[str, Backend] = {}
        self._lifecycle_lock = threading.Lock()

        # Shared thread pool for DirectBackends and infer_async
        self._executor = ThreadPoolExecutor(
            max_workers=max(gpu_exec_slots * 2, 4),
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
        backend: Literal["direct", "subprocess"] = "direct",
        use_gpu: Optional[bool] = None,
        use_cuda_ipc: Optional[bool] = None,
        batch_max_size: int = 0,
        batch_max_delay_ms: float = 10.0,
        warmup_iters: int = 0,
        gpu_exec_slots: int = 1,
        **kwargs,
    ) -> None:
        """Load a model and create its backend.

        Blocks until the model is loaded, warmed up, and ready to serve.

        Raises:
            ValueError: If model_id is already loaded.
            RuntimeError: If loading fails (bad weights, OOM, download error).
        """
        with self._lifecycle_lock:
            if model_id in self._backends:
                raise ValueError(f"Model '{model_id}' is already loaded")

            logger.info(
                "Loading model '%s' with backend=%s, batch_max_size=%d",
                model_id, backend, batch_max_size,
            )

            b = self._create_backend(
                model_id=model_id,
                api_key=api_key,
                backend=backend,
                use_gpu=use_gpu,
                use_cuda_ipc=use_cuda_ipc,
                batch_max_size=batch_max_size,
                batch_max_delay_ms=batch_max_delay_ms,
                gpu_exec_slots=gpu_exec_slots,
                **kwargs,
            )
            self._backends[model_id] = b

        # Warmup outside the lock — model is registered, other models can load
        if warmup_iters > 0:
            self._warmup(model_id, warmup_iters)

        logger.info("Model '%s' loaded (state=%s)", model_id, b.state)

    def _create_backend(
        self,
        model_id: str,
        api_key: str,
        backend: str,
        **kwargs,
    ) -> Backend:
        if backend == "direct":
            from inference_models.backends.direct import DirectBackend

            # DirectBackend uses shared executor for submit() when not batching
            return DirectBackend(
                model_id, api_key,
                executor=self._executor,
                **kwargs,
            )
        elif backend == "subprocess":
            from inference_models.backends.subproc import SubprocessBackend

            return SubprocessBackend(model_id, api_key, **kwargs)
        else:
            raise ValueError(
                f"Unknown backend '{backend}'. Choose 'direct' or 'subprocess'."
            )

    def _warmup(self, model_id: str, iters: int) -> None:
        """Run synthetic inferences to warm up the model."""
        backend = self._backends[model_id]
        logger.info("Warming up '%s' with %d iterations", model_id, iters)
        try:
            import numpy as np

            # Use a small synthetic image — model pre_process handles resizing
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            for i in range(iters):
                backend.infer_sync(dummy)
            logger.info("Warmup complete for '%s'", model_id)
        except Exception:
            logger.warning(
                "Warmup failed for '%s' — model is loaded but not warmed up",
                model_id, exc_info=True,
            )

    def unload(self, model_id: str) -> None:
        """Fully unload a model, releasing all GPU and CPU resources."""
        with self._lifecycle_lock:
            backend = self._backends.pop(model_id, None)
            if backend is None:
                raise KeyError(f"Model '{model_id}' is not loaded")
            self._pinned_bytes.pop(model_id, None)

        logger.info("Unloading model '%s'", model_id)
        backend.unload()

    def sleep(self, model_id: str) -> None:
        """Offload model weights to CPU pinned memory, freeing VRAM.

        Raises:
            RuntimeError: If max_pinned_memory_mb would be exceeded.
            KeyError: If model_id is not loaded.
        """
        with self._lifecycle_lock:
            backend = self._get_backend(model_id)
            pinned = backend.sleep()

            if pinned is not None and pinned > 0:
                total_after = sum(self._pinned_bytes.values()) + pinned
                if (
                    self._max_pinned_memory_bytes > 0
                    and total_after > self._max_pinned_memory_bytes
                ):
                    # Roll back — wake the model back up
                    backend.wake()
                    raise RuntimeError(
                        f"Cannot sleep '{model_id}': would use "
                        f"{total_after / 1024 / 1024:.0f}MB pinned memory, "
                        f"exceeding limit of "
                        f"{self._max_pinned_memory_bytes / 1024 / 1024:.0f}MB"
                    )
                self._pinned_bytes[model_id] = pinned

        logger.info(
            "Model '%s' sleeping (pinned=%s bytes)",
            model_id,
            pinned if pinned is not None else "N/A",
        )

    def wake(self, model_id: str) -> None:
        """Reload a sleeping model's weights from CPU pinned memory to GPU.

        Raises:
            KeyError: If model_id is not registered.
            RuntimeError: If model is not in sleeping state.
        """
        with self._lifecycle_lock:
            backend = self._get_backend(model_id)
            backend.wake()
            self._pinned_bytes.pop(model_id, None)

        logger.info("Model '%s' woken up", model_id)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer_sync(self, model_id: str, *args, **kwargs) -> Any:
        """Run inference synchronously. Blocks until result is ready.

        Raises:
            KeyError: If model_id is not loaded.
        """
        backend = self._get_backend(model_id)
        return backend.infer_sync(*args, **kwargs)

    async def infer_async(self, model_id: str, *args, **kwargs) -> Any:
        """Run inference asynchronously.

        Raises:
            KeyError: If model_id is not loaded.
        """
        backend = self._get_backend(model_id)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: backend.infer_sync(*args, **kwargs),
        )

    def submit(self, model_id: str, pre_processed: Any, *, priority: int = 0) -> Future:
        """Submit a pre-processed item for inference.

        Call ``backend.pre_process()`` first, then pass the result here.
        Returns a Future that resolves to the final post-processed result.

        Raises:
            KeyError: If model_id is not loaded.
        """
        backend = self._get_backend(model_id)
        return backend.submit(pre_processed, priority=priority)

    def pre_process(self, model_id: str, *args, **kwargs) -> Any:
        """Run pre-processing for a model. Returns (tensor, meta).

        Useful when callers want to pre-process + submit separately
        (e.g. to submit multiple items concurrently).
        """
        backend = self._get_backend(model_id)
        return backend.pre_process(*args, **kwargs)

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Snapshot of the full manager state.

        Non-blocking — never contends with inference.
        """
        gpu_mem = self._gpu_memory_info()
        models = []
        gpu_exec_slots_used = 0

        for model_id, backend in self._backends.items():
            s = backend.stats()
            s["model_id"] = model_id
            models.append(s)
            if backend.state == "loaded":
                gpu_exec_slots_used += backend.gpu_exec_slots

        return {
            "gpu_memory_used_mb": gpu_mem["used_mb"],
            "gpu_memory_total_mb": gpu_mem["total_mb"],
            "ram_pinned_used_mb": sum(self._pinned_bytes.values()) / 1024 / 1024,
            "gpu_exec_slots_used": gpu_exec_slots_used,
            "gpu_exec_slots_total": self._gpu_exec_slots_total,
            "models_loaded": self.loaded_models,
            "models_sleeping": self.sleeping_models,
            "models": models,
        }

    def model_stats(self, model_id: str) -> Dict[str, Any]:
        """Stats for a single model."""
        backend = self._get_backend(model_id)
        s = backend.stats()
        s["model_id"] = model_id
        return s

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

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
        return [
            mid for mid, b in self._backends.items()
            if b.state == "loaded"
        ]

    @property
    def sleeping_models(self) -> List[str]:
        return [
            mid for mid, b in self._backends.items()
            if b.state == "sleeping"
        ]

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
                logger.warning("Error unloading '%s' during shutdown", model_id, exc_info=True)

        self._executor.shutdown(wait=False)
        logger.info("ModelManager shut down")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _gpu_memory_info() -> Dict[str, float]:
        """Get GPU memory usage. Returns zeros if CUDA is unavailable."""
        try:
            import torch

            if torch.cuda.is_available():
                used = torch.cuda.memory_allocated() / 1024 / 1024
                total = torch.cuda.get_device_properties(0).total_mem / 1024 / 1024
                return {"used_mb": round(used, 1), "total_mb": round(total, 1)}
        except Exception:
            pass
        return {"used_mb": 0.0, "total_mb": 0.0}
