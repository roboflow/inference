from __future__ import annotations

import asyncio
import functools
import logging
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from inference_model_manager.backends.base import Backend

logger = logging.getLogger(__name__)


class DirectBackend(Backend):
    """Loads and runs a model in the current process.

    All inference happens in-process via ``model.infer()`` — no IPC, no
    worker processes. When batching is enabled, a BatchCollector groups
    concurrent requests and calls ``model.infer(images)`` with the batch,
    achieving a single CUDA sync per batch.

    Preferred for: InferencePipeline, CPU-only / constrained hardware,
    any scenario where per-frame IPC overhead is unacceptable.
    """

    def __init__(
        self,
        model_id: str,
        api_key: str,
        *,
        device: Optional[str] = None,
        decoder: str = "imagecodecs",
        executor: Optional[ThreadPoolExecutor] = None,
        batch_max_size: int = 0,
        batch_max_delay_ms: float = 10.0,
        **kwargs,
    ) -> None:
        from inference_model_manager.backends.decode import make_decoder
        from inference_models.models.auto_loaders.core import AutoModel

        self._model_id = model_id
        self._device_str = device
        self._decoder_name = decoder
        self._executor = executor
        self._batch_max_delay_ms = batch_max_delay_ms
        self._state_value: str = "loading"
        self._gpu_memory_delta_mb: float = 0.0

        self._decode: Callable[[bytes], Any] = make_decoder(
            decoder,
            device=device or "cpu",
        )

        load_kwargs = dict(kwargs)
        if device is not None:
            load_kwargs["device"] = device
        logger.info(
            "DirectBackend(%s): loading model (device=%s, decoder=%s)",
            model_id,
            device or "default",
            decoder,
        )
        gpu_before = self._gpu_mem_snapshot()
        try:
            self._model = AutoModel.from_pretrained(
                model_id, api_key=api_key, **load_kwargs
            )
        except Exception:
            # Ensure GPU memory freed if model partially loaded
            self._model = None
            raise
        self._gpu_memory_delta_mb = (self._gpu_mem_snapshot() - gpu_before) / (
            1024 * 1024
        )
        self._state_value = "loaded"

        self._device_str = self._detect_device()

        # Clamp batch size to model's limit
        from inference_model_manager.backends.base import detect_max_batch_size

        model_max = detect_max_batch_size(self._model)
        if batch_max_size > 0 and model_max is not None:
            self._batch_max_size = min(batch_max_size, model_max)
        else:
            self._batch_max_size = batch_max_size

        # Stats
        self._inference_count = 0
        self._error_count = 0
        self._last_inference_ts = 0.0
        self._start_ts = time.monotonic()
        self._latencies: deque[float] = deque(maxlen=1000)

        # Batching
        self._batch_collector = (
            self._create_batch_collector() if self._batch_max_size > 1 else None
        )

        model_type = type(self._model).__name__
        class_count = len(self.class_names) if self.class_names else 0
        logger.info(
            "DirectBackend(%s): ready | model_type=%s | device=%s | decoder=%s | "
            "class_names=%d | model_max_batch=%s | effective_batch=%s | "
            "batch_delay=%.1fms | executor=%s",
            model_id,
            model_type,
            self._device_str,
            decoder,
            class_count,
            model_max,
            self._batch_max_size or "off",
            batch_max_delay_ms,
            "shared" if executor else "none",
        )

    def _detect_device(self) -> str:
        if self._model is None:
            return self._device_str or "cpu"
        params = (
            list(self._model.parameters()) if hasattr(self._model, "parameters") else []
        )
        if params:
            return str(params[0].device)
        buffers = list(self._model.buffers()) if hasattr(self._model, "buffers") else []
        if buffers:
            return str(buffers[0].device)
        return self._device_str or "cpu"

    def _create_batch_collector(self):
        from inference_model_manager.backends.batch_collector import BatchCollector

        return BatchCollector(
            forward_fn=self._infer_batch,
            max_size=self._batch_max_size,
            max_delay_s=self._batch_max_delay_ms / 1000,
        )

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def _decode_input(self, raw_input: Any) -> Any:
        if isinstance(raw_input, (bytes, bytearray)):
            return self._decode(raw_input)
        return raw_input

    def _infer_batch(self, raw_items: list) -> list:
        """Batch inference via model.infer(). Called by BatchCollector."""
        images = [self._decode_input(inp) for inp, _ in raw_items]
        if len(images) == 1:
            result = self._model.infer(images[0])
            return [result] if not isinstance(result, list) else result
        results = self._model.infer(images)
        if not isinstance(results, list):
            results = [results]
        return results

    # ------------------------------------------------------------------
    # Inference entry points
    # ------------------------------------------------------------------

    def infer_sync(self, raw_input: Any, **kwargs) -> Any:
        """Run model.infer() synchronously.

        When batching is enabled, routes through submit() so the
        BatchCollector can group this call with concurrent ones.
        """
        if self._batch_collector is not None:
            return self.submit(raw_input, **kwargs).result()

        t0 = time.monotonic()
        image = self._decode_input(raw_input)
        result = self._model.infer(image)

        elapsed = time.monotonic() - t0
        self._inference_count += 1
        self._last_inference_ts = t0
        self._latencies.append(elapsed)
        return result

    async def infer_async(self, raw_input: Any, **kwargs) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,  # None falls back to event loop's default executor
            functools.partial(self.infer_sync, raw_input, **kwargs),
        )

    def submit(self, raw_input: Any, *, priority: int = 0, **kwargs) -> Future:
        if not self.is_accepting:
            raise RuntimeError(
                f"DirectBackend('{self._model_id}') is not accepting "
                f"requests (state={self.state})"
            )

        if self._batch_collector is not None:
            t0 = time.monotonic()
            future = self._batch_collector.add(
                raw_input, kwargs or None, priority=priority
            )
            future.add_done_callback(lambda f: self._record_inference(t0, f))
            return future

        # No batching — run directly
        if self._executor is not None:
            return self._executor.submit(self._infer_single, raw_input, kwargs)
        future: Future = Future()
        try:
            result = self._infer_single(raw_input, kwargs)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        return future

    def _infer_single(self, raw_input: Any, kwargs: Any) -> Any:
        t0 = time.monotonic()
        try:
            results = self._infer_batch([(raw_input, kwargs)])
            return results[0]
        except Exception:
            self._error_count += 1
            raise
        finally:
            elapsed = time.monotonic() - t0
            self._inference_count += 1
            self._last_inference_ts = t0
            self._latencies.append(elapsed)

    def record_inference(self, t0: float, error: bool = False) -> None:
        elapsed = time.monotonic() - t0
        self._inference_count += 1
        self._last_inference_ts = t0
        self._latencies.append(elapsed)
        if error:
            self._error_count += 1

    def _record_inference(self, t0: float, future: Future) -> None:
        self.record_inference(t0, error=future.exception() is not None)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def drain_and_unload(self, timeout_s: float = 30.0) -> None:
        """Stop accepting new work, drain batch collector, then unload."""
        self._state_value = "draining"
        logger.info(
            "DirectBackend(%s): draining (timeout=%.1fs)", self._model_id, timeout_s
        )

        if self._batch_collector is not None:
            self._batch_collector.stop(drain=True)
            self._batch_collector = None

        self.unload()

    def unload(self) -> None:
        self._state_value = "unhealthy"
        if self._batch_collector is not None:
            self._batch_collector.stop(drain=False)
            self._batch_collector = None
        del self._model
        self._model = None

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @property
    def model(self) -> Any:
        return self._model

    @property
    def device(self) -> str:
        return self._device_str or "cpu"

    @property
    def state(self) -> str:
        if self._model is None:
            return "unhealthy"
        return self._state_value

    @property
    def is_healthy(self) -> bool:
        return self._model is not None and self._state_value == "loaded"

    @property
    def is_accepting(self) -> bool:
        return self._state_value == "loaded" and self._model is not None

    @property
    def max_batch_size(self) -> Optional[int]:
        from inference_model_manager.backends.base import detect_max_batch_size

        return detect_max_batch_size(self._model)

    @property
    def queue_depth(self) -> int:
        if self._batch_collector is not None:
            return self._batch_collector.queue_depth
        return 0

    @staticmethod
    def _gpu_mem_snapshot() -> int:
        """Return current GPU memory used by this process (bytes) via pynvml.

        Falls back to torch.cuda.memory_allocated, then 0.
        """
        try:
            import os

            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                if proc.pid == os.getpid():
                    return proc.usedGpuMemory or 0
        except Exception:
            pass
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.memory_allocated()
        except Exception:
            pass
        return 0

    def stats(self) -> Dict[str, Any]:
        sorted_lats = sorted(self._latencies) if self._latencies else []
        bc = self._batch_collector.stats() if self._batch_collector else {}

        def _pct(p: float) -> float:
            if not sorted_lats:
                return 0.0
            idx = min(int(len(sorted_lats) * p / 100), len(sorted_lats) - 1)
            return sorted_lats[idx] * 1000

        return {
            "model_id": self._model_id,
            "backend_type": "direct",
            "decoder": self._decoder_name,
            "state": self.state,
            "is_accepting": self.is_accepting,
            "queue_depth": bc.get("queue_depth", 0),
            "queue_depth_by_priority": bc.get("queue_depth_by_priority", {}),
            "max_batch_size": self.max_batch_size,
            "current_batch_fill_pct": bc.get("avg_batch_fill_pct", 0.0),
            "batch_delay_ms": bc.get("avg_batch_delay_ms", 0.0),
            "throughput_fps": (
                self._inference_count / max(time.monotonic() - self._start_ts, 1e-6)
            ),
            "latency_p50_ms": _pct(50),
            "latency_p99_ms": _pct(99),
            "gpu_memory_mb": self._gpu_memory_delta_mb,
            "inference_count": self._inference_count,
            "error_count": self._error_count,
            "last_inference_ts": self._last_inference_ts,
            "model_class_name": type(self._model).__name__ if self._model else None,
        }

    @property
    def class_names(self) -> Optional[List[str]]:
        return getattr(self._model, "class_names", None)
