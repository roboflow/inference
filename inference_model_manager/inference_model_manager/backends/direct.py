from __future__ import annotations

import logging
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from inference_model_manager.backends.base import Backend, attach_model_caches

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

        # batch_max_size / batch_max_delay_ms accepted for API parity with
        # SubprocessBackend (ModelManager.load passes them generically). They
        # are unused in DirectBackend — inference is dispatched per-request by
        # ModelManager.process() via invoke_task(backend.model, ...).
        del batch_max_size, batch_max_delay_ms

        self._model_id = model_id
        self._device_str = device
        self._decoder_name = decoder
        self._executor = executor
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
            attach_model_caches(self._model)
        except Exception:
            self._model = None
            raise
        self._gpu_memory_delta_mb = (self._gpu_mem_snapshot() - gpu_before) / (
            1024 * 1024
        )
        self._state_value = "loaded"

        self._device_str = self._detect_device()

        from inference_model_manager.backends.base import detect_max_batch_size

        model_max = detect_max_batch_size(self._model)

        self._inference_count = 0
        self._error_count = 0
        self._last_inference_ts = 0.0
        self._start_ts = time.monotonic()
        self._latencies: deque[float] = deque(maxlen=1000)
        self._inflight = 0
        self._inflight_lock = threading.Lock()

        model_type = type(self._model).__name__
        class_count = len(self.class_names) if self.class_names else 0
        logger.info(
            "DirectBackend(%s): ready | model_type=%s | device=%s | decoder=%s | "
            "class_names=%d | model_max_batch=%s | executor=%s",
            model_id,
            model_type,
            self._device_str,
            decoder,
            class_count,
            model_max,
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

    def _decode_input(self, raw_input: Any) -> Any:
        if isinstance(raw_input, (bytes, bytearray)):
            return self._decode(raw_input)
        return raw_input

    def record_inference(self, t0: float, error: bool = False) -> None:
        elapsed = time.monotonic() - t0
        self._inference_count += 1
        self._last_inference_ts = t0
        self._latencies.append(elapsed)
        if error:
            self._error_count += 1

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def inflight_begin(self) -> None:
        with self._inflight_lock:
            self._inflight += 1

    def inflight_end(self) -> None:
        with self._inflight_lock:
            self._inflight = max(0, self._inflight - 1)

    def drain_and_unload(self, timeout_s: float = 30.0) -> None:
        self._state_value = "draining"
        logger.info(
            "DirectBackend(%s): draining (timeout=%.1fs)", self._model_id, timeout_s
        )
        # Wait for in-flight forward passes — dropping the model under a live
        # invoke_task crashes (CUDA error / AttributeError).
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and self._inflight > 0:
            time.sleep(0.05)
        if self._inflight > 0:
            logger.warning(
                "DirectBackend(%s): drain timeout — %d inference(s) still "
                "in flight, force-unloading",
                self._model_id,
                self._inflight,
            )
        self.unload()

    def unload(self) -> None:
        self._state_value = "unhealthy"
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
            "queue_depth": 0,
            "queue_depth_by_priority": {},
            "max_batch_size": self.max_batch_size,
            "current_batch_fill_pct": 0.0,
            "batch_delay_ms": 0.0,
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

    @property
    def key_points_classes(self) -> Optional[List[List[str]]]:
        return getattr(self._model, "key_points_classes", None)
