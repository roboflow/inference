from __future__ import annotations

import asyncio
import functools
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from inference_models.backends.base import Backend


class DirectBackend(Backend):
    """Loads and runs a model in the current process.

    All pipeline stages (pre_process, forward, post_process) are in-process
    function calls — no IPC, no worker processes. The model object lives in
    this process's memory.

    Sync callers block the calling thread directly. When an ``executor`` is
    provided, ``submit()`` dispatches work to the thread pool so multiple
    requests can overlap (useful for async callers / ModelManager).

    Preferred for: InferencePipeline, CPU-only / constrained hardware — any
    scenario where IPC round-trip cost per frame is unacceptable.
    """

    def __init__(
        self,
        model_id: str,
        api_key: str,
        *,
        device: Optional[str] = None,
        executor: Optional[ThreadPoolExecutor] = None,
        batch_max_size: int = 0,
        batch_max_delay_ms: float = 10.0,
        **kwargs,
    ) -> None:
        """Load the model in the current process.

        Args:
            device: Device to load the model on (e.g. ``"cpu"``,
                      ``"cuda:0"``, ``"cuda:1"``). ``None`` uses the
                      library default (CUDA if available, else CPU).
            executor: Shared thread-pool executor for ``submit()`` when
                      batching is disabled. Owned by ModelManager, not by
                      this backend. Ignored when batching is enabled (the
                      BatchCollector dispatch thread handles execution).
                      None is fine if only synchronous usage is needed.
            batch_max_size: Maximum batch size. 0 = no batching (each
                      ``submit()`` runs the pipeline immediately).
            batch_max_delay_ms: Maximum time (ms) to wait for a full batch
                      before dispatching a partial one.
        """
        from inference_models.models.auto_loaders.core import AutoModel

        self._model_id = model_id
        self._device_str = device  # resolved after load
        self._executor = executor
        self._batch_max_size = batch_max_size
        self._batch_max_delay_ms = batch_max_delay_ms
        self._state_value: str = "loading"

        load_kwargs = dict(kwargs)
        if device is not None:
            load_kwargs["device"] = device
        self._model = AutoModel.from_pretrained(model_id, api_key=api_key, **load_kwargs)
        self._state_value = "loaded"

        # Resolve actual device from model parameters
        self._device_str = self._detect_device()

        # Device tracking for sleep/wake
        self._sleep_device = None

        # Stats
        self._inference_count = 0
        self._error_count = 0
        self._last_inference_ts = 0.0
        self._latencies: deque[float] = deque(maxlen=1000)

        # Batching
        self._batch_collector = self._create_batch_collector() if batch_max_size > 0 else None

    def _detect_device(self) -> str:
        """Inspect model parameters to find actual device."""
        if self._model is None:
            return self._device_str or "cpu"
        params = list(self._model.parameters()) if hasattr(self._model, "parameters") else []
        if params:
            return str(params[0].device)
        buffers = list(self._model.buffers()) if hasattr(self._model, "buffers") else []
        if buffers:
            return str(buffers[0].device)
        return self._device_str or "cpu"

    def _create_batch_collector(self):
        from inference_models.backends.batch_collector import BatchCollector

        return BatchCollector(
            collate_fn=self.collate,
            forward_fn=self.forward_sync,
            uncollate_fn=self.uncollate,
            post_process_fn=self.post_process,
            max_size=self._batch_max_size,
            max_delay_s=self._batch_max_delay_ms / 1000,
        )

    # ------------------------------------------------------------------
    # Pipeline stages — delegate to the underlying model
    # ------------------------------------------------------------------

    def pre_process(self, *args, **kwargs) -> Tuple[Any, Any]:
        result = self._model.pre_process(*args, **kwargs)
        # Models with metadata return (tensor, meta).
        # Models without metadata (e.g. Classification) return just tensor.
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return (result, None)

    def post_process(self, raw_output: Any, meta: Any, **kwargs) -> Any:
        if meta is not None:
            return self._model.post_process(raw_output, meta, **kwargs)
        return self._model.post_process(raw_output, **kwargs)

    def collate(self, items: List[Tuple[Any, Any]]) -> Any:
        import torch

        tensors = [t for t, _ in items]
        if not tensors:
            return tensors
        if isinstance(tensors[0], torch.Tensor):
            return torch.cat(tensors, dim=0)
        # Non-tensor fallback (let model handle it)
        return tensors

    def uncollate(self, batched_output: Any, count: int) -> List[Any]:
        import torch

        if isinstance(batched_output, (list, tuple)) and len(batched_output) == count:
            return list(batched_output)
        if isinstance(batched_output, torch.Tensor) and batched_output.shape[0] == count:
            return list(batched_output.split(1, dim=0))
        # Single-item batch or unrecognized structure
        if count == 1:
            return [batched_output]
        return [batched_output] * count

    def forward_sync(self, batched_input: Any, **kwargs) -> Any:
        return self._model.forward(batched_input, **kwargs)

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit(self, pre_processed: Tuple[Any, Any], *, priority: int = 0) -> Future:
        if not self.is_accepting:
            raise RuntimeError(
                f"DirectBackend('{self._model_id}') is not accepting "
                f"requests (state={self.state})"
            )

        tensor, meta = pre_processed

        # Batching path — BatchCollector handles collate/forward/uncollate/post_process
        if self._batch_collector is not None:
            t0 = time.monotonic()
            future = self._batch_collector.add(tensor, meta, priority=priority)
            future.add_done_callback(lambda f: self._record_inference(t0, f))
            return future

        # No batching — run pipeline directly
        if self._executor is not None:
            return self._executor.submit(self._run_pipeline, tensor, meta)
        future: Future = Future()
        try:
            result = self._run_pipeline(tensor, meta)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        return future

    def _run_pipeline(self, tensor: Any, meta: Any) -> Any:
        """Non-batching path: forward + post_process for a single item."""
        t0 = time.monotonic()
        try:
            raw = self.forward_sync(tensor)
            return self.post_process(raw, meta)
        except Exception:
            self._error_count += 1
            raise
        finally:
            elapsed = time.monotonic() - t0
            self._inference_count += 1
            self._last_inference_ts = t0
            self._latencies.append(elapsed)

    def _record_inference(self, t0: float, future: Future) -> None:
        """Callback for batching path — records stats when Future resolves."""
        elapsed = time.monotonic() - t0
        self._inference_count += 1
        self._last_inference_ts = t0
        self._latencies.append(elapsed)
        if future.exception() is not None:
            self._error_count += 1

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def unload(self) -> None:
        if self._batch_collector is not None:
            self._batch_collector.stop(drain=False)
            self._batch_collector = None
        del self._model
        self._model = None
        self._sleep_device = None

    def sleep(self) -> Optional[int]:
        if self._state_value != "loaded":
            raise RuntimeError(
                f"Cannot sleep: state is '{self._state_value}', expected 'loaded'"
            )
        if self._batch_collector is not None:
            if self._batch_collector.queue_depth > 0:
                raise RuntimeError("Cannot sleep with pending batched requests")
            self._batch_collector.stop(drain=True)
            self._batch_collector = None
        import torch

        params = list(self._model.parameters()) if hasattr(self._model, "parameters") else []
        buffers = list(self._model.buffers()) if hasattr(self._model, "buffers") else []
        if not params and not buffers:
            return None

        self._sleep_device = params[0].device if params else buffers[0].device
        pinned_bytes = 0
        for p in params:
            p.data = p.data.cpu().pin_memory()
            pinned_bytes += p.numel() * p.element_size()
        for b in buffers:
            if b.is_floating_point() or b.is_complex():
                b.data = b.data.cpu().pin_memory()
                pinned_bytes += b.numel() * b.element_size()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._state_value = "sleeping"
        return pinned_bytes

    def wake(self) -> None:
        if self._state_value != "sleeping":
            raise RuntimeError(
                f"Cannot wake: state is '{self._state_value}', expected 'sleeping'"
            )
        import torch

        device = self._sleep_device or torch.device(self._device_str or "cuda")
        params = list(self._model.parameters()) if hasattr(self._model, "parameters") else []
        buffers = list(self._model.buffers()) if hasattr(self._model, "buffers") else []
        for p in params:
            p.data = p.data.to(device, non_blocking=True)
        for b in buffers:
            if b.is_floating_point() or b.is_complex():
                b.data = b.data.to(device, non_blocking=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        self._device_str = str(device)
        self._sleep_device = None
        self._state_value = "loaded"

        # Restart BatchCollector if batching was configured
        if self._batch_max_size > 0 and self._batch_collector is None:
            self._batch_collector = self._create_batch_collector()

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @property
    def device(self) -> str:
        if self._state_value == "sleeping":
            return "cpu"
        return self._device_str or "cpu"

    @property
    def state(self) -> str:
        if self._model is None:
            return "unhealthy"
        return self._state_value

    @property
    def is_healthy(self) -> bool:
        return self._model is not None and self._state_value in ("loaded", "sleeping")

    @property
    def is_accepting(self) -> bool:
        return self._state_value == "loaded" and self._model is not None

    @property
    def max_batch_size(self) -> Optional[int]:
        return getattr(self._model, "_max_batch_size", None)

    @property
    def queue_depth(self) -> int:
        if self._batch_collector is not None:
            return self._batch_collector.queue_depth
        return 0

    def stats(self) -> Dict[str, Any]:
        sorted_lats = sorted(self._latencies) if self._latencies else []
        bc = self._batch_collector.stats() if self._batch_collector else {}

        def _pct(p: float) -> float:
            if not sorted_lats:
                return 0.0
            idx = min(int(len(sorted_lats) * p / 100), len(sorted_lats) - 1)
            return sorted_lats[idx] * 1000  # seconds → ms

        return {
            "model_id": self._model_id,
            "backend_type": "direct",
            "state": self.state,
            "is_accepting": self.is_accepting,
            "queue_depth": bc.get("queue_depth", 0),
            "queue_depth_by_priority": bc.get("queue_depth_by_priority", {}),
            "max_batch_size": self.max_batch_size,
            "current_batch_fill_pct": bc.get("avg_batch_fill_pct", 0.0),
            "batch_delay_ms": bc.get("avg_batch_delay_ms", 0.0),
            "throughput_fps": (
                self._inference_count / sum(sorted_lats) if sorted_lats else 0.0
            ),
            "latency_p50_ms": _pct(50),
            "latency_p99_ms": _pct(99),
            "gpu_memory_mb": 0.0,  # TODO: per-model GPU memory tracking
            "cpu_pinned_memory_mb": 0.0,  # TODO: track from sleep()
            "inference_count": self._inference_count,
            "error_count": self._error_count,
            "last_inference_ts": self._last_inference_ts,
        }

    @property
    def class_names(self) -> Optional[List[str]]:
        return getattr(self._model, "class_names", None)

    # ------------------------------------------------------------------
    # Convenience — not part of Backend ABC, but keeps backward compat
    # with code that calls backend.infer_sync() / infer_async() directly
    # ------------------------------------------------------------------

    def infer_sync(self, *args, **kwargs) -> Any:
        """Run full pipeline synchronously: pre_process → forward → post_process."""
        pre_processed = self.pre_process(*args, **kwargs)
        future = self.submit(pre_processed)
        return future.result()

    async def infer_async(self, *args, **kwargs) -> Any:
        """Run full pipeline via executor. Requires executor at construction."""
        if self._executor is None:
            raise RuntimeError(
                f"infer_async called on DirectBackend('{self._model_id}') "
                f"but no executor was provided — use infer_sync or pass an "
                f"executor at construction"
            )
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            functools.partial(self.infer_sync, *args, **kwargs),
        )
