from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import Future
from typing import Any, Dict, List, Optional, Tuple


class Backend(ABC):
    """Handle to a single loaded model.

    One instance per model. Loading happens in ``__init__`` (blocks until
    ready). ``ModelManager`` owns a ``Dict[str, Backend]`` and routes calls
    by ``model_id``.

    The serving pipeline is decomposed into stages::

        pre_process  →  collate  →  forward  →  uncollate  →  post_process
        (caller)        (caller)    (worker)     (caller)      (caller)

    Pre-processing and post-processing always run on the caller side
    (ModelManager / FastAPI threads). The worker only runs ``forward()``.
    This keeps the worker simple, avoids leaking pre_processing metadata
    across process boundaries, and allows CPU work to be parallelized
    across connections.

    Batching is owned by the backend, not the caller. The backend provides
    ``collate()`` / ``uncollate()`` so callers never need model-specific
    knowledge. The internal ``BatchCollector`` accumulates pre-processed
    items, calls ``collate()`` when the batch is ready, dispatches
    ``forward()``, then ``uncollate()`` to split results back to
    individual requests.

    Callers interact via ``submit()`` which accepts a single pre-processed
    item and returns a ``Future``. The backend handles batching, transport,
    and scheduling transparently.

    For ``DirectBackend``: all stages are in-process function calls.
    For ``SubprocessBackend``: ``forward()`` runs in a worker process,
    with bulk tensor data flowing through a buffer pool (CUDA IPC / Pinned
    SHM / Plain SHM) and signaling + metadata flowing through a ZMQ PAIR
    socket.
    """

    # ------------------------------------------------------------------
    # Pipeline stages — called by ModelManager, not by end users
    # ------------------------------------------------------------------

    @abstractmethod
    def pre_process(self, *args, **kwargs) -> Tuple[Any, Any]:
        """Convert raw input (image, prompt, etc.) into model-ready tensor + metadata.

        Returns:
            (pre_processed_tensor, pre_processing_meta): The tensor is what
            enters the batch queue. The meta is kept on the caller side and
            passed to ``post_process()`` later (never crosses the process
            boundary in SubprocessBackend).
        """
        ...

    @abstractmethod
    def post_process(self, raw_output: Any, meta: Any, **kwargs) -> Any:
        """Convert raw forward output + pre_processing meta into final result.

        Runs on the caller side. For object detection this applies NMS,
        rescales coordinates using the original image dimensions from meta,
        and returns a ``Detections`` dataclass. For segmentation, it
        reassembles masks, etc.
        """
        ...

    # ------------------------------------------------------------------
    # Batching primitives — backend knows how to stack/unstack
    # ------------------------------------------------------------------

    @abstractmethod
    def collate(self, items: List[Tuple[Any, Any]]) -> Any:
        """Stack multiple pre-processed tensors into a single batched input.

        Called by the internal BatchCollector when the batch is ready
        (size reached or delay expired). Different model types have
        different stacking strategies (YOLO stacks on dim 0, VLMs pad
        token sequences, SAM handles variable prompt counts).

        Args:
            items: List of (pre_processed_tensor, pre_processing_meta) tuples
                from ``pre_process()``.

        Returns:
            Batched input ready for ``forward()``.
        """
        ...

    @abstractmethod
    def uncollate(self, batched_output: Any, count: int) -> List[Any]:
        """Split a batched forward output back into per-request raw outputs.

        Args:
            batched_output: Raw output from ``forward()``.
            count: Number of requests in the batch.

        Returns:
            List of per-request raw outputs, each passed to ``post_process()``.
        """
        ...

    # ------------------------------------------------------------------
    # Forward — runs in the worker (or in-process for DirectBackend)
    # ------------------------------------------------------------------

    @abstractmethod
    def forward_sync(self, batched_input: Any, **kwargs) -> Any:
        """Run the model's forward pass on a collated batch. Blocking.

        For DirectBackend: calls ``model.forward()`` directly.
        For SubprocessBackend: writes batch to buffer pool, signals worker
        via ZMQ, waits for result.
        """
        ...

    async def forward_async(self, batched_input: Any, **kwargs) -> Any:
        """Async version of ``forward_sync()``.

        Default implementation dispatches to a thread executor. Backends
        may override with a native async path (e.g. async ZMQ recv).
        """
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.forward_sync(batched_input, **kwargs)
        )

    # ------------------------------------------------------------------
    # Submission — the main entry point for callers
    # ------------------------------------------------------------------

    @abstractmethod
    def submit(self, pre_processed: Tuple[Any, Any], *, priority: int = 0) -> Future:
        """Submit a pre-processed item for inference. Returns a Future.

        Non-blocking. The item enters the backend's internal BatchCollector.
        When the batch is ready (size or delay), the backend dispatches
        ``collate → forward → uncollate → post_process`` and resolves
        each Future with its individual result.

        Args:
            pre_processed: (tensor, meta) tuple from ``pre_process()``.
            priority: Request priority level (0 = default). Higher priority
                requests are dequeued first when priority queues are enabled.

        Returns:
            Future that resolves to the final post-processed result.
        """
        ...

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def unload(self) -> None:
        """Release all resources (GPU memory, worker processes, buffers).

        Any in-flight requests are failed with an error.
        """
        ...

    @abstractmethod
    def sleep(self) -> Optional[int]:
        """Offload model weights to CPU pinned memory, freeing VRAM.

        Returns the number of bytes of CPU pinned memory now used,
        or None if sleeping is not supported by this backend.

        Raises:
            RuntimeError: If the backend has in-flight requests.
        """
        ...

    @abstractmethod
    def wake(self) -> None:
        """Reload weights from CPU pinned memory to GPU.

        Blocks until the model is ready to serve again.

        Raises:
            RuntimeError: If the backend is not in sleeping state.
        """
        ...

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def device(self) -> str:
        """Device this backend runs forward() on.

        Returns a string like ``"cpu"``, ``"cuda:0"``, ``"cuda:1"``.
        Used by ModelManager to track which backends are on which GPU
        and to report per-device memory usage.
        """
        ...

    @property
    @abstractmethod
    def state(self) -> str:
        """Current state: 'loading', 'loaded', 'sleeping', 'unhealthy'."""
        ...

    @property
    @abstractmethod
    def is_healthy(self) -> bool:
        """Whether this backend is in a usable state.

        For DirectBackend: whether the model object exists and didn't OOM.
        For SubprocessBackend: whether the worker process is alive and
        responsive.
        """
        ...

    @property
    @abstractmethod
    def is_accepting(self) -> bool:
        """Whether this backend can accept new requests.

        False when the admission queue is full. Used by ModelManager to
        reject or reroute requests.
        """
        ...

    @property
    @abstractmethod
    def max_batch_size(self) -> Optional[int]:
        """Maximum batch size this backend supports, or None if unlimited."""
        ...

    @property
    @abstractmethod
    def queue_depth(self) -> int:
        """Number of pending requests in the admission queue."""
        ...

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """Runtime statistics for observability and heartbeat reporting.

        Returns dict with keys:
            model_id, backend_type, state, is_accepting,
            queue_depth, queue_depth_by_priority,
            max_batch_size, current_batch_fill_pct, batch_delay_ms,
            throughput_fps, latency_p50_ms, latency_p99_ms,
            gpu_memory_mb, cpu_pinned_memory_mb,
            inference_count, error_count, last_inference_ts

        Must be non-blocking — never contends with inference.
        """
        ...

    @property
    @abstractmethod
    def class_names(self) -> Optional[List[str]]:
        """Class names for the loaded model, if available."""
        ...
