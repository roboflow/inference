from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import Future
from typing import Any, Dict, List, Optional


class Backend(ABC):
    """Handle to a single loaded model.

    One instance per model. Loading happens in ``__init__`` (blocks until
    ready). ``ModelManager`` owns a ``Dict[str, Backend]`` and routes calls
    by ``model_id``.

    ``model.infer()`` handles the full pipeline internally (decode, resize,
    normalize, forward, post-process). Backends wrap that call with
    transport, batching, and lifecycle management.

    Callers use ``infer_sync()`` / ``infer_async()`` for the simple path,
    or ``submit()`` when they want a non-blocking Future with priority and
    batching.

    For ``DirectBackend``: inference runs in-process via a thread pool.
    For ``SubprocessBackend``: inference runs in a worker subprocess;
    bulk image data flows through a shared ``SHMPool`` and signaling
    flows through a ZMQ PAIR socket.
    """

    # ------------------------------------------------------------------
    # Inference — main entry points
    # ------------------------------------------------------------------

    @abstractmethod
    def infer_sync(self, raw_input: Any, **kwargs) -> Any:
        """Run inference synchronously. Blocks until result is ready.

        ``raw_input`` may be raw bytes (JPEG/PNG), a numpy array, or a
        decoded image — the backend and model handle decoding internally.

        Raises:
            RuntimeError: If the backend is not in 'loaded' state.
        """
        ...

    @abstractmethod
    async def infer_async(self, raw_input: Any, **kwargs) -> Any:
        """Run inference asynchronously. Returns when result is ready.

        Raises:
            RuntimeError: If the backend is not in 'loaded' state.
        """
        ...

    @abstractmethod
    def submit(self, raw_input: Any, *, priority: int = 0, **kwargs) -> Future:
        """Submit a request for inference. Returns a Future immediately.

        Non-blocking. The item enters the backend's internal BatchCollector
        (if batching is enabled). When the batch fires, the Future resolves
        with the post-processed result.

        Args:
            raw_input: Raw image bytes, numpy array, or decoded image.
            priority: Request priority (0 = default). Higher priority
                requests are dequeued first.

        Returns:
            Future that resolves to the model's output.
        """
        ...

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def unload(self) -> None:
        """Release all resources (GPU memory, worker processes, SHM).

        Any in-flight requests are cancelled with an error.
        """
        ...

    def drain_and_unload(self, timeout_s: float = 30.0) -> None:
        """Stop accepting new work, wait for in-flight to finish, then unload.

        1. Set state to 'draining' — new submit/signal calls are rejected.
        2. Wait up to ``timeout_s`` for pending work to complete.
        3. If timeout expires, force-cancel remaining work.
        4. Call ``unload()`` for final cleanup.

        Default implementation just calls ``unload()`` immediately.
        Backends override to implement graceful drain.
        """
        self.unload()

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
        """Reload weights from CPU pinned memory back to GPU.

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
        """Device this backend runs inference on: 'cpu', 'cuda:0', etc."""
        ...

    @property
    @abstractmethod
    def state(self) -> str:
        """Current state: 'loading', 'loaded', 'sleeping', 'unhealthy'."""
        ...

    @property
    @abstractmethod
    def is_healthy(self) -> bool:
        """Whether this backend is in a usable state."""
        ...

    @property
    @abstractmethod
    def is_accepting(self) -> bool:
        """Whether this backend can accept new requests right now."""
        ...

    @property
    @abstractmethod
    def max_batch_size(self) -> Optional[int]:
        """Maximum batch size this backend supports, or None if unlimited."""
        ...

    @property
    @abstractmethod
    def queue_depth(self) -> int:
        """Number of pending requests waiting in the batch queue."""
        ...

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """Runtime statistics snapshot. Must be non-blocking.

        Returns dict with at minimum:
            model_id, backend_type, state, is_accepting,
            queue_depth, max_batch_size,
            throughput_fps, latency_p50_ms, latency_p99_ms,
            inference_count, error_count, last_inference_ts
        """
        ...

    @property
    @abstractmethod
    def class_names(self) -> Optional[List[str]]:
        """Class names for the loaded model, if available."""
        ...

    @property
    def worker_pid(self) -> Optional[int]:
        """OS PID of the worker subprocess, if applicable. None for in-process backends."""
        return None
