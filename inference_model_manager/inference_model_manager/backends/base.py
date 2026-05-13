from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


def detect_max_batch_size(model) -> Optional[int]:
    """Duck-type max batch size from a model instance.

    Models use various attribute names. Returns None if not discoverable.
    """
    bs = (
        getattr(model, "max_batch_size", None)
        or getattr(model, "_max_batch_size", None)
        or getattr(model, "_input_batch_size", None)
    )
    if bs is not None:
        return bs
    # TorchScript/TRT models store it in inference_config
    cfg = getattr(model, "_inference_config", None)
    if cfg is not None:
        fwd = getattr(cfg, "forward_pass", None)
        if fwd is not None:
            sbs = getattr(fwd, "static_batch_size", None)
            if sbs is not None:
                return sbs
            mdb = getattr(fwd, "max_dynamic_batch_size", None)
            if mdb is not None:
                return mdb
    # TRT config (separate from inference_config)
    trt_cfg = getattr(model, "_trt_config", None)
    if trt_cfg is not None:
        sbs = getattr(trt_cfg, "static_batch_size", None)
        if sbs is not None:
            return sbs
        return getattr(trt_cfg, "dynamic_batch_size_max", None)
    return None


class Backend(ABC):
    """Handle to a single loaded model.

    One instance per model. Loading happens in ``__init__`` (blocks until
    ready). ``ModelManager`` owns a ``Dict[str, Backend]`` and routes calls
    by ``model_id``.

    ``model.infer()`` handles the full pipeline internally (decode, resize,
    normalize, forward, post-process). Backends wrap that call with
    transport, batching, and lifecycle management.

    All inference goes through ``ModelManager.process()`` /
    ``process_async()`` / ``submit()`` — backends have no public inference
    entry point. ``ModelManager`` dispatches to ``invoke_task`` (direct) or
    ``signal_slot`` (subprocess, after SHM slot allocation).

    For ``DirectBackend``: inference runs in-process via a thread pool.
    For ``SubprocessBackend``: inference runs in a worker subprocess;
    bulk image data flows through a shared ``SHMPool`` and signaling
    flows through a ZMQ PAIR socket.
    """

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
        """Current state: 'loading', 'loaded', 'draining', 'unhealthy'."""
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

    def record_inference(self, t0: float, error: bool = False) -> None:
        """Record an inference for stats tracking. Called by ModelManager."""
        pass

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
    def model(self) -> Any:
        """Underlying model instance. Used by ModelManager.invoke() for task dispatch.

        Returns None for subprocess backends (model lives in worker process).
        """
        return None

    @property
    def worker_pid(self) -> Optional[int]:
        """OS PID of the worker subprocess, if applicable. None for in-process backends."""
        return None
