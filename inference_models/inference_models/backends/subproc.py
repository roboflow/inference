"""SubprocessBackend — model.forward() runs in a worker subprocess.

Pre-processing and post-processing run in the caller process (on a CPU copy
of the model). The worker only calls forward().

Input transport tiers (chosen at load time):
  CUDA IPC   — pre-allocated GPU buffer pool, ~6-10us overhead
  SHM        — POSIX shared memory (worker calls .cuda() if GPU), ~1-16ms

Results always flow through SHM (efficient for all sizes).
ZMQ PAIR carries metadata and signaling alongside buffer pools.

Configuration flags:
  use_gpu       — worker uses GPU for forward(). Default: True if CUDA available.
  use_cuda_ipc  — use CUDA IPC for input transport. Default: True if CUDA available.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from inference_models.backends.base import Backend

logger = logging.getLogger(__name__)


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


# ─── ZMQ message types ──────────────────────────────────────────────

_MSG_INFER = b"I"
_MSG_RESULT = b"R"
_MSG_ERROR = b"E"
_MSG_SHUTDOWN = b"S"
_MSG_HEARTBEAT = b"H"

_WORKER_HEARTBEAT_MS = 2000
_WORKER_HEARTBEAT_TIMEOUT_S = 30.0


# ─── SHM buffer pool ────────────────────────────────────────────────


class _ShmPool:
    """Paired (input + result) POSIX shared-memory buffer slots.

    Created by the main process. The worker opens existing blocks by name.
    Slot lifecycle (acquire/release) is owned by the main process.
    """

    def __init__(self, num_slots: int, input_bytes: int, result_bytes: int) -> None:
        self._input = [
            shared_memory.SharedMemory(create=True, size=input_bytes)
            for _ in range(num_slots)
        ]
        self._result = [
            shared_memory.SharedMemory(create=True, size=result_bytes)
            for _ in range(num_slots)
        ]
        self._free: deque[int] = deque(range(num_slots))
        self._cond = threading.Condition()

    @property
    def input_names(self) -> List[str]:
        return [b.name for b in self._input]

    @property
    def result_names(self) -> List[str]:
        return [b.name for b in self._result]

    def acquire(self, timeout: float = 10.0) -> int:
        with self._cond:
            deadline = time.monotonic() + timeout
            while not self._free:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._cond.wait(timeout=remaining):
                    raise TimeoutError("No free buffer slots")
            return self._free.popleft()

    def release(self, slot: int) -> None:
        with self._cond:
            self._free.append(slot)
            self._cond.notify()

    def input_buf(self, slot: int) -> memoryview:
        return self._input[slot].buf

    def result_buf(self, slot: int) -> memoryview:
        return self._result[slot].buf

    def close(self) -> None:
        for lst in (self._input, self._result):
            for b in lst:
                try:
                    b.close()
                    b.unlink()
                except Exception:
                    pass


# ─── CUDA pinning helpers ────────────────────────────────────────────

_cudart = None  # lazy-loaded


def _get_cudart():
    """Load libcudart.so once. Returns None if unavailable."""
    global _cudart
    if _cudart is not None:
        return _cudart if _cudart is not False else None
    import ctypes

    for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0"):
        try:
            _cudart = ctypes.CDLL(name)
            return _cudart
        except OSError:
            continue
    _cudart = False
    return None


def _pin_shm_buffer(buf: memoryview) -> bool:
    """Register an SHM buffer as CUDA page-locked memory for DMA.

    Returns True on success. Logs a warning on failure (e.g. platform
    mismatch, locked-memory limit exceeded). Safe to call even if CUDA
    init hasn't happened yet — will just return False.
    """
    import ctypes
    import logging

    lib = _get_cudart()
    if lib is None:
        logging.getLogger(__name__).warning(
            "cudaHostRegister unavailable (libcudart not found) — "
            "SHM will not be pinned, GPU transfers will be slower"
        )
        return False
    arr = (ctypes.c_char * len(buf)).from_buffer(buf)
    ptr = ctypes.cast(arr, ctypes.c_void_p)
    # cudaHostRegisterPortable = 1 (accessible from any CUDA context)
    ret = lib.cudaHostRegister(ptr, ctypes.c_size_t(len(buf)), ctypes.c_uint(1))
    if ret != 0:
        logging.getLogger(__name__).warning(
            "cudaHostRegister failed (error %d) — "
            "SHM will not be pinned, GPU transfers will be slower. "
            "Check locked-memory limits (ulimit -l).",
            ret,
        )
    return ret == 0


def _unpin_shm_buffer(buf: memoryview) -> None:
    """Unregister a previously pinned SHM buffer."""
    import ctypes

    lib = _get_cudart()
    if lib is None:
        return
    arr = (ctypes.c_char * len(buf)).from_buffer(buf)
    ptr = ctypes.cast(arr, ctypes.c_void_p)
    lib.cudaHostUnregister(ptr)


# ─── Tensor ↔ SHM helpers ───────────────────────────────────────────


def _tensor_to_shm(buf: memoryview, tensor: Any) -> dict:
    """Write a tensor to an SHM slot. Returns metadata dict."""
    import torch

    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().contiguous().numpy()
    elif isinstance(tensor, np.ndarray):
        arr = np.ascontiguousarray(tensor) if not tensor.flags["C_CONTIGUOUS"] else tensor
    else:
        arr = np.ascontiguousarray(tensor)
    data = arr.tobytes()
    buf[: len(data)] = data
    return {"shape": list(arr.shape), "dtype": str(arr.dtype), "nbytes": len(data)}


def _tensor_from_shm(
    buf: memoryview, meta: dict, device: str = "cpu", pinned: bool = False,
) -> Any:
    """Read a tensor from an SHM slot.

    When ``device`` is GPU and ``pinned`` is True, creates a zero-copy view
    into the SHM buffer and calls ``.cuda()`` — the GPU DMA-reads directly
    from the pinned SHM pages (~25 GB/s). No intermediate copy.

    When ``device`` is CPU, copies out of SHM (the slot will be reused).
    """
    import torch

    shape = tuple(meta["shape"])
    dtype = np.dtype(meta["dtype"])
    nbytes = meta["nbytes"]
    arr = np.ndarray(shape, dtype=dtype, buffer=buf[:nbytes])

    if device != "cpu" and pinned:
        # Zero-copy view into pinned SHM → .cuda() does DMA directly
        t = torch.from_numpy(arr).cuda()
    else:
        # CPU path or non-pinned: must copy out (SHM will be reused)
        t = torch.from_numpy(arr.copy())
        if device != "cpu":
            t = t.to(device)
    return t


def _result_to_shm(buf: memoryview, output: Any) -> dict:
    """Write forward() output (tensor, tuple of tensors, or arbitrary) to SHM."""
    import torch

    if isinstance(output, torch.Tensor):
        tensors = [output]
    elif isinstance(output, (tuple, list)):
        tensors = list(output)
    else:
        import pickle

        data = pickle.dumps(output)
        buf[: len(data)] = data
        return {"fmt": "pkl", "n": len(data)}

    parts: list = []
    offset = 0
    for t in tensors:
        arr = t.detach().cpu().contiguous().numpy()
        data = arr.tobytes()
        buf[offset : offset + len(data)] = data
        parts.append(
            {"s": list(arr.shape), "d": str(arr.dtype), "o": offset, "n": len(data)}
        )
        offset += len(data)
    return {"fmt": "t", "parts": parts}


def _result_from_shm(buf: memoryview, meta: dict) -> Any:
    """Read forward() output from SHM."""
    import torch

    if meta.get("fmt") == "pkl":
        import pickle

        return pickle.loads(bytes(buf[: meta["n"]]))

    tensors = []
    for p in meta["parts"]:
        arr = np.ndarray(
            tuple(p["s"]),
            dtype=np.dtype(p["d"]),
            buffer=buf[p["o"] : p["o"] + p["n"]],
        )
        tensors.append(torch.from_numpy(arr.copy()))
    return tensors[0] if len(tensors) == 1 else tuple(tensors)


# ─── Worker process ──────────────────────────────────────────────────


def _worker_main(
    model_id: str,
    api_key: str,
    setup_pipe: Any,
    zmq_addr: str,
    use_gpu: bool,
    use_cuda_ipc: bool,
    input_shm_names: Optional[List[str]],
    result_shm_names: List[str],
    model_kwargs: dict,
    gpu_device: Optional[str] = None,
) -> None:
    """Worker subprocess entry point.

    Loads the model, opens buffers, and enters a ZMQ forward() loop.
    Only calls model.forward() — pre/post-processing happens in the caller.

    Args:
        gpu_device: CUDA device string (e.g. ``"cuda:0"``, ``"cuda:1"``).
            ``None`` defaults to ``"cuda"`` (device 0) when ``use_gpu``
            is True.
    """
    import torch
    import zmq

    from inference_models.models.auto_loaders.core import AutoModel

    model = None
    input_shm: Optional[list] = None
    result_shm: Optional[list] = None
    sock = None
    zmq_ctx = None

    _log = logging.getLogger(f"{__name__}.worker")

    try:
        # ── CUDA init ────────────────────────────────────────────────
        device = "cpu"
        if use_gpu and torch.cuda.is_available():
            target = gpu_device or "cuda:0"
            dev = torch.device(target)
            torch.cuda.set_device(dev)
            device = str(dev)
        _log.info("Worker(%s): CUDA init done, device=%s", model_id, device)

        # ── CUDA IPC: receive GPU input buffers from main ────────────
        cuda_buffers = None
        cuda_events = None
        if use_cuda_ipc:
            setup = setup_pipe.recv()
            cuda_buffers = setup["buffers"]  # ForkingPickler opens IPC handles
            cuda_events = [
                torch.cuda.Event.from_ipc_handle(
                    torch.cuda.current_device(), h
                )
                for h in setup["event_handles"]
            ]

        # ── Load model on the target device ──────────────────────────
        _log.info("Worker(%s): loading model with weights on %s", model_id, device)
        model = AutoModel.from_pretrained(
            model_id, api_key=api_key, device=device, **model_kwargs,
        )
        _log.info(
            "Worker(%s): model loaded | type=%s | max_batch=%s",
            model_id, type(model).__name__,
            getattr(model, "max_batch_size", None),
        )

        # ── Open SHM ────────────────────────────────────────────────
        shm_pinned = False
        if input_shm_names:
            input_shm = [shared_memory.SharedMemory(name=n) for n in input_shm_names]
            # Pin input SHM for GPU DMA when CUDA is available
            if device.startswith("cuda"):
                pinned_ok = all(_pin_shm_buffer(s.buf) for s in input_shm)
                shm_pinned = pinned_ok
        result_shm = [shared_memory.SharedMemory(name=n) for n in result_shm_names]

        # ── Signal ready ────────────────────────────────────────────
        _log.info(
            "Worker(%s): ready | shm_pinned=%s | cuda_ipc=%s",
            model_id, shm_pinned, use_cuda_ipc,
        )
        setup_pipe.send("READY")

        # ── Connect ZMQ ─────────────────────────────────────────────
        zmq_ctx = zmq.Context()
        sock = zmq_ctx.socket(zmq.PAIR)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(zmq_addr)

        # ── Forward loop (with heartbeat) ───────────────────────────
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)

        while True:
            events = dict(poller.poll(timeout=_WORKER_HEARTBEAT_MS))
            if sock not in events:
                # Idle — send heartbeat so main knows we're alive
                try:
                    sock.send(_MSG_HEARTBEAT)
                except zmq.ZMQError:
                    break
                continue

            try:
                frames = sock.recv_multipart()
            except zmq.ZMQError:
                break

            if frames[0] == _MSG_SHUTDOWN:
                break
            if frames[0] != _MSG_INFER:
                continue

            meta = json.loads(frames[1])
            slot = meta["slot"]

            try:
                # Read input
                if cuda_buffers is not None:
                    cuda_events[slot].wait()
                    numel = meta["numel"]
                    shape = tuple(meta["shape"])
                    dtype_str = meta.get("dtype", "torch.float32")
                    torch_dtype = getattr(torch, dtype_str.replace("torch.", ""))
                    input_tensor = (
                        cuda_buffers[slot][:numel].to(torch_dtype).reshape(shape)
                    )
                else:
                    input_tensor = _tensor_from_shm(
                        input_shm[slot].buf, meta,
                        device=device, pinned=shm_pinned,
                    )

                # Forward (the only model call in the worker)
                with torch.no_grad():
                    output = model.forward(input_tensor)

                # Write result to SHM (always, even for CUDA IPC inputs)
                rmeta = _result_to_shm(result_shm[slot].buf, output)
                rmeta["slot"] = slot
                sock.send_multipart([_MSG_RESULT, json.dumps(rmeta).encode()])

            except Exception as e:
                sock.send_multipart(
                    [
                        _MSG_ERROR,
                        json.dumps(
                            {"slot": slot, "error": f"{type(e).__name__}: {e}"}
                        ).encode(),
                    ]
                )

    except Exception as e:
        try:
            setup_pipe.send(f"ERROR: {e}")
        except Exception:
            pass

    finally:
        if input_shm:
            for s in input_shm:
                try:
                    if shm_pinned:
                        _unpin_shm_buffer(s.buf)
                    s.close()
                except Exception:
                    pass
        if result_shm:
            for s in result_shm:
                try:
                    s.close()
                except Exception:
                    pass
        del model
        if sock:
            sock.close()
        if zmq_ctx:
            zmq_ctx.term()


# ─── SubprocessBackend ──────────────────────────────────────────────


class SubprocessBackend(Backend):
    """Runs model.forward() in a worker subprocess.

    pre_process() and post_process() execute on a CPU copy of the model in
    the caller process. The worker only calls forward().

    Transport selection::

        use_gpu=True,  use_cuda_ipc=True  → CUDA IPC input + SHM result
        use_gpu=True,  use_cuda_ipc=False → SHM input, worker .cuda(), SHM result
        use_gpu=False                     → SHM input, CPU forward, SHM result

    Both ``use_gpu`` and ``use_cuda_ipc`` default to True when CUDA is
    available.
    """

    def __init__(
        self,
        model_id: str,
        api_key: str,
        *,
        device: Optional[str] = None,
        use_gpu: Optional[bool] = None,
        use_cuda_ipc: Optional[bool] = None,
        batch_max_size: int = 0,
        batch_max_delay_ms: float = 10.0,
        num_buffer_slots: int = 4,
        input_buffer_mb: int = 32,
        result_buffer_mb: int = 128,
        worker_start_timeout: float = 120.0,
        **kwargs,
    ) -> None:
        """
        Args:
            device: CUDA device for the worker (e.g. ``"cuda:0"``,
                ``"cuda:1"``). When set, implies ``use_gpu=True``.
                ``None`` defaults to ``"cuda:0"`` when CUDA is available.
            use_gpu: Worker uses GPU for forward(). Default: True if CUDA
                available. Overridden to True when ``device`` is a CUDA
                device.
            use_cuda_ipc: Use CUDA IPC for input transport (requires
                use_gpu=True and caller CUDA context). Default: True if
                CUDA available.
            batch_max_size: Maximum batch size. 0 = no batching.
            batch_max_delay_ms: Max ms to wait for a full batch.
            num_buffer_slots: Number of paired (input+result) SHM slots.
            input_buffer_mb: Size of each input SHM slot in MB.
            result_buffer_mb: Size of each result SHM slot in MB.
            worker_start_timeout: Seconds to wait for worker to load
                model and signal ready. Raises RuntimeError on timeout.
        """
        import zmq

        self._model_id = model_id
        self._batch_max_size_requested = batch_max_size
        self._batch_max_delay_ms = batch_max_delay_ms
        self._state_value: str = "loading"

        # ── Resolve device + transport ───────────────────────────────
        has_cuda = _cuda_available()

        # device="cuda:1" implies use_gpu=True
        if device is not None and device.startswith("cuda"):
            use_gpu = True

        self._use_gpu = use_gpu if use_gpu is not None else has_cuda
        self._use_cuda_ipc = (
            use_cuda_ipc
            if use_cuda_ipc is not None
            else (has_cuda and self._use_gpu)
        )
        if self._use_cuda_ipc and not self._use_gpu:
            raise ValueError("use_cuda_ipc=True requires use_gpu=True")

        # Resolve device string: "cuda:0", "cuda:1", or "cpu"
        if self._use_gpu:
            self._device_str = device if device and device.startswith("cuda") else "cuda:0"
        else:
            self._device_str = "cpu"

        # ── Load model on CPU for pre/post processing (no weights) ───
        from inference_models.models.auto_loaders.core import AutoModel

        transport = (
            "cuda_ipc" if self._use_cuda_ipc
            else ("gpu_shm" if self._use_gpu else "cpu_shm")
        )
        logger.info(
            "SubprocessBackend(%s): loading pre/post config (no weights) "
            "| worker_device=%s | transport=%s",
            model_id, self._device_str, transport,
        )
        self._model = AutoModel.from_pretrained(
            model_id, api_key=api_key, device="cpu", load_weights=False,
            **kwargs,
        )

        # Clamp batch size to model's limit
        model_max = getattr(self._model, "max_batch_size", None)
        if batch_max_size > 0 and model_max is not None:
            self._batch_max_size = min(batch_max_size, model_max)
        else:
            self._batch_max_size = batch_max_size

        model_type = type(self._model).__name__
        class_count = len(self.class_names) if self.class_names else 0
        logger.info(
            "SubprocessBackend(%s): config loaded | model_type=%s | "
            "class_names=%d | model_max_batch=%s | effective_batch=%s | "
            "shm_slots=%d | input_buf=%dMB | result_buf=%dMB",
            model_id, model_type, class_count,
            model_max, self._batch_max_size or "off",
            num_buffer_slots, input_buffer_mb, result_buffer_mb,
        )

        # ── SHM buffer pool ──────────────────────────────────────────
        input_bytes = input_buffer_mb * 1024 * 1024
        result_bytes = result_buffer_mb * 1024 * 1024
        self._pool = _ShmPool(num_buffer_slots, input_bytes, result_bytes)

        # ── CUDA IPC GPU input buffers (optional) ────────────────────
        self._cuda_input_buffers: Optional[list] = None
        self._cuda_events: Optional[list] = None
        if self._use_cuda_ipc:
            import torch

            ipc_device = torch.device(self._device_str)
            max_elements = input_bytes // 4  # flat float32
            self._cuda_input_buffers = [
                torch.empty(max_elements, dtype=torch.float32, device=ipc_device)
                for _ in range(num_buffer_slots)
            ]
            self._cuda_events = [
                torch.cuda.Event(enable_timing=False, interprocess=True)
                for _ in range(num_buffer_slots)
            ]

        # ── ZMQ ──────────────────────────────────────────────────────
        self._zmq_ctx = zmq.Context()
        self._zmq_sock = self._zmq_ctx.socket(zmq.PAIR)
        self._zmq_sock.setsockopt(zmq.LINGER, 0)
        self._zmq_addr = (
            f"ipc:///tmp/inference_sp_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        )
        self._zmq_sock.bind(self._zmq_addr)
        self._zmq_lock = threading.Lock()

        # ── Spawn worker ─────────────────────────────────────────────
        try:
            import torch.multiprocessing as mp
        except ImportError:
            import multiprocessing as mp

        ctx = mp.get_context("spawn")
        parent_pipe, child_pipe = ctx.Pipe()

        self._worker = ctx.Process(
            target=_worker_main,
            kwargs=dict(
                model_id=model_id,
                api_key=api_key,
                setup_pipe=child_pipe,
                zmq_addr=self._zmq_addr,
                use_gpu=self._use_gpu,
                use_cuda_ipc=self._use_cuda_ipc,
                input_shm_names=(
                    self._pool.input_names if not self._use_cuda_ipc else None
                ),
                result_shm_names=self._pool.result_names,
                model_kwargs=kwargs,
                gpu_device=self._device_str if self._use_gpu else None,
            ),
            daemon=True,
        )
        self._worker.start()

        # Send CUDA IPC data through pipe (ForkingPickler handles handles)
        if self._use_cuda_ipc:
            parent_pipe.send(
                {
                    "buffers": self._cuda_input_buffers,
                    "event_handles": [
                        e.ipc_handle() for e in self._cuda_events
                    ],
                }
            )

        # Wait for worker to load model and signal ready (with timeout)
        if not parent_pipe.poll(timeout=worker_start_timeout):
            self._worker.kill()
            self._worker.join(timeout=5)
            self._pool.close()
            raise RuntimeError(
                f"Worker failed to start: timed out after {worker_start_timeout}s "
                f"waiting for READY (model loading too slow or worker crashed)"
            )
        msg = parent_pipe.recv()
        if isinstance(msg, str) and msg.startswith("ERROR"):
            self._worker.join(timeout=10)
            self._pool.close()
            raise RuntimeError(f"Worker failed to start: {msg}")
        if msg != "READY":
            self._pool.close()
            raise RuntimeError(f"Unexpected worker response: {msg}")

        self._pipe = parent_pipe
        self._last_worker_activity = time.monotonic()
        logger.info(
            "SubprocessBackend(%s): worker ready (pid=%d, device=%s)",
            model_id, self._worker.pid, self._device_str,
        )

        # ── Stats ────────────────────────────────────────────────────
        self._inference_count = 0
        self._error_count = 0
        self._last_inference_ts = 0.0
        self._latencies: deque[float] = deque(maxlen=1000)

        # ── Batching ─────────────────────────────────────────────────
        # Batching (<=1 means no batching — skip collector overhead)
        self._batch_collector = (
            self._create_batch_collector() if self._batch_max_size > 1 else None
        )

        self._state_value = "loaded"

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
    # Pipeline stages — pre/post on caller (CPU model), forward on worker
    # ------------------------------------------------------------------

    def pre_process(self, *args, **kwargs) -> Tuple[Any, Any]:
        result = self._model.pre_process(*args, **kwargs)
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
        return tensors

    def uncollate(self, batched_output: Any, count: int) -> List[Any]:
        import torch

        if isinstance(batched_output, (list, tuple)) and len(batched_output) == count:
            return list(batched_output)
        if isinstance(batched_output, torch.Tensor) and batched_output.shape[0] == count:
            return list(batched_output.split(1, dim=0))
        if count == 1:
            return [batched_output]
        return [batched_output] * count

    def forward_sync(self, batched_input: Any, **kwargs) -> Any:
        """Send batch to worker via buffer pool + ZMQ, block for result."""
        slot = self._pool.acquire()
        try:
            # ── Write input to buffer pool ───────────────────────────
            if self._use_cuda_ipc:
                import torch

                src = batched_input
                if not src.is_cuda:
                    src = src.cuda()
                src = src.contiguous()
                flat = src.reshape(-1).to(torch.float32)
                self._cuda_input_buffers[slot][: flat.numel()].copy_(flat)
                self._cuda_events[slot].record()
                meta: dict = {
                    "slot": slot,
                    "shape": list(batched_input.shape),
                    "dtype": str(batched_input.dtype),
                    "numel": flat.numel(),
                }
            else:
                meta = _tensor_to_shm(self._pool.input_buf(slot), batched_input)
                meta["slot"] = slot

            # ── Send request, block for response ─────────────────────
            with self._zmq_lock:
                self._zmq_sock.send_multipart(
                    [_MSG_INFER, json.dumps(meta).encode()]
                )
                # Drain any heartbeats, wait for actual result
                while True:
                    frames = self._zmq_sock.recv_multipart()
                    self._last_worker_activity = time.monotonic()
                    if frames[0] != _MSG_HEARTBEAT:
                        break

            # ── Handle response ──────────────────────────────────────
            if frames[0] == _MSG_ERROR:
                error_meta = json.loads(frames[1])
                raise RuntimeError(error_meta["error"])

            result_meta = json.loads(frames[1])
            return _result_from_shm(
                self._pool.result_buf(result_meta["slot"]), result_meta
            )
        finally:
            self._pool.release(slot)

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit(self, pre_processed: Tuple[Any, Any], *, priority: int = 0) -> Future:
        if not self.is_accepting:
            raise RuntimeError(
                f"SubprocessBackend('{self._model_id}') is not accepting "
                f"requests (state={self.state})"
            )

        tensor, meta = pre_processed

        if self._batch_collector is not None:
            t0 = time.monotonic()
            future = self._batch_collector.add(tensor, meta, priority=priority)
            future.add_done_callback(lambda f: self._record_inference(t0, f))
            return future

        # No batching — run pipeline directly (blocking)
        future: Future = Future()
        try:
            result = self._run_pipeline(tensor, meta)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        return future

    def _run_pipeline(self, tensor: Any, meta: Any) -> Any:
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

        try:
            with self._zmq_lock:
                self._zmq_sock.send_multipart([_MSG_SHUTDOWN, b""])
        except Exception:
            pass

        if self._worker.is_alive():
            self._worker.join(timeout=5)
            if self._worker.is_alive():
                self._worker.kill()

        self._zmq_sock.close(linger=0)
        self._zmq_ctx.term()
        self._pool.close()
        self._cuda_input_buffers = None
        self._cuda_events = None
        del self._model
        self._model = None

        if self._zmq_addr.startswith("ipc://"):
            try:
                os.unlink(self._zmq_addr[len("ipc://") :])
            except OSError:
                pass

    def sleep(self) -> Optional[int]:
        return None

    def wake(self) -> None:
        raise RuntimeError("SubprocessBackend does not support sleep/wake yet")

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def _drain_heartbeats(self) -> None:
        """Non-blocking drain of pending worker heartbeats."""
        import zmq

        with self._zmq_lock:
            while True:
                try:
                    self._zmq_sock.recv_multipart(flags=zmq.NOBLOCK)
                    self._last_worker_activity = time.monotonic()
                except zmq.Again:
                    break

    def _worker_responsive(self) -> bool:
        """True if worker sent a heartbeat or result recently."""
        self._drain_heartbeats()
        idle = time.monotonic() - self._last_worker_activity
        return idle < _WORKER_HEARTBEAT_TIMEOUT_S

    @property
    def device(self) -> str:
        return self._device_str

    @property
    def state(self) -> str:
        if self._model is None or not self._worker.is_alive():
            return "unhealthy"
        if self._state_value == "loaded" and not self._worker_responsive():
            return "unhealthy"
        return self._state_value

    @property
    def is_healthy(self) -> bool:
        return (
            self._model is not None
            and self._worker.is_alive()
            and self._worker_responsive()
        )

    @property
    def is_accepting(self) -> bool:
        return self.state == "loaded"

    @property
    def max_batch_size(self) -> Optional[int]:
        return getattr(self._model, "max_batch_size", None)

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
            return sorted_lats[idx] * 1000

        transport = (
            "cuda_ipc"
            if self._use_cuda_ipc
            else ("gpu_shm" if self._use_gpu else "cpu_shm")
        )

        return {
            "model_id": self._model_id,
            "backend_type": "subprocess",
            "device": self._device_str,
            "transport": transport,
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
            "gpu_memory_mb": 0.0,
            "cpu_pinned_memory_mb": 0.0,
            "inference_count": self._inference_count,
            "error_count": self._error_count,
            "last_inference_ts": self._last_inference_ts,
            "worker_alive": self._worker.is_alive(),
        }

    @property
    def class_names(self) -> Optional[List[str]]:
        return getattr(self._model, "class_names", None)

    # ------------------------------------------------------------------
    # Convenience (backward compat)
    # ------------------------------------------------------------------

    def infer_sync(self, *args, **kwargs) -> Any:
        pre_processed = self.pre_process(*args, **kwargs)
        future = self.submit(pre_processed)
        return future.result()

    async def infer_async(self, *args, **kwargs) -> Any:
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.infer_sync(*args, **kwargs)
        )
