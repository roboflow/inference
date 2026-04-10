"""SubprocessBackend — full inference pipeline runs in a worker subprocess.

The worker does decode → pre_process → forward → post_process.
The caller sends raw input (compressed bytes or numpy array) over POSIX
shared memory.  The worker returns pickled final results over SHM.

Triple-buffered input transport (configurable via ``num_input_buffers``):
  N input SHM blocks allow the caller to prepare the next batch while
  the worker processes the current one.  The worker reads numpy arrays
  directly from SHM (zero-copy view) — safe because the buffer won't
  be reused until N-1 more batches cycle through.

Batching: a caller-side BatchCollector groups concurrent requests,
respecting item count, time delay, and total byte limits (SHM buffer
capacity).

Transport:
  Input:  raw bytes (e.g. compressed JPEG ~50-200 KB) or numpy → SHM ring
  Result: pickled post-processed output (few KB) → SHM
  Signal: ZMQ PAIR carries JSON metadata (buffer index, offsets, shapes)
"""

from __future__ import annotations

import json
import logging
import os
import pickle
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


# ─── ZMQ message types ──────────────────────────────────────────────

_MSG_INFER = b"I"
_MSG_RESULT = b"R"
_MSG_ERROR = b"E"
_MSG_SHUTDOWN = b"S"
_MSG_HEARTBEAT = b"H"

_WORKER_HEARTBEAT_MS = 2000
_WORKER_HEARTBEAT_TIMEOUT_S = 30.0


# ─── SHM helpers ─────────────────────────────────────────────────────


def _input_nbytes(raw_input: Any) -> int:
    """Return the byte size of a raw input for SHM budget tracking."""
    if isinstance(raw_input, (bytes, bytearray, memoryview)):
        return len(raw_input)
    if isinstance(raw_input, np.ndarray):
        return raw_input.nbytes
    try:
        return raw_input.nelement() * raw_input.element_size()
    except AttributeError:
        return 0


def _write_input_to_shm(buf: memoryview, raw_input: Any) -> dict:
    """Write a single input to SHM at the start of *buf*.  Returns metadata."""
    if isinstance(raw_input, (bytes, bytearray, memoryview)):
        nbytes = len(raw_input)
        buf_bytes = buf.cast("B") if buf.format != "B" else buf
        buf_bytes[:nbytes] = raw_input
        return {"fmt": "bytes", "n": nbytes}

    if not isinstance(raw_input, np.ndarray):
        import torch

        if isinstance(raw_input, torch.Tensor):
            raw_input = raw_input.detach().cpu().contiguous().numpy()
        else:
            raw_input = np.asarray(raw_input)
    if not raw_input.flags["C_CONTIGUOUS"]:
        raw_input = np.ascontiguousarray(raw_input)
    nbytes = raw_input.nbytes
    shm_view = np.ndarray(raw_input.shape, dtype=raw_input.dtype, buffer=buf[:nbytes])
    np.copyto(shm_view, raw_input)
    return {
        "fmt": "np", "shape": list(raw_input.shape),
        "dtype": str(raw_input.dtype), "n": nbytes,
    }


def _read_input_from_shm(buf: memoryview, meta: dict) -> Any:
    """Read a single input from SHM.

    For bytes: returns a copy (needed by image decoders).
    For numpy: returns a zero-copy view into SHM.  Safe when the caller
    won't reuse this buffer until the worker finishes (triple-buffering).
    """
    n = meta["n"]
    if meta["fmt"] == "bytes":
        return bytes(buf[:n])
    shape = tuple(meta["shape"])
    dtype = np.dtype(meta["dtype"])
    return np.ndarray(shape, dtype=dtype, buffer=buf[:n])


# ─── Worker process ─────────────────────────────────────────────────


def _worker_main(
    model_id: str,
    api_key: str,
    setup_pipe: Any,
    zmq_addr: str,
    use_gpu: bool,
    input_shm_names: List[str],
    result_shm_name: str,
    model_kwargs: dict,
    gpu_device: Optional[str] = None,
) -> None:
    """Worker subprocess entry point.

    Loads the full model, receives batches of raw inputs over SHM,
    runs the complete pipeline (decode → pre → forward → post), and
    returns pickled results over SHM.
    """
    import torch
    if os.environ.get("ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND") is None:
        os.environ["ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND"] = "False"
    import zmq

    from inference_models.models.auto_loaders.core import AutoModel

    model = None
    input_shms: Optional[list] = None
    result_shm = None
    sock = None
    zmq_ctx = None
    _log = logging.getLogger(f"{__name__}.worker")

    try:
        device = gpu_device if use_gpu and gpu_device else ("cuda:0" if use_gpu else "cpu")
        _log.info("Worker(%s): loading model on %s", model_id, device)
        model = AutoModel.from_pretrained(
            model_id, api_key=api_key, device=device, **model_kwargs,
        )
        _log.info(
            "Worker(%s): model loaded | type=%s | max_batch=%s",
            model_id, type(model).__name__,
            getattr(model, "max_batch_size", None),
        )

        # ── Open SHM ────────────────────────────────────────────────
        input_shms = [shared_memory.SharedMemory(name=n) for n in input_shm_names]
        result_shm = shared_memory.SharedMemory(name=result_shm_name)

        # ── Signal ready (with model metadata) ──────────────────────
        setup_pipe.send({
            "status": "READY",
            "class_names": getattr(model, "class_names", None),
            "max_batch_size": getattr(model, "max_batch_size", None),
        })

        # ── Connect ZMQ ─────────────────────────────────────────────
        zmq_ctx = zmq.Context()
        sock = zmq_ctx.socket(zmq.PAIR)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(zmq_addr)

        # ── Main loop ───────────────────────────────────────────────
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)

        while True:
            events = dict(poller.poll(timeout=_WORKER_HEARTBEAT_MS))
            if sock not in events:
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

            req = json.loads(frames[1])
            buf_idx = req["buf"]

            try:
                _process_batch(model, req, input_shms[buf_idx], result_shm, sock)
            except Exception as e:
                _log.exception("Worker(%s): batch error", model_id)
                try:
                    sock.send_multipart([
                        _MSG_ERROR,
                        json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(),
                    ])
                except zmq.ZMQError:
                    break

    except Exception as e:
        try:
            setup_pipe.send({"status": f"ERROR: {e}"})
        except Exception:
            pass
    finally:
        if input_shms:
            for s in input_shms:
                try:
                    s.close()
                except Exception:
                    pass
        if result_shm:
            try:
                result_shm.close()
            except Exception:
                pass
        del model
        if sock:
            sock.close()
        if zmq_ctx:
            zmq_ctx.term()


def _decode_raw(raw):
    """Decode compressed bytes to numpy image, or pass through numpy arrays."""
    if isinstance(raw, bytes):
        import imagecodecs

        return imagecodecs.imread(raw)
    return raw


def _process_batch(model, req, input_shm, result_shm, sock):
    """Run the full pipeline on a batch of raw inputs.

    Reads inputs from the specified input SHM buffer at offsets described
    in req["items"].  Writes pickled results into result SHM.
    """
    items = req["items"]

    images = []
    for item in items:
        raw = _read_input_from_shm(input_shm.buf[item["offset"]:], item)
        images.append(_decode_raw(raw))

    if len(images) == 1:
        results = model.infer(images[0])
    else:
        results = model.infer(images)

    if not isinstance(results, list):
        results = [results]
    result_items = []
    offset = 0
    for final in results:
        data = pickle.dumps(final)
        result_shm.buf[offset: offset + len(data)] = data
        result_items.append({"offset": offset, "n": len(data)})
        offset += len(data)

    sock.send_multipart(
        [_MSG_RESULT, json.dumps({"items": result_items}).encode()]
    )


# ─── SubprocessBackend ──────────────────────────────────────────────


class SubprocessBackend(Backend):
    """Full inference pipeline runs in a worker subprocess.

    The worker handles decode, pre_process, forward, and post_process.
    The caller sends raw input (bytes or numpy) over a ring of SHM
    input buffers (triple-buffered by default) and receives pickled
    final results from a single result SHM block.

    When ``batch_max_size > 1`` a caller-side ``BatchCollector`` groups
    concurrent requests, respecting item count, time delay, and total
    byte size (SHM buffer capacity).
    """

    def __init__(
        self,
        model_id: str,
        api_key: str,
        *,
        device: Optional[str] = None,
        use_gpu: Optional[bool] = None,
        batch_max_size: int = 0,
        batch_max_delay_ms: float = 10.0,
        num_input_buffers: int = 3,
        input_buffer_mb: int = 32,
        result_buffer_mb: int = 80,
        worker_start_timeout: float = 120.0,
        **kwargs,
    ) -> None:
        import zmq

        self._model_id = model_id
        self._batch_max_size_requested = batch_max_size
        self._batch_max_delay_ms = batch_max_delay_ms
        self._state_value: str = "loading"

        # ── Resolve device ──────────────────────────────────────────
        if device is not None and device.startswith("cuda"):
            use_gpu = True
        if use_gpu is None:
            use_gpu = device is None or device.startswith("cuda")
        self._use_gpu = use_gpu
        if self._use_gpu:
            self._device_str = (
                device if device and device.startswith("cuda") else "cuda:0"
            )
        else:
            self._device_str = "cpu"

        # ── SHM: input ring + single result block ───────────────────
        self._input_buffer_bytes = input_buffer_mb * 1024 * 1024
        result_bytes = result_buffer_mb * 1024 * 1024

        self._input_shms = [
            shared_memory.SharedMemory(create=True, size=self._input_buffer_bytes)
            for _ in range(num_input_buffers)
        ]
        self._input_free: deque[int] = deque(range(num_input_buffers))
        self._input_free_cond = threading.Condition()

        self._result_shm = shared_memory.SharedMemory(
            create=True, size=result_bytes,
        )
        # Protects ZMQ socket (not thread-safe)
        self._zmq_lock = threading.Lock()

        logger.info(
            "SubprocessBackend(%s): device=%s | batch=%s | "
            "input_bufs=%d×%dMB | result_buf=%dMB",
            model_id, self._device_str,
            batch_max_size or "off",
            num_input_buffers, input_buffer_mb, result_buffer_mb,
        )

        # ── ZMQ ─────────────────────────────────────────────────────
        self._zmq_ctx = zmq.Context()
        self._zmq_sock = self._zmq_ctx.socket(zmq.PAIR)
        self._zmq_sock.setsockopt(zmq.LINGER, 0)
        self._zmq_addr = (
            f"ipc:///tmp/inference_sp_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        )
        self._zmq_sock.bind(self._zmq_addr)

        # ── Spawn worker ────────────────────────────────────────────
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
                input_shm_names=[s.name for s in self._input_shms],
                result_shm_name=self._result_shm.name,
                model_kwargs=kwargs,
                gpu_device=self._device_str if self._use_gpu else None,
            ),
            daemon=True,
        )
        self._worker.start()

        # ── Wait for worker ready ───────────────────────────────────
        if not parent_pipe.poll(timeout=worker_start_timeout):
            self._worker.kill()
            self._worker.join(timeout=5)
            self._cleanup_shm()
            raise RuntimeError(
                f"Worker failed to start: timed out after {worker_start_timeout}s"
            )

        msg = parent_pipe.recv()
        if not isinstance(msg, dict) or not msg.get("status", "").startswith("READY"):
            err = msg if isinstance(msg, str) else msg.get("status", str(msg))
            self._worker.join(timeout=10)
            self._cleanup_shm()
            raise RuntimeError(f"Worker failed to start: {err}")

        # ── Store model metadata from worker ────────────────────────
        self._class_names = msg.get("class_names")
        self._max_batch_size_model = msg.get("max_batch_size")
        self._pipe = parent_pipe
        self._last_worker_activity = time.monotonic()

        # Clamp batch size to model's limit
        if batch_max_size > 0 and self._max_batch_size_model is not None:
            self._batch_max_size = min(batch_max_size, self._max_batch_size_model)
        else:
            self._batch_max_size = batch_max_size

        logger.info(
            "SubprocessBackend(%s): worker ready (pid=%d, device=%s, batch=%s)",
            model_id, self._worker.pid, self._device_str,
            self._batch_max_size or "off",
        )

        # ── Stats ───────────────────────────────────────────────────
        self._inference_count = 0
        self._error_count = 0
        self._last_inference_ts = 0.0
        self._latencies: deque[float] = deque(maxlen=1000)

        # ── Batching ────────────────────────────────────────────────
        self._batch_collector = (
            self._create_batch_collector() if self._batch_max_size > 1 else None
        )
        self._state_value = "loaded"

    # ------------------------------------------------------------------
    # SHM management
    # ------------------------------------------------------------------

    def _acquire_input_buf(self, timeout: float = 30.0) -> int:
        """Get a free input buffer index. Blocks if all buffers in-flight."""
        with self._input_free_cond:
            deadline = time.monotonic() + timeout
            while not self._input_free:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._input_free_cond.wait(timeout=remaining):
                    raise TimeoutError("No free input buffer")
            return self._input_free.popleft()

    def _release_input_buf(self, idx: int) -> None:
        """Mark an input buffer as free (worker finished reading it)."""
        with self._input_free_cond:
            self._input_free.append(idx)
            self._input_free_cond.notify()

    def _cleanup_shm(self):
        for shm in self._input_shms:
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
        try:
            self._result_shm.close()
            self._result_shm.unlink()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Batching
    # ------------------------------------------------------------------

    def _create_batch_collector(self):
        from inference_models.backends.batch_collector import BatchCollector

        return BatchCollector(
            forward_fn=self._send_batch_to_worker,
            max_size=self._batch_max_size,
            max_delay_s=self._batch_max_delay_ms / 1000,
            max_bytes=self._input_buffer_bytes,
        )

    # ------------------------------------------------------------------
    # Worker communication
    # ------------------------------------------------------------------

    def _send_batch_to_worker(self, raw_items: list) -> list:
        """Send a batch of (raw_input, kwargs) to the worker.

        Acquires a free input buffer, packs all inputs sequentially,
        sends metadata over ZMQ, waits for result, releases buffer.
        """
        buf_idx = self._acquire_input_buf()
        try:
            # ── Pack inputs into acquired buffer (outside zmq lock) ──
            buf = self._input_shms[buf_idx].buf
            items_meta: list[dict] = []
            offset = 0
            for raw_input, kwargs in raw_items:
                meta = _write_input_to_shm(buf[offset:], raw_input)
                meta["offset"] = offset
                if kwargs:
                    meta["kwargs"] = kwargs
                offset += meta["n"]
                items_meta.append(meta)

            # ── Send request + wait for response ─────────────────────
            with self._zmq_lock:
                self._zmq_sock.send_multipart([
                    _MSG_INFER,
                    json.dumps({"buf": buf_idx, "items": items_meta}).encode(),
                ])
                while True:
                    frames = self._zmq_sock.recv_multipart()
                    self._last_worker_activity = time.monotonic()
                    if frames[0] != _MSG_HEARTBEAT:
                        break
        finally:
            self._release_input_buf(buf_idx)

        # ── Parse response ──────────────────────────────────────────
        if frames[0] == _MSG_ERROR:
            raise RuntimeError(
                json.loads(frames[1]).get("error", "worker error")
            )

        resp = json.loads(frames[1])
        results = []
        for rm in resp["items"]:
            data = bytes(self._result_shm.buf[rm["offset"]: rm["offset"] + rm["n"]])
            results.append(pickle.loads(data))
        return results

    # ------------------------------------------------------------------
    # Pipeline stages (caller-side stubs — real work is in the worker)
    # ------------------------------------------------------------------

    def pre_process(self, *args, **kwargs) -> Tuple[Any, Any]:
        return (args[0] if args else None, kwargs if kwargs else None)

    def post_process(self, raw_output: Any, meta: Any, **kwargs) -> Any:
        return raw_output

    def collate(self, items: List[Tuple[Any, Any]]) -> Any:
        return items

    def uncollate(self, batched_output: Any, count: int) -> List[Any]:
        if isinstance(batched_output, list):
            return batched_output
        return [batched_output] * count

    def forward_sync(self, raw_items: Any, **kwargs) -> Any:
        return self._send_batch_to_worker(raw_items)

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit(self, pre_processed: Tuple[Any, Any], *, priority: int = 0) -> Future:
        if not self.is_accepting:
            raise RuntimeError(
                f"SubprocessBackend('{self._model_id}') is not accepting "
                f"requests (state={self.state})"
            )

        raw_input, kwargs = pre_processed
        nbytes = _input_nbytes(raw_input)

        if nbytes > self._input_buffer_bytes:
            raise ValueError(
                f"Input too large ({nbytes} bytes) for SHM buffer "
                f"({self._input_buffer_bytes} bytes). "
                f"Increase input_buffer_mb (currently "
                f"{self._input_buffer_bytes // (1024 * 1024)}MB)."
            )

        if self._batch_collector is not None:
            t0 = time.monotonic()
            future = self._batch_collector.add(
                raw_input, kwargs, priority=priority, nbytes=nbytes,
            )
            future.add_done_callback(lambda f: self._record_inference(t0, f))
            return future

        # No batching — single-item request
        future: Future = Future()
        t0 = time.monotonic()
        try:
            results = self._send_batch_to_worker([(raw_input, kwargs)])
            future.set_result(results[0])
        except Exception as e:
            future.set_exception(e)
        finally:
            elapsed = time.monotonic() - t0
            self._inference_count += 1
            self._last_inference_ts = t0
            self._latencies.append(elapsed)
            if future.exception() is not None:
                self._error_count += 1
        return future

    def _record_inference(self, t0: float, future: Future) -> None:
        elapsed = time.monotonic() - t0
        self._inference_count += 1
        self._last_inference_ts = t0
        self._latencies.append(elapsed)
        if future.exception() is not None:
            self._error_count += 1

    # ------------------------------------------------------------------
    # Convenience
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
        self._cleanup_shm()

        if self._zmq_addr.startswith("ipc://"):
            try:
                os.unlink(self._zmq_addr[len("ipc://"):])
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
        import zmq

        with self._zmq_lock:
            while True:
                try:
                    self._zmq_sock.recv_multipart(flags=zmq.NOBLOCK)
                    self._last_worker_activity = time.monotonic()
                except zmq.Again:
                    break

    def _worker_responsive(self) -> bool:
        self._drain_heartbeats()
        idle = time.monotonic() - self._last_worker_activity
        return idle < _WORKER_HEARTBEAT_TIMEOUT_S

    @property
    def device(self) -> str:
        return self._device_str

    @property
    def state(self) -> str:
        if not self._worker.is_alive():
            return "unhealthy"
        if self._state_value == "loaded" and not self._worker_responsive():
            return "unhealthy"
        return self._state_value

    @property
    def is_healthy(self) -> bool:
        return self._worker.is_alive() and self._worker_responsive()

    @property
    def is_accepting(self) -> bool:
        return self.state == "loaded"

    @property
    def max_batch_size(self) -> Optional[int]:
        return self._max_batch_size_model

    @property
    def queue_depth(self) -> int:
        if self._batch_collector is not None:
            return self._batch_collector.queue_depth
        return 0

    @property
    def input_buffer_capacity(self) -> int:
        """Capacity of each input SHM buffer in bytes."""
        return self._input_buffer_bytes

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
            "backend_type": "subprocess",
            "device": self._device_str,
            "transport": "shm_ring",
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
            "input_buffers": len(self._input_shms),
            "input_buffer_mb": self._input_buffer_bytes // (1024 * 1024),
        }

    @property
    def class_names(self) -> Optional[List[str]]:
        return self._class_names
