"""SubprocessBackend v2 — SHMPool + worker-side greedy batching.

Transport:
  Input:  raw bytes written to SHMPool.input_memoryview(slot_id)
  Output: pickled Python object in SHMPool.result_memoryview(slot_id)
  Signal: ZMQ PAIR — parent sends T_SLOT_READY per request;
          worker sends T_RESULT per completed slot

Thread-safety (ZMQ):
  ZMQ sockets are not thread-safe.  All socket operations run in _recv_thread.
  signal_slot() and unload() enqueue work to _outbound (Queue); the recv
  thread drains _outbound before polling for inbound messages.

Modes:
  standalone:
    SubprocessBackend owns the SHMPool.  submit() allocates a slot, writes
    input, signals the worker, and returns a Future that the recv thread
    resolves when T_RESULT arrives.  infer_sync() calls Future.result().

  orchestrated (MMP):
    SubprocessBackend attaches to MMP's existing SHMPool.  MMP calls
    signal_slot() for each request; the recv thread calls on_result_callback
    when T_RESULT arrives.  submit()/infer_sync() raise RuntimeError.

Input formats (both modes):
  - bytes / bytearray / memoryview:  written directly.
  - numpy ndarray:  serialised with np.save() (magic b'\\x93NUMPY').
  Worker detects format by inspecting the first 6 bytes of the slot.
"""

from __future__ import annotations

import io
import logging
import os
import pickle
import queue
import struct
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from inference_models.backends.base import Backend
from inference_models.backends.utils.shm_pool import SHMPool
from inference_models.backends.utils.transport import default_transport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PAIR protocol constants (parent ↔ worker)
# ---------------------------------------------------------------------------

_MSG_SLOT_READY = b"\x01"   # parent→worker: struct.pack(">IQ", slot_id, req_id)  [12 B]
_MSG_RESULT     = b"\x02"   # worker→parent: struct.pack(">QII", req_id, slot_id, result_sz) [16 B]
_MSG_HEARTBEAT  = b"\x03"   # worker→parent: keepalive (no payload)
_MSG_SHUTDOWN   = b"\x04"   # parent→worker: stop gracefully (no payload)

_HEARTBEAT_INTERVAL_S   = 2.0
_WORKER_HEARTBEAT_TIMEOUT = 30.0   # seconds of silence → unhealthy

_DEFAULT_BATCH_MAX_SIZE    = 8
_DEFAULT_BATCH_MAX_WAIT_MS = 5.0


# ---------------------------------------------------------------------------
# Input serialisation helper (parent side)
# ---------------------------------------------------------------------------

_NP_MAGIC = b"\x93NUMPY"   # first 6 bytes of every np.save() file


def _to_bytes(raw_input: Any) -> bytes:
    """Serialise any input value to bytes for writing to the SHMPool input slot.

    Returns:
        bytes, bytearray, memoryview  →  bytes (zero-copy when possible)
        numpy ndarray                 →  numpy .npy bytes (magic b'\\x93NUMPY')
        anything else                 →  pickle
    """
    if isinstance(raw_input, (bytes, bytearray)):
        return bytes(raw_input)
    if isinstance(raw_input, memoryview):
        return bytes(raw_input)
    if isinstance(raw_input, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, raw_input, allow_pickle=False)
        return buf.getvalue()
    return pickle.dumps(raw_input)


# ---------------------------------------------------------------------------
# Worker subprocess
# ---------------------------------------------------------------------------

def _worker_main(
    model_id:          str,
    api_key:           str,
    setup_pipe:        Any,
    zmq_addr:          str,
    use_gpu:           bool,
    gpu_device:        Optional[str],
    shm_pool_name:     str,
    n_slots:           int,
    input_mb:          float,
    result_mb:         float,
    batch_max_size:    int,
    batch_max_wait_ms: float,
    decoder_name:      str,
    model_kwargs:      dict,
) -> None:
    """Worker subprocess entry point.

    Loads model, attaches SHMPool, signals READY, then greedy-batches
    T_SLOT_READY messages and sends T_RESULT per completed slot.
    """
    if os.environ.get("ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND") is None:
        os.environ["ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND"] = "False"

    import zmq  # noqa: PLC0415

    from inference_models.models.auto_loaders.core import AutoModel  # noqa: PLC0415
    from inference_models.backends.decode import make_decoder          # noqa: PLC0415

    _log = logging.getLogger(f"{__name__}.worker")
    pool = sock = zmq_ctx = model = None

    try:
        device = gpu_device if (use_gpu and gpu_device) else ("cuda:0" if use_gpu else "cpu")
        _log.info("Worker(%s): loading on %s", model_id, device)
        model = AutoModel.from_pretrained(model_id, api_key=api_key, device=device,
                                          **model_kwargs)
        _log.info("Worker(%s): model ready (%s)", model_id, type(model).__name__)

        decode_fn = make_decoder(decoder_name, device=device)
        pool      = SHMPool.attach(shm_pool_name, n_slots=n_slots,
                                   input_mb=input_mb, result_mb=result_mb)

        setup_pipe.send({
            "status":        "READY",
            "class_names":   getattr(model, "class_names", None),
            "max_batch_size": getattr(model, "max_batch_size", None),
        })

        zmq_ctx = zmq.Context()
        sock    = zmq_ctx.socket(zmq.PAIR)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(zmq_addr)

        _worker_loop(model, pool, sock, decode_fn,
                     batch_max_size, batch_max_wait_ms, _log)

    except Exception as exc:
        try:
            setup_pipe.send({"status": f"ERROR: {exc}"})
        except Exception:
            pass
    finally:
        if pool:
            pool.close()
        if sock:
            sock.close()
        if zmq_ctx:
            zmq_ctx.term()
        del model


def _worker_loop(
    model,
    pool:              SHMPool,
    sock,
    decode_fn,
    batch_max_size:    int,
    batch_max_wait_ms: float,
    log,
) -> None:
    """Greedy batch loop — accumulate T_SLOT_READY, fire on size-or-timeout."""
    import zmq

    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)

    batch_max_wait_s = batch_max_wait_ms / 1000.0
    pending: list[tuple[int, int]] = []   # (slot_id, req_id)
    batch_start   = 0.0
    last_heartbeat = time.monotonic()

    while True:
        # Compute poll timeout
        now = time.monotonic()
        if pending:
            wait_left  = batch_start + batch_max_wait_s - now
            timeout_ms = max(0, int(wait_left * 1000))
        else:
            hb_due     = last_heartbeat + _HEARTBEAT_INTERVAL_S
            timeout_ms = max(1, int((hb_due - now) * 1000))

        events = dict(poller.poll(timeout=timeout_ms))

        if sock in events:
            try:
                frames = sock.recv_multipart()
            except zmq.ZMQError:
                break

            msg = frames[0]
            if msg == _MSG_SHUTDOWN:
                break
            elif msg == _MSG_SLOT_READY and len(frames) > 1 and len(frames[1]) == 12:
                slot_id, req_id = struct.unpack(">IQ", frames[1])
                if not pending:
                    batch_start = time.monotonic()
                pending.append((slot_id, req_id))
        else:
            # Timeout while idle → heartbeat
            if not pending:
                now = time.monotonic()
                if now - last_heartbeat >= _HEARTBEAT_INTERVAL_S:
                    try:
                        sock.send_multipart([_MSG_HEARTBEAT, b""])
                    except zmq.ZMQError:
                        break
                    last_heartbeat = now

        # Fire batch when ready
        now = time.monotonic()
        if pending and (
            len(pending) >= batch_max_size
            or (now - batch_start) >= batch_max_wait_s
        ):
            _process_slots(model, pool, pending, sock, decode_fn, log)
            pending.clear()
            batch_start = 0.0


def _process_slots(
    model,
    pool:     SHMPool,
    batch:    list[tuple[int, int]],
    sock,
    decode_fn,
    log,
) -> None:
    """Process a batch of (slot_id, req_id), write results to SHM, send T_RESULT."""
    import zmq

    images: list[Any] = []
    for slot_id, _req_id in batch:
        hdr = pool.read_header(slot_id)
        raw = bytes(pool.input_memoryview(slot_id)[: hdr.input_size])
        # Numpy .npy magic vs raw image bytes (JPEG/PNG/etc.)
        if raw[:6] == _NP_MAGIC:
            img = np.load(io.BytesIO(raw), allow_pickle=False)
        else:
            img = decode_fn(raw)
        images.append(img)

    results: list[Any]
    try:
        raw_out = model.infer(images[0]) if len(images) == 1 else model.infer(images)
        results = raw_out if isinstance(raw_out, list) else [raw_out]
    except Exception:
        log.exception("Worker: model.infer() failed")
        results = [None] * len(batch)

    for (slot_id, req_id), result in zip(batch, results):
        if result is None:
            pool.mark_error(slot_id)
            try:
                sock.send_multipart([_MSG_RESULT,
                                     struct.pack(">QII", req_id, slot_id, 0)])
            except zmq.ZMQError:
                return
            continue

        data = pickle.dumps(result)
        mv   = pool.result_memoryview(slot_id)
        mv[:len(data)] = data
        mv.release()
        pool.mark_done(slot_id, len(data))
        try:
            sock.send_multipart([_MSG_RESULT,
                                 struct.pack(">QII", req_id, slot_id, len(data))])
        except zmq.ZMQError:
            return


# ---------------------------------------------------------------------------
# SubprocessBackend
# ---------------------------------------------------------------------------

class SubprocessBackend(Backend):
    """Inference backend running the model in a worker subprocess.

    v2 transport: SHMPool for data, ZMQ PAIR for signals.
    Worker accumulates slot signals and greedy-batches them.
    """

    def __init__(
        self,
        model_id: str,
        api_key: str,
        *,
        # SHMPool
        shm_pool_name:       Optional[str]      = None,
        n_slots:             int                = 8,
        input_mb:            float              = 20.0,
        result_mb:           float              = 4.0,
        # Worker batching
        batch_max_size:      int                = _DEFAULT_BATCH_MAX_SIZE,
        batch_max_delay_ms:  float              = _DEFAULT_BATCH_MAX_WAIT_MS,
        # Orchestrated-mode callback
        on_result_callback:  Optional[Callable] = None,
        # Device
        device:              Optional[str]      = None,
        use_gpu:             Optional[bool]     = None,
        use_cuda_ipc:        Optional[bool]     = None,   # reserved, unused
        # Misc
        decoder:             str                = "cv2",
        worker_start_timeout: float             = 120.0,
        **kwargs,
    ) -> None:
        import zmq  # noqa: PLC0415

        self._model_id    = model_id
        self._state_value: str = "loading"

        # ── Device resolution ────────────────────────────────────────
        if device is not None and device.startswith("cuda"):
            use_gpu = True
        if use_gpu is None:
            use_gpu = (device is None or device.startswith("cuda"))
        self._use_gpu    = use_gpu
        self._device_str = (
            (device if device and device.startswith("cuda") else "cuda:0")
            if self._use_gpu else "cpu"
        )

        # ── SHMPool ─────────────────────────────────────────────────
        if shm_pool_name is None:
            self._pool     = SHMPool.create(n_slots, input_mb, result_mb)
            self._own_pool = True
        else:
            self._pool     = SHMPool.attach(shm_pool_name, n_slots=n_slots,
                                            input_mb=input_mb, result_mb=result_mb)
            self._own_pool = False

        logger.info(
            "SubprocessBackend(%s): device=%s pool=%s slots=%d "
            "input=%.0fMB result=%.0fMB batch=%d/%.0fms",
            model_id, self._device_str,
            "owned" if self._own_pool else f"attached:{shm_pool_name[:8]}…",
            n_slots, input_mb, result_mb, batch_max_size, batch_max_delay_ms,
        )

        # ── ZMQ PAIR ─────────────────────────────────────────────────
        self._zmq_ctx  = zmq.Context()
        self._zmq_sock = self._zmq_ctx.socket(zmq.PAIR)
        self._zmq_sock.setsockopt(zmq.LINGER, 0)

        _transport = os.environ.get("INFERENCE_ZMQ_TRANSPORT", default_transport())
        _sock_id   = f"sp_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        if _transport == "ipc":
            self._zmq_addr = f"ipc:///tmp/inference_{_sock_id}.ipc"
        else:
            self._zmq_addr = "tcp://127.0.0.1:*"
        self._zmq_sock.bind(self._zmq_addr)
        if _transport != "ipc":
            self._zmq_addr = self._zmq_sock.getsockopt_string(zmq.LAST_ENDPOINT)

        # ── Spawn worker ─────────────────────────────────────────────
        import multiprocessing as mp  # noqa: PLC0415

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
                gpu_device=self._device_str if self._use_gpu else None,
                shm_pool_name=self._pool.name,
                n_slots=n_slots,
                input_mb=input_mb,
                result_mb=result_mb,
                batch_max_size=batch_max_size,
                batch_max_wait_ms=batch_max_delay_ms,
                decoder_name=decoder,
                model_kwargs=kwargs,
            ),
            daemon=True,
        )
        self._worker.start()

        if not parent_pipe.poll(timeout=worker_start_timeout):
            self._worker.kill()
            self._worker.join(timeout=5)
            self._pool.close()
            raise RuntimeError(
                f"SubprocessBackend({model_id!r}): worker timeout "
                f"after {worker_start_timeout}s"
            )

        msg = parent_pipe.recv()
        if not isinstance(msg, dict) or not msg.get("status", "").startswith("READY"):
            err = msg if isinstance(msg, str) else msg.get("status", str(msg))
            self._worker.join(timeout=10)
            self._pool.close()
            raise RuntimeError(f"SubprocessBackend({model_id!r}): {err}")

        self._class_names           = msg.get("class_names")
        self._max_batch_size_model  = msg.get("max_batch_size")
        self._last_worker_activity  = time.monotonic()

        logger.info(
            "SubprocessBackend(%s): worker ready (pid=%d, device=%s)",
            model_id, self._worker.pid, self._device_str,
        )

        # ── Stats ─────────────────────────────────────────────────────
        self._inference_count   = 0
        self._error_count       = 0
        self._last_inference_ts = 0.0
        self._latencies: deque[float] = deque(maxlen=1000)

        # ── Recv thread ───────────────────────────────────────────────
        # All ZMQ socket I/O runs here.  Other threads communicate via _outbound.
        self._outbound: queue.Queue = queue.Queue()
        self._slot_futures: Dict[int, tuple] = {}   # slot_id → (req_id, Future)
        self._slot_lock                       = threading.Lock()
        self._on_result_callback: Optional[Callable] = on_result_callback
        self._recv_running = True
        self._recv_thread  = threading.Thread(
            target=self._recv_loop,
            daemon=True,
            name=f"subproc-recv-{model_id[:20]}",
        )
        self._recv_thread.start()

        self._state_value = "loaded"

    # ------------------------------------------------------------------
    # Orchestrated-mode API (called by MMP)
    # ------------------------------------------------------------------

    def signal_slot(self, slot_id: int, req_id: int) -> None:
        """Enqueue T_SLOT_READY for the recv thread to send. Thread-safe."""
        self._outbound.put((slot_id, req_id))

    def set_on_result_callback(self, callback: Callable[[int, int, int], None]) -> None:
        """Set MMP callback: called on each T_RESULT from worker."""
        self._on_result_callback = callback

    # ------------------------------------------------------------------
    # Recv thread — sole owner of _zmq_sock
    # ------------------------------------------------------------------

    def _recv_loop(self) -> None:
        import zmq  # noqa: PLC0415

        poller = zmq.Poller()
        poller.register(self._zmq_sock, zmq.POLLIN)

        while self._recv_running:
            # Drain outbound queue first
            while True:
                try:
                    item = self._outbound.get_nowait()
                except queue.Empty:
                    break
                if item is None:   # shutdown sentinel
                    try:
                        self._zmq_sock.send_multipart([_MSG_SHUTDOWN, b""])
                    except Exception:
                        pass
                    return
                slot_id, req_id = item
                try:
                    self._zmq_sock.send_multipart(
                        [_MSG_SLOT_READY, struct.pack(">IQ", slot_id, req_id)]
                    )
                except zmq.ZMQError:
                    return

            # Poll with 10ms timeout so outbound queue drains promptly
            events = dict(poller.poll(timeout=10))
            if self._zmq_sock not in events:
                continue

            try:
                frames = self._zmq_sock.recv_multipart()
            except zmq.ZMQError:
                break

            self._last_worker_activity = time.monotonic()
            msg = frames[0]

            if msg == _MSG_HEARTBEAT:
                pass   # timestamp already updated
            elif msg == _MSG_RESULT and len(frames) > 1 and len(frames[1]) == 16:
                req_id, slot_id, result_sz = struct.unpack(">QII", frames[1])
                self._handle_result(req_id, slot_id, result_sz)

    def _handle_result(self, req_id: int, slot_id: int, result_sz: int) -> None:
        """Called from recv thread on each T_RESULT."""
        # Standalone mode: resolve Future for this slot
        if self._own_pool:
            with self._slot_lock:
                entry = self._slot_futures.pop(slot_id, None)
            if entry:
                _, future = entry
                if result_sz > 0:
                    try:
                        data   = bytes(self._pool.result_memoryview(slot_id)[:result_sz])
                        result = pickle.loads(data)
                        if not future.done():
                            future.set_result(result)
                    except Exception as exc:
                        if not future.done():
                            future.set_exception(exc)
                else:
                    if not future.done():
                        future.set_exception(RuntimeError("worker inference error"))
                self._pool.free_slot(slot_id)

        # Orchestrated mode: notify MMP
        if self._on_result_callback is not None:
            try:
                self._on_result_callback(req_id, slot_id, result_sz)
            except Exception:
                logger.exception("SubprocessBackend: on_result_callback raised")

    # ------------------------------------------------------------------
    # Standalone inference
    # ------------------------------------------------------------------

    def submit(self, raw_input: Any, *, priority: int = 0, **kwargs) -> Future:
        """Submit a request and return a Future. Standalone mode only."""
        if not self._own_pool:
            raise RuntimeError(
                "submit() is unavailable in orchestrated mode — use signal_slot()"
            )
        if not self.is_accepting:
            raise RuntimeError(
                f"SubprocessBackend('{self._model_id}') not accepting "
                f"(state={self.state})"
            )

        input_bytes = _to_bytes(raw_input)
        if len(input_bytes) > self._pool.input_slot_bytes:
            raise ValueError(
                f"Input {len(input_bytes)} B > slot capacity "
                f"{self._pool.input_slot_bytes} B — increase input_mb"
            )

        req_id  = uuid.uuid4().int & 0xFFFF_FFFF_FFFF_FFFF
        slot_id = self._pool.alloc_slot()
        self._pool.mark_allocated(slot_id, req_id)
        self._pool.input_memoryview(slot_id)[:len(input_bytes)] = input_bytes
        self._pool.mark_written(slot_id, len(input_bytes))

        future: Future = Future()
        t0 = time.monotonic()
        future.add_done_callback(lambda f: self._record_inference(t0, f))

        with self._slot_lock:
            self._slot_futures[slot_id] = (req_id, future)

        self.signal_slot(slot_id, req_id)
        return future

    def _record_inference(self, t0: float, future: Future) -> None:
        elapsed = time.monotonic() - t0
        self._inference_count   += 1
        self._last_inference_ts  = t0
        self._latencies.append(elapsed)
        if future.exception() is not None:
            self._error_count += 1

    def infer_sync(self, raw_input: Any, **kwargs) -> Any:
        return self.submit(raw_input, **kwargs).result()

    async def infer_async(self, raw_input: Any, **kwargs) -> Any:
        import asyncio  # noqa: PLC0415
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.infer_sync(raw_input, **kwargs)
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def unload(self) -> None:
        self._state_value = "loading"   # block new submits immediately

        # Signal recv thread: send T_SHUTDOWN to worker, then exit
        self._recv_running = False
        self._outbound.put(None)        # None = shutdown sentinel
        self._recv_thread.join(timeout=5.0)

        # Kill worker if still alive
        if self._worker.is_alive():
            self._worker.join(timeout=5)
        if self._worker.is_alive():
            self._worker.kill()

        # Cancel pending futures (standalone mode)
        with self._slot_lock:
            for slot_id, (_, future) in self._slot_futures.items():
                if not future.done():
                    future.set_exception(RuntimeError("backend unloaded"))
            self._slot_futures.clear()

        self._zmq_sock.close(linger=0)
        self._zmq_ctx.term()

        self._pool.close()   # owner unlinks; attached just detaches

        if self._zmq_addr.startswith("ipc://"):
            try:
                os.unlink(self._zmq_addr[len("ipc://"):])
            except OSError:
                pass

        logger.info("SubprocessBackend(%s): unloaded", self._model_id)

    def sleep(self) -> Optional[int]:
        return None   # Issue #8: not yet implemented

    def wake(self) -> None:
        raise RuntimeError(
            "SubprocessBackend.sleep/wake not yet implemented (Issue #8)"
        )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @property
    def device(self) -> str:
        return self._device_str

    @property
    def state(self) -> str:
        if not self._worker.is_alive():
            return "unhealthy"
        idle = time.monotonic() - self._last_worker_activity
        if self._state_value == "loaded" and idle > _WORKER_HEARTBEAT_TIMEOUT:
            return "unhealthy"
        return self._state_value

    @property
    def is_healthy(self) -> bool:
        return self.state == "loaded"

    @property
    def is_accepting(self) -> bool:
        return self.state == "loaded"

    @property
    def max_batch_size(self) -> Optional[int]:
        return self._max_batch_size_model

    @property
    def queue_depth(self) -> int:
        with self._slot_lock:
            return len(self._slot_futures)

    def stats(self) -> Dict[str, Any]:
        sorted_lats = sorted(self._latencies) if self._latencies else []

        def _pct(p: float) -> float:
            if not sorted_lats:
                return 0.0
            idx = min(int(len(sorted_lats) * p / 100), len(sorted_lats) - 1)
            return sorted_lats[idx] * 1000

        return {
            "model_id":             self._model_id,
            "backend_type":         "subprocess",
            "device":               self._device_str,
            "transport":            "shm_pool",
            "state":                self.state,
            "is_accepting":         self.is_accepting,
            "queue_depth":          self.queue_depth,
            "max_batch_size":       self.max_batch_size,
            "throughput_fps":       (
                self._inference_count / sum(sorted_lats) if sorted_lats else 0.0
            ),
            "latency_p50_ms":       _pct(50),
            "latency_p99_ms":       _pct(99),
            "gpu_memory_mb":        0.0,
            "cpu_pinned_memory_mb": 0.0,
            "inference_count":      self._inference_count,
            "error_count":          self._error_count,
            "last_inference_ts":    self._last_inference_ts,
            "worker_alive":         self._worker.is_alive(),
            "shm_pool_name":        self._pool.name,
            "shm_own_pool":         self._own_pool,
            "shm_free_slots":       self._pool.free_count if self._own_pool else -1,
        }

    @property
    def class_names(self) -> Optional[List[str]]:
        return self._class_names
