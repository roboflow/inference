"""SubprocessBackend v2 — SHMPool + worker-side greedy batching.

Transport:
  Input:  raw bytes written to SHMPool.data_memoryview(slot_id)
  Output: pickled Python object written to same SHMPool.data_memoryview(slot_id)
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

import asyncio
import io
import json
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

from inference_model_manager.backends.base import Backend
from inference_model_manager.backends.utils.shm_pool import SHMPool
from inference_model_manager.backends.utils.transport import default_transport
from inference_model_manager.dispatch import invoke_task

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PAIR protocol constants (parent ↔ worker)
# ---------------------------------------------------------------------------

_MSG_SLOT_READY = b"\x01"  # parent→worker: struct.pack(">IQ", slot_id, req_id)  [12 B]
_MSG_RESULT = (
    b"\x02"  # worker→parent: struct.pack(">QII", req_id, slot_id, result_sz) [16 B]
)
_MSG_HEARTBEAT = b"\x03"  # worker→parent: keepalive + JSON stats payload
_MSG_SHUTDOWN = b"\x04"  # parent→worker: stop gracefully (no payload)
_MSG_STATS_REQ = b"\x05"  # parent→worker: force stats refresh (no payload)

_STATS_SENTINEL = "STATS"  # sentinel for outbound queue → sends _MSG_STATS_REQ

_HEARTBEAT_INTERVAL_S = 2.0
_WORKER_HEARTBEAT_TIMEOUT = 30.0  # seconds of silence → unhealthy

_DEFAULT_BATCH_MAX_SIZE = 8
_DEFAULT_BATCH_MAX_WAIT_MS = 5.0


# ---------------------------------------------------------------------------
# Input serialisation helper (parent side)
# ---------------------------------------------------------------------------

_NP_MAGIC = b"\x93NUMPY"  # first 6 bytes of every np.save() file


class _InferenceError:
    """Sentinel for failed inference — distinguishes from legitimate None result."""

    __slots__ = ("message",)

    def __init__(self, message: str) -> None:
        self.message = message


def _write_error_to_slot(pool: "SHMPool", slot_id: int, message: str) -> None:
    """Write error detail to DATA area before mark_error, so app.py can read it."""
    err_bytes = message.encode("utf-8", errors="replace")[:1024]  # cap at 1KB
    mv = pool.data_memoryview(slot_id)
    mv[: len(err_bytes)] = err_bytes
    mv.release()
    pool.mark_error(slot_id, error_code=1, error_size=len(err_bytes))


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
    model_id: str,
    api_key: str,
    setup_pipe: Any,
    zmq_addr: str,
    use_gpu: bool,
    gpu_device: Optional[str],
    shm_pool_name: str,
    n_slots: int,
    input_mb: float,
    batch_max_size: int,
    batch_max_wait_ms: float,
    use_nvjpeg: bool,
    model_kwargs: dict,
) -> None:
    """Worker subprocess entry point.

    Loads model, attaches SHMPool, signals READY, then greedy-batches
    T_SLOT_READY messages and sends T_RESULT per completed slot.
    """
    if os.environ.get("ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND") is None:
        os.environ["ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND"] = "False"

    import zmq  # noqa: PLC0415

    from inference_model_manager.backends.decode import (  # noqa: PLC0415
        make_batch_decoder,
    )
    from inference_models.models.auto_loaders.core import AutoModel  # noqa: PLC0415

    _log = logging.getLogger(f"{__name__}.worker")
    pool = sock = zmq_ctx = model = None

    try:
        device = (
            gpu_device if (use_gpu and gpu_device) else ("cuda:0" if use_gpu else "cpu")
        )
        _log.info("Worker(%s): loading on %s", model_id, device)
        if os.environ.get("DEBUG_BENCHMARK_MODE"):
            from inference_model_manager.backends.passthrough_model import (
                PassthroughModel,
            )

            model = PassthroughModel()
            _log.info("Worker(%s): using PassthroughModel (benchmark mode)", model_id)
        else:
            model = AutoModel.from_pretrained(
                model_id, api_key=api_key, device=device, **model_kwargs
            )
            _log.info("Worker(%s): model ready (%s)", model_id, type(model).__name__)

        batch_decode_fn = make_batch_decoder(device, use_nvjpeg=use_nvjpeg)
        pool = SHMPool.attach(shm_pool_name, n_slots=n_slots, input_mb=input_mb)

        from inference_model_manager.backends.base import detect_max_batch_size

        model_max_bs = detect_max_batch_size(model)
        effective_bs = model_max_bs if batch_max_size <= 0 else batch_max_size
        if model_max_bs is not None and effective_bs > model_max_bs:
            effective_bs = model_max_bs
        _log.info(
            "Worker(%s): batch_max_size=%s (model=%s, configured=%s)",
            model_id,
            effective_bs,
            model_max_bs,
            batch_max_size,
        )

        setup_pipe.send(
            {
                "status": "READY",
                "class_names": getattr(model, "class_names", None),
                "max_batch_size": model_max_bs,
                "model_class_name": type(model).__name__,
                "model_mro_names": [cls.__name__ for cls in type(model).__mro__],
            }
        )

        zmq_ctx = zmq.Context()
        sock = zmq_ctx.socket(zmq.PAIR)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(zmq_addr)

        _worker_loop(
            model,
            pool,
            sock,
            batch_decode_fn,
            effective_bs or _DEFAULT_BATCH_MAX_SIZE,
            batch_max_wait_ms,
            _log,
        )

    except KeyboardInterrupt:
        pass  # clean exit on Ctrl+C
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
    pool: SHMPool,
    sock,
    batch_decode_fn,
    batch_max_size: int,
    batch_max_wait_ms: float,
    log,
) -> None:
    """Greedy batch loop — accumulate T_SLOT_READY, fire on size-or-timeout."""
    import zmq  # noqa: PLC0415 — subprocess; already in sys.modules from _worker_main

    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)

    batch_max_wait_s = batch_max_wait_ms / 1000.0
    pending: list[tuple[int, int, bytes]] = []  # (slot_id, req_id, params_bytes)
    batch_start = 0.0
    last_heartbeat = time.monotonic()

    # Worker-side stats — updated by _process_slots, read by _MSG_STATS_REQ
    worker_stats: dict[str, Any] = {
        "inference_count": 0,
        "error_count": 0,
        "batch_count": 0,
        "latencies": deque(maxlen=1000),  # end-to-end per-batch (seconds)
        "batch_sizes": deque(maxlen=1000),
        "start_ts": time.monotonic(),
    }

    while True:
        # Compute poll timeout
        now = time.monotonic()
        if pending:
            wait_left = batch_start + batch_max_wait_s - now
            timeout_ms = max(0, int(wait_left * 1000))
        else:
            hb_due = last_heartbeat + _HEARTBEAT_INTERVAL_S
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
            elif msg == _MSG_SLOT_READY and len(frames) > 1 and len(frames[1]) >= 12:
                slot_id, req_id = struct.unpack(">IQ", frames[1][:12])
                params_bytes = frames[2] if len(frames) > 2 else b"{}"
                if not pending:
                    batch_start = time.monotonic()
                pending.append((slot_id, req_id, params_bytes))
            elif msg == _MSG_STATS_REQ:
                try:
                    sock.send_multipart(
                        [
                            _MSG_HEARTBEAT,
                            _build_worker_stats_payload(worker_stats),
                        ]
                    )
                except zmq.ZMQError:
                    break
        else:
            # Timeout while idle → heartbeat with stats
            if not pending:
                now = time.monotonic()
                if now - last_heartbeat >= _HEARTBEAT_INTERVAL_S:
                    try:
                        sock.send_multipart(
                            [
                                _MSG_HEARTBEAT,
                                _build_worker_stats_payload(worker_stats),
                            ]
                        )
                    except zmq.ZMQError:
                        break
                    last_heartbeat = now

        # Fire batch when ready
        now = time.monotonic()
        if pending and (
            len(pending) >= batch_max_size or (now - batch_start) >= batch_max_wait_s
        ):
            _process_slots(
                model, pool, pending, sock, batch_decode_fn, log, worker_stats
            )
            pending.clear()
            batch_start = 0.0


def _build_worker_stats_payload(worker_stats: dict) -> bytes:
    """Build JSON stats payload for heartbeat."""
    lats = list(worker_stats["latencies"])
    lats.sort()
    bs = list(worker_stats["batch_sizes"])
    uptime = time.monotonic() - worker_stats["start_ts"]
    infer_count = worker_stats["inference_count"]

    def _pct(p: float) -> float:
        if not lats:
            return 0.0
        idx = min(int(len(lats) * p / 100), len(lats) - 1)
        return lats[idx] * 1000

    return json.dumps(
        {
            "inference_count": infer_count,
            "error_count": worker_stats["error_count"],
            "batch_count": worker_stats["batch_count"],
            "throughput_fps": infer_count / max(uptime, 1e-6),
            "latency_p50_ms": _pct(50),
            "latency_p95_ms": _pct(95),
            "latency_p99_ms": _pct(99),
            "avg_batch_size": sum(bs) / len(bs) if bs else 0.0,
            "uptime_s": round(uptime, 1),
        }
    ).encode()


def _process_slots(
    model,
    pool: SHMPool,
    batch: list[tuple[int, int, bytes]],
    sock,
    batch_decode_fn,
    log,
    worker_stats: dict,
) -> None:
    """Process a batch of (slot_id, req_id, params_bytes), write results to SHM, send T_RESULT."""
    import zmq

    t0 = time.monotonic()

    # Parse per-slot params
    slot_params: list[dict] = []
    for _, _, params_bytes in batch:
        slot_params.append(json.loads(params_bytes) if params_bytes else {})

    # Gather memoryviews; classify as .npy (standalone numpy) vs raw bytes
    mvs: list[Any] = []
    is_npy: list[bool] = []
    for slot_id, _, _ in batch:
        hdr = pool.read_header(slot_id)
        mv = pool.data_memoryview(slot_id)[: hdr.input_size]
        mvs.append(mv)
        is_npy.append(bytes(mv[:6]) == _NP_MAGIC)

    images: list[Any] = [None] * len(batch)
    decode_errors: list[bool] = [False] * len(batch)

    # Guard: empty slots → mark as decode error immediately
    for i, mv in enumerate(mvs):
        if len(mv) == 0:
            log.error("Worker: slot %d has 0 bytes — skipping", batch[i][0])
            decode_errors[i] = True

    # .npy slots — standalone mode (numpy arrays serialised via np.save)
    for i, (mv, npy) in enumerate(zip(mvs, is_npy)):
        if npy:
            try:
                images[i] = np.load(io.BytesIO(bytes(mv)), allow_pickle=False)
            except Exception:
                log.exception("Worker: failed to load .npy slot %d", batch[i][0])
                decode_errors[i] = True

    # Raw image bytes (JPEG + non-JPEG) — single batch_decode_fn call
    raw_indices = [
        i for i, npy in enumerate(is_npy) if not npy and not decode_errors[i]
    ]
    if raw_indices:
        try:
            raw_decoded = batch_decode_fn([mvs[i] for i in raw_indices])
            for i, img in zip(raw_indices, raw_decoded):
                images[i] = img
        except Exception:
            log.exception(
                "Worker: batch decode failed for %d slot(s)", len(raw_indices)
            )
            for i in raw_indices:
                decode_errors[i] = True

    # Release memoryviews
    for mv in mvs:
        try:
            mv.release()
        except Exception:
            pass

    # Short-circuit slots that failed to decode — send error without calling infer
    error_indices = {i for i, err in enumerate(decode_errors) if err}
    if error_indices:
        for i in error_indices:
            slot_id, req_id, _ = batch[i]
            _write_error_to_slot(pool, slot_id, "image decode failed")
            try:
                sock.send_multipart(
                    [_MSG_RESULT, struct.pack(">QII", req_id, slot_id, 0)]
                )
            except zmq.ZMQError:
                log.warning("Worker: ZMQ send failed for error slot %d", slot_id)
        if len(error_indices) == len(batch):
            return
        # Filter to only successfully decoded slots for infer
        good = [
            (i, batch[i], images[i])
            for i in range(len(batch))
            if i not in error_indices
        ]
        good_batch = [b for _, b, _ in good]
        good_images = [img for _, _, img in good]
        good_params = [slot_params[i] for i, _, _ in good]
    else:
        good_batch = batch
        good_images = images
        good_params = slot_params

    # Group by (task, params) so that slots with different tasks/params are
    # processed in separate sub-batches instead of silently using only the
    # first slot's params for the entire batch.
    sub_batches: dict[str, list[int]] = {}
    for idx, p in enumerate(good_params):
        # Build a hashable key from the params dict (task + sorted remaining params)
        key = json.dumps(p, sort_keys=True)
        sub_batches.setdefault(key, []).append(idx)

    results: list[Any] = [None] * len(good_batch)
    for params_key, indices in sub_batches.items():
        sub_params = json.loads(params_key)
        task = sub_params.pop("task", None)
        sub_images = [good_images[i] for i in indices]
        try:
            images_arg = sub_images[0] if len(sub_images) == 1 else sub_images
            raw_out = invoke_task(model, task=task, images=images_arg, **sub_params)
            sub_results = raw_out if isinstance(raw_out, list) else [raw_out]
            for i, r in zip(indices, sub_results):
                results[i] = r
        except Exception as exc:
            log.exception("Worker: invoke_task(task=%r) failed", task)
            for i in indices:
                results[i] = _InferenceError(str(exc))

    n_ok = 0
    n_err = len(error_indices) if error_indices else 0

    for (slot_id, req_id, _), result in zip(good_batch, results):
        if result is None or isinstance(result, _InferenceError):
            err_msg = (
                result.message
                if isinstance(result, _InferenceError)
                else "inference returned None"
            )
            _write_error_to_slot(pool, slot_id, err_msg)
            try:
                sock.send_multipart(
                    [_MSG_RESULT, struct.pack(">QII", req_id, slot_id, 0)]
                )
            except zmq.ZMQError:
                return
            n_err += 1
            continue

        # Move tensors to CPU before pickle — result travels through SHM (CPU
        # memory), so serialising with device='cuda' just forces the receiver
        # to have a GPU for no reason.
        if hasattr(result, "xyxy"):
            result.xyxy = result.xyxy.cpu()
            result.confidence = result.confidence.cpu()
            result.class_id = result.class_id.cpu()
        data = pickle.dumps(result)
        mv = pool.data_memoryview(slot_id)
        if len(data) > len(mv):
            log.error(
                "Worker: result %d B exceeds slot capacity %d B for slot %d — marking error",
                len(data),
                len(mv),
                slot_id,
            )
            mv.release()
            _write_error_to_slot(
                pool,
                slot_id,
                f"result {len(data)}B exceeds slot capacity {len(mv)}B",
            )
            try:
                sock.send_multipart(
                    [_MSG_RESULT, struct.pack(">QII", req_id, slot_id, 0)]
                )
            except zmq.ZMQError:
                pass
            n_err += 1
            continue
        mv[: len(data)] = data
        mv.release()
        pool.mark_done(slot_id, len(data))
        try:
            sock.send_multipart(
                [_MSG_RESULT, struct.pack(">QII", req_id, slot_id, len(data))]
            )
        except zmq.ZMQError:
            return
        n_ok += 1

    # Update worker stats
    elapsed = time.monotonic() - t0
    worker_stats["inference_count"] += n_ok
    worker_stats["error_count"] += n_err
    worker_stats["batch_count"] += 1
    worker_stats["latencies"].append(elapsed)
    worker_stats["batch_sizes"].append(len(batch))


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
        # SHMPool (mandatory — pool is always created externally)
        shm_pool_name: str,
        n_slots: int,
        input_mb: float,
        # Worker batching
        batch_max_size: int = _DEFAULT_BATCH_MAX_SIZE,
        batch_max_delay_ms: float = _DEFAULT_BATCH_MAX_WAIT_MS,
        # Orchestrated-mode callbacks
        on_result_callback: Optional[Callable] = None,
        on_death_callback: Optional[Callable] = None,
        # Device
        device: Optional[str] = None,
        use_gpu: Optional[bool] = None,
        use_cuda_ipc: Optional[bool] = None,  # reserved, unused
        # Misc
        decoder: str = "imagecodecs",
        worker_start_timeout: float = 120.0,
        **kwargs,
    ) -> None:
        import zmq  # noqa: PLC0415

        self._model_id = model_id
        self._state_value: str = "loading"

        # ── Device resolution ────────────────────────────────────────
        if device is not None and device.startswith("cuda"):
            use_gpu = True
        if use_gpu is None:
            import torch  # noqa: PLC0415

            use_gpu = (device is not None and device.startswith("cuda")) or (
                device is None and torch.cuda.is_available()
            )
        self._use_gpu = use_gpu
        self._device_str = (
            (device if device and device.startswith("cuda") else "cuda:0")
            if self._use_gpu
            else "cpu"
        )

        # ── SHMPool (always attach — never create) ─────────────────
        self._pool = SHMPool.attach(shm_pool_name, n_slots=n_slots, input_mb=input_mb)

        logger.info(
            "SubprocessBackend(%s): device=%s pool=%s slots=%d "
            "input=%.0fMB batch=%d/%.0fms",
            model_id,
            self._device_str,
            f"attached:{shm_pool_name[:8]}…",
            n_slots,
            input_mb,
            batch_max_size,
            batch_max_delay_ms,
        )

        # ── ZMQ PAIR ─────────────────────────────────────────────────
        self._zmq_ctx = zmq.Context()
        self._zmq_sock = self._zmq_ctx.socket(zmq.PAIR)
        self._zmq_sock.setsockopt(zmq.LINGER, 0)

        _transport = os.environ.get("INFERENCE_ZMQ_TRANSPORT", default_transport())
        _sock_id = f"sp_{os.getpid()}_{uuid.uuid4().hex[:8]}"
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
                batch_max_size=batch_max_size,
                batch_max_wait_ms=batch_max_delay_ms,
                use_nvjpeg=(decoder == "nvjpeg"),
                model_kwargs=kwargs,
            ),
            daemon=True,
        )
        self._worker.start()

        if not parent_pipe.poll(timeout=worker_start_timeout):
            self._worker.kill()
            self._worker.join(timeout=5)
            raise RuntimeError(
                f"SubprocessBackend({model_id!r}): worker timeout "
                f"after {worker_start_timeout}s"
            )

        msg = parent_pipe.recv()
        if not isinstance(msg, dict) or not msg.get("status", "").startswith("READY"):
            err = msg if isinstance(msg, str) else msg.get("status", str(msg))
            self._worker.join(timeout=10)
            raise RuntimeError(f"SubprocessBackend({model_id!r}): {err}")

        self._class_names = msg.get("class_names")
        self._max_batch_size_model = msg.get("max_batch_size")
        self._model_class_name = msg.get("model_class_name")
        self._model_mro_names = msg.get("model_mro_names", [])
        self._last_worker_activity = time.monotonic()

        logger.info(
            "SubprocessBackend(%s): worker ready (pid=%d, device=%s)",
            model_id,
            self._worker.pid,
            self._device_str,
        )

        # ── Recv thread ───────────────────────────────────────────────
        # All ZMQ socket I/O runs here.  Other threads communicate via _outbound.
        self._outbound: queue.Queue = queue.Queue()
        self._slot_futures: Dict[int, tuple] = {}  # slot_id → (req_id, Future)
        self._slot_lock = threading.Lock()
        self._on_result_callback: Optional[Callable] = on_result_callback
        self._on_death_callback: Optional[Callable] = on_death_callback
        self._recv_running = True
        self._recv_dead = False
        self._worker_stats: dict[str, Any] = {}  # latest snapshot from worker
        self._worker_stats_event: Optional[threading.Event] = None
        self._recv_thread = threading.Thread(
            target=self._recv_loop,
            daemon=True,
            name=f"subproc-recv-{model_id[:20]}",
        )
        self._recv_thread.start()

        self._state_value = "loaded"

    # ------------------------------------------------------------------
    # Orchestrated-mode API (called by MMP)
    # ------------------------------------------------------------------

    def signal_slot(
        self, slot_id: int, req_id: int, params_bytes: bytes = b"{}"
    ) -> None:
        """Enqueue T_SLOT_READY for the recv thread to send. Thread-safe."""
        if self._recv_dead:
            raise RuntimeError(
                f"SubprocessBackend({self._model_id!r}): recv thread is dead, "
                "cannot enqueue work"
            )
        self._outbound.put((slot_id, req_id, params_bytes))
        self._last_worker_activity = time.monotonic()

    def set_on_result_callback(self, callback: Callable[[int, int, int], None]) -> None:
        """Set MMP callback: called on each T_RESULT from worker."""
        self._on_result_callback = callback

    def set_on_death_callback(self, callback: Callable[[str], None]) -> None:
        """Set MMP callback: called with model_id when worker dies."""
        self._on_death_callback = callback

    def refresh_worker_stats(self, timeout_s: float = 1.0) -> dict[str, Any]:
        """Request fresh stats from worker. Blocks up to timeout_s.

        Returns latest worker stats dict (may be stale if worker is busy).
        Thread-safe — sends via outbound queue, recv thread handles response.
        """
        if self._recv_dead:
            return self._worker_stats
        evt = threading.Event()
        self._worker_stats_event = evt
        # Enqueue stats request — recv thread sends _MSG_STATS_REQ
        self._outbound.put(_STATS_SENTINEL)
        evt.wait(timeout=timeout_s)
        self._worker_stats_event = None
        return self._worker_stats

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
                if item is None:  # shutdown sentinel
                    try:
                        self._zmq_sock.send_multipart([_MSG_SHUTDOWN, b""])
                    except Exception:
                        pass
                    return
                if item is _STATS_SENTINEL:
                    try:
                        self._zmq_sock.send_multipart([_MSG_STATS_REQ, b""])
                    except zmq.ZMQError:
                        pass
                    continue
                slot_id, req_id, params_bytes = item
                try:
                    self._zmq_sock.send_multipart(
                        [
                            _MSG_SLOT_READY,
                            struct.pack(">IQ", slot_id, req_id),
                            params_bytes,
                        ]
                    )
                except zmq.ZMQError:
                    return

            # Poll with 10ms timeout so outbound queue drains promptly
            events = dict(poller.poll(timeout=10))
            if self._zmq_sock not in events:
                if not self._worker.is_alive():
                    self._handle_worker_death()
                    return
                continue

            try:
                frames = self._zmq_sock.recv_multipart()
            except zmq.ZMQError:
                logger.error(
                    "SubprocessBackend(%s): recv thread exiting due to ZMQError",
                    self._model_id,
                )
                self._recv_dead = True
                return

            self._last_worker_activity = time.monotonic()
            msg = frames[0]

            if msg == _MSG_HEARTBEAT:
                if len(frames) > 1 and frames[1]:
                    try:
                        self._worker_stats = json.loads(frames[1])
                    except Exception:
                        pass
                    evt = self._worker_stats_event
                    if evt is not None:
                        evt.set()
            elif msg == _MSG_RESULT and len(frames) > 1 and len(frames[1]) == 16:
                req_id, slot_id, result_sz = struct.unpack(">QII", frames[1])
                self._handle_result(req_id, slot_id, result_sz)

    def _handle_worker_death(self) -> None:
        """Called from recv thread when worker process exits unexpectedly."""
        logger.error(
            "SubprocessBackend(%s): worker died (pid=%s exitcode=%s)",
            self._model_id,
            self._worker.pid,
            self._worker.exitcode,
        )
        self._state_value = "unhealthy"
        self._recv_dead = True

        with self._slot_lock:
            pending = list(self._slot_futures.items())
            self._slot_futures.clear()

        for slot_id, (req_id, future) in pending:
            # Reject pending futures
            if future is not None and not future.done():
                future.set_exception(
                    RuntimeError(f"SubprocessBackend({self._model_id!r}): worker died")
                )
            # Notify owner (MMP or ModelManager) — result_sz=0 → error
            if self._on_result_callback is not None:
                try:
                    self._on_result_callback(req_id, slot_id, 0)
                except Exception:
                    logger.exception(
                        "SubprocessBackend(%s): on_result_callback raised "
                        "during worker-death cleanup",
                        self._model_id,
                    )

        # Notify MMP to trigger reload
        if self._on_death_callback is not None:
            try:
                self._on_death_callback(self._model_id)
            except Exception:
                logger.exception(
                    "SubprocessBackend(%s): on_death_callback raised",
                    self._model_id,
                )

    def _handle_result(self, req_id: int, slot_id: int, result_sz: int) -> None:
        """Called from recv thread on each T_RESULT."""
        # Resolve Future if one exists (standalone submit path)
        with self._slot_lock:
            entry = self._slot_futures.pop(slot_id, None)
        if entry is not None:
            _, future = entry
            if future is not None and not future.done():
                if result_sz > 0:
                    try:
                        data = bytes(self._pool.data_memoryview(slot_id)[:result_sz])
                        result = pickle.loads(data)
                        future.set_result(result)
                    except Exception as exc:
                        future.set_exception(exc)
                else:
                    future.set_exception(RuntimeError("worker inference error"))

        # Notify owner (MMP or ModelManager)
        if self._on_result_callback is not None:
            try:
                self._on_result_callback(req_id, slot_id, result_sz)
            except Exception:
                logger.exception("SubprocessBackend: on_result_callback raised")

    # ------------------------------------------------------------------
    # Standalone inference
    # ------------------------------------------------------------------

    def submit_slot(
        self,
        slot_id: int,
        req_id: int,
        future: Optional[Future] = None,
        params_bytes: bytes = b"{}",
    ) -> None:
        """Register a future for this slot and signal the worker.

        Called by ModelManager after it writes input to the pool.
        If ``future`` is provided, it will be resolved when the worker
        sends T_RESULT for this slot.
        """
        with self._slot_lock:
            self._slot_futures[slot_id] = (req_id, future)

        self.signal_slot(slot_id, req_id, params_bytes)

    def infer_sync(self, raw_input: Any, **kwargs) -> Any:
        raise RuntimeError(
            "SubprocessBackend.infer_sync() is not available — "
            "use ModelManager.infer_sync() which handles pool allocation"
        )

    async def infer_async(self, raw_input: Any, **kwargs) -> Any:
        raise RuntimeError(
            "SubprocessBackend.infer_async() is not available — "
            "use ModelManager.infer_async() which handles pool allocation"
        )

    def submit(self, raw_input: Any, *, priority: int = 0, **kwargs) -> Future:
        raise RuntimeError(
            "SubprocessBackend.submit() is not available — "
            "use ModelManager.submit() which handles pool allocation"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def drain_and_unload(self, timeout_s: float = 30.0) -> None:
        """Stop accepting new work, wait for in-flight to finish, then unload."""
        self._state_value = "draining"
        logger.info(
            "SubprocessBackend(%s): draining (timeout=%.1fs)", self._model_id, timeout_s
        )

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._slot_lock:
                if not self._slot_futures:
                    break
                n = len(self._slot_futures)
            logger.debug(
                "SubprocessBackend(%s): %d slots still in-flight", self._model_id, n
            )
            time.sleep(0.1)
        else:
            with self._slot_lock:
                n = len(self._slot_futures)
            if n > 0:
                logger.warning(
                    "SubprocessBackend(%s): drain timeout — %d slots still in-flight, force-unloading",
                    self._model_id,
                    n,
                )

        self.unload()

    def unload(self) -> None:
        self._state_value = "unhealthy"  # block new submits immediately

        # Signal recv thread: send T_SHUTDOWN to worker, then exit
        self._recv_running = False
        self._outbound.put(None)  # None = shutdown sentinel
        self._recv_thread.join(timeout=5.0)

        # Kill worker if still alive
        if self._worker.is_alive():
            self._worker.join(timeout=5)
        if self._worker.is_alive():
            self._worker.kill()

        # Cancel pending futures
        with self._slot_lock:
            for slot_id, (_, future) in self._slot_futures.items():
                if future is not None and not future.done():
                    future.set_exception(RuntimeError("backend unloaded"))
            self._slot_futures.clear()

        self._zmq_sock.close(linger=0)
        self._zmq_ctx.term()

        # Pool is externally owned — never close/unlink here

        if self._zmq_addr.startswith("ipc://"):
            try:
                os.unlink(self._zmq_addr[len("ipc://") :])
            except OSError:
                pass

        logger.info("SubprocessBackend(%s): unloaded", self._model_id)

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
        ws = self._worker_stats
        return {
            "model_id": self._model_id,
            "backend_type": "subprocess",
            "device": self._device_str,
            "transport": "shm_pool",
            "state": self.state,
            "is_accepting": self.is_accepting,
            "queue_depth": self.queue_depth,
            "max_batch_size": self.max_batch_size,
            "throughput_fps": ws.get("throughput_fps", 0.0),
            "latency_p50_ms": ws.get("latency_p50_ms", 0.0),
            "latency_p95_ms": ws.get("latency_p95_ms", 0.0),
            "latency_p99_ms": ws.get("latency_p99_ms", 0.0),
            "inference_count": ws.get("inference_count", 0),
            "error_count": ws.get("error_count", 0),
            "batch_count": ws.get("batch_count", 0),
            "avg_batch_size": ws.get("avg_batch_size", 0.0),
            "worker_uptime_s": ws.get("uptime_s", 0.0),
            "worker_alive": self._worker.is_alive(),
            "shm_pool_name": self._pool.name,
            "model_class_name": self._model_class_name,
        }

    @property
    def worker_pid(self) -> Optional[int]:
        return self._worker.pid if self._worker.is_alive() else None

    @property
    def class_names(self) -> Optional[List[str]]:
        return self._class_names
