"""Shared-base subprocess backend: one worker hosts a base model + many heads.

`SharedHeadControlPlane` owns the parent-side head lifecycle — load_head/drop_head
request/ack correlation over the worker control channel, and the per-head metadata
registry. Transport is injected (`send`) so the correlation, timeout, and
worker-death paths are testable without a live socket.

`SharedHeadBackend` is a Backend view: it implements the Backend interface by
delegating worker liveness to the owner and serving per-head metadata. Many views
share one owner; a view never touches the worker process.
"""

import logging
import os
import pickle
import queue
import struct
import threading
import time
import uuid
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import zmq

from inference_model_manager import configuration as cfg
from inference_model_manager.backends.base import Backend
from inference_model_manager.backends.shared_base_protocol import (
    MSG_DROP_HEAD,
    MSG_DROP_HEAD_ACK,
    MSG_HEAD_SLOT_READY,
    MSG_LOAD_HEAD,
    MSG_LOAD_HEAD_ACK,
    decode_control,
    encode_control,
    pack_head_slot,
)
from inference_model_manager.backends.subproc import (
    _MSG_HEARTBEAT,
    _MSG_RESULT,
    _MSG_SHUTDOWN,
    _MSG_STATS_REQ,
)
from inference_model_manager.backends.utils.shm_pool import SHMPool
from inference_model_manager.backends.utils.transport import default_transport

logger = logging.getLogger(__name__)

_DEFAULT_CONTROL_TIMEOUT_S = 120.0
_SEND_TIMEOUT_MS = 5000


@dataclass(frozen=True)
class HeadMetadata:
    head_index: int
    model_mro_names: List[str] = field(default_factory=list)
    max_batch_size: Optional[int] = None
    class_names: Optional[List[str]] = None


class SharedHeadControlPlane:
    """Parent-side head load/drop over the worker control channel.

    `send(tag, payload)` ships one control frame; it must raise if the worker is
    gone. Acks arrive out-of-band (recv thread) via `on_ack`; `fail_all` is called
    on worker death so no `load_head`/`drop_head` caller hangs.
    """

    def __init__(self, send: Callable[[bytes, bytes], None]) -> None:
        self._send = send
        self._lock = threading.Lock()
        self._pending: Dict[int, Future] = {}
        self._heads: Dict[str, HeadMetadata] = {}
        self._req_counter = 0
        self._dead = False

    # ---- ack / death plumbing (called from the owner recv thread) ----

    def on_ack(self, payload: Dict[str, Any]) -> None:
        req_id = payload.get("req_id")
        with self._lock:
            future = self._pending.pop(req_id, None)
        if future is not None and not future.done():
            future.set_result(payload)

    def fail_all(self, exc: BaseException) -> None:
        with self._lock:
            self._dead = True
            pending = list(self._pending.values())
            self._pending.clear()
        for future in pending:
            if not future.done():
                future.set_exception(exc)

    # ---- request/ack round-trip ----

    def _round_trip(
        self, tag: bytes, req_id: int, fields: Dict[str, Any], timeout_s: float
    ) -> Dict[str, Any]:
        future: Future = Future()
        with self._lock:
            if self._dead:
                raise RuntimeError("shared-base worker is dead")
            self._pending[req_id] = future
        try:
            self._send(tag, encode_control(req_id, **fields))
        except Exception:
            with self._lock:
                self._pending.pop(req_id, None)
            raise
        try:
            return future.result(timeout=timeout_s)
        except FutureTimeoutError:
            with self._lock:
                self._pending.pop(req_id, None)
            raise TimeoutError(f"shared-base control timed out (req_id={req_id})")

    def _next_req_id(self) -> int:
        self._req_counter += 1
        return self._req_counter

    def load_head(
        self,
        head_id: str,
        fields: Dict[str, Any],
        timeout_s: float = _DEFAULT_CONTROL_TIMEOUT_S,
    ) -> HeadMetadata:
        with self._lock:
            if head_id in self._heads:
                raise ValueError(f"head '{head_id}' already loaded")
            req_id = self._next_req_id()
        ack = self._round_trip(
            MSG_LOAD_HEAD, req_id, {"head_id": head_id, **fields}, timeout_s
        )
        if not ack.get("ok"):
            raise RuntimeError(ack.get("error", f"load_head failed for '{head_id}'"))
        metadata = HeadMetadata(
            head_index=ack["head_index"],
            model_mro_names=ack.get("model_mro_names", []),
            max_batch_size=ack.get("max_batch_size"),
            class_names=ack.get("class_names"),
        )
        with self._lock:
            self._heads[head_id] = metadata
        return metadata

    def drop_head(
        self, head_id: str, timeout_s: float = _DEFAULT_CONTROL_TIMEOUT_S
    ) -> None:
        with self._lock:
            if head_id not in self._heads:
                return
            req_id = self._next_req_id()
        ack = self._round_trip(MSG_DROP_HEAD, req_id, {"head_id": head_id}, timeout_s)
        if not ack.get("ok"):
            raise RuntimeError(ack.get("error", f"drop_head failed for '{head_id}'"))
        # Only forget the head once the worker has confirmed removal.
        with self._lock:
            self._heads.pop(head_id, None)

    # ---- read-only views ----

    def has_head(self, head_id: str) -> bool:
        with self._lock:
            return head_id in self._heads

    def metadata(self, head_id: str) -> Optional[HeadMetadata]:
        with self._lock:
            return self._heads.get(head_id)

    def head_count(self) -> int:
        with self._lock:
            return len(self._heads)


class _HeadSlotTracker:
    """Per-head in-flight slot accounting: maps each signalled slot to its head so a
    result decrements the right head, and drain waits only on one head's slots."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._slot_head: Dict[int, str] = {}
        self._outstanding: Dict[str, int] = {}

    def signal(self, head_id: str, slot_id: int) -> None:
        with self._lock:
            self._slot_head[slot_id] = head_id
            self._outstanding[head_id] = self._outstanding.get(head_id, 0) + 1

    def complete(self, slot_id: int) -> Optional[str]:
        with self._lock:
            head_id = self._slot_head.pop(slot_id, None)
            if head_id is not None and self._outstanding.get(head_id):
                self._outstanding[head_id] -= 1
            return head_id

    def outstanding(self, head_id: str) -> int:
        with self._lock:
            return self._outstanding.get(head_id, 0)

    def total(self) -> int:
        with self._lock:
            return sum(self._outstanding.values())

    def forget(self, head_id: str) -> None:
        with self._lock:
            self._outstanding.pop(head_id, None)
            for slot_id, hid in list(self._slot_head.items()):
                if hid == head_id:
                    del self._slot_head[slot_id]


class SharedHeadBackend(Backend):
    """Backend view onto one head hosted by a shared-base owner worker.

    Worker liveness (device/state/health) delegates to the owner; per-head metadata
    is fixed at load time. The view never kills or drains the worker process —
    `unload` drops just this head; the owner tears the worker down when its last
    head is gone.
    """

    def __init__(self, owner: Any, head_id: str, metadata: HeadMetadata) -> None:
        self._owner = owner
        self._model_id = head_id
        self._metadata = metadata
        # Attribute (not just property) so registry lazy-registration can read it.
        self._model_mro_names = metadata.model_mro_names

    # ---- lifecycle ----

    def unload(self) -> None:
        self._owner.drop_head(self._model_id)

    def drain_and_unload(self, timeout_s: float = 30.0) -> None:
        self._owner.drain_head(self._model_id, timeout_s=timeout_s)
        self._owner.drop_head(self._model_id)

    # ---- inference routing ----

    def signal_slot(
        self, slot_id: int, req_id: int, params_bytes: bytes = b"{}"
    ) -> None:
        self._owner.signal_slot(self._model_id, slot_id, req_id, params_bytes)

    def submit_slot(
        self,
        slot_id: int,
        req_id: int,
        future: Optional[Any] = None,
        params_bytes: bytes = b"{}",
    ) -> None:
        self._owner.submit_slot(self._model_id, slot_id, req_id, future, params_bytes)

    def set_on_result_callback(self, callback: Callable[[int, int, int], None]) -> None:
        # All views over one owner route to the same (req_id-keyed) result callback.
        self._owner.set_on_result_callback(callback)

    # ---- observability (worker liveness via owner; metadata per head) ----

    @property
    def device(self) -> str:
        return self._owner.device

    @property
    def state(self) -> str:
        return self._owner.state

    @property
    def is_healthy(self) -> bool:
        return self._owner.is_healthy and self._owner.has_head(self._model_id)

    @property
    def is_accepting(self) -> bool:
        return self._owner.is_accepting and self._owner.has_head(self._model_id)

    @property
    def max_batch_size(self) -> Optional[int]:
        return self._metadata.max_batch_size

    @property
    def class_names(self) -> Optional[List[str]]:
        return self._metadata.class_names

    @property
    def queue_depth(self) -> int:
        return self._owner.head_queue_depth(self._model_id)

    @property
    def worker_pid(self) -> Optional[int]:
        # Head views never own pid attribution — the owner entry counts base VRAM.
        return None

    def refresh_worker_stats(self, timeout_s: float = 1.0) -> Dict[str, Any]:
        return self._owner.refresh_worker_stats(timeout_s=timeout_s)

    def stats(self) -> Dict[str, Any]:
        ws = self._owner.worker_stats()
        return {
            "model_id": self._model_id,
            "backend_type": "shared-head",
            "device": self._owner.device,
            "state": self.state,
            "is_accepting": self.is_accepting,
            "queue_depth": self.queue_depth,
            "max_batch_size": self._metadata.max_batch_size,
            "worker_pid": None,
            "throughput_fps": ws.get("throughput_fps", 0.0),
            "latency_p50_ms": ws.get("latency_p50_ms", 0.0),
            "latency_p99_ms": ws.get("latency_p99_ms", 0.0),
            "inference_count": ws.get("inference_count", 0),
            "error_count": ws.get("error_count", 0),
        }


class SharedBaseSubprocessBackend:
    """Owner of one shared-base worker: a resident base model + N heads.

    Mirrors SubprocessBackend transport (recv thread sole socket owner, outbound
    queue for sends) but multiplexes heads. The base loads eagerly in __init__ on the
    head device. Not a Backend in ModelManager._backends — it lives in
    MMP._shared_workers[base_key], referenced by its SharedHeadBackend views.
    """

    def __init__(
        self,
        base_key: str,
        resolution: Any,
        api_key: str,
        *,
        shm_pool_name: str,
        n_slots: int,
        input_mb: float,
        device: Optional[str] = None,
        batch_max_size: int = 0,
        batch_max_delay_ms: float = 0.0,
        decoder: str = "imagecodecs",
        on_shared_worker_death: Optional[Callable[[str], None]] = None,
        on_empty: Optional[Callable[[str, "SharedBaseSubprocessBackend"], None]] = None,
        on_result_callback: Optional[Callable[[int, int, int], None]] = None,
        worker_start_timeout: float = cfg.INFERENCE_WORKER_START_TIMEOUT_S,
        base_model_kwargs: Optional[dict] = None,
    ) -> None:
        self._base_key = base_key
        self._resolution = resolution
        self._state_value = "loading"
        self._on_shared_worker_death = on_shared_worker_death
        self._on_empty = on_empty
        self._on_result_callback = on_result_callback
        self._death_handled = False
        self._retired = False

        # Treat "" as unset (CUDA-if-available), matching `device or None` callers.
        device = device or None
        use_gpu = device is not None and device.startswith("cuda")
        if device is None:
            import torch  # noqa: PLC0415

            use_gpu = torch.cuda.is_available()
        self._device_str = (
            (device if device and device.startswith("cuda") else "cuda:0")
            if use_gpu
            else "cpu"
        )

        # Defaults so the failure-cleanup helper can run after a partial __init__.
        self._pool = None
        self._zmq_ctx = None
        self._zmq_sock = None
        self._zmq_addr = ""
        self._worker = None
        try:
            self._pool = SHMPool.attach(
                shm_pool_name, n_slots=n_slots, input_mb=input_mb
            )

            self._zmq_ctx = zmq.Context()
            self._zmq_sock = self._zmq_ctx.socket(zmq.PAIR)
            self._zmq_sock.setsockopt(zmq.LINGER, 0)
            self._zmq_sock.setsockopt(zmq.SNDTIMEO, _SEND_TIMEOUT_MS)
            _transport = os.environ.get(
                cfg.INFERENCE_ZMQ_TRANSPORT_ENV, default_transport()
            )
            _sock_id = f"shb_{os.getpid()}_{uuid.uuid4().hex[:8]}"
            if _transport == "ipc":
                self._zmq_addr = f"ipc:///tmp/inference_{_sock_id}.ipc"
            else:
                self._zmq_addr = "tcp://127.0.0.1:*"
            self._zmq_sock.bind(self._zmq_addr)
            if _transport != "ipc":
                self._zmq_addr = self._zmq_sock.getsockopt_string(zmq.LAST_ENDPOINT)

            import multiprocessing as mp  # noqa: PLC0415

            from inference_model_manager.backends.shared_base_worker import (
                _shared_worker_main,
            )

            ctx = mp.get_context("spawn")
            parent_pipe, child_pipe = ctx.Pipe()
            self._worker = ctx.Process(
                target=_shared_worker_main,
                kwargs=dict(
                    base_model_id=resolution.dep_model_id,
                    base_resolved_package_id=resolution.resolved_package_id,
                    api_key=api_key,
                    setup_pipe=child_pipe,
                    zmq_addr=self._zmq_addr,
                    device=self._device_str,
                    shm_pool_name=self._pool.name,
                    n_slots=n_slots,
                    input_mb=input_mb,
                    batch_max_size=batch_max_size,
                    batch_max_wait_ms=batch_max_delay_ms,
                    decoder=decoder,
                    dep_name=resolution.dep_name,
                    dep_model_id=resolution.dep_model_id,
                    dep_metadata_package_id=resolution.dep_metadata_package_id,
                    base_model_kwargs=base_model_kwargs or {},
                ),
                daemon=True,
            )
            self._worker.start()

            if not parent_pipe.poll(timeout=worker_start_timeout):
                raise RuntimeError(f"SharedBase({base_key!r}): worker timeout")
            msg = parent_pipe.recv()
            if not isinstance(msg, dict) or not msg.get("status", "").startswith(
                "READY"
            ):
                err = msg if isinstance(msg, str) else msg.get("status", str(msg))
                raise RuntimeError(f"SharedBase({base_key!r}): {err}")
        except BaseException:
            self._kill_worker()
            self._close_transport()
            raise

        self._last_worker_activity = time.monotonic()
        self._worker_stats: Dict[str, Any] = {}
        self._worker_stats_event: Optional[threading.Event] = None
        self._tracker = _HeadSlotTracker()
        self._slot_lock = threading.Lock()
        self._slot_meta: Dict[int, tuple] = {}  # slot_id → (req_id, future, head_id)
        self._load_lock = threading.Lock()
        self._inflight_loads = 0
        self._control = SharedHeadControlPlane(send=self._send_control)
        self._outbound: queue.Queue = queue.Queue()
        self._recv_running = True
        self._recv_dead = False
        self._recv_thread = threading.Thread(
            target=self._recv_loop, daemon=True, name=f"shb-recv-{base_key[:12]}"
        )
        self._recv_thread.start()
        self._state_value = "loaded"

    # ---- head lifecycle (called by ModelManager.load_shared_head / views) ----

    def begin_load(self) -> bool:
        """Reserve an in-flight load. Returns False if the owner is retired OR its
        worker has died (death sets _recv_dead before the cache entry is popped, so a
        reservation here would otherwise reserve a dead worker) — the caller must
        obtain a fresh owner. Held until end_load() so the worker is never reaped
        between obtaining it and calling load_head()."""
        with self._load_lock:
            if self._retired or self._recv_dead:
                return False
            self._inflight_loads += 1
            return True

    def end_load(self) -> None:
        with self._load_lock:
            self._inflight_loads -= 1
        # A failed-and-only head leaves the worker empty; reap it so the cache never
        # hands back a base with no heads. A concurrent reservation keeps it alive.
        self._maybe_retire()

    def load_head(self, head_id: str, api_key: str) -> HeadMetadata:
        return self._control.load_head(
            head_id, {"api_key": api_key, "device": self._device_str}
        )

    def drop_head(self, head_id: str) -> None:
        self._control.drop_head(head_id)
        self._tracker.forget(head_id)
        self._maybe_retire()

    def _maybe_retire(self) -> None:
        with self._load_lock:
            if self._retired or self._inflight_loads > 0 or self._control.head_count() > 0:
                return
            self._retired = True
        self.unload()
        if self._on_empty is not None:
            try:
                self._on_empty(self._base_key, self)
            except Exception:
                logger.exception("SharedBase: on_empty raised")

    def drain_head(self, head_id: str, timeout_s: float = 30.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._tracker.outstanding(head_id) == 0:
                return
            time.sleep(0.05)
        logger.warning(
            "SharedBase(%s): drain timeout for head %s — %d slots in-flight",
            self._base_key,
            head_id,
            self._tracker.outstanding(head_id),
        )

    def has_head(self, head_id: str) -> bool:
        return self._control.has_head(head_id)

    def head_metadata(self, head_id: str) -> Optional[HeadMetadata]:
        return self._control.metadata(head_id)

    def head_count(self) -> int:
        return self._control.head_count()

    def head_queue_depth(self, head_id: str) -> int:
        return self._tracker.outstanding(head_id)

    # ---- inference routing ----

    def signal_slot(
        self, head_id: str, slot_id: int, req_id: int, params_bytes: bytes = b"{}"
    ) -> None:
        self._enqueue_slot(head_id, slot_id, req_id, None, params_bytes)

    def submit_slot(
        self,
        head_id: str,
        slot_id: int,
        req_id: int,
        future: Optional[Future] = None,
        params_bytes: bytes = b"{}",
    ) -> None:
        self._enqueue_slot(head_id, slot_id, req_id, future, params_bytes)

    def _enqueue_slot(
        self,
        head_id: str,
        slot_id: int,
        req_id: int,
        future: Optional[Future],
        params_bytes: bytes,
    ) -> None:
        if self._recv_dead:
            raise RuntimeError(f"SharedBase({self._base_key!r}): worker dead")
        metadata = self._control.metadata(head_id)
        if metadata is None:
            raise RuntimeError(f"SharedBase: head '{head_id}' not loaded")
        with self._slot_lock:
            self._slot_meta[slot_id] = (req_id, future, head_id)
        self._tracker.signal(head_id, slot_id)
        self._outbound.put(("slot", metadata.head_index, slot_id, req_id, params_bytes))

    def set_on_result_callback(self, callback: Callable[[int, int, int], None]) -> None:
        self._on_result_callback = callback

    def worker_stats(self) -> Dict[str, Any]:
        return self._worker_stats

    def refresh_worker_stats(self, timeout_s: float = 1.0) -> Dict[str, Any]:
        if self._recv_dead:
            return self._worker_stats
        evt = threading.Event()
        self._worker_stats_event = evt
        self._outbound.put(("control", _MSG_STATS_REQ, b""))
        evt.wait(timeout=timeout_s)
        self._worker_stats_event = None
        return self._worker_stats

    # ---- transport ----

    def _send_control(self, tag: bytes, payload: bytes) -> None:
        if self._recv_dead:
            raise RuntimeError(f"SharedBase({self._base_key!r}): worker dead")
        self._outbound.put(("control", tag, payload))

    def _recv_loop(self) -> None:
        poller = zmq.Poller()
        poller.register(self._zmq_sock, zmq.POLLIN)
        while self._recv_running:
            while True:
                try:
                    item = self._outbound.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    try:
                        self._zmq_sock.send_multipart([_MSG_SHUTDOWN, b""])
                    except Exception:
                        pass
                    return
                try:
                    if item[0] == "control":
                        self._zmq_sock.send_multipart([item[1], item[2]])
                    else:
                        _, head_index, slot_id, req_id, params_bytes = item
                        self._zmq_sock.send_multipart(
                            [
                                MSG_HEAD_SLOT_READY,
                                pack_head_slot(slot_id, req_id, head_index),
                                params_bytes,
                            ]
                        )
                except zmq.ZMQError:
                    self._handle_worker_death("send failed")
                    return

            events = dict(poller.poll(timeout=10))
            if self._zmq_sock not in events:
                if not self._worker.is_alive():
                    self._handle_worker_death()
                    return
                continue
            try:
                frames = self._zmq_sock.recv_multipart()
            except zmq.ZMQError:
                self._handle_worker_death("recv failed")
                return
            self._last_worker_activity = time.monotonic()
            msg = frames[0]
            if msg in (MSG_LOAD_HEAD_ACK, MSG_DROP_HEAD_ACK):
                try:
                    self._control.on_ack(decode_control(frames[1]))
                except Exception:
                    logger.exception("SharedBase: bad control ack")
            elif msg == _MSG_HEARTBEAT:
                if len(frames) > 1 and frames[1]:
                    try:
                        import json  # noqa: PLC0415

                        self._worker_stats = json.loads(frames[1])
                    except Exception:
                        pass
                    evt = self._worker_stats_event
                    if evt is not None:
                        evt.set()
            elif msg == _MSG_RESULT and len(frames) > 1 and len(frames[1]) == 16:
                req_id, slot_id, result_sz = struct.unpack(">QII", frames[1])
                self._handle_result(req_id, slot_id, result_sz)

    def _handle_result(self, req_id: int, slot_id: int, result_sz: int) -> None:
        self._tracker.complete(slot_id)
        with self._slot_lock:
            entry = self._slot_meta.pop(slot_id, None)
        if entry is not None:
            _, future, _ = entry
            if future is not None and not future.done():
                if result_sz > 0:
                    try:
                        data = bytes(self._pool.data_memoryview(slot_id)[:result_sz])
                        future.set_result(pickle.loads(data))
                    except Exception as exc:  # noqa: BLE001
                        future.set_exception(exc)
                else:
                    future.set_exception(RuntimeError("worker inference error"))
        if self._on_result_callback is not None:
            try:
                self._on_result_callback(req_id, slot_id, result_sz)
            except Exception:
                logger.exception("SharedBase: on_result_callback raised")

    def _handle_worker_death(self, reason: str = "worker died") -> None:
        if self._death_handled:
            return
        self._death_handled = True
        self._state_value = "unhealthy"
        self._recv_dead = True
        logger.error("SharedBase(%s): %s", self._base_key, reason)
        # Fail in-flight head loads/drops so no caller hangs.
        self._control.fail_all(RuntimeError(f"SharedBase({self._base_key!r}): {reason}"))
        # Complete every tracked slot as an error so the owner (MMP/ModelManager)
        # clears its pending/inflight state instead of waiting forever.
        with self._slot_lock:
            slots = list(self._slot_meta.items())
            self._slot_meta.clear()
        for slot_id, (req_id, future, _) in slots:
            # Clear per-head outstanding too, else queue_depth/drain stay stuck.
            self._tracker.complete(slot_id)
            if future is not None and not future.done():
                future.set_exception(
                    RuntimeError(f"SharedBase({self._base_key!r}): worker died")
                )
            if self._on_result_callback is not None:
                try:
                    self._on_result_callback(req_id, slot_id, 0)
                except Exception:
                    logger.exception("SharedBase: on_result_callback raised on death")
        if self._on_shared_worker_death is not None:
            try:
                self._on_shared_worker_death(self._base_key)
            except Exception:
                logger.exception("SharedBase: on_shared_worker_death raised")

    # ---- teardown ----

    def _kill_worker(self) -> None:
        worker = getattr(self, "_worker", None)
        if worker is None:
            return
        try:
            if worker.is_alive():
                worker.kill()
                worker.join(timeout=5)
        except Exception:
            pass

    def _close_transport(self) -> None:
        # Detaches our pool attachment (owner=False → never unlinks the shared block)
        # and tears down our ZMQ endpoint + ipc file. Safe to call partially built.
        pool = getattr(self, "_pool", None)
        if pool is not None:
            try:
                pool.close()
            except Exception:
                pass
        sock = getattr(self, "_zmq_sock", None)
        if sock is not None:
            try:
                sock.close(linger=0)
            except Exception:
                pass
        ctx = getattr(self, "_zmq_ctx", None)
        if ctx is not None:
            try:
                ctx.term()
            except Exception:
                pass
        addr = getattr(self, "_zmq_addr", "") or ""
        if addr.startswith("ipc://"):
            try:
                os.unlink(addr[len("ipc://") :])
            except OSError:
                pass

    def unload(self) -> None:
        self._state_value = "unhealthy"
        self._recv_dead = True
        self._outbound.put(None)
        self._recv_thread.join(timeout=5.0)
        if self._recv_thread.is_alive():
            self._recv_running = False
            self._worker.kill()
            self._recv_thread.join(timeout=2.0)
        self._recv_running = False
        if self._worker.is_alive():
            self._worker.join(timeout=5)
        self._kill_worker()
        if not self._recv_thread.is_alive():
            try:
                self._zmq_sock.close(linger=0)
                self._zmq_ctx.term()
            except Exception:
                pass
            if self._zmq_addr.startswith("ipc://"):
                try:
                    os.unlink(self._zmq_addr[len("ipc://") :])
                except OSError:
                    pass
        logger.info("SharedBase(%s): unloaded", self._base_key)

    # ---- observability (worker liveness) ----

    @property
    def base_key(self) -> str:
        return self._base_key

    @property
    def retired(self) -> bool:
        return self._retired

    @property
    def device(self) -> str:
        return self._device_str

    @property
    def state(self) -> str:
        if not self._worker.is_alive():
            return "unhealthy"
        return self._state_value

    @property
    def is_healthy(self) -> bool:
        return self.state == "loaded"

    @property
    def is_accepting(self) -> bool:
        return self.state == "loaded"

    @property
    def worker_pid(self) -> Optional[int]:
        return self._worker.pid if self._worker.is_alive() else None
