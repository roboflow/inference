"""Worker subprocess for the shared-base backend.

Loads one base model eagerly (resolved package + head device), then accepts heads
via the control channel. Each head is a full model that reuses the already-resident
base as an injected dependency, so base weights live in VRAM once. Slots carry a
head_index; the loop routes each batch to its head and reuses the plain worker's
`_process_slots` per head.

The pure handlers (`handle_load_head`, `handle_drop_head`, `group_pending_by_head`)
hold the lifecycle/routing logic and are unit-tested; the main/loop wiring is
exercised by integration (needs a real base model + GPU) and kept thin.
"""

import logging
import struct
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import zmq

from inference_model_manager.backends.base import detect_max_batch_size
from inference_model_manager.backends.decode import Decoder, make_batch_decoder
from inference_model_manager.backends.utils.shm_pool import SHMPool
from inference_model_manager.backends.shared_base_protocol import (
    MSG_DROP_HEAD,
    MSG_DROP_HEAD_ACK,
    MSG_HEAD_SLOT_READY,
    MSG_LOAD_HEAD,
    MSG_LOAD_HEAD_ACK,
    HEAD_SLOT_SIZE,
    HeadIndexRegistry,
    decode_control,
    encode_control,
    unpack_head_slot,
)
from inference_model_manager.backends.subproc import (
    _MSG_HEARTBEAT,
    _MSG_RESULT,
    _MSG_SHUTDOWN,
    _MSG_STATS_REQ,
    _build_worker_stats_payload,
    _model_supports_rle,
    _process_slots,
    _slot_erroneable,
    _write_error_to_slot,
)

PendingSlot = Tuple[int, int, int, bytes]  # (slot_id, req_id, head_index, params_bytes)


def group_pending_by_head(
    pending: List[PendingSlot], registry: HeadIndexRegistry
) -> Tuple[Dict[int, List[Tuple[int, int, bytes]]], List[Tuple[int, int]]]:
    """Split accumulated slots into per-head batches; collect slots whose head_index
    is unknown/retired (to be errored, never misrouted)."""
    groups: Dict[int, List[Tuple[int, int, bytes]]] = {}
    unknown: List[Tuple[int, int]] = []
    for slot_id, req_id, head_index, params_bytes in pending:
        if registry.get(head_index) is None:
            unknown.append((slot_id, req_id))
            continue
        groups.setdefault(head_index, []).append((slot_id, req_id, params_bytes))
    return groups, unknown


def handle_load_head(
    payload: Dict[str, Any],
    registry: HeadIndexRegistry,
    load_fn: Callable[[Dict[str, Any]], Tuple[Any, Dict[str, Any]]],
) -> Dict[str, Any]:
    """Load a head and register it. Any failure (incl. CUDA OOM) returns a negative
    ack — the base and existing heads are never touched.

    Idempotent: an already-loaded head re-acks positively with its existing
    index, so a parent retry after a control timeout converges instead of
    erroring forever."""
    req_id = payload.get("req_id")
    head_id = payload.get("head_id")
    try:
        if head_id in registry:
            head_index = registry.index_for(head_id)
            model = registry.get(head_index)
            meta = {
                "model_mro_names": [cls.__name__ for cls in type(model).__mro__],
                "max_batch_size": detect_max_batch_size(model),
                "class_names": getattr(model, "class_names", None),
            }
        else:
            model, meta = load_fn(payload)
            head_index = registry.add(head_id, model)
        return {
            "req_id": req_id,
            "ok": True,
            "head_id": head_id,
            "head_index": head_index,
            **meta,
        }
    except Exception as exc:  # noqa: BLE001 — failure must stay isolated to this head
        return {
            "req_id": req_id,
            "ok": False,
            "head_id": head_id,
            "error": f"{type(exc).__name__}: {exc}",
        }


def handle_drop_head(
    payload: Dict[str, Any],
    registry: HeadIndexRegistry,
    on_removed: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, Any]:
    """Remove a head's mapping, THEN ack. After the ack the index is retired, so any
    late slot for it is rejected by routing rather than hitting a new head."""
    req_id = payload.get("req_id")
    head_id = payload.get("head_id")
    index = registry.remove(head_id)
    if index is not None and on_removed is not None:
        on_removed(index, head_id)
    return {"req_id": req_id, "ok": True, "head_id": head_id}


def make_head_loader(
    base_instance: Any,
    dep_name: str,
    dep_model_id: str,
    dep_metadata_package_id: Optional[str],
    device: str,
) -> Callable[[Dict[str, Any]], Tuple[Any, Dict[str, Any]]]:
    """Build the real head loader: inject the resident base as a preloaded dependency."""

    def load_fn(payload: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        from inference_models.models.auto_loaders.core import AutoModel
        from inference_models.models.auto_loaders.entities import SuppliedDependency

        preloaded = {
            dep_name: SuppliedDependency(
                model_id=dep_model_id,
                model_package_id=dep_metadata_package_id,
                instance=base_instance,
            )
        }
        model = AutoModel.from_pretrained(
            payload.get("model_id_or_path") or payload["head_id"],
            api_key=payload.get("api_key") or None,
            device=device,
            preloaded_model_dependencies=preloaded,
        )
        meta = {
            "model_mro_names": [cls.__name__ for cls in type(model).__mro__],
            "max_batch_size": detect_max_batch_size(model),
            "class_names": getattr(model, "class_names", None),
        }
        return model, meta

    return load_fn


def _shared_worker_main(
    base_model_id: str,
    base_resolved_package_id: str,
    api_key: str,
    setup_pipe: Any,
    zmq_addr: str,
    device: str,
    shm_pool_name: str,
    n_slots: int,
    input_mb: float,
    batch_max_size: int,
    batch_max_wait_ms: float,
    decoder: Decoder,
    dep_name: str,
    dep_model_id: str,
    dep_metadata_package_id: Optional[str],
    base_model_kwargs: dict,
) -> None:
    """Entry point: load the base on ``device``, signal READY, then serve heads."""
    log = logging.getLogger(f"{__name__}.worker")
    pool = sock = zmq_ctx = base = None
    try:
        from inference_models.models.auto_loaders.core import AutoModel

        log.info("SharedWorker(%s): loading base on %s", base_model_id, device)
        base = AutoModel.from_pretrained(
            base_model_id,
            api_key=api_key or None,
            model_package_id=base_resolved_package_id,
            device=device,
            **base_model_kwargs,
        )
        registry = HeadIndexRegistry()
        load_fn = make_head_loader(
            base, dep_name, dep_model_id, dep_metadata_package_id, device
        )
        batch_decode_fn = make_batch_decoder(device, decoder=decoder)
        pool = SHMPool.attach(shm_pool_name, n_slots=n_slots, input_mb=input_mb)

        setup_pipe.send({"status": "READY", "base_model_id": base_model_id})

        zmq_ctx = zmq.Context()
        sock = zmq_ctx.socket(zmq.PAIR)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(zmq_addr)

        _shared_worker_loop(
            pool,
            sock,
            batch_decode_fn,
            registry,
            load_fn,
            batch_max_size or 8,
            batch_max_wait_ms,
            log,
        )
    except KeyboardInterrupt:
        pass
    except Exception as exc:  # noqa: BLE001
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
        del base


def _error_slots(pool, sock, slots: List[Tuple[int, int]], reason: str, log) -> None:
    for slot_id, req_id in slots:
        # Gate like the plain worker's crash path: a slot already resulted
        # (DONE/ERROR) or rebound by the reaper must not be stomped or get a
        # duplicate _MSG_RESULT.
        if not _slot_erroneable(pool, slot_id, req_id):
            continue
        try:
            _write_error_to_slot(pool, slot_id, reason, request_id=req_id)
            sock.send_multipart([_MSG_RESULT, struct.pack(">QII", req_id, slot_id, 0)])
        except Exception:
            log.warning("SharedWorker: failed to error slot %d", slot_id)


def _shared_worker_loop(
    pool,
    sock,
    batch_decode_fn,
    registry: HeadIndexRegistry,
    load_fn: Callable[[Dict[str, Any]], Tuple[Any, Dict[str, Any]]],
    batch_max_size: int,
    batch_max_wait_ms: float,
    log,
) -> None:
    """Accumulate head slots, fire per-head batches; serve load/drop control inline."""
    from collections import deque

    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    batch_max_wait_s = batch_max_wait_ms / 1000.0
    pending: List[PendingSlot] = []
    batch_start = 0.0
    worker_stats: Dict[str, Any] = {
        "inference_count": 0,
        "error_count": 0,
        "batch_count": 0,
        "latencies": deque(maxlen=1000),
        "batch_sizes": deque(maxlen=1000),
        "decode_ms": deque(maxlen=1000),
        "infer_ms": deque(maxlen=1000),
        "write_ms": deque(maxlen=1000),
        "start_ts": time.monotonic(),
        "last_empty_cache_check_ts": 0.0,
    }

    while True:
        timeout_ms = (
            max(0, int((batch_start + batch_max_wait_s - time.monotonic()) * 1000))
            if pending
            else 50
        )
        events = dict(poller.poll(timeout=timeout_ms))
        if sock in events:
            try:
                frames = sock.recv_multipart()
            except zmq.ZMQError:
                break
            msg = frames[0]
            if msg == _MSG_SHUTDOWN:
                break
            elif msg == MSG_LOAD_HEAD:
                ack = handle_load_head(decode_control(frames[1]), registry, load_fn)
                sock.send_multipart([MSG_LOAD_HEAD_ACK, encode_control(**ack)])
            elif msg == MSG_DROP_HEAD:
                ack = handle_drop_head(decode_control(frames[1]), registry)
                sock.send_multipart([MSG_DROP_HEAD_ACK, encode_control(**ack)])
            elif msg == MSG_HEAD_SLOT_READY and len(frames[1]) >= HEAD_SLOT_SIZE:
                slot_id, req_id, head_index = unpack_head_slot(frames[1])
                params_bytes = frames[2] if len(frames) > 2 else b"{}"
                # Refresh the reaper timestamp like the plain worker: a slot
                # queued behind a long batch or an inline head load must not
                # age out and get reclaimed while it waits.
                pool.touch_slot(slot_id)
                if not pending:
                    batch_start = time.monotonic()
                pending.append((slot_id, req_id, head_index, params_bytes))
            elif msg == _MSG_STATS_REQ:
                sock.send_multipart(
                    [_MSG_HEARTBEAT, _build_worker_stats_payload(worker_stats)]
                )

        if pending and (
            len(pending) >= batch_max_size
            or (time.monotonic() - batch_start) >= batch_max_wait_s
        ):
            groups, unknown = group_pending_by_head(pending, registry)
            pending = []
            batch_start = 0.0
            if unknown:
                _error_slots(pool, sock, unknown, "unknown head", log)
            for head_index, batch in groups.items():
                model = registry.get(head_index)
                try:
                    _process_slots(
                        model,
                        pool,
                        batch,
                        sock,
                        batch_decode_fn,
                        log,
                        worker_stats,
                        supports_rle=_model_supports_rle(model),
                    )
                except Exception:
                    log.exception("SharedWorker: _process_slots crashed for head %d", head_index)
                    _error_slots(
                        pool, sock, [(s, r) for s, r, _ in batch], "batch crashed", log
                    )
