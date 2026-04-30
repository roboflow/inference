"""POST /infer — fast binary inference endpoint."""

from __future__ import annotations

import asyncio
import logging
import os
import pickle
import time

from fastapi import APIRouter, Depends, Request, Response
from starlette.requests import ClientDisconnect

from inference_server import state
from inference_server.serializers import serialize_json

logger = logging.getLogger(__name__)

router = APIRouter()


def _bearer_token(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    return auth[7:] if auth.startswith("Bearer ") else ""


@router.post("/infer")
async def infer(request: Request, api_key: str = Depends(_bearer_token)) -> Response:
    """Fast binary inference endpoint.

    Headers:
        Authorization: Bearer <api_key>    Required.

    Query params:
        model_id    Required.
        task        Optional. Task name.
        instance    Optional. Multi-instance routing.
        device      Optional. Device hint for cold-path load.
        format      Optional. "json" or "pickle" (default: pickle).
        *           Additional params forwarded to backend.

    Body:
        Raw image bytes.
    """
    params = dict(request.query_params)
    model_id = params.pop("model_id", "")
    instance = params.pop("instance", "")
    device = params.pop("device", "")
    fmt = params.pop("format", "pickle")
    if fmt not in ("json", "pickle"):
        return Response(
            status_code=400,
            content=f"Invalid format={fmt!r}; must be 'json' or 'pickle'".encode(),
        )
    if not model_id:
        return Response(status_code=400, content=b"model_id query param required")

    _T = state.TIMING
    if _T:
        _t0 = time.monotonic()

    status = await state.ensure_loaded(model_id, instance, api_key, device)
    if _T:
        _t1 = time.monotonic()
    if status[0] == "load_timeout":
        return Response(
            status_code=503,
            headers={"Retry-After": str(status[1])},
            content=b"model loading, try again shortly",
        )
    if status[0] == "error":
        return Response(status_code=500, content=b"model load failed")

    try:
        slot_id = await state.alloc_slot(model_id, instance)
    except asyncio.TimeoutError:
        return Response(
            status_code=503,
            headers={"Retry-After": "1"},
            content=b"server busy, try again",
        )
    except RuntimeError:
        return Response(status_code=503, content=b"server busy, try again")
    if _T:
        _t2 = time.monotonic()

    try:
        pos = 0
        try:
            async for chunk in request.stream():
                if pos + len(chunk) > state.SHM_DATA_SIZE:
                    return Response(status_code=413, content=b"payload too large")
                if pos == 0 and not state.looks_like_image(chunk):
                    return Response(
                        status_code=415, content=b"body is not a recognized image format"
                    )
                state.write_input(slot_id, chunk, pos)
                pos += len(chunk)
        except ClientDisconnect:
            logger.debug("[infer] client disconnected during body stream, slot=%d", slot_id)
            return Response(status_code=499)

        if _T:
            _t3 = time.monotonic()

        result = await state.submit_and_wait(slot_id, model_id, instance, pos, params, request=request)
        if _T:
            _t4 = time.monotonic()

        if result[0] == "error":
            if _T:
                state.pipeline_record({
                    "timestamp": time.time(), "worker_pid": os.getpid(),
                    "endpoint": "/infer", "payload_bytes": pos, "status": 500,
                    "body_extract_ms": 0,
                    "ensure_loaded_ms": (_t1 - _t0) * 1000,
                    "zmq_alloc_ms": (_t2 - _t1) * 1000,
                    "shm_write_ms": (_t3 - _t2) * 1000,
                    "zmq_submit_ms": 0,
                    "zmq_result_wait_ms": (_t4 - _t3) * 1000,
                    "result_read_ms": 0, "serialize_ms": 0,
                    "total_ms": (_t4 - _t0) * 1000,
                })
            return Response(status_code=500, content=b"inference failed")
        if result[0] != "result":
            return Response(status_code=500, content=b"internal error")

        _, result_slot_id, result_sz = result

        hdr = state.read_slot_header(result_slot_id)
        if hdr is not None and hdr.status == state.SLOT_STATUS_ERROR:
            if hdr.result_size > 0:
                err_msg = state.read_result(result_slot_id, hdr.result_size).decode(
                    "utf-8", errors="replace"
                )
            else:
                err_msg = "inference failed"
            if _T:
                state.pipeline_record({
                    "timestamp": time.time(), "worker_pid": os.getpid(),
                    "endpoint": "/infer", "payload_bytes": pos, "status": 500,
                    "body_extract_ms": 0,
                    "ensure_loaded_ms": (_t1 - _t0) * 1000,
                    "zmq_alloc_ms": (_t2 - _t1) * 1000,
                    "shm_write_ms": (_t3 - _t2) * 1000,
                    "zmq_submit_ms": 0,
                    "zmq_result_wait_ms": (_t4 - _t3) * 1000,
                    "result_read_ms": 0, "serialize_ms": 0,
                    "total_ms": (time.monotonic() - _t0) * 1000,
                })
            return Response(status_code=500, content=err_msg.encode())

        raw = state.read_result(result_slot_id, result_sz)

        if fmt == "json":
            try:
                obj = pickle.loads(raw)
            except Exception:
                return Response(
                    status_code=500, content=b"result deserialization failed"
                )
            content = serialize_json(obj)
            if _T:
                _t_end = time.monotonic()
                state.pipeline_record({
                    "timestamp": time.time(), "worker_pid": os.getpid(),
                    "endpoint": "/infer", "payload_bytes": pos, "status": 200,
                    "body_extract_ms": 0,
                    "ensure_loaded_ms": (_t1 - _t0) * 1000,
                    "zmq_alloc_ms": (_t2 - _t1) * 1000,
                    "shm_write_ms": (_t3 - _t2) * 1000,
                    "zmq_submit_ms": 0,
                    "zmq_result_wait_ms": (_t4 - _t3) * 1000,
                    "result_read_ms": (_t_end - _t4) * 1000,
                    "serialize_ms": 0,
                    "total_ms": (_t_end - _t0) * 1000,
                })
            return Response(content=content, media_type="application/json")

        if _T:
            _t_end = time.monotonic()
            state.pipeline_record({
                "timestamp": time.time(), "worker_pid": os.getpid(),
                "endpoint": "/infer", "payload_bytes": pos, "status": 200,
                "body_extract_ms": 0,
                "ensure_loaded_ms": (_t1 - _t0) * 1000,
                "zmq_alloc_ms": (_t2 - _t1) * 1000,
                "shm_write_ms": (_t3 - _t2) * 1000,
                "zmq_submit_ms": 0,
                "zmq_result_wait_ms": (_t4 - _t3) * 1000,
                "result_read_ms": (_t_end - _t4) * 1000,
                "serialize_ms": 0,
                "total_ms": (_t_end - _t0) * 1000,
            })
        return Response(content=raw, media_type="application/octet-stream")

    except asyncio.TimeoutError:
        if _T:
            state.pipeline_record({
                "timestamp": time.time(), "worker_pid": os.getpid(),
                "endpoint": "/infer", "payload_bytes": pos, "status": 504,
                "body_extract_ms": 0,
                "ensure_loaded_ms": (_t1 - _t0) * 1000,
                "zmq_alloc_ms": (_t2 - _t1) * 1000,
                "shm_write_ms": (_t3 - _t2) * 1000 if '_t3' in dir() else 0,
                "zmq_submit_ms": 0,
                "zmq_result_wait_ms": (time.monotonic() - _t3) * 1000 if '_t3' in dir() else 0,
                "result_read_ms": 0, "serialize_ms": 0,
                "total_ms": (time.monotonic() - _t0) * 1000,
            })
        return Response(status_code=504, content=b"inference timeout")

    except state._ClientDisconnected:
        logger.debug("[infer] client disconnected during inference wait, slot=%d", slot_id)
        return Response(status_code=499)

    finally:
        state.free_slot(slot_id)
