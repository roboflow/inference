"""v2 model endpoints — /v2/models/*"""

from __future__ import annotations

import asyncio
import json
import struct

from fastapi import APIRouter, Depends, Request, Response

from inference_server import state

router = APIRouter(prefix="/v2/models")

_V2_TODO = Response(
    status_code=501,
    content=b'{"error_code":"NOT_IMPLEMENTED","description":"endpoint not yet implemented"}',
    media_type="application/json",
)


def _bearer_token(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    return auth[7:] if auth.startswith("Bearer ") else ""


@router.post("/infer")
async def v2_infer(request: Request, api_key: str = Depends(_bearer_token)) -> Response:
    """v2 model inference.

    TODO: Multiple input methods (multipart, base64, URL, raw body).
    TODO: Response envelope {type, model_info, usage, predictions}.
    TODO: Typed prediction formats.
    TODO: ?style=compact|rich format selection.
    TODO: Batch inference.
    """
    return _V2_TODO


@router.get("/interface")
async def v2_model_interface(
    request: Request, api_key: str = Depends(_bearer_token)
) -> Response:
    """Discover model interface — supported tasks, params, response types.

    Model must be loaded first. Returns tasks from model registry.

    Query params:
        model_id    Required. Must be already loaded.
    """
    model_id = request.query_params.get("model_id", "")
    if not model_id:
        return Response(
            status_code=400,
            content=b'{"error_code":"MISSING_PARAM","description":"model_id query param required"}',
            media_type="application/json",
        )

    try:
        stats = await state.fetch_stats(timeout_s=5.0)
    except (asyncio.TimeoutError, Exception):
        return Response(
            status_code=503,
            content=b'{"error":"stats_unavailable"}',
            media_type="application/json",
        )

    models = stats.get("mmp_models", {})
    model_info = models.get(model_id)
    if model_info is None:
        return Response(
            status_code=404,
            content=json.dumps({"error_code": "MODEL_NOT_LOADED", "description": f"model '{model_id}' is not loaded"}).encode(),
            media_type="application/json",
        )

    tasks = model_info.get("tasks", {})
    return Response(
        content=json.dumps({"model_id": model_id, "tasks": tasks}).encode(),
        media_type="application/json",
    )


@router.get("/compatibility")
async def v2_model_compatibility(
    request: Request, api_key: str = Depends(_bearer_token)
) -> Response:
    """Discover models compatible with current server configuration.

    TODO: Query model registry for packages matching server runtime.
    """
    return _V2_TODO


@router.get("")
async def v2_list_models(
    request: Request, api_key: str = Depends(_bearer_token)
) -> Response:
    """List currently loaded models with state, device, memory, queue depth."""
    try:
        stats = await state.fetch_stats(timeout_s=5.0)
    except (asyncio.TimeoutError, Exception):
        return Response(
            status_code=503,
            content=b'{"error":"stats_unavailable"}',
            media_type="application/json",
        )

    models = stats.get("models", {})
    return Response(
        content=json.dumps({"models": models}).encode(),
        media_type="application/json",
    )


@router.post("/load")
async def v2_load_model(
    request: Request, api_key: str = Depends(_bearer_token)
) -> Response:
    """Load specified model."""
    model_id = request.query_params.get("model_id", "")
    if not model_id:
        return Response(
            status_code=400,
            content=b'{"error_code":"MISSING_PARAM","description":"model_id query param required"}',
            media_type="application/json",
        )

    mid_bytes = model_id.encode()
    key_bytes = api_key.encode()
    payload = (
        struct.pack(">H", len(mid_bytes))
        + mid_bytes
        + struct.pack(">H", len(key_bytes))
        + key_bytes
    )

    try:
        result = await state.lifecycle_req(state.T_LOAD, payload)
    except asyncio.TimeoutError:
        return Response(
            status_code=504,
            content=b'{"error_code":"TIMEOUT","description":"load request timeout"}',
            media_type="application/json",
        )

    if result[0] == "error":
        return Response(
            status_code=500,
            content=b'{"error_code":"LOAD_FAILED","description":"model load failed"}',
            media_type="application/json",
        )
    return Response(
        content=json.dumps({"model_id": model_id, "status": "loaded"}).encode(),
        media_type="application/json",
    )


@router.post("/unload")
async def v2_unload_model(
    request: Request, api_key: str = Depends(_bearer_token)
) -> Response:
    """Unload specified model."""
    model_id = request.query_params.get("model_id", "")
    if not model_id:
        return Response(
            status_code=400,
            content=b'{"error_code":"MISSING_PARAM","description":"model_id query param required"}',
            media_type="application/json",
        )

    mid_bytes = model_id.encode()
    payload = struct.pack(">H", len(mid_bytes)) + mid_bytes

    try:
        result = await state.lifecycle_req(state.T_UNLOAD, payload)
    except asyncio.TimeoutError:
        return Response(
            status_code=504,
            content=b'{"error_code":"TIMEOUT","description":"unload request timeout"}',
            media_type="application/json",
        )

    if result[0] == "error":
        return Response(
            status_code=500,
            content=b'{"error_code":"UNLOAD_FAILED","description":"model unload failed"}',
            media_type="application/json",
        )
    return Response(
        content=json.dumps({"model_id": model_id, "status": "unloaded"}).encode(),
        media_type="application/json",
    )


@router.delete("")
async def v2_unload_all(
    request: Request, api_key: str = Depends(_bearer_token)
) -> Response:
    """Unload all models."""
    try:
        stats = await state.fetch_stats(timeout_s=5.0)
    except (asyncio.TimeoutError, Exception):
        return Response(
            status_code=503,
            content=b'{"error":"stats_unavailable"}',
            media_type="application/json",
        )

    models = stats.get("models", {})
    errors = []
    for model_id in list(models.keys()):
        mid_bytes = model_id.encode()
        payload = struct.pack(">H", len(mid_bytes)) + mid_bytes
        try:
            result = await state.lifecycle_req(state.T_UNLOAD, payload, timeout_s=10.0)
            if result[0] == "error":
                errors.append(model_id)
        except asyncio.TimeoutError:
            errors.append(model_id)

    if errors:
        return Response(
            content=json.dumps({"status": "partial", "failed": errors}).encode(),
            media_type="application/json",
        )
    return Response(
        content=json.dumps({"status": "all_unloaded", "count": len(models)}).encode(),
        media_type="application/json",
    )
