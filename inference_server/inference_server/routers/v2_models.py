"""v2 model endpoints — /v2/models/*"""

from __future__ import annotations

import asyncio
import json
import pickle
import struct
from typing import Any

from fastapi import APIRouter, Depends, Request, Response

from inference_server import state
from inference_server.serializers import serialize_json
from inference_model_manager.serializers_typed import (
    serialize_detections_compact,
    serialize_classification_compact,
    serialize_multilabel_classification_compact,
    serialize_instance_segmentation_compact,
    serialize_semantic_segmentation_compact,
    serialize_keypoints_compact,
    serialize_embeddings,
    serialize_text,
    serialize_depth_compact,
    serialize_passthrough,
)

router = APIRouter(prefix="/v2/models")

_V2_TODO = Response(
    status_code=501,
    content=b'{"error_code":"NOT_IMPLEMENTED","description":"endpoint not yet implemented"}',
    media_type="application/json",
)


def _bearer_token(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    return auth[7:] if auth.startswith("Bearer ") else ""


# ---------------------------------------------------------------------------
# Typed serialization helper
# ---------------------------------------------------------------------------

# Map prediction class name → compact serializer
_SERIALIZER_BY_TYPE: dict[str, Any] = {
    "Detections": serialize_detections_compact,
    "ClassificationPrediction": serialize_classification_compact,
    "MultiLabelClassificationPrediction": serialize_multilabel_classification_compact,
    "InstanceSegmentationPrediction": serialize_instance_segmentation_compact,
    "SemanticSegmentationPrediction": serialize_semantic_segmentation_compact,
    "KeypointsPrediction": serialize_keypoints_compact,
    "EmbeddingResult": serialize_embeddings,
    "DepthEstimationPrediction": serialize_depth_compact,
}

# class_names cache — populated from MMP stats on each v2 infer request
_class_names_cache: dict[str, list | None] = {}


class _ModelProxy:
    """Lightweight proxy providing class_names for typed serializers."""
    __slots__ = ("class_names",)

    def __init__(self, class_names: list | None):
        self.class_names = class_names


def _typed_serialize(predictions: object, class_names: list | None) -> object:
    """Typed serialization by prediction class name, fallback to passthrough."""
    proxy = _ModelProxy(class_names)

    # List of predictions (batch result)
    if isinstance(predictions, list) and predictions:
        cls_name = type(predictions[0]).__name__
    else:
        cls_name = type(predictions).__name__

    serializer = _SERIALIZER_BY_TYPE.get(cls_name)
    if serializer is not None:
        return serializer(predictions, proxy)
    if isinstance(predictions, str):
        return serialize_text(predictions, proxy)
    return serialize_passthrough(predictions, proxy)


# ---------------------------------------------------------------------------
# POST /v2/models/infer
# ---------------------------------------------------------------------------


@router.post("/infer")
async def v2_infer(request: Request, api_key: str = Depends(_bearer_token)) -> Response:
    """v2 model inference — structured JSON response with typed predictions.

    Query params:
        model_id    Required.
        task        Optional. Task name (default task if omitted).
        instance    Optional. Multi-instance routing.
        device      Optional. Device hint for cold-path load.
        style       Optional. "compact" (default) or "rich" (501 for now).
        *           Additional params forwarded to backend.

    Body:
        Raw image bytes (Content-Type: image/jpeg etc.).
        TODO: multipart form, base64 JSON, URL-based input.
    """
    params = dict(request.query_params)
    model_id = params.pop("model_id", "")
    task = params.pop("task", None)
    instance = params.pop("instance", "")
    device = params.pop("device", "")
    style = params.pop("style", "compact")

    if not model_id:
        return Response(
            status_code=400,
            content=b'{"error_code":"MISSING_PARAM","description":"model_id query param required"}',
            media_type="application/json",
        )
    if style not in ("compact", "rich"):
        return Response(
            status_code=400,
            content=b'{"error_code":"INVALID_PARAM","description":"style must be compact or rich"}',
            media_type="application/json",
        )
    if style == "rich":
        return Response(
            status_code=501,
            content=b'{"error_code":"NOT_IMPLEMENTED","description":"rich format not yet implemented"}',
            media_type="application/json",
        )

    if task:
        params["task"] = task

    # 1. Ensure model loaded
    status = await state.ensure_loaded(model_id, instance, api_key, device)
    if status[0] == "load_timeout":
        return Response(
            status_code=503,
            headers={"Retry-After": str(status[1])},
            content=b'{"error_code":"MODEL_LOADING","description":"model loading, try again shortly"}',
            media_type="application/json",
        )
    if status[0] == "error":
        return Response(
            status_code=500,
            content=b'{"error_code":"LOAD_FAILED","description":"model load failed"}',
            media_type="application/json",
        )

    # 2. Alloc SHM slot
    try:
        slot_id = await state.alloc_slot(model_id, instance)
    except (asyncio.TimeoutError, RuntimeError):
        return Response(
            status_code=503,
            content=b'{"error_code":"SERVER_BUSY","description":"no slots available, try again"}',
            media_type="application/json",
        )

    try:
        # 3. Stream body into SHM
        pos = 0
        async for chunk in request.stream():
            if pos + len(chunk) > state.SHM_DATA_SIZE:
                return Response(
                    status_code=413,
                    content=b'{"error_code":"PAYLOAD_TOO_LARGE","description":"image exceeds slot size"}',
                    media_type="application/json",
                )
            if pos == 0 and not state.looks_like_image(chunk):
                return Response(
                    status_code=415,
                    content=b'{"error_code":"UNSUPPORTED_FORMAT","description":"body is not a recognized image format"}',
                    media_type="application/json",
                )
            state.write_input(slot_id, chunk, pos)
            pos += len(chunk)

        # 4. Submit and wait
        result = await state.submit_and_wait(slot_id, model_id, instance, pos, params)

        if result[0] == "error":
            return Response(
                status_code=500,
                content=b'{"error_code":"INFERENCE_FAILED","description":"inference failed"}',
                media_type="application/json",
            )
        if result[0] != "result":
            return Response(
                status_code=500,
                content=b'{"error_code":"INTERNAL_ERROR","description":"unexpected result type"}',
                media_type="application/json",
            )

        _, result_slot_id, result_sz = result

        # Check error slot
        hdr = state.read_slot_header(result_slot_id)
        if hdr is not None and hdr.status == state.SLOT_STATUS_ERROR:
            err_msg = "inference failed"
            if hdr.result_size > 0:
                err_msg = state.read_result(result_slot_id, hdr.result_size).decode(
                    "utf-8", errors="replace"
                )
            return Response(
                status_code=500,
                content=json.dumps({"error_code": "INFERENCE_FAILED", "description": err_msg}).encode(),
                media_type="application/json",
            )

        # 5. Deserialize + typed serialization + envelope
        raw = state.read_result(result_slot_id, result_sz)
        try:
            predictions = pickle.loads(raw)
        except Exception:
            return Response(
                status_code=500,
                content=b'{"error_code":"DESERIALIZATION_FAILED","description":"result deserialization failed"}',
                media_type="application/json",
            )

        class_names = _class_names_cache.get(model_id)
        typed_predictions = _typed_serialize(predictions, class_names)

        envelope = {
            "type": "roboflow-inference-server-response-v1",
            "model_info": {"model_id": model_id, "task": task},
            "usage": {},
            "predictions": typed_predictions if isinstance(typed_predictions, list) else [typed_predictions],
        }

        return Response(
            content=json.dumps(envelope, default=str).encode(),
            media_type="application/json",
        )

    except asyncio.TimeoutError:
        return Response(
            status_code=504,
            content=b'{"error_code":"TIMEOUT","description":"inference timeout"}',
            media_type="application/json",
        )

    finally:
        state.free_slot(slot_id)


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

    models = stats.get("mmp_models", {})
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

    models = stats.get("mmp_models", {})
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
