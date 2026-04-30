"""v2 model endpoints — /v2/models/*"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import pickle
import struct
import time
from typing import Any, Optional

import aiohttp
from fastapi import APIRouter, Depends, Request, Response
from starlette.requests import ClientDisconnect

from inference_server import state
from inference_server.errors import error_response
from inference_server.serializers import serialize_json

logger = logging.getLogger(__name__)
from inference_model_manager.serializers_typed import (
    serialize_classification_compact,
    serialize_classification_rich,
    serialize_depth_compact,
    serialize_detections_compact,
    serialize_detections_rich,
    serialize_embeddings,
    serialize_instance_segmentation_compact,
    serialize_instance_segmentation_rich,
    serialize_keypoints_compact,
    serialize_multilabel_classification_compact,
    serialize_passthrough,
    serialize_semantic_segmentation_compact,
    serialize_text,
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

_COMPACT_SERIALIZERS: dict[str, Any] = {
    "Detections": serialize_detections_compact,
    "ClassificationPrediction": serialize_classification_compact,
    "MultiLabelClassificationPrediction": serialize_multilabel_classification_compact,
    "InstanceSegmentationPrediction": serialize_instance_segmentation_compact,
    "SemanticSegmentationPrediction": serialize_semantic_segmentation_compact,
    "KeypointsPrediction": serialize_keypoints_compact,
    "EmbeddingResult": serialize_embeddings,
    "DepthEstimationPrediction": serialize_depth_compact,
}

_RICH_SERIALIZERS: dict[str, Any] = {
    "Detections": serialize_detections_rich,
    "ClassificationPrediction": serialize_classification_rich,
    "MultiLabelClassificationPrediction": serialize_classification_rich,
    "InstanceSegmentationPrediction": serialize_instance_segmentation_rich,
    # Types without a rich variant fall back to compact
}

_class_names_cache: dict[str, list | None] = {}


class _ModelProxy:
    """Lightweight proxy providing class_names for typed serializers."""

    __slots__ = ("class_names",)

    def __init__(self, class_names: list | None):
        self.class_names = class_names


def _typed_serialize(
    predictions: object, class_names: list | None, style: str = "rich"
) -> object:
    proxy = _ModelProxy(class_names)
    if isinstance(predictions, list) and predictions:
        cls_name = type(predictions[0]).__name__
    else:
        cls_name = type(predictions).__name__

    serializers = _RICH_SERIALIZERS if style == "rich" else _COMPACT_SERIALIZERS
    serializer = serializers.get(cls_name) or _COMPACT_SERIALIZERS.get(cls_name)
    if serializer is not None:
        return serializer(predictions, proxy)
    if isinstance(predictions, str):
        return serialize_text(predictions, proxy)
    return serialize_passthrough(predictions, proxy)


# ---------------------------------------------------------------------------
# URL image fetch
# ---------------------------------------------------------------------------

_URL_FETCH_TIMEOUT_S = 10
_URL_FETCH_MAX_BYTES = 50 * 1024 * 1024  # 50 MB


async def _fetch_image_from_url(url: str) -> tuple[Optional[bytes], Optional[Response]]:
    """Fetch image bytes from URL. Returns (bytes, None) or (None, error_response)."""
    if not url.startswith(("http://", "https://")):
        return None, error_response(
            400, "INVALID_URL", "image URL must start with http:// or https://"
        )
    timeout = aiohttp.ClientTimeout(total=_URL_FETCH_TIMEOUT_S)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None, error_response(
                        502,
                        "URL_FETCH_FAILED",
                        f"fetching image URL returned status {resp.status}",
                    )
                content_length = resp.content_length or 0
                if content_length > _URL_FETCH_MAX_BYTES:
                    return None, error_response(
                        413,
                        "URL_IMAGE_TOO_LARGE",
                        f"image at URL exceeds {_URL_FETCH_MAX_BYTES // (1024*1024)}MB limit",
                    )
                data = await resp.read()
                if len(data) > _URL_FETCH_MAX_BYTES:
                    return None, error_response(
                        413,
                        "URL_IMAGE_TOO_LARGE",
                        f"image at URL exceeds {_URL_FETCH_MAX_BYTES // (1024*1024)}MB limit",
                    )
                return data, None
    except asyncio.TimeoutError:
        return None, error_response(
            504,
            "URL_FETCH_TIMEOUT",
            f"fetching image URL timed out after {_URL_FETCH_TIMEOUT_S}s",
        )
    except aiohttp.ClientError as exc:
        return None, error_response(
            502, "URL_FETCH_FAILED", f"fetching image URL failed: {exc}"
        )


# ---------------------------------------------------------------------------
# Input extraction — detect Content-Type, produce image bytes + extra params
# ---------------------------------------------------------------------------


async def _extract_images_and_params(
    request: Request,
) -> tuple[list[bytes], dict, Optional[Response]]:
    """Extract image bytes (possibly multiple) and extra params from request body.

    Supports:
      - Raw body (image/jpeg, image/png, application/octet-stream) — single image
      - Multipart form — one or more image= file parts + optional scalar fields + optional inputs= JSON part
      - JSON body — base64 image(s): single {"type":"base64","value":"..."} or list of them

    Returns (images_list, extra_params, error_response).
    If error_response is not None, return it immediately.
    """
    content_type = (
        (request.headers.get("content-type") or "").lower().split(";")[0].strip()
    )

    # --- Multipart form ---
    if content_type == "multipart/form-data":
        form = await request.form()
        images: list[bytes] = []
        extra_params: dict = {}
        for key, value in form.multi_items():
            if key == "image":
                images.append(await value.read())
            elif key == "inputs" and isinstance(value, str):
                try:
                    extra_params.update(json.loads(value))
                except json.JSONDecodeError:
                    logger.warning("v2_infer: invalid JSON in 'inputs' form field")
            elif isinstance(value, str):
                extra_params[key] = value
        if not images:
            return (
                [],
                {},
                error_response(
                    400,
                    "MISSING_IMAGE",
                    "multipart form must include at least one 'image' file part",
                ),
            )
        return images, extra_params, None

    # --- JSON body with base64 ---
    if content_type == "application/json":
        try:
            body = await request.json()
        except Exception:
            return (
                [],
                {},
                error_response(400, "INVALID_JSON", "request body is not valid JSON"),
            )

        inputs = body.get("inputs", {})
        if not isinstance(inputs, dict):
            return (
                [],
                {},
                error_response(400, "INVALID_INPUTS", "'inputs' must be an object"),
            )

        image_spec = inputs.pop("image", None)
        if image_spec is None:
            return (
                [],
                {},
                error_response(
                    400,
                    "MISSING_IMAGE",
                    "JSON body must include inputs.image",
                ),
            )

        # Normalize to list
        specs = image_spec if isinstance(image_spec, list) else [image_spec]
        images = []
        for i, spec in enumerate(specs):
            if not isinstance(spec, dict) or spec.get("type") != "base64":
                return (
                    [],
                    {},
                    error_response(
                        400,
                        "INVALID_IMAGE",
                        f'inputs.image[{i}] must be {{"type": "base64", "value": "..."}}',
                    ),
                )
            try:
                images.append(base64.b64decode(spec["value"]))
            except Exception:
                return (
                    [],
                    {},
                    error_response(
                        400,
                        "DECODE_FAILED",
                        f"base64 decode failed for inputs.image[{i}]",
                    ),
                )

        return images, inputs, None

    # --- Raw body (default) — single image ---
    chunks = []
    async for chunk in request.stream():
        chunks.append(chunk)
    image_bytes = b"".join(chunks) if chunks else b""
    return [image_bytes] if image_bytes else [], {}, None


# ---------------------------------------------------------------------------
# SHM round-trip: write image → submit → read result → build envelope
# ---------------------------------------------------------------------------


async def _submit_one_slot(
    image_bytes: bytes,
    model_id: str,
    instance: str,
    params: dict,
    *,
    request: Optional[Request] = None,
    _t_body_extract_ms: float = 0.0,
    _t_ensure_loaded_ms: float = 0.0,
) -> tuple[Optional[object], Optional[Response]]:
    """Alloc slot, write image, submit, read result. Returns (predictions, None) or (None, error)."""
    _T = state.TIMING
    if _T:
        _t0 = time.monotonic()

    try:
        slot_id = await state.alloc_slot(model_id, instance)
    except (asyncio.TimeoutError, RuntimeError):
        if _T:
            state.pipeline_record({
                "timestamp": time.time(), "worker_pid": os.getpid(),
                "endpoint": "/v2/models/infer", "payload_bytes": len(image_bytes),
                "status": 503,
                "body_extract_ms": _t_body_extract_ms,
                "ensure_loaded_ms": _t_ensure_loaded_ms,
                "zmq_alloc_ms": (time.monotonic() - _t0) * 1000,
                "shm_write_ms": 0, "zmq_submit_ms": 0, "zmq_result_wait_ms": 0,
                "result_read_ms": 0, "serialize_ms": 0,
                "total_ms": (time.monotonic() - _t0) * 1000,
            })
        return None, error_response(
            503, "SERVER_BUSY", "no slots available, try again", follow_up="retry in 1s"
        )

    if _T:
        _t_alloc = time.monotonic()

    try:
        state.write_input(slot_id, image_bytes, 0)
        if _T:
            _t_write = time.monotonic()

        result = await state.submit_and_wait(
            slot_id, model_id, instance, len(image_bytes), params, request=request
        )
        if _T:
            _t_result = time.monotonic()

        if result[0] == "error":
            if _T:
                state.pipeline_record({
                    "timestamp": time.time(), "worker_pid": os.getpid(),
                    "endpoint": "/v2/models/infer", "payload_bytes": len(image_bytes),
                    "status": 500,
                    "body_extract_ms": _t_body_extract_ms,
                    "ensure_loaded_ms": _t_ensure_loaded_ms,
                    "zmq_alloc_ms": (_t_alloc - _t0) * 1000,
                    "shm_write_ms": (_t_write - _t_alloc) * 1000,
                    "zmq_submit_ms": 0,
                    "zmq_result_wait_ms": (_t_result - _t_write) * 1000,
                    "result_read_ms": 0, "serialize_ms": 0,
                    "total_ms": (_t_result - _t0) * 1000,
                })
            return None, error_response(500, "INFERENCE_FAILED", "inference failed")
        if result[0] != "result":
            return None, error_response(500, "INTERNAL_ERROR", "unexpected result type")

        _, result_slot_id, result_sz = result

        hdr = state.read_slot_header(result_slot_id)
        if hdr is not None and hdr.status == state.SLOT_STATUS_ERROR:
            err_msg = "inference failed"
            if hdr.result_size > 0:
                err_msg = state.read_result(result_slot_id, hdr.result_size).decode(
                    "utf-8", errors="replace"
                )
            if _T:
                state.pipeline_record({
                    "timestamp": time.time(), "worker_pid": os.getpid(),
                    "endpoint": "/v2/models/infer", "payload_bytes": len(image_bytes),
                    "status": 500,
                    "body_extract_ms": _t_body_extract_ms,
                    "ensure_loaded_ms": _t_ensure_loaded_ms,
                    "zmq_alloc_ms": (_t_alloc - _t0) * 1000,
                    "shm_write_ms": (_t_write - _t_alloc) * 1000,
                    "zmq_submit_ms": 0,
                    "zmq_result_wait_ms": (_t_result - _t_write) * 1000,
                    "result_read_ms": 0, "serialize_ms": 0,
                    "total_ms": (time.monotonic() - _t0) * 1000,
                })
            return None, error_response(500, "INFERENCE_FAILED", err_msg)

        raw = state.read_result(result_slot_id, result_sz)

        try:
            obj = pickle.loads(raw)
        except Exception:
            return None, error_response(
                500, "DESERIALIZATION_FAILED", "result deserialization failed"
            )

        if _T:
            _t_end = time.monotonic()
            state.pipeline_record({
                "timestamp": time.time(), "worker_pid": os.getpid(),
                "endpoint": "/v2/models/infer", "payload_bytes": len(image_bytes),
                "status": 200,
                "body_extract_ms": _t_body_extract_ms,
                "ensure_loaded_ms": _t_ensure_loaded_ms,
                "zmq_alloc_ms": (_t_alloc - _t0) * 1000,
                "shm_write_ms": (_t_write - _t_alloc) * 1000,
                "zmq_submit_ms": 0,
                "zmq_result_wait_ms": (_t_result - _t_write) * 1000,
                "result_read_ms": (_t_end - _t_result) * 1000,
                "serialize_ms": 0,
                "total_ms": (_t_end - _t0) * 1000,
            })
        return obj, None

    except asyncio.TimeoutError:
        if _T:
            state.pipeline_record({
                "timestamp": time.time(), "worker_pid": os.getpid(),
                "endpoint": "/v2/models/infer", "payload_bytes": len(image_bytes),
                "status": 504,
                "body_extract_ms": _t_body_extract_ms,
                "ensure_loaded_ms": _t_ensure_loaded_ms,
                "zmq_alloc_ms": (_t_alloc - _t0) * 1000,
                "shm_write_ms": 0, "zmq_submit_ms": 0,
                "zmq_result_wait_ms": (time.monotonic() - _t_alloc) * 1000,
                "result_read_ms": 0, "serialize_ms": 0,
                "total_ms": (time.monotonic() - _t0) * 1000,
            })
        return None, error_response(504, "TIMEOUT", "inference timeout")

    except state._ClientDisconnected:
        logger.debug("[v2_slot] client disconnected during inference wait, slot=%d", slot_id)
        return None, Response(status_code=499)

    finally:
        state.free_slot(slot_id)


def _build_envelope(
    predictions_list: list,
    model_id: str,
    task: Optional[str],
    style: str = "rich",
) -> Response:
    """Wrap typed predictions in v2 response envelope."""
    class_names = _class_names_cache.get(model_id)
    typed = [_typed_serialize(p, class_names, style=style) for p in predictions_list]
    envelope = {
        "type": "roboflow-inference-server-response-v1",
        "model_info": {"model_id": model_id, "task": task},
        "usage": {},
        "predictions": typed,
    }
    return Response(
        content=json.dumps(envelope, default=str).encode(),
        media_type="application/json",
    )


async def _infer_images(
    images: list[bytes],
    model_id: str,
    instance: str,
    task: Optional[str],
    params: dict,
    style: str = "rich",
    request: Optional[Request] = None,
    _t_body_extract_ms: float = 0.0,
    _t_ensure_loaded_ms: float = 0.0,
) -> Response:
    """Infer on one or more images. Returns v2 envelope with N predictions."""
    if not images:
        return error_response(400, "EMPTY_BODY", "no image data provided")

    for i, img in enumerate(images):
        if len(img) > state.SHM_DATA_SIZE:
            return error_response(
                413, "PAYLOAD_TOO_LARGE", f"image[{i}] exceeds slot size"
            )
        if not state.looks_like_image(img):
            return error_response(
                415,
                "UNSUPPORTED_FORMAT",
                f"image[{i}] is not a recognized image format",
            )

    if len(images) == 1:
        predictions, err = await _submit_one_slot(
            images[0], model_id, instance, params,
            request=request,
            _t_body_extract_ms=_t_body_extract_ms,
            _t_ensure_loaded_ms=_t_ensure_loaded_ms,
        )
        if err is not None:
            return err
        return _build_envelope([predictions], model_id, task, style=style)

    # Batch: submit all concurrently
    slot_tasks = [
        _submit_one_slot(
            img, model_id, instance, params,
            request=request,
            _t_body_extract_ms=_t_body_extract_ms,
            _t_ensure_loaded_ms=_t_ensure_loaded_ms,
        )
        for img in images
    ]
    results = await asyncio.gather(*slot_tasks)

    predictions_list = []
    for predictions, err in results:
        if err is not None:
            return err
        predictions_list.append(predictions)

    return _build_envelope(predictions_list, model_id, task, style=style)


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

    Image input (one of, supports batch via repeated image param/parts):
        - Query param: image=<URL> (server fetches, max 50MB, 10s timeout). Repeat for batch.
        - Raw body: image/jpeg, image/png, application/octet-stream (single image only)
        - Multipart form: image= file parts (one or more), optional scalar fields, optional inputs= JSON part
        - JSON body: {"inputs": {"image": <spec_or_list>, ...}} where spec is {"type":"base64","value":"..."}
    """
    params = dict(request.query_params)
    model_id = params.pop("model_id", "")
    task = params.pop("task", None)
    instance = params.pop("instance", "")
    device = params.pop("device", "")
    style = params.pop("style", "rich")

    if not model_id:
        return error_response(400, "MISSING_PARAM", "model_id query param required")
    if style not in ("compact", "rich"):
        return error_response(400, "INVALID_PARAM", "style must be 'compact' or 'rich'")

    if task:
        params["task"] = task

    # URL-based input: image=<url> in query params (supports batch via repeated param)
    image_urls = request.query_params.getlist("image")
    image_urls = [u for u in image_urls if u.startswith(("http://", "https://"))]
    params.pop("image", None)

    _t_body_start = time.monotonic()

    try:
        if image_urls:
            fetch_tasks = [_fetch_image_from_url(u) for u in image_urls]
            fetch_results = await asyncio.gather(*fetch_tasks)
            images = []
            for img_bytes, err in fetch_results:
                if err is not None:
                    return err
                images.append(img_bytes)
            extra_params = {}
        else:
            # Extract from request body (raw, multipart, JSON+base64)
            images, extra_params, err = await _extract_images_and_params(request)
            if err is not None:
                return err
        params.update(extra_params)
    except ClientDisconnect:
        logger.debug("[v2_infer] client disconnected during body extraction")
        return Response(status_code=499)

    _t_body_done = time.monotonic()
    _t_body_extract_ms = (_t_body_done - _t_body_start) * 1000

    # Ensure model loaded
    status = await state.ensure_loaded(model_id, instance, api_key, device)
    _t_ensure_done = time.monotonic()
    _t_ensure_loaded_ms = (_t_ensure_done - _t_body_done) * 1000

    if status[0] == "load_timeout":
        return error_response(
            503,
            "MODEL_LOADING",
            "model loading, try again shortly",
            follow_up="retry after Retry-After seconds",
            headers={"Retry-After": str(status[1])},
        )
    if status[0] == "error":
        return error_response(500, "LOAD_FAILED", "model load failed")

    return await _infer_images(
        images, model_id, instance, task, params, style=style,
        request=request,
        _t_body_extract_ms=_t_body_extract_ms,
        _t_ensure_loaded_ms=_t_ensure_loaded_ms,
    )


# ---------------------------------------------------------------------------
# GET /v2/models/interface
# ---------------------------------------------------------------------------


@router.get("/interface")
async def v2_model_interface(request: Request) -> Response:
    """Discover model interface — supported tasks, params, response types."""
    model_id = request.query_params.get("model_id", "")
    if not model_id:
        return error_response(400, "MISSING_PARAM", "model_id query param required")

    try:
        stats = await state.fetch_stats(timeout_s=5.0)
    except (asyncio.TimeoutError, Exception):
        return error_response(503, "STATS_UNAVAILABLE", "could not reach model manager")

    models = stats.get("mmp_models", {})
    model_info = models.get(model_id)
    if model_info is None:
        return error_response(
            404,
            "MODEL_NOT_LOADED",
            f"model '{model_id}' is not loaded",
            follow_up="load the model first via POST /v2/models/load",
        )

    tasks = model_info.get("tasks", {})
    return Response(
        content=json.dumps({"model_id": model_id, "tasks": tasks}).encode(),
        media_type="application/json",
    )


# ---------------------------------------------------------------------------
# GET /v2/models/compatibility
# ---------------------------------------------------------------------------


@router.get("/compatibility")
async def v2_model_compatibility() -> Response:
    """Discover models compatible with current server configuration.

    TODO: Query model registry for packages matching server runtime.
    """
    return _V2_TODO


# ---------------------------------------------------------------------------
# GET /v2/models (list)
# ---------------------------------------------------------------------------


@router.get("")
async def v2_list_models() -> Response:
    """List currently loaded models with state, device, memory, queue depth."""
    try:
        stats = await state.fetch_stats(timeout_s=5.0)
    except (asyncio.TimeoutError, Exception):
        return error_response(503, "STATS_UNAVAILABLE", "could not reach model manager")

    models = stats.get("mmp_models", {})
    return Response(
        content=json.dumps({"models": models}).encode(),
        media_type="application/json",
    )


# ---------------------------------------------------------------------------
# POST /v2/models/load
# ---------------------------------------------------------------------------


@router.post("/load")
async def v2_load_model(
    request: Request, api_key: str = Depends(_bearer_token)
) -> Response:
    """Load specified model."""
    model_id = request.query_params.get("model_id", "")
    if not model_id:
        return error_response(400, "MISSING_PARAM", "model_id query param required")

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
        return error_response(504, "TIMEOUT", "load request timeout")

    if result[0] == "error":
        return error_response(500, "LOAD_FAILED", "model load failed")
    return Response(
        content=json.dumps({"model_id": model_id, "status": "loaded"}).encode(),
        media_type="application/json",
    )


# ---------------------------------------------------------------------------
# POST /v2/models/unload
# ---------------------------------------------------------------------------


@router.post("/unload")
async def v2_unload_model(request: Request) -> Response:
    """Unload specified model."""
    model_id = request.query_params.get("model_id", "")
    if not model_id:
        return error_response(400, "MISSING_PARAM", "model_id query param required")

    mid_bytes = model_id.encode()
    payload = struct.pack(">H", len(mid_bytes)) + mid_bytes

    try:
        result = await state.lifecycle_req(state.T_UNLOAD, payload)
    except asyncio.TimeoutError:
        return error_response(504, "TIMEOUT", "unload request timeout")

    if result[0] == "error":
        return error_response(500, "UNLOAD_FAILED", "model unload failed")
    return Response(
        content=json.dumps({"model_id": model_id, "status": "unloaded"}).encode(),
        media_type="application/json",
    )


# ---------------------------------------------------------------------------
# DELETE /v2/models (unload all)
# ---------------------------------------------------------------------------


@router.delete("")
async def v2_unload_all() -> Response:
    """Unload all models."""
    try:
        stats = await state.fetch_stats(timeout_s=5.0)
    except (asyncio.TimeoutError, Exception):
        return error_response(503, "STATS_UNAVAILABLE", "could not reach model manager")

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
