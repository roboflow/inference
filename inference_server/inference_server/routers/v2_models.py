"""v2 model endpoints — /v2/models/*"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import struct
from typing import Any, Optional

import aiohttp
from fastapi import APIRouter, Depends, Request, Response
from starlette.requests import ClientDisconnect

from inference_server.dependencies import get_model_manager
from inference_server.errors import error_response
from inference_server.proxies.base import ClientDisconnected, ModelManagerProxy
from inference_server.proxies.mmp_client import looks_like_image

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

logger = logging.getLogger(__name__)

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
    """Extract image bytes (possibly multiple) and extra params from request body."""
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
                    400, "MISSING_IMAGE", "JSON body must include inputs.image"
                ),
            )

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
# Single-image inference via proxy
# ---------------------------------------------------------------------------


async def _infer_one(
    mm: ModelManagerProxy,
    image_bytes: bytes,
    model_id: str,
    instance: str,
    task: Optional[str],
    params: dict,
    request: Optional[Request],
) -> tuple[Optional[object], Optional[Response]]:
    """Run inference on one image via proxy. Returns (prediction, None) or (None, error)."""
    try:
        prediction = await mm.infer(
            model_id=model_id,
            image=image_bytes,
            task=task,
            instance=instance,
            params=params,
            request=request,
        )
        return prediction, None
    except ValueError as exc:
        # MMPClient raises ValueError on payload-too-large
        return None, error_response(413, "PAYLOAD_TOO_LARGE", str(exc))
    except asyncio.TimeoutError:
        return None, error_response(504, "TIMEOUT", "inference timeout")
    except ClientDisconnected:
        return None, Response(status_code=499)
    except RuntimeError as exc:
        msg = str(exc) or "inference failed"
        if "no slots" in msg.lower() or "alloc" in msg.lower():
            return None, error_response(
                503,
                "SERVER_BUSY",
                "no slots available, try again",
                follow_up="retry in 1s",
            )
        return None, error_response(500, "INFERENCE_FAILED", msg)


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
    mm: ModelManagerProxy,
    images: list[bytes],
    model_id: str,
    instance: str,
    task: Optional[str],
    params: dict,
    style: str = "rich",
    request: Optional[Request] = None,
) -> Response:
    """Infer on one or more images. Returns v2 envelope with N predictions."""
    if not images:
        return error_response(400, "EMPTY_BODY", "no image data provided")

    for i, img in enumerate(images):
        if not looks_like_image(img):
            return error_response(
                415,
                "UNSUPPORTED_FORMAT",
                f"image[{i}] is not a recognized image format",
            )

    if len(images) == 1:
        prediction, err = await _infer_one(
            mm, images[0], model_id, instance, task, params, request
        )
        if err is not None:
            return err
        return _build_envelope([prediction], model_id, task, style=style)

    # Batch: fan out concurrently
    results = await asyncio.gather(
        *(
            _infer_one(mm, img, model_id, instance, task, params, request)
            for img in images
        )
    )

    predictions_list = []
    for prediction, err in results:
        if err is not None:
            return err
        predictions_list.append(prediction)

    return _build_envelope(predictions_list, model_id, task, style=style)


# ---------------------------------------------------------------------------
# POST /v2/models/infer
# ---------------------------------------------------------------------------


@router.post("/infer")
async def v2_infer(
    request: Request,
    api_key: str = Depends(_bearer_token),
    mm: ModelManagerProxy = Depends(get_model_manager),
) -> Response:
    """v2 model inference — structured JSON response with typed predictions."""
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

    try:
        if image_urls:
            fetch_results = await asyncio.gather(
                *(_fetch_image_from_url(u) for u in image_urls)
            )
            images = []
            for img_bytes, err in fetch_results:
                if err is not None:
                    return err
                images.append(img_bytes)
            extra_params = {}
        else:
            images, extra_params, err = await _extract_images_and_params(request)
            if err is not None:
                return err
        params.update(extra_params)
    except ClientDisconnect:
        logger.debug("[v2_infer] client disconnected during body extraction")
        return Response(status_code=499)

    # Ensure model loaded
    status = await mm.ensure_loaded(model_id, instance, api_key, device)

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
        mm, images, model_id, instance, task, params,
        style=style, request=request,
    )


# ---------------------------------------------------------------------------
# GET /v2/models/interface
# ---------------------------------------------------------------------------


@router.get("/interface")
async def v2_model_interface(
    request: Request,
    mm: ModelManagerProxy = Depends(get_model_manager),
) -> Response:
    """Discover model interface — supported tasks, params, response types."""
    model_id = request.query_params.get("model_id", "")
    if not model_id:
        return error_response(400, "MISSING_PARAM", "model_id query param required")

    try:
        info = await mm.interface(model_id)
    except RuntimeError as exc:
        return error_response(
            404,
            "MODEL_NOT_LOADED",
            str(exc),
            follow_up="load the model first via POST /v2/models/load",
        )
    except Exception:
        return error_response(503, "STATS_UNAVAILABLE", "could not reach model manager")

    return Response(
        content=json.dumps(info).encode(),
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
async def v2_list_models(
    mm: ModelManagerProxy = Depends(get_model_manager),
) -> Response:
    """List currently loaded models with state, device, memory, queue depth."""
    try:
        stats = await mm.stats()
    except Exception:
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
    request: Request,
    api_key: str = Depends(_bearer_token),
    mm: ModelManagerProxy = Depends(get_model_manager),
) -> Response:
    """Load specified model."""
    model_id = request.query_params.get("model_id", "")
    if not model_id:
        return error_response(400, "MISSING_PARAM", "model_id query param required")

    try:
        result = await mm.load(model_id, api_key)
    except asyncio.TimeoutError:
        return error_response(504, "TIMEOUT", "load request timeout")

    if result[0] != "ok":
        return error_response(500, "LOAD_FAILED", "model load failed")
    return Response(
        content=json.dumps({"model_id": model_id, "status": "loaded"}).encode(),
        media_type="application/json",
    )


# ---------------------------------------------------------------------------
# POST /v2/models/unload
# ---------------------------------------------------------------------------


@router.post("/unload")
async def v2_unload_model(
    request: Request,
    mm: ModelManagerProxy = Depends(get_model_manager),
) -> Response:
    """Unload specified model."""
    model_id = request.query_params.get("model_id", "")
    if not model_id:
        return error_response(400, "MISSING_PARAM", "model_id query param required")

    try:
        result = await mm.unload(model_id)
    except asyncio.TimeoutError:
        return error_response(504, "TIMEOUT", "unload request timeout")

    if result[0] != "ok":
        return error_response(500, "UNLOAD_FAILED", "model unload failed")
    return Response(
        content=json.dumps({"model_id": model_id, "status": "unloaded"}).encode(),
        media_type="application/json",
    )


# ---------------------------------------------------------------------------
# DELETE /v2/models (unload all)
# ---------------------------------------------------------------------------


@router.delete("")
async def v2_unload_all(
    mm: ModelManagerProxy = Depends(get_model_manager),
) -> Response:
    """Unload all models."""
    try:
        stats = await mm.stats()
    except Exception:
        return error_response(503, "STATS_UNAVAILABLE", "could not reach model manager")

    models = stats.get("mmp_models", {})
    errors = []
    for model_id in list(models.keys()):
        try:
            result = await mm.unload(model_id)
            if result[0] != "ok":
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
