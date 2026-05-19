"""POST /infer — fast binary inference endpoint.

Minimum-overhead path: raw image bytes in, raw pickle (or JSON) bytes out.
Body is buffered into bytes before submission (one extra memcpy vs the
previous direct stream-to-SHM path). If profiling shows this matters,
add a streaming variant to the proxy.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, Request, Response
from starlette.requests import ClientDisconnect

from inference_server.dependencies import get_model_manager
from inference_server.proxies.base import ClientDisconnected, ModelManagerProxy
from inference_server.proxies.mmp_client import looks_like_image
from inference_server.serializers import serialize_json

logger = logging.getLogger(__name__)

router = APIRouter()


def _bearer_token(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    return auth[7:] if auth.startswith("Bearer ") else ""


@router.post("/infer")
async def infer(
    request: Request,
    api_key: str = Depends(_bearer_token),
    mm: ModelManagerProxy = Depends(get_model_manager),
) -> Response:
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
    task = params.pop("task", None)
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

    status = await mm.ensure_loaded(model_id, instance, api_key, device)
    if status[0] == "load_timeout":
        return Response(
            status_code=503,
            headers={"Retry-After": str(status[1])},
            content=b"model loading, try again shortly",
        )
    if status[0] == "error":
        return Response(status_code=500, content=b"model load failed")

    # Buffer body
    try:
        chunks: list[bytes] = []
        first_checked = False
        async for chunk in request.stream():
            if not first_checked and chunk:
                if not looks_like_image(chunk):
                    return Response(
                        status_code=415,
                        content=b"body is not a recognized image format",
                    )
                first_checked = True
            chunks.append(chunk)
        image_bytes = b"".join(chunks)
    except ClientDisconnect:
        logger.debug("[infer] client disconnected during body stream")
        return Response(status_code=499)

    if not image_bytes:
        return Response(status_code=400, content=b"empty body")

    try:
        result = await mm.infer(
            model_id=model_id,
            image=image_bytes,
            task=task,
            instance=instance,
            params=params,
            request=request,
            raw_pickle=(fmt == "pickle"),
        )
    except ValueError:
        return Response(status_code=413, content=b"payload too large")
    except asyncio.TimeoutError:
        return Response(status_code=504, content=b"inference timeout")
    except ClientDisconnected:
        logger.debug("[infer] client disconnected during inference wait")
        return Response(status_code=499)
    except RuntimeError as exc:
        msg = str(exc).encode() or b"inference failed"
        if b"no slots" in msg.lower() or b"alloc" in msg.lower():
            return Response(
                status_code=503,
                headers={"Retry-After": "1"},
                content=b"server busy, try again",
            )
        return Response(status_code=500, content=msg)

    if fmt == "json":
        return Response(content=serialize_json(result), media_type="application/json")

    # fmt == "pickle": result is already raw pickle bytes (raw_pickle=True).
    return Response(content=result, media_type="application/octet-stream")
