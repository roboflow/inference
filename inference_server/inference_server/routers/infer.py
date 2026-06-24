"""POST /infer — fast binary inference endpoint.

Minimum-overhead path: raw image bytes in, raw pickle (or JSON) bytes out.
Known-length request bodies stream into SHM after a small fast prefix check.
Slow, unknown-length, or non-MMP callers use the buffered fallback.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, Depends, Request, Response
from starlette.requests import ClientDisconnect

from inference_server import configuration
from inference_server.auth import extract_bearer
from inference_server.dependencies import get_model_manager
from inference_server.errors import (
    PayloadTooLargeError,
    ServerBusyError,
    UploadTooSlowError,
)
from inference_server.framework.input_parsers.image_check import looks_like_image
from inference_server.image_headers import image_pixels
from inference_server.proxies.base import ClientDisconnected, ModelManagerProxy
from inference_server.serializers import serialize_json

logger = logging.getLogger(__name__)

router = APIRouter()
_SNIFF_BYTES = 262
_MAX_DECODED_PIXELS = (
    int(configuration.MAX_DECODED_MEGAPIXELS * 1_000_000)
    if configuration.MAX_DECODED_MEGAPIXELS > 0
    else 0
)


def _bearer_token(request: Request) -> str:
    return extract_bearer(request.headers.get("authorization", ""))


def _content_length(request: Request) -> int | None:
    raw = request.headers.get("content-length")
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


def _upload_too_slow(received: int, started: float) -> bool:
    elapsed = time.monotonic() - started
    if elapsed <= 0:
        return False
    return received / elapsed < configuration.INFER_STREAM_MIN_BYTES_PER_S


def _upload_timeout_s(content_length: int) -> float:
    by_rate = content_length / configuration.INFER_STREAM_MIN_BYTES_PER_S
    return max(configuration.INFER_STREAM_UPLOAD_TIMEOUT_S, by_rate)


async def _chain_chunks(
    initial: list[bytes], rest: AsyncIterator[bytes]
) -> AsyncIterator[bytes]:
    for chunk in initial:
        yield chunk
    async for chunk in rest:
        yield chunk


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

    content_length = _content_length(request)
    stream_infer = getattr(mm, "infer_stream", None)
    slot_size = getattr(mm, "shm_data_size", None)
    if (
        content_length is not None
        and slot_size is not None
        and content_length > slot_size
    ):
        return Response(status_code=413, content=b"payload too large")

    # Known-length HTTP bodies can stream into SHM, but only after a tiny
    # prefix arrives fast enough. Slow clients stay on the old RAM path so they
    # cannot pin scarce SHM slots while drip-uploading.
    body_iter = request.stream().__aiter__()
    initial_chunks: list[bytes] = []
    prefix = bytearray()
    received = 0
    checked_type = False
    stream_exhausted = False
    use_shm_stream = False

    try:
        if content_length is not None and stream_infer is not None:
            prefix_target = (
                _SNIFF_BYTES
                if content_length <= configuration.INFER_STREAM_SMALL_BODY_BYTES
                else configuration.INFER_STREAM_DIM_PREFIX_BYTES
            )
            started = time.monotonic()
            while len(prefix) < prefix_target:
                try:
                    chunk = await body_iter.__anext__()
                except StopAsyncIteration:
                    stream_exhausted = True
                    break
                if not chunk:
                    continue
                initial_chunks.append(chunk)
                received += len(chunk)
                if len(prefix) < prefix_target:
                    prefix.extend(chunk[: prefix_target - len(prefix)])

                if not checked_type and len(prefix) >= _SNIFF_BYTES:
                    if not looks_like_image(prefix):
                        return Response(
                            status_code=415,
                            content=b"body is not a recognized image format",
                        )
                    checked_type = True
                    if _upload_too_slow(received, started):
                        break
                    if content_length <= configuration.INFER_STREAM_SMALL_BODY_BYTES:
                        break

                if (
                    checked_type
                    and content_length > configuration.INFER_STREAM_SMALL_BODY_BYTES
                    and _MAX_DECODED_PIXELS
                ):
                    px = image_pixels(prefix)
                    if px:
                        if px > _MAX_DECODED_PIXELS:
                            return Response(
                                status_code=413,
                                content=(
                                    f"image {px / 1e6:.1f}MP exceeds max "
                                    f"{_MAX_DECODED_PIXELS / 1e6:.0f}MP"
                                ).encode(),
                            )
                        break

            if initial_chunks:
                if not checked_type and not looks_like_image(prefix):
                    return Response(
                        status_code=415,
                        content=b"body is not a recognized image format",
                    )
                checked_type = True
                use_shm_stream = not _upload_too_slow(received, started)

        if use_shm_stream:
            result = await stream_infer(
                model_id=model_id,
                image_chunks=_chain_chunks(initial_chunks, body_iter),
                content_length=content_length,
                task=task,
                instance=instance,
                params=params,
                request=request,
                raw_pickle=(fmt == "pickle"),
                upload_timeout_s=_upload_timeout_s(content_length),
            )
        else:
            chunks = initial_chunks
            if not stream_exhausted:
                async for chunk in body_iter:
                    if not checked_type and chunk:
                        if not looks_like_image(chunk):
                            return Response(
                                status_code=415,
                                content=b"body is not a recognized image format",
                            )
                        checked_type = True
                    chunks.append(chunk)
            image_bytes = b"".join(chunks)
            del chunks  # drop the per-chunk copies; the joined buffer is enough

            if not image_bytes:
                return Response(status_code=400, content=b"empty body")

            result = await mm.infer(
                model_id=model_id,
                image=image_bytes,
                task=task,
                instance=instance,
                params=params,
                request=request,
                raw_pickle=(fmt == "pickle"),
            )
    except ClientDisconnect:
        logger.debug("[infer] client disconnected during body stream")
        return Response(status_code=499)
    except PayloadTooLargeError:
        return Response(status_code=413, content=b"payload too large")
    except UploadTooSlowError:
        return Response(status_code=408, content=b"upload too slow")
    except ServerBusyError:
        return Response(
            status_code=503,
            headers={"Retry-After": "1"},
            content=b"server busy, try again",
        )
    except ValueError as exc:
        return Response(status_code=400, content=str(exc).encode() or b"bad request")
    except asyncio.TimeoutError:
        return Response(status_code=504, content=b"inference timeout")
    except ClientDisconnected:
        logger.debug("[infer] client disconnected during inference wait")
        return Response(status_code=499)
    except RuntimeError as exc:
        # Full text logged with a correlation id; never echoed to the client.
        ref = uuid.uuid4().hex[:8]
        logger.error("[infer] inference failed (ref %s): %s", ref, exc)
        return Response(
            status_code=500, content=f"inference failed (ref {ref})".encode()
        )

    if fmt == "json":
        return Response(content=serialize_json(result), media_type="application/json")

    # fmt == "pickle": result is already raw pickle bytes (raw_pickle=True).
    return Response(content=result, media_type="application/octet-stream")
