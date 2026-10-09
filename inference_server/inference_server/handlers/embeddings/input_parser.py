from __future__ import annotations

from fastapi import Request, Response
from starlette.requests import ClientDisconnect

from inference_server.errors import PayloadTooLargeError, error_response
from inference_server.framework.entities import CommonRequestParams, InputParseError
from inference_server.framework.input_parsers import (
    extract_images_and_params,
    fetch_images_from_urls,
)
from inference_server.framework.input_parsers.image_check import looks_like_image


async def _imageless_json_inputs(request: Request, action: str) -> dict | None:
    if action not in ("embed_text", "compare"):
        return None
    content_type = (
        (request.headers.get("content-type") or "").lower().split(";")[0].strip()
    )
    if content_type != "application/json":
        return None
    try:
        body = await request.json()
    except (PayloadTooLargeError, ClientDisconnect):
        raise
    except Exception:
        return None
    inputs = body.get("inputs", {}) if isinstance(body, dict) else None
    if not isinstance(inputs, dict) or inputs.get("image") is not None:
        return None
    return {k: v for k, v in inputs.items() if k != "image"}


async def parse_embeddings_input(request: Request, common: CommonRequestParams) -> dict:
    image_urls = [
        u
        for u in request.query_params.getlist("image")
        if u.startswith(("http://", "https://"))
    ]

    extra_params: dict = {}
    images: list[bytes] = []
    if image_urls:
        images, err = await fetch_images_from_urls(image_urls)
        if err is not None:
            raise InputParseError(err)
    else:
        try:
            imageless_inputs = await _imageless_json_inputs(request, common.action)
            if imageless_inputs is not None:
                extra_params.update(imageless_inputs)
            else:
                images, body_params, err = await extract_images_and_params(request)
                if err is not None:
                    raise InputParseError(err)
                extra_params.update(body_params)
        except ClientDisconnect:
            raise InputParseError(Response(status_code=499))

    merged = dict(common.extra)
    merged.update(extra_params)

    action = common.action
    if action == "embed_images" and not images:
        raise InputParseError(error_response(400, "EMPTY_BODY", "no image data provided"))
    if action == "embed_text" and not merged.get("texts"):
        raise InputParseError(error_response(400, "MISSING_PARAM", "'texts' param required"))
    if action == "compare" and not images and not merged.get("prompt_texts"):
        raise InputParseError(
            error_response(400, "MISSING_PARAM", "compare needs images or prompt_texts")
        )
    if action == "compare" and not images and not merged.get("subject_text"):
        raise InputParseError(
            error_response(
                400,
                "MISSING_PARAM",
                "compare needs a subject: provide an image or subject_text",
            )
        )

    for i, img in enumerate(images):
        if not looks_like_image(img):
            raise InputParseError(
                error_response(
                    415,
                    "UNSUPPORTED_FORMAT",
                    f"image[{i}] is not a recognized image format",
                )
            )

    return {"images": images, "params": merged}
