from __future__ import annotations

import asyncio

from fastapi import Request, Response
from starlette.requests import ClientDisconnect

from inference_server.errors import error_response
from inference_server.framework.entities import (
    CommonRequestParams,
    InputParseError,
)
from inference_server.framework.input_parsers import (
    extract_images_and_params,
    fetch_image_from_url,
)
from inference_server.proxies.mmp_client import looks_like_image


async def parse_multilabel_classification_input(
    request: Request, common: CommonRequestParams
) -> dict:
    image_urls = [
        u
        for u in request.query_params.getlist("image")
        if u.startswith(("http://", "https://"))
    ]

    extra_params: dict = {}
    if image_urls:
        results = await asyncio.gather(
            *(fetch_image_from_url(u) for u in image_urls)
        )
        images: list[bytes] = []
        for img_bytes, err in results:
            if err is not None:
                raise InputParseError(err)
            images.append(img_bytes)
    else:
        try:
            images, body_params, err = await extract_images_and_params(request)
        except ClientDisconnect:
            raise InputParseError(Response(status_code=499))
        if err is not None:
            raise InputParseError(err)
        extra_params.update(body_params)

    if not images:
        raise InputParseError(
            error_response(400, "EMPTY_BODY", "no image data provided")
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

    merged = dict(common.extra)
    merged.update(extra_params)
    return {"images": images, "params": merged}
