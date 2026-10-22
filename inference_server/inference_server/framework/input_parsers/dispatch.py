"""Content-Type dispatch — pick the right body parser."""

from __future__ import annotations

from typing import Optional

from fastapi import Request, Response

from inference_server.framework.input_parsers.json_base64 import extract_json_base64
from inference_server.framework.input_parsers.multipart import extract_multipart
from inference_server.framework.input_parsers.raw_body import extract_raw_body


async def extract_images_and_params(
    request: Request,
) -> tuple[list[bytes], dict, Optional[Response]]:
    """Extract images + scalar params from the request body.

    Dispatch:
      - `multipart/form-data` → `extract_multipart` (one or more `image=` parts)
      - `application/json` → `extract_json_base64` (base64-encoded image(s))
      - everything else → `extract_raw_body` (single image as raw bytes)
    """
    content_type = (
        (request.headers.get("content-type") or "").lower().split(";")[0].strip()
    )
    if content_type == "multipart/form-data":
        return await extract_multipart(request)
    if content_type == "application/json":
        return await extract_json_base64(request)
    return await extract_raw_body(request)
