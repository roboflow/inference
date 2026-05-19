"""JSON body with base64-encoded image(s).

Payload shape::

    {
      "inputs": {
        "image": {"type": "base64", "value": "..."},
        ...                                       # other scalar params
      }
    }

`inputs.image` may also be a list for batch.
"""

from __future__ import annotations

import base64
from typing import Optional

from fastapi import Request, Response

from inference_server.errors import error_response


async def extract_json_base64(
    request: Request,
) -> tuple[list[bytes], dict, Optional[Response]]:
    """Parse JSON body, decode base64 image(s), return (images, extra_params, err)."""
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
            error_response(400, "MISSING_IMAGE", "JSON body must include inputs.image"),
        )

    specs = image_spec if isinstance(image_spec, list) else [image_spec]
    images: list[bytes] = []
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
