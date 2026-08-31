"""Multipart form image extraction (one or more images + scalar fields)."""

from __future__ import annotations

import json
import logging
from typing import Optional

from fastapi import Request, Response

from inference_server.framework.input_parsers.image_limits import too_many_images

logger = logging.getLogger(__name__)


async def extract_multipart(
    request: Request,
) -> tuple[list[bytes], dict, Optional[Response]]:
    """Read multipart form. Returns (images, extra_params, error_response_or_None).

    Recognized fields:
      - `image` (file part, repeated for batch) — collected into images list
      - `inputs` (string, JSON object) — merged into extra_params
      - any other string field — added to extra_params
    """
    form = await request.form()
    images: list[bytes] = []
    extra_params: dict = {}
    for key, value in form.multi_items():
        if key == "image":
            # Checked before the read so an over-long part list is refused
            # without buffering every part.
            limit_error = too_many_images(len(images) + 1)
            if limit_error is not None:
                return [], {}, limit_error
            images.append(await value.read())
        elif key == "inputs" and isinstance(value, str):
            try:
                extra_params.update(json.loads(value))
            except json.JSONDecodeError:
                logger.warning("multipart: invalid JSON in 'inputs' form field")
        elif isinstance(value, str):
            extra_params[key] = value
    # Zero image parts is legal at this layer: params-only requests carry
    # inputs without a payload. Each handler-family parser enforces its own
    # image requirement.
    return images, extra_params, None
