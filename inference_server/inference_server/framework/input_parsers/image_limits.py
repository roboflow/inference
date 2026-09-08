"""Shared per-request image-count cap.

The body byte budget bounds total SIZE, not COUNT: a compact payload can carry
a long list of tiny images and spawn one gateway call per entry. Every parser
enforces the same cap.
"""

from __future__ import annotations

from typing import Optional

from fastapi import Response

from inference_server import configuration
from inference_server.errors import error_response


def too_many_images(count: int) -> Optional[Response]:
    """Error response when ``count`` exceeds the cap, else None."""
    limit = configuration.MAX_IMAGES_PER_REQUEST
    if limit > 0 and count > limit:
        return error_response(
            400,
            "TOO_MANY_IMAGES",
            f"at most {limit} images per request",
        )
    return None
