"""Small image-header helpers for HTTP hot paths."""

from __future__ import annotations

import io
from typing import Any

from PIL import Image

_OVERSIZED = 1 << 62
_HEADER_MAX = 1 << 20


def image_pixels(data: Any) -> int | None:
    try:
        with Image.open(io.BytesIO(bytes(data[:_HEADER_MAX]))) as im:
            w, h = im.size
            return w * h
    except Image.DecompressionBombError:
        return _OVERSIZED
    except Exception:
        return None
