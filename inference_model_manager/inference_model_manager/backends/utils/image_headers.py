"""Lightweight image-header reads (no pixel decode).

Pillow-only (no imagecodecs/torch) so the HTTP client processes can import it
for the resolution reject gate without pulling the full decode stack.
"""

from __future__ import annotations

import io
from typing import Any

from PIL import Image

# Sentinel returned when Pillow's decompression-bomb guard trips: the image is
# so large Pillow refuses to even report its size. Larger than any sane cap, so
# callers reject it. (Pillow raises only above 2*MAX_IMAGE_PIXELS, ~178 MP.)
_OVERSIZED = 1 << 62

# Only the header is read, so we copy at most this many bytes into the decoder
# buffer — bounds the cost when callers pass a full (e.g. multi-MB SHM) payload.
# 1 MiB comfortably covers any real JPEG SOF / PNG IHDR / WebP header, even
# behind large EXIF/ICC; a header deeper than this does not occur in practice.
_HEADER_MAX = 1 << 20


def image_pixels(data: Any) -> int | None:
    """Pixel count (width*height) from an image header via Pillow, no decode.

    ``Image.open`` is lazy: it reads only the header for ``.size`` and never
    decodes pixels here. Pass the full payload — we copy at most ``_HEADER_MAX``
    bytes into the buffer (enough for any real header, so no false None from a
    truncated prefix, without copying a whole multi-MB image). Returns None only
    when the header genuinely can't be parsed — callers must NOT reject on None.

    A ``DecompressionBombError`` (image beyond Pillow's hard limit) returns the
    ``_OVERSIZED`` sentinel rather than None, so a bomb is rejected, not let
    through. We never mutate the process-global ``MAX_IMAGE_PIXELS``.
    """
    try:
        with Image.open(io.BytesIO(bytes(data[:_HEADER_MAX]))) as im:
            w, h = im.size
            return w * h
    except Image.DecompressionBombError:
        return _OVERSIZED
    except Exception:
        return None
