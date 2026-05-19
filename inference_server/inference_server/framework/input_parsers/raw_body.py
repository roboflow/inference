"""Raw-body image extraction (single image)."""

from __future__ import annotations

from typing import Optional

from fastapi import Request, Response


async def extract_raw_body(
    request: Request,
) -> tuple[list[bytes], dict, Optional[Response]]:
    """Read raw request body. Returns ([bytes], {}, None) — single image."""
    chunks = []
    async for chunk in request.stream():
        chunks.append(chunk)
    image_bytes = b"".join(chunks) if chunks else b""
    return [image_bytes] if image_bytes else [], {}, None
