from typing import Optional

from fastapi import Request


async def request_body_content(request: Request) -> Optional[bytes]:
    content_type = request.headers.get("Content-Type")
    image_reference_in_query = request.query_params.get("image")
    if (
        content_type is None
        or "multipart/form-data" in content_type
        or image_reference_in_query
    ):
        return None
    return await request.body()
