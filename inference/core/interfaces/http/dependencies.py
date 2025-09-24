from typing import Optional, Union

from fastapi import Request
from starlette.datastructures import UploadFile

from inference.core.exceptions import InputImageLoadError


async def parse_body_content_for_legacy_request_handler(
    request: Request,
) -> Optional[Union[bytes, UploadFile]]:
    content_type = request.headers.get("Content-Type")
    if content_type is None:
        return None
    image_reference_in_query = request.query_params.get("image")
    if "multipart/form-data" in content_type:
        form_data = await request.form()
        if "file" not in form_data:
            raise InputImageLoadError(
                message="Expected image to be send in part named 'file' of multipart/form-data request",
                public_message="Expected image to be send in part named 'file' of multipart/form-data request",
            )
        return form_data["file"]
    if content_type is None or image_reference_in_query:
        return None
    return await request.body()
