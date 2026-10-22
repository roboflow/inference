from typing import Optional, Union

from fastapi import HTTPException, Request
from starlette.datastructures import UploadFile
from starlette.requests import ClientDisconnect

from inference.core.exceptions import InputImageLoadError


async def parse_body_content_for_legacy_request_handler(
    request: Request,
) -> Optional[Union[bytes, UploadFile]]:
    """Read the image body for a legacy inference request.

    Args:
        request (Request): Incoming HTTP request.

    Returns:
        Optional[Union[bytes, UploadFile]]: Uploaded image, or None when the
        request has no content type or supplies the image through its query.

    Raises:
        HTTPException: If the client disconnects before finishing the upload.
        InputImageLoadError: If a multipart request has no file part.
    """
    content_type = request.headers.get("Content-Type")
    if content_type is None:
        return None

    image_reference_in_query = request.query_params.get("image")
    try:
        if "multipart/form-data" in content_type:
            form_data = await request.form()
            if "file" not in form_data:
                raise InputImageLoadError(
                    message="Expected image to be send in part named 'file' of multipart/form-data request",
                    public_message="Expected image to be send in part named 'file' of multipart/form-data request",
                )
            return form_data["file"]

        if image_reference_in_query:
            return None

        body = await request.body()
        return body
    except ClientDisconnect as error:
        raise HTTPException(
            status_code=400,
            detail="Client disconnected before the request body was complete.",
        ) from error
