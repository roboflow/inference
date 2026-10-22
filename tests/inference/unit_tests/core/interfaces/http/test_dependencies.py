import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from fastapi import Depends, FastAPI
from starlette.testclient import TestClient

from inference.core.interfaces.http.dependencies import (
    parse_body_content_for_legacy_request_handler,
)


def _app_with_legacy_body_dependency():
    app = FastAPI()
    received_bodies = []

    @app.post("/test/1")
    async def _infer(body=Depends(parse_body_content_for_legacy_request_handler)):
        received_bodies.append(body)
        return {"ok": True}

    return app, received_bodies


@pytest.mark.parametrize(
    ("content_type", "partial_body"),
    [
        (b"application/octet-stream", b"partial image"),
        (
            b"multipart/form-data; boundary=test-boundary",
            b'--test-boundary\r\nContent-Disposition: form-data; name="file"; '
            b'filename="image.jpg"\r\nContent-Type: image/jpeg\r\n\r\npartial image',
        ),
    ],
)
def test_disconnected_upload_is_a_client_error_and_never_reaches_inference(
    content_type, partial_body
):
    app, received_bodies = _app_with_legacy_body_dependency()
    receive = AsyncMock(
        side_effect=[
            {"type": "http.request", "body": partial_body, "more_body": True},
            {"type": "http.disconnect"},
        ]
    )
    send = AsyncMock()
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/test/1",
        "query_string": b"",
        "headers": [(b"content-type", content_type)],
    }

    asyncio.run(app(scope, receive, send))

    assert send.call_args_list[0].args[0]["status"] == 400
    response_body = json.loads(send.call_args_list[1].args[0]["body"])
    assert response_body == {
        "detail": "Client disconnected before the request body was complete."
    }
    assert received_bodies == []


def test_complete_upload_still_reaches_inference():
    app, received_bodies = _app_with_legacy_body_dependency()

    response = TestClient(app).post(
        "/test/1", content=b"complete image", headers={"content-type": "image/jpeg"}
    )

    assert response.status_code == 200
    assert received_bodies == [b"complete image"]


def test_complete_multipart_upload_still_reaches_inference():
    app, received_bodies = _app_with_legacy_body_dependency()

    response = TestClient(app).post(
        "/test/1", files={"file": ("image.jpg", b"complete image", "image/jpeg")}
    )

    assert response.status_code == 200
    assert len(received_bodies) == 1
    assert received_bodies[0].filename == "image.jpg"
