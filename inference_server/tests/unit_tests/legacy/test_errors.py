import asyncio
import json

import pytest
from inference_models.errors import (
    ModelNotFoundError,
    ModelRetrievalError,
    UnauthorizedModelAccessError,
)

from inference_server.errors import PayloadTooLargeError, ServerBusyError
from inference_server.legacy.errors import (
    LegacyHTTPError,
    legacy_error_response,
    with_legacy_errors,
)


def _registry_runtime_error(cause):
    try:
        raise RuntimeError(str(cause) or "Roboflow registry unreachable") from cause
    except RuntimeError as error:
        return error


def _retrieval_error(status_code):
    error = ModelRetrievalError("denied")
    error.status_code = status_code
    return error


@pytest.mark.parametrize(
    "error,status",
    [
        (LegacyHTTPError(418, "tea"), 418),
        (PayloadTooLargeError("big"), 413),
        (ServerBusyError("busy"), 503),
        (PermissionError("nope"), 401),
        (UnauthorizedModelAccessError("nope"), 401),
        (LookupError("missing"), 404),
        (ModelNotFoundError("missing"), 404),
        (ValueError("bad"), 400),
        (asyncio.TimeoutError(), 504),
        (_registry_runtime_error(OSError("down")), 503),
        (_registry_runtime_error(ModelRetrievalError("x")), 503),
        (_registry_runtime_error(_retrieval_error(402)), 402),
        (_registry_runtime_error(_retrieval_error(403)), 403),
        (_registry_runtime_error(_retrieval_error(423)), 423),
        (RuntimeError("boom"), 500),
    ],
)
def test_status_mapping(error, status):
    response = legacy_error_response(error)
    assert response.status_code == status
    assert "message" in json.loads(response.body)


@pytest.mark.asyncio
async def test_decorator_turns_exception_into_response():
    @with_legacy_errors
    async def route():
        raise LookupError("x")

    assert (await route()).status_code == 404


def test_body_limit_middleware_rejects_oversized_declared_and_streamed_bodies(
    monkeypatch,
):
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient

    from inference_server.legacy.errors import (
        _BodyLimitMiddleware,
        install_legacy_exception_handlers,
    )

    monkeypatch.setattr("inference_server.configuration.MAX_BODY_BYTES", 10)
    app = FastAPI()

    @app.post("/echo")
    async def _echo(request: Request):
        return {"n": len(await request.body())}

    @app.post("/v2/models/infer")
    async def _v2(request: Request):
        return {"n": len(await request.body())}

    install_legacy_exception_handlers(app)
    app.add_middleware(_BodyLimitMiddleware)
    client = TestClient(app)
    assert client.post("/echo", content=b"x" * 11).status_code == 413
    assert client.post("/echo", content=b"x" * 5).json() == {"n": 5}
    assert (
        client.post(
            "/echo",
            content=iter([b"x" * 6, b"x" * 6]),
            headers={"Transfer-Encoding": "chunked"},
        ).status_code
        == 413
    )
    assert client.post("/v2/models/infer", content=b"x" * 11).status_code == 200


@pytest.mark.asyncio
async def test_body_limit_middleware_rejects_streamed_typed_body(monkeypatch):
    from fastapi import FastAPI
    from pydantic import BaseModel

    from inference_server.legacy.errors import (
        _BodyLimitMiddleware,
        install_legacy_exception_handlers,
    )

    monkeypatch.setattr("inference_server.configuration.MAX_BODY_BYTES", 10)

    class _Payload(BaseModel):
        value: str

    app = FastAPI()

    @app.post("/typed")
    async def _typed(payload: _Payload):
        return {"n": len(payload.value)}

    install_legacy_exception_handlers(app)
    app.add_middleware(_BodyLimitMiddleware)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/typed",
        "raw_path": b"/typed",
        "root_path": "",
        "query_string": b"",
        "headers": [(b"host", b"testserver"), (b"content-type", b"application/json")],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }
    chunks = [
        {"type": "http.request", "body": b'{"valu', "more_body": True},
        {"type": "http.request", "body": b'e":"x"}', "more_body": False},
    ]
    sent = []

    async def receive():
        return chunks.pop(0)

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)

    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 413
    assert json.loads(sent[1]["body"]) == {"message": "Request payload too large."}
    assert not chunks
