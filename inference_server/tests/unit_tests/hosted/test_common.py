import pytest

from inference_server import configuration
from inference_server.hosted.common import HostedRequest, resolve_api_key


def _scope(method="POST", path="/x", query=b"", headers=None):
    return {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": query,
        "headers": [
            (k.encode("latin-1"), v.encode("latin-1")) for k, v in (headers or [])
        ],
    }


def _receive_of(*messages):
    pending = list(messages)

    async def _receive():
        if pending:
            return pending.pop(0)
        return {"type": "http.disconnect"}

    return _receive


def _json_headers(body: bytes):
    return [("content-type", "application/json"), ("content-length", str(len(body)))]


def test_query_params_and_header_parsing():
    request = HostedRequest(
        _scope(
            query=b"api_key=a&api_key=b&flag=",
            headers=[("Content-Type", "application/json; charset=utf-8")],
        ),
        _receive_of(),
    )

    assert request.query_params == {"api_key": "b", "flag": ""}
    assert request.content_type == "application/json"
    assert request.content_length == 0
    assert request.has_json_body is False


def test_garbage_content_length_reads_as_zero():
    request = HostedRequest(_scope(headers=[("content-length", "lots")]), _receive_of())

    assert request.content_length == 0


@pytest.mark.asyncio
async def test_oversized_json_body_is_still_parsed(monkeypatch):
    monkeypatch.setattr(configuration, "MAX_BODY_BYTES", 10)
    body = b'{"api_key": "k", "dynamic_blocks_definitions": [1]}'
    request = HostedRequest(
        _scope(headers=_json_headers(body)),
        _receive_of({"type": "http.request", "body": body, "more_body": False}),
    )

    assert request.has_json_body is True
    assert await resolve_api_key(request) == "k"


@pytest.mark.asyncio
async def test_chunked_body_is_joined_and_replayed_once():
    body = b'{"api_key": "k", "n": 1}'
    request = HostedRequest(
        _scope(headers=_json_headers(body)),
        _receive_of(
            {"type": "http.request", "body": body[:5], "more_body": True},
            {"type": "http.request", "body": body[5:], "more_body": False},
        ),
    )

    assert await request.json() == {"api_key": "k", "n": 1}
    assert await request.json() == {"api_key": "k", "n": 1}
    replayed = await request.receive()
    assert replayed == {"type": "http.request", "body": body, "more_body": False}
    assert (await request.receive())["type"] == "http.disconnect"


@pytest.mark.asyncio
async def test_empty_buffered_body_replays_terminal_request_message():
    request = HostedRequest(
        _scope(headers=_json_headers(b"x")),
        _receive_of({"type": "http.request", "body": b"", "more_body": False}),
    )

    assert await request.json() == {}
    assert await request.receive() == {
        "type": "http.request",
        "body": b"",
        "more_body": False,
    }


@pytest.mark.asyncio
async def test_disconnect_while_buffering_is_replayed():
    request = HostedRequest(
        _scope(headers=_json_headers(b"12345")),
        _receive_of(
            {"type": "http.request", "body": b"12", "more_body": True},
            {"type": "http.disconnect"},
        ),
    )

    assert await request.json() == {}
    assert await request.receive() == {"type": "http.disconnect"}


@pytest.mark.asyncio
async def test_receive_passes_through_when_body_untouched():
    message = {"type": "http.request", "body": b"raw", "more_body": False}
    request = HostedRequest(_scope(), _receive_of(message))

    assert await request.receive() is message


@pytest.mark.asyncio
async def test_non_dict_json_body_yields_no_params():
    body = b"[1, 2]"
    request = HostedRequest(
        _scope(headers=_json_headers(body)),
        _receive_of({"type": "http.request", "body": body, "more_body": False}),
    )

    assert await request.json() == {}
    assert await resolve_api_key(request) is None
