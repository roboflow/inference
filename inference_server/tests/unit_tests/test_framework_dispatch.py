from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request, Response

from inference_server.framework.dispatch import (
    decode_common_request_params,
    handle_model_inference_request,
)
from inference_server.framework.entities import (
    InputParseError,
    ModelHandlerDescription,
    ModelInterfaceDescription,
)
from inference_server.framework.registry import _HANDLERS
from inference_server.proxies.base import ClientDisconnected

_JPEG = bytes(
    [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46]
    + [0x00] * 12
    + [0xFF, 0xD9]
)


def _request(query: bytes = b"", headers=None, body: bytes = b""):
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v2/models/infer",
        "query_string": query,
        "headers": headers or [(b"authorization", b"Bearer k1")],
    }
    state = {"sent": False}

    async def recv():
        if state["sent"]:
            return {"type": "http.disconnect"}
        state["sent"] = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, recv)


def _mock_proxy():
    proxy = MagicMock()
    proxy.ensure_loaded = AsyncMock(return_value=("model_ready",))
    proxy.infer = AsyncMock(return_value=MagicMock())
    return proxy


def _stat_returns(value):
    return patch(
        "inference_server.framework.dispatch.stat_model_while_checking_auth",
        new=AsyncMock(return_value=value),
    )


def _stat_raises(exc):
    return patch(
        "inference_server.framework.dispatch.stat_model_while_checking_auth",
        new=AsyncMock(side_effect=exc),
    )


@pytest.fixture
def fake_handler_entry():
    keys_before = set(_HANDLERS)
    params: dict[str, dict] = {}
    interface = ModelInterfaceDescription(task="t", params=params, output_schema={})
    parser = AsyncMock(return_value={"images": [b"x"], "params": {}})
    handler = AsyncMock(return_value=MagicMock())
    serializer = MagicMock(return_value=Response(status_code=200, content=b"ok"))
    desc = ModelHandlerDescription(
        input_parser=parser,
        handler=handler,
        output_serializer=serializer,
        interface_provider=lambda: interface,
    )
    _HANDLERS[("fake-task", "infer")] = desc
    try:
        yield {
            "interface": interface,
            "parser": parser,
            "handler": handler,
            "serializer": serializer,
        }
    finally:
        for key in set(_HANDLERS) - keys_before:
            del _HANDLERS[key]


def test_decode_extracts_model_id_action_style_and_extra():
    req = _request(
        query=b"model_id=acme/1&action=embed_text&response_style=rich&confidence=0.5"
    )
    common = decode_common_request_params(req)
    assert common.model_id == "acme/1"
    assert common.action == "embed_text"
    assert common.response_style == "rich"
    assert common.api_key == "k1"
    assert common.extra == {"confidence": "0.5"}


def test_decode_falls_back_style_alias_to_response_style():
    req = _request(query=b"model_id=acme/1&style=rich")
    common = decode_common_request_params(req)
    assert common.response_style == "rich"


@pytest.mark.asyncio
async def test_missing_model_id_returns_400():
    r = await handle_model_inference_request(_request(query=b""), _mock_proxy())
    assert r.status_code == 400
    assert b"MISSING_PARAM" in r.body


@pytest.mark.asyncio
async def test_bad_response_style_returns_400():
    r = await handle_model_inference_request(
        _request(query=b"model_id=m&response_style=bogus"), _mock_proxy()
    )
    assert r.status_code == 400
    assert b"INVALID_PARAM" in r.body


@pytest.mark.asyncio
async def test_unauthorized_returns_401():
    with _stat_raises(PermissionError("bad key")):
        r = await handle_model_inference_request(
            _request(query=b"model_id=m"), _mock_proxy()
        )
    assert r.status_code == 401
    assert b"UNAUTHORIZED" in r.body


@pytest.mark.asyncio
async def test_unknown_model_returns_404():
    with _stat_raises(LookupError("no such")):
        r = await handle_model_inference_request(
            _request(query=b"model_id=m"), _mock_proxy()
        )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_registry_unreachable_returns_503():
    with _stat_raises(RuntimeError("down")):
        r = await handle_model_inference_request(
            _request(query=b"model_id=m"), _mock_proxy()
        )
    assert r.status_code == 503
    assert b"REGISTRY_UNAVAILABLE" in r.body


@pytest.mark.asyncio
async def test_unregistered_model_type_returns_none_for_router_fallthrough():
    with _stat_returns(("not-a-registered-task", "infer")):
        r = await handle_model_inference_request(
            _request(query=b"model_id=m"), _mock_proxy()
        )
    assert r is None


@pytest.mark.asyncio
async def test_registered_model_type_with_bad_action_returns_400(fake_handler_entry):
    with _stat_returns(("fake-task", "infer")):
        r = await handle_model_inference_request(
            _request(query=b"model_id=m&action=bogus"), _mock_proxy()
        )
    assert r.status_code == 400
    assert b"INVALID_ACTION" in r.body


@pytest.mark.asyncio
async def test_param_validation_rejects_missing_required(fake_handler_entry):
    fake_handler_entry["interface"].params["confidence"] = {
        "type": "float",
        "required": True,
    }
    with _stat_returns(("fake-task", "infer")):
        r = await handle_model_inference_request(
            _request(query=b"model_id=m"), _mock_proxy()
        )
    assert r.status_code == 400
    assert b"MISSING_PARAM" in r.body
    assert b"confidence" in r.body


@pytest.mark.asyncio
async def test_param_validation_rejects_wrong_type(fake_handler_entry):
    fake_handler_entry["interface"].params["confidence"] = {
        "type": "float",
        "required": False,
    }
    fake_handler_entry["parser"].return_value = {
        "images": [b"x"],
        "params": {"confidence": "notafloat"},
    }
    with _stat_returns(("fake-task", "infer")):
        r = await handle_model_inference_request(
            _request(query=b"model_id=m&confidence=notafloat"), _mock_proxy()
        )
    assert r.status_code == 400
    assert b"INVALID_PARAM" in r.body


@pytest.mark.asyncio
async def test_param_validation_skips_non_coercible_types(fake_handler_entry):
    fake_handler_entry["interface"].params["images"] = {
        "type": "image",
        "required": True,
    }
    proxy = _mock_proxy()
    with _stat_returns(("fake-task", "infer")):
        r = await handle_model_inference_request(_request(query=b"model_id=m"), proxy)
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_param_validation_accepts_required_str_from_body(fake_handler_entry):
    fake_handler_entry["interface"].params["prompt"] = {
        "type": "str",
        "required": True,
    }
    fake_handler_entry["parser"].return_value = {
        "images": [b"x"],
        "params": {"prompt": "hello"},
    }
    with _stat_returns(("fake-task", "infer")):
        r = await handle_model_inference_request(
            _request(query=b"model_id=m"), _mock_proxy()
        )
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_param_validation_rejects_required_str_missing_everywhere(
    fake_handler_entry,
):
    fake_handler_entry["interface"].params["prompt"] = {
        "type": "str",
        "required": True,
    }
    with _stat_returns(("fake-task", "infer")):
        r = await handle_model_inference_request(
            _request(query=b"model_id=m"), _mock_proxy()
        )
    assert r.status_code == 400
    assert b"MISSING_PARAM" in r.body
    assert b"prompt" in r.body


@pytest.mark.asyncio
async def test_input_parser_error_short_circuits(fake_handler_entry):
    fake_handler_entry["parser"].side_effect = InputParseError(
        Response(status_code=415, content=b"bad fmt")
    )
    proxy = _mock_proxy()
    with _stat_returns(("fake-task", "infer")):
        r = await handle_model_inference_request(_request(query=b"model_id=m"), proxy)
    assert r.status_code == 415
    proxy.ensure_loaded.assert_not_awaited()


@pytest.mark.asyncio
async def test_ensure_loaded_load_timeout_returns_503(fake_handler_entry):
    proxy = _mock_proxy()
    proxy.ensure_loaded = AsyncMock(return_value=("load_timeout", 7))
    with _stat_returns(("fake-task", "infer")):
        r = await handle_model_inference_request(_request(query=b"model_id=m"), proxy)
    assert r.status_code == 503
    assert r.headers["Retry-After"] == "7"


@pytest.mark.asyncio
async def test_ensure_loaded_error_returns_500(fake_handler_entry):
    proxy = _mock_proxy()
    proxy.ensure_loaded = AsyncMock(return_value=("error", "x"))
    with _stat_returns(("fake-task", "infer")):
        r = await handle_model_inference_request(_request(query=b"model_id=m"), proxy)
    assert r.status_code == 500


@pytest.mark.asyncio
async def test_handler_value_error_returns_400(fake_handler_entry):
    """ValueError is bad input (registry validation etc.), NOT payload-too-large."""
    fake_handler_entry["handler"].side_effect = ValueError("'prompt' param required")
    with _stat_returns(("fake-task", "infer")):
        r = await handle_model_inference_request(
            _request(query=b"model_id=m"), _mock_proxy()
        )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_handler_payload_too_large_returns_413(fake_handler_entry):
    from inference_server.errors import PayloadTooLargeError

    fake_handler_entry["handler"].side_effect = PayloadTooLargeError("too big")
    with _stat_returns(("fake-task", "infer")):
        r = await handle_model_inference_request(
            _request(query=b"model_id=m"), _mock_proxy()
        )
    assert r.status_code == 413


@pytest.mark.asyncio
async def test_cuda_oom_is_500_not_busy(fake_handler_entry):
    """Regression: substring matching classified 'Tried to allocate' as 503."""
    fake_handler_entry["handler"].side_effect = RuntimeError(
        "CUDA out of memory. Tried to allocate 2.00 GiB"
    )
    with _stat_returns(("fake-task", "infer")):
        r = await handle_model_inference_request(
            _request(query=b"model_id=m"), _mock_proxy()
        )
    assert r.status_code == 500
    body = r.body.decode()
    assert "Tried to allocate" not in body  # no internal text leaked
    assert "ref " in body  # correlation id present


@pytest.mark.asyncio
async def test_server_busy_error_returns_503_with_retry_after(fake_handler_entry):
    from inference_server.errors import ServerBusyError

    fake_handler_entry["handler"].side_effect = ServerBusyError("no capacity")
    with _stat_returns(("fake-task", "infer")):
        r = await handle_model_inference_request(
            _request(query=b"model_id=m"), _mock_proxy()
        )
    assert r.status_code == 503
    assert r.headers.get("retry-after") == "1"


@pytest.mark.asyncio
async def test_handler_timeout_returns_504(fake_handler_entry):
    fake_handler_entry["handler"].side_effect = asyncio.TimeoutError()
    with _stat_returns(("fake-task", "infer")):
        r = await handle_model_inference_request(
            _request(query=b"model_id=m"), _mock_proxy()
        )
    assert r.status_code == 504


@pytest.mark.asyncio
async def test_plain_runtime_error_is_500_even_if_text_mentions_slots(
    fake_handler_entry,
):
    """Busy classification is by ServerBusyError TYPE now — message text is
    never inspected (substring matching misclassified CUDA OOM as busy)."""
    fake_handler_entry["handler"].side_effect = RuntimeError("no slots available")
    with _stat_returns(("fake-task", "infer")):
        r = await handle_model_inference_request(
            _request(query=b"model_id=m"), _mock_proxy()
        )
    assert r.status_code == 500


@pytest.mark.asyncio
async def test_handler_client_disconnected_returns_499(fake_handler_entry):
    fake_handler_entry["handler"].side_effect = ClientDisconnected()
    with _stat_returns(("fake-task", "infer")):
        r = await handle_model_inference_request(
            _request(query=b"model_id=m"), _mock_proxy()
        )
    assert r.status_code == 499


@pytest.mark.asyncio
async def test_happy_path_invokes_full_pipeline(fake_handler_entry):
    proxy = _mock_proxy()
    with _stat_returns(("fake-task", "infer")):
        r = await handle_model_inference_request(_request(query=b"model_id=m"), proxy)
    assert r.status_code == 200
    fake_handler_entry["parser"].assert_awaited_once()
    proxy.ensure_loaded.assert_awaited_once()
    fake_handler_entry["handler"].assert_awaited_once()
    fake_handler_entry["serializer"].assert_called_once()


@pytest.mark.asyncio
async def test_client_action_overrides_api_default(fake_handler_entry):
    _HANDLERS[("fake-task", "embed_text")] = _HANDLERS[("fake-task", "infer")]
    with _stat_returns(("fake-task", "infer")):
        r = await handle_model_inference_request(
            _request(query=b"model_id=m&action=embed_text"), _mock_proxy()
        )
    assert r.status_code == 200
