from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request, Response
from starlette.requests import ClientDisconnect

from inference_server.framework.entities import (
    CommonRequestParams,
    InputParseError,
    ServerHooks,
)
from inference_server.handlers.object_detection.description import _DESCRIPTION
from inference_server.handlers.object_detection.handler import handle_object_detection
from inference_server.handlers.object_detection.input_parser import (
    parse_object_detection_input,
)
from inference_server.handlers.object_detection.introspection import (
    get_object_detection_interface,
)
from inference_server.handlers.object_detection.output_serializer import (
    serialize_object_detection,
)

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
        "headers": headers or [],
    }
    state = {"sent": False}

    async def recv():
        if state["sent"]:
            return {"type": "http.disconnect"}
        state["sent"] = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, recv)


# ---------- introspection ----------


def test_introspection_exposes_object_detection_task_and_param_spec():
    interface = get_object_detection_interface()
    assert interface.task == "object-detection"
    assert "images" in interface.params
    assert interface.params["images"]["type"] == "image"
    for p in ("confidence", "iou_threshold", "max_detections", "class_agnostic_nms"):
        assert p in interface.params


def test_description_registers_under_object_detection_infer():
    assert _DESCRIPTION.input_parser is parse_object_detection_input
    assert _DESCRIPTION.handler is handle_object_detection
    assert _DESCRIPTION.output_serializer is serialize_object_detection
    assert _DESCRIPTION.interface_provider is get_object_detection_interface


# ---------- input parser ----------


@pytest.mark.asyncio
async def test_input_parser_reads_raw_body_and_packs_extra_params():
    req = _request(headers=[(b"content-type", b"application/octet-stream")], body=_JPEG)
    common = CommonRequestParams(model_id="m", api_key="", extra={"confidence": "0.5"})
    out = await parse_object_detection_input(req, common)
    assert out["images"] == [_JPEG]
    assert out["params"]["confidence"] == "0.5"


@pytest.mark.asyncio
async def test_input_parser_fetches_image_urls_when_query_image_is_url():
    req = _request(query=b"image=https://x/a.jpg&image=https://x/b.jpg")
    common = CommonRequestParams(model_id="m", api_key="")
    with patch(
        "inference_server.handlers.object_detection.input_parser.fetch_image_from_url",
        new=AsyncMock(side_effect=[(_JPEG, None), (_JPEG, None)]),
    ) as fetch:
        out = await parse_object_detection_input(req, common)
    assert out["images"] == [_JPEG, _JPEG]
    assert fetch.await_count == 2


@pytest.mark.asyncio
async def test_input_parser_propagates_url_fetch_error():
    req = _request(query=b"image=https://x/a.jpg")
    err_resp = Response(status_code=502, content=b"upstream down")
    common = CommonRequestParams(model_id="m", api_key="")
    with patch(
        "inference_server.handlers.object_detection.input_parser.fetch_image_from_url",
        new=AsyncMock(return_value=(None, err_resp)),
    ):
        with pytest.raises(InputParseError) as exc:
            await parse_object_detection_input(req, common)
    assert exc.value.response.status_code == 502


@pytest.mark.asyncio
async def test_input_parser_returns_499_on_client_disconnect():
    req = _request(headers=[(b"content-type", b"application/octet-stream")], body=_JPEG)
    common = CommonRequestParams(model_id="m", api_key="")
    with patch(
        "inference_server.handlers.object_detection.input_parser.extract_images_and_params",
        new=AsyncMock(side_effect=ClientDisconnect()),
    ):
        with pytest.raises(InputParseError) as exc:
            await parse_object_detection_input(req, common)
    assert exc.value.response.status_code == 499


@pytest.mark.asyncio
async def test_input_parser_rejects_empty_body_with_400():
    req = _request(headers=[(b"content-type", b"application/octet-stream")], body=b"")
    common = CommonRequestParams(model_id="m", api_key="")
    with pytest.raises(InputParseError) as exc:
        await parse_object_detection_input(req, common)
    assert exc.value.response.status_code == 400
    assert b"EMPTY_BODY" in exc.value.response.body


@pytest.mark.asyncio
async def test_input_parser_rejects_unrecognized_format_with_415():
    req = _request(
        headers=[(b"content-type", b"application/octet-stream")],
        body=b"not-an-image-payload-12345",
    )
    common = CommonRequestParams(model_id="m", api_key="")
    with pytest.raises(InputParseError) as exc:
        await parse_object_detection_input(req, common)
    assert exc.value.response.status_code == 415
    assert b"UNSUPPORTED_FORMAT" in exc.value.response.body


# ---------- handler ----------


@pytest.mark.asyncio
async def test_handler_single_image_calls_proxy_infer_once_with_task_none():
    proxy = MagicMock()
    fake_prediction = MagicMock()
    proxy.infer = AsyncMock(return_value=fake_prediction)
    hooks = ServerHooks(
        request=None, common=CommonRequestParams(model_id="acme/1", api_key="k")
    )
    out = await handle_object_detection(
        "infer", {"images": [_JPEG], "params": {"confidence": "0.5"}}, proxy, hooks
    )
    assert out is fake_prediction
    proxy.infer.assert_awaited_once()
    kwargs = proxy.infer.await_args.kwargs
    assert kwargs["model_id"] == "acme/1"
    assert kwargs["image"] == _JPEG
    assert kwargs["task"] is None
    assert kwargs["params"]["confidence"] == "0.5"


@pytest.mark.asyncio
async def test_handler_batch_fans_out_concurrent_proxy_calls():
    proxy = MagicMock()
    proxy.infer = AsyncMock(side_effect=[MagicMock() for _ in range(3)])
    hooks = ServerHooks(
        request=None, common=CommonRequestParams(model_id="acme/1", api_key="")
    )
    out = await handle_object_detection(
        "infer", {"images": [_JPEG, _JPEG, _JPEG], "params": {}}, proxy, hooks
    )
    assert isinstance(out, list) and len(out) == 3
    assert proxy.infer.await_count == 3


@pytest.mark.asyncio
async def test_handler_passes_non_infer_action_as_task_kwarg():
    proxy = MagicMock()
    proxy.infer = AsyncMock(return_value=MagicMock())
    hooks = ServerHooks(
        request=None, common=CommonRequestParams(model_id="m", api_key="")
    )
    await handle_object_detection(
        "detect_objects", {"images": [_JPEG], "params": {}}, proxy, hooks
    )
    assert proxy.infer.await_args.kwargs["task"] == "detect_objects"


# ---------- output serializer ----------


def test_serializer_wraps_single_prediction_in_envelope():
    common = CommonRequestParams(
        model_id="acme/1", api_key="", response_style="compact"
    )
    with patch(
        "inference_server.handlers.object_detection.output_serializer.serialize_detections_compact",
        return_value={"detections": [{"cls": 0, "conf": 0.9}]},
    ) as s:
        resp = serialize_object_detection(MagicMock(), common)
    assert resp.status_code == 200
    body = json.loads(resp.body)
    assert body["type"] == "roboflow-inference-server-response-v1"
    assert body["model_info"] == {"model_id": "acme/1", "task": "object-detection"}
    assert len(body["predictions"]) == 1
    s.assert_called_once()


def test_serializer_picks_rich_serializer_for_rich_style():
    common = CommonRequestParams(model_id="m", api_key="", response_style="rich")
    with (
        patch(
            "inference_server.handlers.object_detection.output_serializer.serialize_detections_rich",
            return_value={"detections": []},
        ) as rich,
        patch(
            "inference_server.handlers.object_detection.output_serializer.serialize_detections_compact"
        ) as compact,
    ):
        serialize_object_detection(MagicMock(), common)
    rich.assert_called_once()
    compact.assert_not_called()


def test_serializer_handles_batch_of_predictions():
    common = CommonRequestParams(model_id="m", api_key="", response_style="compact")
    with patch(
        "inference_server.handlers.object_detection.output_serializer.serialize_detections_compact",
        return_value={"detections": []},
    ) as s:
        resp = serialize_object_detection([MagicMock(), MagicMock()], common)
    body = json.loads(resp.body)
    assert len(body["predictions"]) == 2
    assert s.call_count == 2
