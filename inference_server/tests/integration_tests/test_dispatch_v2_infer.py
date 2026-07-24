from __future__ import annotations

import json
from typing import Any, Optional
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_asyncio
from fastapi import Request

_JPEG = bytes(
    [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46]
    + [0x00] * 12
    + [0xFF, 0xD9]
)


class _FakeProxy:
    def __init__(self):
        self.ensure_loaded = AsyncMock(return_value=("model_ready",))
        self.infer = AsyncMock()
        self.load = AsyncMock(return_value=("ok",))
        self.unload = AsyncMock(return_value=("ok",))
        self.stats = AsyncMock(return_value={"mmp_models": {}})
        self.interface = AsyncMock(return_value={"model_id": "m", "tasks": {}})


@pytest_asyncio.fixture()
async def app_with_fake_proxy():
    import inference_server.app as app_mod
    from inference_server.framework.model_stat import _reset_cache_for_tests

    _reset_cache_for_tests()
    prev_validate = app_mod.validate_api_key
    app_mod.validate_api_key = AsyncMock(return_value=(True, None))
    proxy = _FakeProxy()
    app_mod.app.state.model_manager = proxy
    try:
        yield app_mod.app, proxy
    finally:
        app_mod.validate_api_key = prev_validate


@pytest_asyncio.fixture()
async def client(app_with_fake_proxy):
    app, _ = app_with_fake_proxy
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        yield c


def _patch_stat(task_type: str, default_action: str = "infer"):
    return patch(
        "inference_server.framework.dispatch.stat_model_while_checking_auth",
        new=AsyncMock(return_value=(task_type, default_action)),
    )


# ---------------------------------------------------------------------------
# Object detection — dispatcher cuts in, envelope returned
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_object_detection_dispatch_returns_envelope(client, app_with_fake_proxy):
    _, proxy = app_with_fake_proxy

    class _FakeDetections:
        pass

    proxy.infer.return_value = _FakeDetections()
    with (
        patch(
            "inference_server.handlers.object_detection.output_serializer."
            "serialize_detections_compact",
            return_value={"detections": [{"cls": 0, "conf": 0.9}]},
        ),
        _patch_stat("object-detection"),
    ):
        resp = await client.post(
            "/v2/models/infer?model_id=acme/1",
            content=_JPEG,
            headers={
                "authorization": "Bearer k",
                "content-type": "application/octet-stream",
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "roboflow-inference-server-response-v1"
    assert body["model_info"]["task"] == "object-detection"
    assert body["model_info"]["model_id"] == "acme/1"
    assert len(body["predictions"]) == 1
    proxy.ensure_loaded.assert_awaited_once()
    proxy.infer.assert_awaited_once()
    call = proxy.infer.await_args.kwargs
    assert call["model_id"] == "acme/1"
    assert call["image"] == _JPEG


@pytest.mark.asyncio
async def test_classification_dispatch_returns_envelope(client, app_with_fake_proxy):
    _, proxy = app_with_fake_proxy
    proxy.infer.return_value = object()
    with (
        patch(
            "inference_server.handlers.classification.output_serializer."
            "serialize_classification_compact",
            return_value={"top": "cat", "confidence": 0.9},
        ),
        _patch_stat("classification"),
    ):
        resp = await client.post(
            "/v2/models/infer?model_id=acme/1",
            content=_JPEG,
            headers={"authorization": "Bearer k", "content-type": "image/jpeg"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_info"]["task"] == "classification"


@pytest.mark.asyncio
async def test_vlm_prompt_dispatch_threads_prompt_into_params(
    client, app_with_fake_proxy
):
    _, proxy = app_with_fake_proxy
    proxy.infer.return_value = "a cat sitting on a mat"
    with (
        patch(
            "inference_server.handlers.vlm.output_serializer.serialize_text",
            return_value={"text": "a cat sitting on a mat"},
        ),
        _patch_stat("vlm", default_action="prompt"),
    ):
        resp = await client.post(
            "/v2/models/infer?model_id=acme/1&prompt=describe%20this",
            content=_JPEG,
            headers={"authorization": "Bearer k", "content-type": "image/jpeg"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_info"]["task"] == "vlm"
    call = proxy.infer.await_args.kwargs
    assert call["task"] == "prompt"
    assert call["params"].get("prompt") == "describe this"


@pytest.mark.asyncio
async def test_vlm_prompt_missing_required_prompt_returns_400(
    client, app_with_fake_proxy
):
    _, proxy = app_with_fake_proxy
    with _patch_stat("vlm", default_action="prompt"):
        resp = await client.post(
            "/v2/models/infer?model_id=acme/1",
            content=_JPEG,
            headers={"authorization": "Bearer k", "content-type": "image/jpeg"},
        )
    assert resp.status_code == 400
    body = resp.json()
    assert body["error_code"] == "MISSING_PARAM"
    assert "prompt" in body["description"]
    proxy.infer.assert_not_awaited()


# ---------------------------------------------------------------------------
# Open-vocabulary OD — classes list collection across query/body
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_open_vocab_od_collects_classes_from_query_list(
    client, app_with_fake_proxy
):
    _, proxy = app_with_fake_proxy
    proxy.infer.return_value = object()
    with (
        patch(
            "inference_server.handlers.open_vocabulary_object_detection."
            "output_serializer.serialize_detections_compact",
            return_value={"detections": []},
        ),
        _patch_stat("open-vocabulary-object-detection"),
    ):
        resp = await client.post(
            "/v2/models/infer?model_id=acme/1&classes=cat&classes=dog",
            content=_JPEG,
            headers={"authorization": "Bearer k", "content-type": "image/jpeg"},
        )
    assert resp.status_code == 200
    call = proxy.infer.await_args.kwargs
    assert call["params"]["classes"] == ["cat", "dog"]


@pytest.mark.asyncio
async def test_open_vocab_od_missing_classes_returns_400(client, app_with_fake_proxy):
    _, proxy = app_with_fake_proxy
    with _patch_stat("open-vocabulary-object-detection"):
        resp = await client.post(
            "/v2/models/infer?model_id=acme/1",
            content=_JPEG,
            headers={"authorization": "Bearer k", "content-type": "image/jpeg"},
        )
    assert resp.status_code == 400
    assert "classes" in resp.json()["description"]
    proxy.infer.assert_not_awaited()


# ---------------------------------------------------------------------------
# Registry miss — dispatcher returns None, router falls through to inline body
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unregistered_model_type_returns_501(client, app_with_fake_proxy):
    _, proxy = app_with_fake_proxy
    with _patch_stat("not-a-registered-task"):
        resp = await client.post(
            "/v2/models/infer?model_id=acme/1",
            content=_JPEG,
            headers={"authorization": "Bearer k", "content-type": "image/jpeg"},
        )
    assert resp.status_code == 501
    assert resp.json()["error_code"] == "NOT_IMPLEMENTED"
    proxy.infer.assert_not_awaited()


# ---------------------------------------------------------------------------
# Dispatcher error mapping end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unauthorized_returns_401(client, app_with_fake_proxy):
    with patch(
        "inference_server.framework.dispatch.stat_model_while_checking_auth",
        new=AsyncMock(side_effect=PermissionError("bad key")),
    ):
        resp = await client.post(
            "/v2/models/infer?model_id=acme/1",
            content=_JPEG,
            headers={
                "authorization": "Bearer bad",
                "content-type": "image/jpeg",
            },
        )
    assert resp.status_code == 401
    assert resp.json()["error_code"] == "UNAUTHORIZED"


@pytest.mark.asyncio
async def test_model_not_found_returns_404(client, app_with_fake_proxy):
    with patch(
        "inference_server.framework.dispatch.stat_model_while_checking_auth",
        new=AsyncMock(side_effect=LookupError("no model")),
    ):
        resp = await client.post(
            "/v2/models/infer?model_id=ghost",
            content=_JPEG,
            headers={"authorization": "Bearer k", "content-type": "image/jpeg"},
        )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_unrecognized_image_returns_415(client, app_with_fake_proxy):
    with _patch_stat("object-detection"):
        resp = await client.post(
            "/v2/models/infer?model_id=acme/1",
            content=b"not an image at all just plain text payload",
            headers={
                "authorization": "Bearer k",
                "content-type": "application/octet-stream",
            },
        )
    assert resp.status_code == 415
    assert resp.json()["error_code"] == "UNSUPPORTED_FORMAT"


@pytest.mark.asyncio
async def test_empty_body_returns_400(client, app_with_fake_proxy):
    with _patch_stat("object-detection"):
        resp = await client.post(
            "/v2/models/infer?model_id=acme/1",
            content=b"",
            headers={
                "authorization": "Bearer k",
                "content-type": "application/octet-stream",
            },
        )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_bad_action_for_registered_model_type_returns_400(
    client, app_with_fake_proxy
):
    with _patch_stat("object-detection"):
        resp = await client.post(
            "/v2/models/infer?model_id=acme/1&action=embed_text",
            content=_JPEG,
            headers={"authorization": "Bearer k", "content-type": "image/jpeg"},
        )
    assert resp.status_code == 400
    body = resp.json()
    assert body["error_code"] == "INVALID_ACTION"
    assert "supported actions" in body["actionable_follow_up"]


# ---------------------------------------------------------------------------
# /v2/models/interface — proxy first, then registry for unloaded models
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interface_returns_proxy_data_when_model_loaded(
    client, app_with_fake_proxy
):
    _, proxy = app_with_fake_proxy
    proxy.interface.return_value = {
        "model_id": "acme/1",
        "tasks": {"infer": {"params": {}}},
    }
    resp = await client.get(
        "/v2/models/interface?model_id=acme/1",
        headers={"authorization": "Bearer k"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["tasks"] == {"infer": {"params": {}}}


@pytest.mark.asyncio
async def test_interface_falls_back_to_registry_when_model_unloaded(
    client, app_with_fake_proxy
):
    _, proxy = app_with_fake_proxy
    proxy.interface.side_effect = RuntimeError("not loaded")
    with patch(
        "inference_server.routers.v2_models.stat_model_while_checking_auth",
        new=AsyncMock(return_value=("object-detection", "infer")),
    ):
        resp = await client.get(
            "/v2/models/interface?model_id=acme/1",
            headers={"authorization": "Bearer k"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_type"] == "object-detection"
    assert "infer" in body["actions"]
    assert body["actions"]["infer"]["task"] == "object-detection"
    assert "images" in body["actions"]["infer"]["params"]


@pytest.mark.asyncio
async def test_interface_falls_back_to_registry_lists_all_vlm_actions(
    client, app_with_fake_proxy
):
    _, proxy = app_with_fake_proxy
    proxy.interface.side_effect = RuntimeError("not loaded")
    with patch(
        "inference_server.routers.v2_models.stat_model_while_checking_auth",
        new=AsyncMock(return_value=("vlm", "prompt")),
    ):
        resp = await client.get(
            "/v2/models/interface?model_id=acme/1",
            headers={"authorization": "Bearer k"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_type"] == "vlm"
    assert "prompt" in body["actions"]
    assert "caption" in body["actions"]
    assert "detect" in body["actions"]


@pytest.mark.asyncio
async def test_interface_unauthorized_in_registry_path_returns_401(
    client, app_with_fake_proxy
):
    _, proxy = app_with_fake_proxy
    proxy.interface.side_effect = RuntimeError("not loaded")
    with patch(
        "inference_server.routers.v2_models.stat_model_while_checking_auth",
        new=AsyncMock(side_effect=PermissionError("bad key")),
    ):
        resp = await client.get(
            "/v2/models/interface?model_id=acme/1",
            headers={"authorization": "Bearer bad"},
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_interface_unloaded_unknown_taskType_returns_404(
    client, app_with_fake_proxy
):
    _, proxy = app_with_fake_proxy
    proxy.interface.side_effect = RuntimeError("not loaded")
    with patch(
        "inference_server.routers.v2_models.stat_model_while_checking_auth",
        new=AsyncMock(side_effect=LookupError("no such model")),
    ):
        resp = await client.get(
            "/v2/models/interface?model_id=ghost",
            headers={"authorization": "Bearer k"},
        )
    assert resp.status_code == 404
    assert resp.json()["error_code"] == "MODEL_NOT_LOADED"
