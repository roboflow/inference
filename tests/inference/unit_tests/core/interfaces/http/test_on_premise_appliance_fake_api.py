"""Appliance API key checks against a fake appliance API over real HTTP.

The fake API listens on 127.0.0.1. Nothing that talks to it is mocked, so the
tests run the real usage check, the real model-data request, the
authorization middleware and the per-model access check.
"""

import asyncio
import threading
from collections import Counter
from typing import Iterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web
from starlette.testclient import TestClient

import inference.core.interfaces.http.http_api as http_api
from inference.core import roboflow_api
from inference.core.entities.responses.clip import ClipEmbeddingResponse
from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    ObjectDetectionInferenceResponse,
)
from inference.core.exceptions import RoboflowAPINotAuthorizedError
from inference.core.registries import roboflow as registry

WORKSPACE_MODEL = "project-a/1"
KEY_A = "key-workspace-a"
KEY_B = "key-workspace-b"
KEY_PAUSED = "key-paused"
KEY_OVER_QUOTA = "key-workspace-over-quota"
KEY_BOGUS = "key-does-not-exist"
ALL_KEYS = (KEY_A, KEY_B, KEY_PAUSED, KEY_OVER_QUOTA, KEY_BOGUS)
UNAUTHORIZED_BODY = {
    "error": {
        "message": "This API key does not exist (or has been revoked).",
        "status": 401,
    }
}


class FakeApplianceApi:
    """Aiohttp server that answers the appliance usage check, on a free port."""

    def __init__(self) -> None:
        self.workspace_calls: Counter = Counter()
        self.model_data_calls: Counter = Counter()
        self.outage = False
        self.base_url = ""
        self._loop = asyncio.new_event_loop()
        self._runner = None

    async def _usage_check(self, request: web.Request) -> web.Response:
        # Answers like app/functions/services/billingUsage/
        # billingUsageRequestHandlers.js on the appliance stack.
        api_key = request.query.get("api_key")
        self.workspace_calls[api_key] += 1
        if self.outage:
            return web.json_response({"error": "unavailable"}, status=503)
        if api_key == KEY_A:
            return web.json_response(
                {
                    "workspaceId": "db-workspace-a",
                    "workspace": "workspace-a",
                    "underCap": True,
                    "creditMetering": "not_metered",
                }
            )
        if api_key == KEY_B:
            return web.json_response(
                {
                    "workspaceId": "db-workspace-b",
                    "workspace": "workspace-b",
                    "underCap": True,
                }
            )
        if api_key == KEY_PAUSED:
            return web.json_response({"error": {"message": "paused"}}, status=423)
        if api_key == KEY_OVER_QUOTA:
            return web.json_response(
                {
                    "workspaceId": "db-workspace-c",
                    "workspace": "workspace-c",
                    "underCap": False,
                    "error": {
                        "type": "AccessException",
                        "reason": "credit_cap_exceeded",
                        "message": "Oh no! It seems your workspace ran out of credits.",
                        "planCycleUsed": 120,
                        "cycleCreditCap": 100,
                    },
                },
                status=402,
            )
        return web.json_response(UNAUTHORIZED_BODY, status=401)

    async def _model_data(self, request: web.Request) -> web.Response:
        api_key = request.query.get("api_key")
        self.model_data_calls[api_key] += 1
        if api_key != KEY_A:
            return web.json_response(UNAUTHORIZED_BODY, status=401)
        return web.json_response({"ort": {"model": "project-a"}})

    def start(self) -> None:
        started = threading.Event()

        async def serve() -> None:
            app = web.Application()
            app.router.add_get("/serverless/usage-check", self._usage_check)
            app.router.add_get("/ort/{dataset}/{version}", self._model_data)
            self._runner = web.AppRunner(app)
            await self._runner.setup()
            site = web.TCPSite(self._runner, "127.0.0.1", 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]
            self.base_url = f"http://127.0.0.1:{port}"
            started.set()

        def run() -> None:
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(serve())
            self._loop.run_forever()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        assert started.wait(timeout=5)

    def stop(self) -> None:
        asyncio.run_coroutine_threadsafe(self._runner.cleanup(), self._loop).result(5)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        self._loop.close()


@pytest.fixture
def fake_api() -> Iterator[FakeApplianceApi]:
    api = FakeApplianceApi()
    api.start()
    try:
        yield api
    finally:
        api.stop()


@pytest.fixture
def appliance_client(monkeypatch, fake_api: FakeApplianceApi) -> Iterator[TestClient]:
    monkeypatch.setattr(roboflow_api, "API_BASE_URL", fake_api.base_url)
    monkeypatch.setattr(roboflow_api, "INTERNAL_WEIGHTS_URL_SUFFIX", "")
    monkeypatch.setattr(roboflow_api, "MODELS_CACHE_AUTH_ENABLED", True)
    monkeypatch.setattr(roboflow_api, "OFFLINE_MODE", False)
    monkeypatch.setattr(http_api, "API_BASE_URL", fake_api.base_url, raising=False)
    monkeypatch.setattr(http_api, "InferenceInstrumentator", _DummyInstrumentator)
    monkeypatch.setattr(
        http_api.usage_collector, "async_push_usage_payloads", AsyncMock()
    )
    monkeypatch.setattr(http_api, "ON_PREMISE_APPLIANCE", True)
    monkeypatch.setattr(http_api, "GCP_SERVERLESS", False)
    monkeypatch.setattr(http_api, "LAMBDA", False)
    monkeypatch.setattr(http_api, "OFFLINE_MODE", False)
    monkeypatch.setattr(http_api, "DEDICATED_DEPLOYMENT_WORKSPACE_URL", None)
    monkeypatch.setattr(http_api, "WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT", None)
    monkeypatch.setattr(registry, "ON_PREMISE_APPLIANCE", True)
    monkeypatch.setattr(registry, "USE_INFERENCE_MODELS", False)
    registry._check_if_api_key_has_access_to_model_cached.cache_clear()

    model_manager = MagicMock()
    model_manager.pingback = None
    model_manager.num_errors = 0
    model_manager.get_task_type.return_value = "object-detection"
    model_manager.infer_from_request_sync.side_effect = _fake_inference_response

    def add_model(model_id, api_key, **kwargs) -> None:
        # Mirrors ModelManager.add_model under MODELS_CACHE_AUTH_ENABLED.
        has_access = registry._check_if_api_key_has_access_to_model(
            api_key=api_key,
            model_id=model_id,
            endpoint_type=kwargs.get(
                "endpoint_type", roboflow_api.ModelEndpointType.ORT
            ),
        )
        if not has_access:
            raise RoboflowAPINotAuthorizedError("denied")

    model_manager.add_model.side_effect = add_model
    interface = http_api.HttpInterface(model_manager=model_manager)
    with TestClient(interface.app) as client:
        yield client
    registry._check_if_api_key_has_access_to_model_cached.cache_clear()


class _DummyInstrumentator:
    def __init__(self, app, model_manager, endpoint="/metrics"):
        self.app = app
        self.model_manager = model_manager
        self.endpoint = endpoint

    def set_stream_manager_client(self, stream_manager_client) -> None:
        self.stream_manager_client = stream_manager_client


def _fake_inference_response(model_id, request, **kwargs):
    if model_id.startswith("clip"):
        return ClipEmbeddingResponse(embeddings=[[0.1, 0.2]], time=0.01)
    return ObjectDetectionInferenceResponse(
        image=InferenceResponseImage(width=1, height=1), predictions=[], time=0.01
    )


def _workspace_model_request(api_key=None) -> dict:
    body = {
        "model_id": WORKSPACE_MODEL,
        "image": {"type": "url", "value": "https://example.com/a.jpg"},
    }
    if api_key is not None:
        body["api_key"] = api_key
    return body


def _core_model_request(api_key=None) -> dict:
    body = {"text": "a cat"}
    if api_key is not None:
        body["api_key"] = api_key
    return body


def _post_workspace_model(client: TestClient, api_key=None):
    return client.post(
        "/infer/object_detection", json=_workspace_model_request(api_key)
    )


def test_request_without_key_is_refused_without_calling_the_appliance(
    appliance_client, fake_api
) -> None:
    response = _post_workspace_model(appliance_client)

    assert response.status_code == 401
    assert "Missing api_key" in response.json()["message"]
    assert sum(fake_api.workspace_calls.values()) == 0


def test_unknown_key_is_refused_and_the_refusal_is_cached(
    appliance_client, fake_api
) -> None:
    first = _post_workspace_model(appliance_client, KEY_BOGUS)
    second = _post_workspace_model(appliance_client, KEY_BOGUS)

    assert first.status_code == second.status_code == 401
    assert first.json()["message"] == "Unauthorized api_key."
    assert fake_api.workspace_calls[KEY_BOGUS] == 1
    assert sum(fake_api.model_data_calls.values()) == 0


def test_paused_key_is_refused_with_403(appliance_client, fake_api) -> None:
    response = _post_workspace_model(appliance_client, KEY_PAUSED)

    assert response.status_code == 403
    assert "paused" in response.json()["message"]
    assert fake_api.workspace_calls[KEY_PAUSED] == 1
    assert sum(fake_api.model_data_calls.values()) == 0


def test_core_model_route_does_not_ask_the_appliance_about_the_model(
    appliance_client, fake_api
) -> None:
    response = appliance_client.post(
        "/clip/embed_text", json=_core_model_request(KEY_A)
    )

    assert response.status_code == 200
    assert fake_api.workspace_calls[KEY_A] == 1
    assert sum(fake_api.model_data_calls.values()) == 0


def test_workspace_model_is_checked_per_key(appliance_client, fake_api) -> None:
    allowed = _post_workspace_model(appliance_client, KEY_A)
    refused = _post_workspace_model(appliance_client, KEY_B)

    assert allowed.status_code == 200
    assert refused.status_code == 401
    # Key B passed the middleware (a valid key) and failed the model check.
    assert fake_api.workspace_calls[KEY_B] == 1
    assert fake_api.model_data_calls[KEY_A] == 1
    assert fake_api.model_data_calls[KEY_B] == 1


def test_workspace_over_its_quota_is_refused_with_403(
    appliance_client, fake_api
) -> None:
    first = _post_workspace_model(appliance_client, KEY_OVER_QUOTA)
    second = appliance_client.post(
        "/clip/embed_text", json=_core_model_request(KEY_OVER_QUOTA)
    )

    assert first.status_code == second.status_code == 403
    assert first.json()["message"] == (
        "Workspace workspace-c has used its inference quota (120 of 100 credits). "
        "Ask your organization administrator to raise the workspace quota."
    )
    # The refusal is cached, and a core model is refused like a workspace model.
    assert fake_api.workspace_calls[KEY_OVER_QUOTA] == 1
    assert sum(fake_api.model_data_calls.values()) == 0


def test_appliance_outage_answers_503_and_caches_nothing(
    appliance_client, fake_api
) -> None:
    fake_api.outage = True

    first = _post_workspace_model(appliance_client, KEY_A)
    second = _post_workspace_model(appliance_client, KEY_A)

    assert first.status_code == second.status_code == 503
    assert first.headers["Retry-After"] == "5"
    assert "RoboflowAPIUnsuccessfulRequestError" in first.json()["message"]
    assert fake_api.workspace_calls[KEY_A] == 2

    fake_api.outage = False
    recovered = _post_workspace_model(appliance_client, KEY_A)

    assert recovered.status_code == 200
    assert fake_api.workspace_calls[KEY_A] == 3


def test_no_response_body_contains_an_api_key(appliance_client, fake_api) -> None:
    fake_api.outage = False
    responses = [_post_workspace_model(appliance_client, key) for key in ALL_KEYS]
    responses.append(_post_workspace_model(appliance_client))
    responses.append(
        appliance_client.post("/clip/embed_text", json=_core_model_request(KEY_B))
    )
    fake_api.outage = True
    responses.append(_post_workspace_model(appliance_client, "key-during-outage"))

    for response in responses:
        for key in (*ALL_KEYS, "key-during-outage"):
            assert key not in response.text
