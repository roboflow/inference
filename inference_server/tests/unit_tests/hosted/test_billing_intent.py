import json
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from inference_server import configuration
from inference_server.framework import model_stat
from inference_server.hosted.assume_identity import enforce_credits_verification
from inference_server.hosted.common import BillingIntentMiddleware
from tests.unit_tests.legacy.conftest import FakeGateway

CREDITS_HEADER = "x-enforce-credits-verification"


@pytest.fixture
def client():
    inner = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @inner.api_route("/{full_path:path}", methods=["GET", "POST", "PUT"])
    async def _probe(request: Request):
        body = await request.body()
        return JSONResponse(
            {"body": body.decode(), "enforce": enforce_credits_verification.get()}
        )

    inner.add_middleware(BillingIntentMiddleware)
    return TestClient(inner)


@pytest.fixture
def secret(monkeypatch):
    monkeypatch.setattr(configuration, "ROBOFLOW_SERVICE_SECRET", "s3cret")


def test_enforced_by_default(client, secret):
    assert client.get("/x").json()["enforce"] is True
    assert client.get("/x?countinference=false").json()["enforce"] is True
    assert (
        client.get("/x?countinference=false&service_secret=nope").json()["enforce"]
        is True
    )
    assert (
        client.get("/x?countinference=true&service_secret=s3cret").json()["enforce"]
        is True
    )


def test_valid_non_billable_intent_from_query(client, secret):
    response = client.put("/x?countinference=false&service_secret=s3cret")

    assert response.json()["enforce"] is False


def test_valid_non_billable_intent_from_body_and_body_replayed(client, secret):
    payload = {"countinference": False, "service_secret": "s3cret", "v": 1}

    response = client.post("/x", json=payload)

    assert response.json()["enforce"] is False
    assert json.loads(response.json()["body"]) == payload


def test_body_wins_over_query(client, secret):
    response = client.post(
        "/x?countinference=false&service_secret=s3cret",
        json={"countinference": True},
    )

    assert response.json()["enforce"] is True


def test_unencodable_secret_is_rejected(client, secret):
    response = client.post(
        "/x",
        content=b'{"countinference": false, "service_secret": "\\ud800"}',
        headers={"Content-Type": "application/json"},
    )

    assert response.json()["enforce"] is True


def test_without_configured_secret_body_is_untouched(client, monkeypatch):
    monkeypatch.setattr(configuration, "ROBOFLOW_SERVICE_SECRET", None)
    seen = []

    async def _json(self):
        seen.append(True)
        return {}

    monkeypatch.setattr("inference_server.hosted.common.HostedRequest.json", _json)
    response = client.post(
        "/x", json={"countinference": False, "service_secret": "anything"}
    )

    assert response.json()["enforce"] is True
    assert seen == []


def test_contextvar_reset_after_request(client, secret):
    client.get("/x?countinference=false&service_secret=s3cret")

    assert enforce_credits_verification.get() is True


@pytest.fixture
def stat_recorder(monkeypatch):
    calls = []

    def _metadata(model_id, api_key=None, extra_headers=None, proxy_url_builder=None):
        calls.append(extra_headers)
        return MagicMock(task_type="object-detection")

    monkeypatch.setattr(model_stat, "get_one_page_of_model_metadata", _metadata)
    model_stat._reset_cache_for_tests()
    yield calls
    model_stat._reset_cache_for_tests()


def test_legacy_route_omits_credits_header_for_valid_non_billable_request(
    legacy_client, stat_recorder, monkeypatch
):
    monkeypatch.setattr(configuration, "ENFORCE_CREDITS_VERIFICATION", True)
    monkeypatch.setattr(configuration, "ROBOFLOW_SERVICE_SECRET", "s3cret")
    client = legacy_client(FakeGateway())

    client.get(
        "/ds/1?api_key=k&countinference=false&service_secret=s3cret"
        "&image=http://example.com/a.jpg"
    )
    refused = client.get(
        "/ds/1?api_key=k&countinference=false&service_secret=wrong"
        "&image=http://example.com/a.jpg"
    )
    client.post(
        "/infer/object_detection?countinference=false&service_secret=wrong",
        json={
            "model_id": "ds/1",
            "api_key": "k",
            "image": {"type": "base64", "value": "x"},
        },
    )

    assert refused.status_code == 500
    assert stat_recorder == [None, {CREDITS_HEADER: "true"}]


def test_v2_route_omits_credits_header_for_valid_non_billable_request(
    legacy_client, stat_recorder, monkeypatch
):
    monkeypatch.setattr(configuration, "ENFORCE_CREDITS_VERIFICATION", True)
    monkeypatch.setattr(configuration, "ROBOFLOW_SERVICE_SECRET", "s3cret")
    monkeypatch.setattr(
        "inference_server.app.validate_api_key",
        _async_return((True, "ws")),
    )
    client = legacy_client(FakeGateway())

    client.post(
        "/v2/models/infer?model_id=ds/1&countinference=false&service_secret=s3cret",
        headers={"Authorization": "Bearer k"},
        json={"model_id": "ds/1", "images": []},
    )
    client.post(
        "/v2/models/infer?model_id=ds/1&countinference=false&service_secret=wrong",
        headers={"Authorization": "Bearer k"},
        json={"model_id": "ds/1", "images": []},
    )

    assert stat_recorder == [None, {CREDITS_HEADER: "true"}]


def _async_return(value):
    async def _call(*args, **kwargs):
        return value

    return _call
