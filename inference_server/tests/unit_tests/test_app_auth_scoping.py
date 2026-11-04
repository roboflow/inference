from __future__ import annotations

import pytest
from fastapi import FastAPI, Response
from fastapi.testclient import TestClient

import inference_server.app as app_mod


def _app_with_probe():
    inner = FastAPI()

    @inner.post("/v2/models/infer")
    @inner.get("/v2/models")
    @inner.get("/infer/object_detection")
    @inner.post("/model/add")
    @inner.get("/anything/else")
    async def _ok():
        return Response(content=b"ok")

    inner.add_middleware(app_mod._AuthMiddleware)
    return inner


def test_v2_path_without_bearer_is_rejected():
    client = TestClient(_app_with_probe())
    assert client.post("/v2/models/infer").status_code == 401


def test_v2_control_plane_still_gated(monkeypatch):
    monkeypatch.setattr(app_mod._cfg, "ENABLE_CONTROL_PLANE_ROUTES", False)
    assert TestClient(_app_with_probe()).get("/v2/models").status_code == 403


def test_non_v2_paths_pass_through_without_bearer():
    client = TestClient(_app_with_probe())
    assert client.get("/infer/object_detection").status_code == 200
    assert client.post("/model/add").status_code == 200
    assert client.get("/anything/else").status_code == 200


def test_v2_path_under_root_path_without_bearer_is_rejected():
    client = TestClient(_app_with_probe(), root_path="/svc")
    assert client.post("/svc/v2/models/infer").status_code == 401


def test_v2_control_plane_under_root_path_still_gated(monkeypatch):
    monkeypatch.setattr(app_mod._cfg, "ENABLE_CONTROL_PLANE_ROUTES", False)
    client = TestClient(_app_with_probe(), root_path="/svc")
    assert client.get("/svc/v2/models").status_code == 403


def test_request_for_the_exact_root_path_is_the_mount_root():
    scope = {"type": "http", "path": "/v2/models", "root_path": "/v2/models"}

    assert app_mod._route_path(scope) == ""


def _local_callers_client(monkeypatch, allow_origins=("*",), builder=False):
    monkeypatch.setattr(app_mod._cfg, "ALLOW_ORIGINS", list(allow_origins))
    monkeypatch.setattr(app_mod._cfg, "ENABLE_BUILDER", builder)
    monkeypatch.setattr(app_mod._cfg, "BUILDER_ORIGIN", "https://app.roboflow.com")
    inner = FastAPI()

    @inner.api_route("/workflows/run", methods=["POST", "OPTIONS"])
    async def _ok():
        return Response(content=b"ok")

    inner.add_middleware(app_mod._LocalCallersOnlyMiddleware)
    return TestClient(inner, base_url="http://localhost:9001")


@pytest.mark.parametrize(
    "headers,allowed",
    [
        ({}, True),
        ({"Host": "127.0.0.1:9001"}, True),
        ({"Host": "[::1]:9001"}, True),
        ({"Host": "localhost"}, True),
        # DNS rebinding: an attacker's name that resolves to 127.0.0.1.
        ({"Host": "rebind.evil.example:9001"}, False),
        ({"Host": "localhost.evil.example"}, False),
        ({"Host": "192.168.1.10:9001"}, False),
        ({"Origin": "http://localhost:3000"}, True),
        ({"Origin": "http://127.0.0.1:9001"}, True),
        ({"Origin": "https://evil.example"}, False),
        ({"Origin": "null"}, False),
        ({"Origin": "https://app.roboflow.com"}, False),
        ({"Sec-Fetch-Site": "cross-site"}, False),
        ({"Sec-Fetch-Site": "same-origin"}, True),
        ({"Sec-Fetch-Site": "none"}, True),
        # A relaying proxy hides the real caller.
        ({"Forwarded": "for=203.0.113.7"}, False),
        ({"X-Forwarded-For": "203.0.113.7"}, False),
        ({"X-Forwarded-Host": "evil.example"}, False),
        ({"X-Real-IP": "203.0.113.7"}, False),
        ({"Via": "1.1 proxy"}, False),
    ],
)
def test_local_callers_only_gate(monkeypatch, headers, allowed):
    client = _local_callers_client(monkeypatch)

    response = client.post("/workflows/run", headers=headers)

    if allowed:
        assert response.status_code == 200, response.text
    else:
        assert response.status_code == 403
        assert response.json()["message"] == app_mod.LOCAL_CALLERS_ONLY_MESSAGE


def test_local_callers_only_gate_refuses_cross_site_preflight(monkeypatch):
    client = _local_callers_client(monkeypatch)

    response = client.options("/workflows/run", headers={"Origin": "https://x.example"})

    assert response.status_code == 403


@pytest.mark.parametrize(
    "allow_origins,builder,origin,allowed",
    [
        (("https://trusted.example",), False, "https://trusted.example", True),
        (("https://trusted.example",), False, "https://other.example", False),
        (("*",), False, "https://anything.example", False),
        (("*",), True, "https://app.roboflow.com", True),
        (("*",), False, "https://app.roboflow.com", False),
    ],
)
def test_local_callers_only_gate_trusts_configured_origins(
    monkeypatch, allow_origins, builder, origin, allowed
):
    client = _local_callers_client(monkeypatch, allow_origins, builder)

    response = client.post("/workflows/run", headers={"Origin": origin})

    assert (response.status_code == 200) is allowed
