from __future__ import annotations

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
