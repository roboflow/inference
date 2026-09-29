from tests.unit_tests.legacy.conftest import FakeGateway, route_paths


def test_info(legacy_client, monkeypatch):
    monkeypatch.setattr("inference_server.configuration.INFERENCE_SERVER_ID", "srv-42")
    r = legacy_client(FakeGateway()).get("/info")
    assert (
        r.status_code == 200
        and r.json()["name"] == "Roboflow Inference Server"
        and r.json()["uuid"] == "srv-42"
    )


def test_openapi_metadata_matches_legacy(legacy_client):
    spec = legacy_client(FakeGateway()).get("/openapi.json").json()
    assert (
        spec["info"]["title"] == "Roboflow Inference Server"
        and spec["info"]["contact"]["email"] == "help@roboflow.com"
    )


def test_healthz_and_readiness(legacy_client):
    c = legacy_client(FakeGateway())
    assert c.get("/healthz").json() == {"status": "healthy"}
    assert c.get("/readiness").json() == {"status": "ready"}


def test_model_add_registry_remove_clear(legacy_client, fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gw = FakeGateway(model_info={"ds/1": {"actions": {"infer": {}}}})
    c = legacy_client(gw)
    r = c.post("/model/add", json={"model_id": "ds/1", "api_key": "k"})
    assert r.status_code == 200
    body = r.json()
    assert body["models"][0] == {
        "model_id": "ds/1",
        "task_type": "object-detection",
        "batch_size": None,
        "input_height": None,
        "input_width": None,
        "vram_bytes": None,
        "request_aliases": [],
        "request_paths": ["/model/add"],
    }
    assert {
        "total_vram_bytes",
        "gpu_memory_used",
        "gpu_memory_total",
        "torch_cuda_allocated",
        "torch_cuda_reserved",
        "torch_cuda_allocator_cache",
        "non_torch_gpu_memory",
    } <= set(body)
    assert ("load", "ds/1", "k") in gw.calls
    assert c.get("/model/registry").json()["models"][0]["model_id"] == "ds/1"
    assert c.post("/model/remove", json={"model_id": "ds/1"}).json()["models"] == []
    c.post("/model/add", json={"model_id": "ds/1", "api_key": "k"})
    assert c.post("/model/clear").json()["models"] == []
    assert c.get("/clear_cache").json() == "Cache Cleared"


def test_registry_lists_models_loaded_via_v2(legacy_client, fake_stat):
    gw = FakeGateway(
        model_info={"other/2": {"model_mro_names": ["ClassificationModel"]}}
    )
    c = legacy_client(gw)
    import asyncio

    asyncio.run(gw.load("other/2"))
    assert c.get("/model/registry").json()["models"][0] == {
        "model_id": "other/2",
        "task_type": "classification",
        "batch_size": None,
        "input_height": None,
        "input_width": None,
        "vram_bytes": None,
        "request_aliases": [],
        "request_paths": [],
    }


def test_control_plane_routes_registered_per_flags(monkeypatch):
    from fastapi import FastAPI

    from inference_server.legacy.router import include_legacy_routers

    def _paths(**flags):
        for name, value in flags.items():
            monkeypatch.setattr(f"inference_server.configuration.{name}", value)
        app = FastAPI()
        include_legacy_routers(app)
        return route_paths(app)

    assert {"/model/add", "/model/registry", "/clear_cache"} <= _paths(
        LEGACY_CONTROL_PLANE_ROUTES_ENABLED=True, GET_MODEL_REGISTRY_ENABLED=True
    )
    assert "/model/registry" not in _paths(
        LEGACY_CONTROL_PLANE_ROUTES_ENABLED=True, GET_MODEL_REGISTRY_ENABLED=False
    )
    disabled = _paths(
        LEGACY_CONTROL_PLANE_ROUTES_ENABLED=False, GET_MODEL_REGISTRY_ENABLED=True
    )
    assert not (
        {
            "/model/add",
            "/model/remove",
            "/model/clear",
            "/clear_cache",
            "/start/{dataset_id}/{version_id}",
        }
        & disabled
    )
    assert "/infer/object_detection" in disabled


def test_healthz_reports_cuda_failure(legacy_client, monkeypatch):
    monkeypatch.setattr(
        "inference_server.legacy.router.check_cuda_health", lambda: (False, "boom")
    )
    r = legacy_client(FakeGateway()).get("/healthz")
    assert r.status_code == 503 and r.json() == {
        "status": "unhealthy",
        "reason": "cuda_error",
    }


def test_start_legacy(legacy_client, fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    r = legacy_client(FakeGateway()).get("/start/ds/1?api_key=k")
    assert r.status_code == 200 and r.json()["status"] == 200


def test_unknown_model_is_404(legacy_client, fake_stat):
    r = legacy_client(FakeGateway()).post("/model/add", json={"model_id": "nope/1"})
    assert r.status_code == 404 and "message" in r.json()
