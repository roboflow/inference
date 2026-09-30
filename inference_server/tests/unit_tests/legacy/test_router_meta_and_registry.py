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


def _wait_for_status(client, path, status_code):
    import time

    deadline = time.monotonic() + 5
    response = client.get(path)
    while response.status_code != status_code and time.monotonic() < deadline:
        time.sleep(0.01)
        response = client.get(path)
    return response


def test_readiness_after_failed_preload(legacy_client, monkeypatch):
    class _FailingGateway(FakeGateway):
        async def load(self, model_id, api_key="", timeout_s=None, pinned=True):
            self.calls.append(("load", model_id, api_key))
            return ("error", 3)

    monkeypatch.setenv("INFERENCE_PRELOAD_MODELS", "ds/1")
    monkeypatch.setenv("PINNED_MODELS", "ds/2")
    gw = _FailingGateway()
    c = legacy_client(gw)

    r = _wait_for_status(c, "/readiness", 200)
    assert r.json() == {"status": "ready"}
    r = _wait_for_status(c, "/v2/server/ready", 200)
    assert r.json() == {"ready": True}
    assert {call[1] for call in gw.calls if call[0] == "load"} == {"ds/1", "ds/2"}


def test_readiness_waits_for_preload_to_finish(legacy_client, monkeypatch):
    import threading

    release = threading.Event()

    class _BlockingGateway(FakeGateway):
        async def load(self, model_id, api_key="", timeout_s=None, pinned=True):
            import asyncio

            while not release.is_set():
                await asyncio.sleep(0.01)
            return await super().load(model_id, api_key, timeout_s, pinned)

    monkeypatch.setenv("INFERENCE_PRELOAD_MODELS", "ds/1")
    c = legacy_client(_BlockingGateway())

    r = c.get("/readiness")
    assert r.status_code == 503 and r.json() == {"status": "not ready"}
    r = c.get("/v2/server/ready")
    assert r.status_code == 503
    assert r.json() == {
        "error_code": "MODEL_NOT_READY",
        "description": "model ds/1 not ready",
        "actionable_follow_up": "wait for model to finish loading",
    }

    release.set()
    assert _wait_for_status(c, "/readiness", 200).json() == {"status": "ready"}
    assert _wait_for_status(c, "/v2/server/ready", 200).json() == {"ready": True}


def test_v2_ready_without_pending_id_reports_preload_not_finished(
    legacy_client, monkeypatch
):
    import threading

    release = threading.Event()

    class _SlowReturnGateway(FakeGateway):
        async def load(self, model_id, api_key="", timeout_s=None, pinned=True):
            import asyncio

            result = await super().load(model_id, api_key, timeout_s, pinned)
            while not release.is_set():
                await asyncio.sleep(0.01)
            return result

    monkeypatch.setenv("INFERENCE_PRELOAD_MODELS", "ds/1")
    gw = _SlowReturnGateway()
    c = legacy_client(gw)
    try:
        import time

        deadline = time.monotonic() + 5
        while "ds/1" not in gw.loaded and time.monotonic() < deadline:
            time.sleep(0.01)
        r = c.get("/v2/server/ready")

        assert r.status_code == 503
        assert r.json() == {
            "error_code": "MODEL_NOT_READY",
            "description": "startup preload not finished",
            "actionable_follow_up": "wait for model to finish loading",
        }
    finally:
        release.set()


def test_model_add_does_not_pin(legacy_client, fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    fake_stat["ds/2"] = ("object-detection", "infer")
    gw = FakeGateway()
    c = legacy_client(gw)

    assert (
        c.post("/model/add", json={"model_id": "ds/1", "api_key": "k"}).status_code
        == 200
    )
    assert c.get("/start/ds/2?api_key=k").status_code == 200

    assert ("load", "ds/1", "k") in gw.calls
    assert ("load", "ds/2", "k") in gw.calls
    assert gw.pinned == []


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
