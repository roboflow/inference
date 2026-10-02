import asyncio

import pytest

from inference_server.gateway import ModelManagerGateway
from tests.unit_tests.legacy.conftest import FakeGateway, route_paths
from tests.unit_tests.legacy.test_router_infer import (
    LOAD_FAILURE_MATRIX,
    FailingLoadManager,
)


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


_DESCRIBED_COCO = {
    "coco/3": {
        "actions": {"infer": {}},
        "class_names": ["cat"],
        "model_mro_names": ["ObjectDetectionModel"],
        "batch_size": 4,
        "max_batch_size": 8,
        "input_height": 640,
        "input_width": 480,
        "vram_bytes": 250,
    }
}


def _infer_through_alias(c):
    from tests.unit_tests.legacy.test_router_infer import _det, _jpeg_b64

    c.app.state.model_manager.predictions[("coco/3", "infer")] = _det()
    response = c.post(
        "/infer/object_detection",
        json={
            "model_id": "yolov8n-640",
            "api_key": "k",
            "image": {"type": "base64", "value": _jpeg_b64()},
        },
    )
    return response


def test_registry_rows_keyed_by_requested_alias(legacy_client, fake_stat):
    fake_stat["coco/3"] = ("object-detection", "infer")
    c = legacy_client(FakeGateway(model_info=_DESCRIBED_COCO))

    assert _infer_through_alias(c).status_code == 200

    body = c.get("/model/registry").json()
    assert body["models"] == [
        {
            "model_id": "yolov8n-640",
            "task_type": "object-detection",
            "batch_size": None,
            "input_height": 640,
            "input_width": 480,
            "vram_bytes": 250,
            "request_aliases": ["coco/3"],
            "request_paths": ["/infer/object_detection"],
        }
    ]
    assert body["total_vram_bytes"] == 250


def test_registry_rows_for_alias_and_canonical_keep_their_own_paths(
    legacy_client, fake_stat
):
    fake_stat["coco/3"] = ("object-detection", "infer")
    c = legacy_client(FakeGateway(model_info=_DESCRIBED_COCO))

    _infer_through_alias(c)
    c.get("/start/coco/3?api_key=k")
    body = c.get("/model/registry").json()

    assert [
        (m["model_id"], m["request_aliases"], m["request_paths"])
        for m in body["models"]
    ] == [
        ("coco/3", [], ["/start/coco/3"]),
        ("yolov8n-640", ["coco/3"], ["/infer/object_detection"]),
    ]
    assert all(m["vram_bytes"] == 250 for m in body["models"])
    assert body["total_vram_bytes"] == 250


def test_registry_row_for_canonical_request_has_no_aliases(legacy_client, fake_stat):
    fake_stat["coco/3"] = ("object-detection", "infer")
    c = legacy_client(FakeGateway(model_info=_DESCRIBED_COCO))

    body = c.post("/model/add", json={"model_id": "coco/3", "api_key": "k"}).json()

    assert [
        (m["model_id"], m["request_aliases"], m["request_paths"])
        for m in body["models"]
    ] == [("coco/3", [], ["/model/add"])]


def test_registry_reports_null_vram_when_not_measured(legacy_client, fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    c = legacy_client(FakeGateway(model_info={"ds/1": {"vram_bytes": None}}))

    body = c.post("/model/add", json={"model_id": "ds/1", "api_key": "k"}).json()

    assert body["models"][0]["vram_bytes"] is None
    assert body["total_vram_bytes"] is None


def test_registry_row_for_preloaded_alias_before_any_request(
    legacy_client, fake_stat, monkeypatch
):
    fake_stat["coco/3"] = ("object-detection", "infer")
    monkeypatch.setenv("INFERENCE_PRELOAD_MODELS", "yolov8n-640")
    gw = FakeGateway(model_info=_DESCRIBED_COCO)
    c = legacy_client(gw)
    _wait_for_status(c, "/readiness", 200)

    assert c.get("/model/registry").json()["models"] == [
        {
            "model_id": "yolov8n-640",
            "task_type": "object-detection",
            "batch_size": None,
            "input_height": 640,
            "input_width": 480,
            "vram_bytes": 250,
            "request_aliases": ["coco/3"],
            "request_paths": [],
        }
    ]

    removed = c.post("/model/remove", json={"model_id": "coco/3"}).json()

    assert _registry_rows(removed) == [("yolov8n-640", ["coco/3"], [])]
    assert _unloads(gw) == []


def test_registry_after_clear_drops_preloaded_alias(
    legacy_client, fake_stat, monkeypatch
):
    fake_stat["coco/3"] = ("object-detection", "infer")
    monkeypatch.setenv("INFERENCE_PRELOAD_MODELS", "yolov8n-640")
    c = legacy_client(FakeGateway(model_info=_DESCRIBED_COCO))
    _wait_for_status(c, "/readiness", 200)

    assert c.post("/model/clear").json()["models"] == []
    body = c.post("/model/add", json={"model_id": "coco/3", "api_key": "k"}).json()

    assert [m["model_id"] for m in body["models"]] == ["coco/3"]


def _unloads(gw):
    return [call for call in gw.calls if call[0] == "unload"]


class _LoadCountingGateway(FakeGateway):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fresh_loads = []

    async def ensure_loaded(self, model_id, instance="", api_key="", device=""):
        if model_id not in self.loaded:
            self.fresh_loads.append(model_id)
        return await super().ensure_loaded(model_id, instance, api_key, device)


def _client_with_alias_and_canonical_rows(legacy_client, fake_stat):
    fake_stat["coco/3"] = ("object-detection", "infer")
    gw = _LoadCountingGateway(model_info=_DESCRIBED_COCO)
    c = legacy_client(gw)
    _infer_through_alias(c)
    c.get("/start/coco/3?api_key=k")
    return c, gw


def test_remove_by_alias_drops_only_the_canonical_row(legacy_client, fake_stat):
    c, gw = _client_with_alias_and_canonical_rows(legacy_client, fake_stat)

    body = c.post("/model/remove", json={"model_id": "yolov8n-640"}).json()

    assert _registry_rows(body) == [
        ("yolov8n-640", ["coco/3"], ["/infer/object_detection"])
    ]
    assert _unloads(gw) == []


def test_remove_by_canonical_drops_only_the_canonical_row(legacy_client, fake_stat):
    c, gw = _client_with_alias_and_canonical_rows(legacy_client, fake_stat)

    body = c.post("/model/remove", json={"model_id": "coco/3"}).json()

    assert _registry_rows(body) == [
        ("yolov8n-640", ["coco/3"], ["/infer/object_detection"])
    ]
    assert _unloads(gw) == []


def test_model_kept_after_canonical_row_removal_serves_alias_and_canonical(
    legacy_client, fake_stat
):
    from tests.unit_tests.legacy.test_router_infer import _det, _jpeg_b64

    c, gw = _client_with_alias_and_canonical_rows(legacy_client, fake_stat)
    gw.predictions[("coco/3", "infer")] = _det()
    c.post("/model/remove", json={"model_id": "coco/3"})

    r = c.post(
        "/infer/object_detection",
        json={
            "model_id": "yolov8n-640",
            "api_key": "k",
            "image": {"type": "base64", "value": _jpeg_b64()},
        },
    )

    assert r.status_code == 200
    assert gw.fresh_loads == ["coco/3"]
    body = c.post("/model/add", json={"model_id": "coco/3", "api_key": "k"}).json()
    assert _registry_rows(body) == [
        ("coco/3", [], ["/model/add"]),
        ("yolov8n-640", ["coco/3"], ["/infer/object_detection"]),
    ]
    assert gw.fresh_loads == ["coco/3"]


def test_remove_when_only_aliases_were_requested_changes_nothing(
    legacy_client, fake_stat
):
    fake_stat["coco/3"] = ("object-detection", "infer")
    gw = FakeGateway(model_info=_DESCRIBED_COCO)
    c = legacy_client(gw)
    _infer_through_alias(c)

    body = c.post("/model/remove", json={"model_id": "yolov8n-640"}).json()

    assert _registry_rows(body) == [
        ("yolov8n-640", ["coco/3"], ["/infer/object_detection"])
    ]
    assert _unloads(gw) == []


def test_remove_by_canonical_unloads_a_model_requested_only_canonically(
    legacy_client, fake_stat
):
    fake_stat["coco/3"] = ("object-detection", "infer")
    gw = FakeGateway(model_info=_DESCRIBED_COCO)
    c = legacy_client(gw)
    c.post("/model/add", json={"model_id": "coco/3", "api_key": "k"})

    assert c.post("/model/remove", json={"model_id": "coco/3"}).json()["models"] == []
    assert _unloads(gw) == [("unload", "coco/3")]

    body = c.post("/model/add", json={"model_id": "yolov8n-640", "api_key": "k"}).json()

    assert _registry_rows(body) == [("coco/3", [], ["/model/add"])]

    c.post("/model/clear")
    body = c.post("/model/add", json={"model_id": "coco/3", "api_key": "k"}).json()

    assert _registry_rows(body) == [("coco/3", [], ["/model/add"])]


def test_remove_by_alias_unloads_a_model_requested_only_canonically(
    legacy_client, fake_stat
):
    fake_stat["coco/3"] = ("object-detection", "infer")
    gw = FakeGateway(model_info=_DESCRIBED_COCO)
    c = legacy_client(gw)
    c.post("/model/add", json={"model_id": "coco/3", "api_key": "k"})

    body = c.post("/model/remove", json={"model_id": "yolov8n-640"}).json()

    assert body["models"] == []
    assert _unloads(gw) == [("unload", "coco/3")]


def test_remove_unloads_a_model_loaded_only_through_v2(legacy_client, fake_stat):
    import asyncio

    gw = FakeGateway(model_info=_DESCRIBED_COCO)
    c = legacy_client(gw)
    asyncio.run(gw.load("coco/3"))

    body = c.post("/model/remove", json={"model_id": "coco/3"}).json()

    assert body["models"] == []
    assert _unloads(gw) == [("unload", "coco/3")]


def test_remove_of_a_model_that_is_not_loaded_changes_nothing(legacy_client, fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gw = FakeGateway()
    c = legacy_client(gw)
    c.post("/model/add", json={"model_id": "ds/1", "api_key": "k"})

    body = c.post("/model/remove", json={"model_id": "coco/3"}).json()

    assert _registry_rows(body) == [("ds/1", [], ["/model/add"])]
    assert _unloads(gw) == []


def test_remove_drops_a_preloaded_canonical_row_and_keeps_alias_rows(
    legacy_client, fake_stat, monkeypatch
):
    fake_stat["coco/3"] = ("object-detection", "infer")
    monkeypatch.setenv("INFERENCE_PRELOAD_MODELS", "coco/3")
    gw = FakeGateway(model_info=_DESCRIBED_COCO)
    c = legacy_client(gw)
    _wait_for_status(c, "/readiness", 200)
    _infer_through_alias(c)

    body = c.post("/model/remove", json={"model_id": "coco/3"}).json()

    assert _registry_rows(body) == [
        ("yolov8n-640", ["coco/3"], ["/infer/object_detection"])
    ]
    assert _registry_rows(c.get("/model/registry").json()) == _registry_rows(body)
    assert _unloads(gw) == []


def test_remove_of_a_preloaded_canonical_id_unloads(
    legacy_client, fake_stat, monkeypatch
):
    monkeypatch.setenv("INFERENCE_PRELOAD_MODELS", "coco/3")
    gw = FakeGateway(model_info=_DESCRIBED_COCO)
    c = legacy_client(gw)
    _wait_for_status(c, "/readiness", 200)

    assert _registry_rows(c.get("/model/registry").json()) == [("coco/3", [], [])]
    assert c.post("/model/remove", json={"model_id": "coco/3"}).json()["models"] == []
    assert _unloads(gw) == [("unload", "coco/3")]


class _StampingGateway(FakeGateway):
    def __init__(self, clock, **kwargs):
        super().__init__(**kwargs)
        self.clock = clock

    async def ensure_loaded(self, model_id, instance="", api_key="", device=""):
        fresh = model_id not in self.loaded
        result = await super().ensure_loaded(model_id, instance, api_key, device)
        if fresh:
            self.loaded[model_id]["loaded_monotonic"] = self.clock["now"]
        return result

    async def load(self, model_id, api_key="", timeout_s=None, pinned=True):
        fresh = model_id not in self.loaded
        result = await super().load(model_id, api_key, timeout_s, pinned)
        if fresh:
            self.loaded[model_id]["loaded_monotonic"] = self.clock["now"]
        return result


def _clock(monkeypatch, now):
    from inference_server.legacy import bridge as bridge_mod

    clock = {"now": now}
    monkeypatch.setattr(bridge_mod, "_clock", lambda: clock["now"])
    return clock


def _registry_rows(body):
    return [
        (m["model_id"], m["request_aliases"], m["request_paths"])
        for m in body["models"]
    ]


def test_registry_after_v2_unload_and_load_keeps_only_new_request(
    legacy_client, fake_stat, monkeypatch
):
    from unittest.mock import AsyncMock

    import inference_server.app as app_mod

    monkeypatch.setattr(app_mod._cfg, "ENABLE_CONTROL_PLANE_ROUTES", True)
    monkeypatch.setattr(
        app_mod, "validate_api_key", AsyncMock(return_value=(True, "ws-1"))
    )
    fake_stat["coco/3"] = ("object-detection", "infer")
    clock = _clock(monkeypatch, 10.0)
    c = legacy_client(_StampingGateway(clock, model_info=_DESCRIBED_COCO))
    _infer_through_alias(c)

    clock["now"] = 20.0
    headers = {"Authorization": "Bearer k"}
    assert (
        c.post("/v2/models/unload?model_id=coco/3", headers=headers).status_code == 200
    )
    assert c.post("/v2/models/load?model_id=coco/3", headers=headers).status_code == 200
    clock["now"] = 30.0
    c.get("/start/coco/3?api_key=k")

    assert _registry_rows(c.get("/model/registry").json()) == [
        ("coco/3", [], ["/start/coco/3"])
    ]


def test_registry_drops_paths_recorded_before_reload_of_a_current_id(
    legacy_client, fake_stat, monkeypatch
):
    fake_stat["coco/3"] = ("object-detection", "infer")
    clock = _clock(monkeypatch, 10.0)
    gw = _StampingGateway(clock, model_info=_DESCRIBED_COCO)
    c = legacy_client(gw)
    _infer_through_alias(c)
    gw.loaded.pop("coco/3")

    clock["now"] = 20.0
    c.get("/start/coco/3?api_key=k")
    bridge = c.app.state.legacy_bridge
    bridge.record_request(
        bridge._routes["coco/3"], "yolov8n-640", "/infer/classification"
    )

    assert _registry_rows(c.get("/model/registry").json()) == [
        ("coco/3", [], ["/start/coco/3"]),
        ("yolov8n-640", [], ["/infer/classification"]),
    ]


def test_registry_keeps_requests_joining_a_loaded_model(
    legacy_client, fake_stat, monkeypatch
):
    fake_stat["coco/3"] = ("object-detection", "infer")
    clock = _clock(monkeypatch, 10.0)
    c = legacy_client(_StampingGateway(clock, model_info=_DESCRIBED_COCO))
    _infer_through_alias(c)
    clock["now"] = 20.0
    c.get("/start/coco/3?api_key=k")

    assert _registry_rows(c.get("/model/registry").json()) == [
        ("coco/3", [], ["/start/coco/3"]),
        ("yolov8n-640", ["coco/3"], ["/infer/object_detection"]),
    ]


def test_registry_keeps_a_request_recorded_while_describing(
    legacy_client, fake_stat, monkeypatch
):
    fake_stat["coco/3"] = ("object-detection", "infer")
    clock = _clock(monkeypatch, 10.0)
    hooks = []

    class _RacingGateway(_StampingGateway):
        async def stats(self):
            snapshot = await super().stats()
            while hooks:
                hooks.pop()()
            return snapshot

    gw = _RacingGateway(clock, model_info=_DESCRIBED_COCO)
    c = legacy_client(gw)
    _infer_through_alias(c)
    gw.loaded.pop("coco/3")
    bridge = c.app.state.legacy_bridge

    def _reload_and_record():
        clock["now"] = 20.0
        gw.loaded["coco/3"] = dict(_DESCRIBED_COCO["coco/3"], loaded_monotonic=20.0)
        bridge.record_request(bridge._routes["coco/3"], "coco/3", "/start/coco/3")

    hooks.append(_reload_and_record)

    assert c.get("/model/registry").json()["models"] == []
    assert _registry_rows(c.get("/model/registry").json()) == [
        ("coco/3", [], ["/start/coco/3"])
    ]


def test_registry_filters_old_requests_after_resolve_outside_a_request(
    legacy_client, fake_stat, monkeypatch
):
    import asyncio

    fake_stat["coco/3"] = ("object-detection", "infer")
    clock = _clock(monkeypatch, 10.0)
    gw = _StampingGateway(clock, model_info=_DESCRIBED_COCO)
    c = legacy_client(gw)
    _infer_through_alias(c)
    gw.loaded.pop("coco/3")

    clock["now"] = 20.0
    asyncio.run(c.app.state.legacy_bridge.resolve("coco/3", "k"))

    assert _registry_rows(c.get("/model/registry").json()) == [("coco/3", [], [])]


def test_registry_drops_evicted_preloaded_alias_on_reload(
    legacy_client, fake_stat, monkeypatch
):
    fake_stat["coco/3"] = ("object-detection", "infer")
    monkeypatch.setenv("INFERENCE_PRELOAD_MODELS", "yolov8n-640")
    clock = _clock(monkeypatch, 10.0)
    gw = _StampingGateway(clock, model_info=_DESCRIBED_COCO)
    c = legacy_client(gw)
    _wait_for_status(c, "/readiness", 200)

    assert _registry_rows(c.get("/model/registry").json()) == [
        ("yolov8n-640", ["coco/3"], [])
    ]
    gw.loaded.pop("coco/3")
    clock["now"] = 20.0
    body = c.post("/model/add", json={"model_id": "coco/3", "api_key": "k"}).json()

    assert _registry_rows(body) == [("coco/3", [], ["/model/add"])]


def test_registry_reports_every_request_without_load_stamp(
    legacy_client, fake_stat, monkeypatch
):
    fake_stat["coco/3"] = ("object-detection", "infer")
    clock = _clock(monkeypatch, 10.0)
    gw = FakeGateway(model_info=_DESCRIBED_COCO)
    c = legacy_client(gw)
    _infer_through_alias(c)
    gw.loaded.pop("coco/3")
    clock["now"] = 20.0
    c.get("/start/coco/3?api_key=k")

    assert _registry_rows(c.get("/model/registry").json()) == [
        ("coco/3", [], ["/start/coco/3"]),
        ("yolov8n-640", ["coco/3"], ["/infer/object_detection"]),
    ]


def test_registry_counts_requests_recorded_at_load_time_as_current(
    legacy_client, fake_stat, monkeypatch
):
    fake_stat["coco/3"] = ("object-detection", "infer")
    clock = _clock(monkeypatch, 10.0)
    gw = _StampingGateway(clock, model_info=_DESCRIBED_COCO)
    c = legacy_client(gw)
    _infer_through_alias(c)

    assert gw.loaded["coco/3"]["loaded_monotonic"] == 10.0
    assert _registry_rows(c.get("/model/registry").json()) == [
        ("yolov8n-640", ["coco/3"], ["/infer/object_detection"])
    ]


def test_model_add_through_alias_records_the_canonical_row(legacy_client, fake_stat):
    fake_stat["coco/3"] = ("object-detection", "infer")
    c = legacy_client(FakeGateway(model_info=_DESCRIBED_COCO))

    body = c.post("/model/add", json={"model_id": "yolov8n-640", "api_key": "k"}).json()

    assert _registry_rows(body) == [("coco/3", [], ["/model/add"])]


def test_model_add_through_alias_then_remove_by_alias_unloads(legacy_client, fake_stat):
    fake_stat["coco/3"] = ("object-detection", "infer")
    gw = FakeGateway(model_info=_DESCRIBED_COCO)
    c = legacy_client(gw)
    c.post("/model/add", json={"model_id": "yolov8n-640", "api_key": "k"})

    body = c.post("/model/remove", json={"model_id": "yolov8n-640"}).json()

    assert body["models"] == []
    assert _unloads(gw) == [("unload", "coco/3")]


def test_model_add_through_alias_then_inference_through_alias_gives_two_rows(
    legacy_client, fake_stat
):
    fake_stat["coco/3"] = ("object-detection", "infer")
    c = legacy_client(FakeGateway(model_info=_DESCRIBED_COCO))
    c.post("/model/add", json={"model_id": "yolov8n-640", "api_key": "k"})

    _infer_through_alias(c)

    assert _registry_rows(c.get("/model/registry").json()) == [
        ("coco/3", [], ["/model/add"]),
        ("yolov8n-640", ["coco/3"], ["/infer/object_detection"]),
    ]


class _EvictBeforeInferenceGateway(_StampingGateway):
    def __init__(self, clock, *, reload_result=None, **kwargs):
        super().__init__(clock, **kwargs)
        self.reload_result = reload_result
        self.ensure_calls = 0

    async def ensure_loaded(self, model_id, instance="", api_key="", device=""):
        self.ensure_calls += 1
        if self.ensure_calls == 2:
            self.loaded.pop(model_id)
            self.clock["now"] = 20.0
            if self.reload_result is not None:
                return self.reload_result
        return await super().ensure_loaded(model_id, instance, api_key, device)


def test_registry_keeps_a_request_whose_model_reloads_during_the_request(
    legacy_client, fake_stat, monkeypatch
):
    fake_stat["coco/3"] = ("object-detection", "infer")
    clock = _clock(monkeypatch, 10.0)
    gw = _EvictBeforeInferenceGateway(clock, model_info=_DESCRIBED_COCO)
    c = legacy_client(gw)

    assert _infer_through_alias(c).status_code == 200

    assert gw.loaded["coco/3"]["loaded_monotonic"] == 20.0
    assert _registry_rows(c.get("/model/registry").json()) == [
        ("yolov8n-640", ["coco/3"], ["/infer/object_detection"])
    ]


def test_registry_keeps_a_request_whose_model_reloads_inside_inference(
    legacy_client, fake_stat, monkeypatch
):
    fake_stat["coco/3"] = ("object-detection", "infer")
    clock = _clock(monkeypatch, 10.0)

    class _ReloadDuringInferGateway(_StampingGateway):
        async def infer(self, **kwargs):
            self.clock["now"] = 20.0
            self.loaded["coco/3"] = dict(
                _DESCRIBED_COCO["coco/3"], loaded_monotonic=20.0
            )
            return await super().infer(**kwargs)

    c = legacy_client(_ReloadDuringInferGateway(clock, model_info=_DESCRIBED_COCO))

    assert _infer_through_alias(c).status_code == 200

    assert _registry_rows(c.get("/model/registry").json()) == [
        ("yolov8n-640", ["coco/3"], ["/infer/object_detection"])
    ]


def test_ensure_loaded_refreshes_the_current_request_record(fake_stat, monkeypatch):
    import asyncio

    from inference_server.legacy.bridge import LegacyModelBridge

    fake_stat["coco/3"] = ("object-detection", "infer")
    clock = _clock(monkeypatch, 10.0)
    gw = _StampingGateway(clock, model_info=_DESCRIBED_COCO)
    bridge = LegacyModelBridge(gw)

    async def _request():
        route = await bridge.resolve("yolov8n-640", "k")
        bridge.record_request(route, "yolov8n-640", "/infer/object_detection")
        gw.loaded.pop("coco/3")
        clock["now"] = 20.0
        await bridge.ensure_loaded(route, "k")

        return route

    route = asyncio.run(_request())

    assert route.requested_at == {"yolov8n-640": 20.0}
    assert route.request_paths_by_id == {
        "yolov8n-640": {"/infer/object_detection": 20.0}
    }


def test_ensure_loaded_without_a_recorded_request_refreshes_nothing(
    fake_stat, monkeypatch
):
    import asyncio

    from inference_server.legacy.bridge import LegacyModelBridge

    fake_stat["coco/3"] = ("object-detection", "infer")
    clock = _clock(monkeypatch, 10.0)
    bridge = LegacyModelBridge(_StampingGateway(clock, model_info=_DESCRIBED_COCO))

    async def _workflow_step():
        route = await bridge.resolve("coco/3", "k")
        await bridge.ensure_loaded(route, "k")

        return route

    route = asyncio.run(_workflow_step())

    assert route.requested_at == {}


def test_registry_after_a_failed_reload_during_the_request_is_empty(
    legacy_client, fake_stat, monkeypatch
):
    fake_stat["coco/3"] = ("object-detection", "infer")
    clock = _clock(monkeypatch, 10.0)
    gw = _EvictBeforeInferenceGateway(
        clock, reload_result=("error", 3), model_info=_DESCRIBED_COCO
    )
    c = legacy_client(gw)

    assert _infer_through_alias(c).status_code == 500

    registry = c.get("/model/registry")
    assert registry.status_code == 200
    assert registry.json()["models"] == []


def test_registry_row_survives_a_wall_clock_jump_backwards(
    legacy_client, fake_stat, monkeypatch
):
    import time

    fake_stat["coco/3"] = ("object-detection", "infer")
    gw = FakeGateway(model_info=_DESCRIBED_COCO)
    c = legacy_client(gw)
    c.post("/model/add", json={"model_id": "coco/3", "api_key": "k"})
    gw.loaded["coco/3"]["loaded_monotonic"] = time.monotonic()

    with monkeypatch.context() as patched:
        patched.setattr(time, "time", lambda: 0.0)
        _infer_through_alias(c)

    assert _registry_rows(c.get("/model/registry").json()) == [
        ("yolov8n-640", ["coco/3"], ["/infer/object_detection"])
    ]


def test_registry_after_bridge_unload_drops_preloaded_alias(
    legacy_client, fake_stat, monkeypatch
):
    import asyncio

    fake_stat["coco/3"] = ("object-detection", "infer")
    monkeypatch.setenv("INFERENCE_PRELOAD_MODELS", "yolov8n-640")
    c = legacy_client(FakeGateway(model_info=_DESCRIBED_COCO))
    _wait_for_status(c, "/readiness", 200)

    asyncio.run(c.app.state.legacy_bridge.unload("coco/3"))
    body = c.post("/model/add", json={"model_id": "coco/3", "api_key": "k"}).json()

    assert _registry_rows(body) == [("coco/3", [], ["/model/add"])]


_PE_REGISTRY_ID = "perception-encoder/PE-Core-L14-336"
_PE_LEGACY_ID = "perception_encoder/PE-Core-L14-336"


def _perception_encoder_gateway():
    import numpy as np

    return FakeGateway(
        predictions={(_PE_REGISTRY_ID, "embed_text"): np.array([[1.0]])},
        model_info={_PE_REGISTRY_ID: {"actions": {"embed_text": {}}}},
    )


def test_registry_row_for_core_request_keeps_the_legacy_id(legacy_client, fake_stat):
    gw = _perception_encoder_gateway()
    c = legacy_client(gw)

    assert (
        c.post("/perception_encoder/embed_text", json={"text": "hi"}).status_code == 200
    )

    assert _registry_rows(c.get("/model/registry").json()) == [
        (_PE_LEGACY_ID, [], ["/perception_encoder/embed_text"])
    ]
    assert (
        c.post("/model/remove", json={"model_id": _PE_LEGACY_ID}).json()["models"] == []
    )
    assert _unloads(gw) == [("unload", _PE_REGISTRY_ID)]


def test_model_add_keeps_the_legacy_id_and_remove_unloads(legacy_client, fake_stat):
    gw = _perception_encoder_gateway()
    c = legacy_client(gw)

    body = c.post("/model/add", json={"model_id": _PE_LEGACY_ID, "api_key": "k"}).json()

    assert _registry_rows(body) == [(_PE_LEGACY_ID, [], ["/model/add"])]
    assert (
        c.post("/model/remove", json={"model_id": _PE_LEGACY_ID}).json()["models"] == []
    )
    assert _unloads(gw) == [("unload", _PE_REGISTRY_ID)]


def test_start_records_the_path_id_without_aliases(legacy_client, fake_stat):
    c = legacy_client(_perception_encoder_gateway())

    c.get(f"/start/{_PE_LEGACY_ID}?api_key=k")

    assert _registry_rows(c.get("/model/registry").json()) == [
        (_PE_LEGACY_ID, [], [f"/start/{_PE_LEGACY_ID}"])
    ]


def test_depth_request_through_an_alias_records_it_without_aliases(
    legacy_client, fake_stat
):
    import numpy as np

    from tests.unit_tests.legacy.test_router_infer import _jpeg_b64

    fake_stat["yolo26-pretrains/yolo26n-depth"] = ("depth-estimation", "infer")
    gw = FakeGateway(
        predictions={
            ("yolo26-pretrains/yolo26n-depth", "infer"): np.array(
                [[0.0, 1.0], [2.0, 3.0]], dtype=np.float32
            )
        },
        model_info={"yolo26-pretrains/yolo26n-depth": {"actions": {"infer": {}}}},
    )
    c = legacy_client(gw)

    r = c.post(
        "/infer/depth-estimation/yolov26n-depth-768",
        json={"image": {"type": "base64", "value": _jpeg_b64()}},
    )

    assert r.status_code == 200, r.text
    assert _registry_rows(c.get("/model/registry").json()) == [
        ("yolov26n-depth-768", [], ["/infer/depth-estimation/yolov26n-depth-768"])
    ]


def test_lmm_request_through_an_alias_records_the_legacy_alias(
    legacy_client, fake_stat
):
    from tests.unit_tests.legacy.test_router_infer import _jpeg_b64

    fake_stat["qwen-pretrains/1"] = ("vlm", "prompt")
    gw = FakeGateway(
        predictions={("qwen-pretrains/1", "prompt"): ["a cat"]},
        model_info={"qwen-pretrains/1": {"actions": {"prompt": {}}}},
    )
    c = legacy_client(gw)

    r = c.post(
        "/infer/lmm",
        json={
            "model_id": "qwen25-vl-7b",
            "image": {"type": "base64", "value": _jpeg_b64()},
            "prompt": "what is it?",
        },
    )

    assert r.status_code == 200, r.text
    assert _registry_rows(c.get("/model/registry").json()) == [
        ("qwen25-vl-7b", ["qwen-pretrains/1"], ["/infer/lmm"])
    ]


_DEPTH_ALIAS = "yolov26n-depth-768"
_DEPTH_REGISTRY_ID = "yolo26-pretrains/yolo26n-depth"
_DEPTH_PATH = f"/infer/depth-estimation/{_DEPTH_ALIAS}"


def _depth_gateway(gateway_cls, *args):
    import numpy as np

    return gateway_cls(
        *args,
        predictions={
            (_DEPTH_REGISTRY_ID, "infer"): np.array(
                [[0.0, 1.0], [2.0, 3.0]], dtype=np.float32
            )
        },
        model_info={_DEPTH_REGISTRY_ID: {"actions": {"infer": {}}}},
    )


def _depth_request(c):
    from tests.unit_tests.legacy.test_router_infer import _jpeg_b64

    response = c.post(
        _DEPTH_PATH, json={"image": {"type": "base64", "value": _jpeg_b64()}}
    )
    return response


def _object_detection_request_through_depth_alias(c):
    from tests.unit_tests.legacy.test_router_infer import _jpeg_b64

    response = c.post(
        "/infer/object_detection",
        json={
            "model_id": _DEPTH_ALIAS,
            "image": {"type": "base64", "value": _jpeg_b64()},
        },
    )
    return response


def test_registry_drops_the_alias_of_an_expired_preload(
    legacy_client, fake_stat, monkeypatch
):
    fake_stat[_DEPTH_REGISTRY_ID] = ("depth-estimation", "infer")
    monkeypatch.setenv("INFERENCE_PRELOAD_MODELS", _DEPTH_ALIAS)
    clock = _clock(monkeypatch, 10.0)
    gw = _depth_gateway(_StampingGateway, clock)
    c = legacy_client(gw)
    _wait_for_status(c, "/readiness", 200)
    assert _registry_rows(c.get("/model/registry").json()) == [
        (_DEPTH_ALIAS, [_DEPTH_REGISTRY_ID], [])
    ]

    gw.loaded.pop(_DEPTH_REGISTRY_ID)
    clock["now"] = 20.0

    assert _depth_request(c).status_code == 200
    assert _registry_rows(c.get("/model/registry").json()) == [
        (_DEPTH_ALIAS, [], [_DEPTH_PATH])
    ]


def test_registry_drops_an_alias_recorded_before_the_reload(
    legacy_client, fake_stat, monkeypatch
):
    fake_stat[_DEPTH_REGISTRY_ID] = ("depth-estimation", "infer")
    clock = _clock(monkeypatch, 10.0)
    gw = _depth_gateway(_StampingGateway, clock)
    c = legacy_client(gw)
    assert _object_detection_request_through_depth_alias(c).status_code == 400
    assert _registry_rows(c.get("/model/registry").json()) == [
        (_DEPTH_ALIAS, [_DEPTH_REGISTRY_ID], ["/infer/object_detection"])
    ]

    gw.loaded.pop(_DEPTH_REGISTRY_ID)
    clock["now"] = 20.0

    assert _depth_request(c).status_code == 200
    assert _registry_rows(c.get("/model/registry").json()) == [
        (_DEPTH_ALIAS, [], [_DEPTH_PATH])
    ]


def test_registry_keeps_every_alias_without_load_stamp(
    legacy_client, fake_stat, monkeypatch
):
    fake_stat[_DEPTH_REGISTRY_ID] = ("depth-estimation", "infer")
    clock = _clock(monkeypatch, 10.0)
    gw = _depth_gateway(FakeGateway)
    c = legacy_client(gw)
    _object_detection_request_through_depth_alias(c)

    gw.loaded.pop(_DEPTH_REGISTRY_ID)
    clock["now"] = 20.0

    assert _depth_request(c).status_code == 200
    assert _registry_rows(c.get("/model/registry").json()) == [
        (
            _DEPTH_ALIAS,
            [_DEPTH_REGISTRY_ID],
            ["/infer/depth-estimation/yolov26n-depth-768", "/infer/object_detection"],
        )
    ]


def _bridge_with_recorded_request(fake_stat, monkeypatch, gateway_cls):
    from inference_server.legacy.bridge import LegacyModelBridge

    fake_stat["coco/3"] = ("object-detection", "infer")
    clock = _clock(monkeypatch, 10.0)
    bridge = LegacyModelBridge(gateway_cls(model_info=_DESCRIBED_COCO))
    return bridge, clock


def test_failed_ensure_loaded_refreshes_the_current_request_record(
    fake_stat, monkeypatch
):
    import asyncio

    import pytest

    from inference_server.legacy.errors import LegacyHTTPError

    bridge, clock = _bridge_with_recorded_request(fake_stat, monkeypatch, FakeGateway)

    async def _request():
        route = await bridge.resolve("yolov8n-640", "k")
        bridge.record_request(
            route, "yolov8n-640", "/infer/object_detection", alias="coco/3"
        )
        clock["now"] = 20.0
        bridge.gateway.ensure_results.append(("error", 3))
        with pytest.raises(LegacyHTTPError):
            await bridge.ensure_loaded(route, "k")

        return route

    route = asyncio.run(_request())

    assert route.requested_at == {"yolov8n-640": 20.0}
    assert route.request_paths_by_id == {
        "yolov8n-640": {"/infer/object_detection": 20.0}
    }


def test_failed_infer_refreshes_the_current_request_record(fake_stat, monkeypatch):
    import asyncio

    import pytest

    class _FailingInferGateway(FakeGateway):
        async def infer(self, **kwargs):
            clock["now"] = 20.0
            raise RuntimeError("inference failed")

    bridge, clock = _bridge_with_recorded_request(
        fake_stat, monkeypatch, _FailingInferGateway
    )

    async def _request():
        route = await bridge.resolve("yolov8n-640", "k")
        bridge.record_request(
            route, "yolov8n-640", "/infer/object_detection", alias="coco/3"
        )
        with pytest.raises(RuntimeError):
            await bridge.infer(route, "k", "infer", [None], {})

        return route

    route = asyncio.run(_request())

    assert route.requested_at == {"yolov8n-640": 20.0}
    assert route.request_paths_by_id == {
        "yolov8n-640": {"/infer/object_detection": 20.0}
    }
    assert route.request_aliases_by_id == {"yolov8n-640": {"coco/3": 20.0}}


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


class _FailingLoadGateway(_StampingGateway):
    def __init__(self, clock, **kwargs):
        super().__init__(clock, **kwargs)
        self.ensure_failures = 0
        self.load_failures = 0

    async def ensure_loaded(self, model_id, instance="", api_key="", device=""):
        if self.ensure_failures:
            self.ensure_failures -= 1
            return ("error", 3)
        return await super().ensure_loaded(model_id, instance, api_key, device)

    async def load(self, model_id, api_key="", timeout_s=None, pinned=True):
        if self.load_failures:
            self.load_failures -= 1
            return ("error", 3)
        return await super().load(model_id, api_key, timeout_s, pinned)


def _object_detection_request(c, model_id):
    from tests.unit_tests.legacy.test_router_infer import _det, _jpeg_b64

    c.app.state.model_manager.predictions[("coco/3", "infer")] = _det()
    response = c.post(
        "/infer/object_detection",
        json={
            "model_id": model_id,
            "api_key": "k",
            "image": {"type": "base64", "value": _jpeg_b64()},
        },
    )
    return response


def test_registry_row_lists_the_path_of_a_request_whose_load_failed(
    legacy_client, fake_stat, monkeypatch
):
    fake_stat["coco/3"] = ("object-detection", "infer")
    clock = _clock(monkeypatch, 10.0)
    gw = _FailingLoadGateway(clock, model_info=_DESCRIBED_COCO)
    c = legacy_client(gw)

    gw.ensure_failures = 1
    assert _object_detection_request(c, "coco/3").status_code == 500
    assert c.get("/model/registry").json()["models"] == []

    assert c.get("/start/coco/3?api_key=k").status_code == 200

    assert _registry_rows(c.get("/model/registry").json()) == [
        ("coco/3", [], ["/infer/object_detection", "/start/coco/3"])
    ]


def test_registry_row_lists_the_alias_of_a_request_whose_load_failed(
    legacy_client, fake_stat, monkeypatch
):
    fake_stat[_DEPTH_REGISTRY_ID] = ("depth-estimation", "infer")
    clock = _clock(monkeypatch, 10.0)
    gw = _depth_gateway(_FailingLoadGateway, clock)
    c = legacy_client(gw)

    gw.ensure_failures = 1
    assert _object_detection_request_through_depth_alias(c).status_code == 500
    assert c.get("/model/registry").json()["models"] == []

    assert _depth_request(c).status_code == 200

    assert _registry_rows(c.get("/model/registry").json()) == [
        (_DEPTH_ALIAS, [_DEPTH_REGISTRY_ID], [_DEPTH_PATH, "/infer/object_detection"])
    ]


def test_registry_keeps_a_failed_request_off_another_row_of_the_same_model(
    legacy_client, fake_stat, monkeypatch
):
    fake_stat["coco/3"] = ("object-detection", "infer")
    clock = _clock(monkeypatch, 10.0)
    gw = _FailingLoadGateway(clock, model_info=_DESCRIBED_COCO)
    c = legacy_client(gw)

    gw.ensure_failures = 1
    assert _object_detection_request(c, "yolov8n-640").status_code == 500
    assert c.get("/start/coco/3?api_key=k").status_code == 200

    assert _registry_rows(c.get("/model/registry").json()) == [
        ("coco/3", [], ["/start/coco/3"])
    ]

    assert _object_detection_request(c, "yolov8n-640").status_code == 200

    assert _registry_rows(c.get("/model/registry").json()) == [
        ("coco/3", [], ["/start/coco/3"]),
        ("yolov8n-640", ["coco/3"], ["/infer/object_detection"]),
    ]


def test_registry_row_lists_the_path_of_a_request_that_failed_the_model_lookup(
    legacy_client, fake_stat, monkeypatch
):
    clock = _clock(monkeypatch, 10.0)
    c = legacy_client(_FailingLoadGateway(clock, model_info=_DESCRIBED_COCO))

    assert _object_detection_request(c, "coco/3").status_code == 404
    assert c.get("/model/registry").json()["models"] == []

    fake_stat["coco/3"] = ("object-detection", "infer")
    assert c.get("/start/coco/3?api_key=k").status_code == 200

    assert _registry_rows(c.get("/model/registry").json()) == [
        ("coco/3", [], ["/infer/object_detection", "/start/coco/3"])
    ]


def test_registry_keeps_a_failed_request_across_clear_and_remove(
    legacy_client, fake_stat, monkeypatch
):
    fake_stat["coco/3"] = ("object-detection", "infer")
    clock = _clock(monkeypatch, 10.0)
    gw = _FailingLoadGateway(clock, model_info=_DESCRIBED_COCO)
    c = legacy_client(gw)

    gw.ensure_failures = 1
    assert _object_detection_request(c, "coco/3").status_code == 500
    assert c.post("/model/clear").json()["models"] == []
    assert c.post("/model/remove", json={"model_id": "coco/3"}).json()["models"] == []
    assert c.get("/start/coco/3?api_key=k").status_code == 200

    assert _registry_rows(c.get("/model/registry").json()) == [
        ("coco/3", [], ["/infer/object_detection", "/start/coco/3"])
    ]


def test_registry_drops_a_joined_failed_request_after_a_reload(
    legacy_client, fake_stat, monkeypatch
):
    fake_stat["coco/3"] = ("object-detection", "infer")
    clock = _clock(monkeypatch, 10.0)
    gw = _FailingLoadGateway(clock, model_info=_DESCRIBED_COCO)
    c = legacy_client(gw)

    gw.ensure_failures = 1
    assert _object_detection_request(c, "coco/3").status_code == 500
    clock["now"] = 20.0
    assert c.get("/start/coco/3?api_key=k").status_code == 200
    assert _registry_rows(c.get("/model/registry").json()) == [
        ("coco/3", [], ["/infer/object_detection", "/start/coco/3"])
    ]

    gw.loaded.pop("coco/3")
    clock["now"] = 30.0
    response = c.post("/model/add", json={"model_id": "coco/3", "api_key": "k"})

    assert _registry_rows(response.json()) == [("coco/3", [], ["/model/add"])]


def test_registry_row_lists_a_model_add_whose_load_failed(
    legacy_client, fake_stat, monkeypatch
):
    fake_stat["coco/3"] = ("object-detection", "infer")
    clock = _clock(monkeypatch, 10.0)
    gw = _FailingLoadGateway(clock, model_info=_DESCRIBED_COCO)
    c = legacy_client(gw)

    gw.load_failures = 1
    response = c.post("/model/add", json={"model_id": "yolov8n-640", "api_key": "k"})
    assert response.status_code == 500
    assert c.get("/start/coco/3?api_key=k").status_code == 200

    assert _registry_rows(c.get("/model/registry").json()) == [
        ("coco/3", [], ["/model/add", "/start/coco/3"])
    ]


def test_registry_row_lists_a_start_whose_load_failed(
    legacy_client, fake_stat, monkeypatch
):
    fake_stat["coco/3"] = ("object-detection", "infer")
    clock = _clock(monkeypatch, 10.0)
    gw = _FailingLoadGateway(clock, model_info=_DESCRIBED_COCO)
    c = legacy_client(gw)

    gw.ensure_failures = 1
    assert c.get("/start/coco/3?api_key=k").status_code == 500
    assert _object_detection_request(c, "coco/3").status_code == 200

    assert _registry_rows(c.get("/model/registry").json()) == [
        ("coco/3", [], ["/infer/object_detection", "/start/coco/3"])
    ]


@pytest.mark.parametrize("error,status,body", LOAD_FAILURE_MATRIX)
def test_load_failure_on_model_add_answers_like_legacy(
    legacy_client, fake_stat, error, status, body
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    manager = FailingLoadManager(error)
    c = legacy_client(ModelManagerGateway(manager))

    response = c.post("/model/add", json={"model_id": "ds/1", "api_key": "k"})

    assert response.status_code == status
    assert response.json() == body
    assert response.headers["content-type"] == "application/json"
    assert "retry-after" not in response.headers
    assert manager.load_calls == 1


class _ExplicitLoadGateway(FakeGateway):
    def __init__(self, outcome):
        super().__init__()
        self.outcome = outcome

    async def load(self, model_id, api_key="", timeout_s=None, pinned=True):
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


@pytest.mark.parametrize(
    "outcome,status,body",
    [
        (("error", 5), 500, {"message": "Model package is broken."}),
        (("error", 6), 500, {"message": "Model package is broken."}),
        (
            (
                "error",
                5,
                {"error_type": "ModelNotFoundError", "message": "missing"},
            ),
            404,
            {
                "message": "Requested Roboflow resource not found. Make sure that "
                "workspace, project or model you referred in request exists."
            },
        ),
        (
            ("error", 5, {"error_type": "KeyError", "message": "'x'"}),
            500,
            {"message": "Internal error."},
        ),
    ],
)
@pytest.mark.parametrize("path", ["/model/add", "/start/ds/1"])
def test_failure_of_the_explicit_load_answers_like_legacy(
    legacy_client, fake_stat, path, outcome, status, body
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    c = legacy_client(_ExplicitLoadGateway(outcome))

    if path == "/model/add":
        response = c.post(path, json={"model_id": "ds/1", "api_key": "k"})
    else:
        response = c.get(f"{path}?api_key=k")

    assert response.status_code == status
    assert response.json() == body
    assert "retry-after" not in response.headers


@pytest.mark.parametrize("path", ["/model/add", "/start/ds/1"])
def test_explicit_load_timeout_answers_not_ready_with_retry_after(
    legacy_client, fake_stat, path
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    c = legacy_client(_ExplicitLoadGateway(asyncio.TimeoutError()))

    if path == "/model/add":
        response = c.post(path, json={"model_id": "ds/1", "api_key": "k"})
    else:
        response = c.get(f"{path}?api_key=k")

    assert response.status_code == 503
    assert response.json() == {
        "message": "Model is temporarily not ready - retry request."
    }
    assert response.headers["retry-after"] == "1"


@pytest.mark.parametrize("switch, expected_key", [(True, "H"), (False, "")])
def test_bearer_key_follows_header_switch_on_legacy_infer_route(
    legacy_client, fake_stat, monkeypatch, switch, expected_key
):
    from tests.unit_tests.legacy.test_router_infer import _det, _jpeg_b64

    monkeypatch.setattr(
        "inference_server.configuration.ALLOW_API_KEY_FROM_HEADERS", switch
    )
    monkeypatch.setattr("inference_server.legacy.common.DEFAULT_API_KEY", None)
    fake_stat["ds/1"] = ("object-detection", "infer")
    gw = FakeGateway(
        predictions={("ds/1", "infer"): _det()},
        model_info={"ds/1": {"class_names": ["cat"]}},
    )
    legacy_client(gw).post(
        "/infer/object_detection",
        headers={"Authorization": "Bearer H"},
        json={"model_id": "ds/1", "image": {"type": "base64", "value": _jpeg_b64()}},
    )
    assert ("ensure_loaded", "ds/1", expected_key) in gw.calls
