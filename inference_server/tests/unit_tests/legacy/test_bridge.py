import asyncio
import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from inference_models.errors import ModelInputError

from inference_server.legacy import bridge as bridge_mod
from inference_server.legacy.bridge import (
    LegacyModelBridge,
    LoopBridge,
    SyncLegacyBridge,
    resolved_model_for,
)
from inference_server.legacy.common import ImagePayload
from inference_server.legacy.errors import LegacyHTTPError
from tests.unit_tests.legacy.conftest import FakeGateway


@pytest.mark.asyncio
async def test_resolve_uses_registry_stat_and_fills_metadata(fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gw = FakeGateway(
        model_info={
            "ds/1": {
                "class_names": ["a", "b"],
                "actions": {"infer": {}},
                "model_class_name": "X",
            }
        }
    )
    bridge = LegacyModelBridge(gw)
    route = await bridge.resolve("ds/1", "key")
    assert (
        route.task_type == "object-detection"
        and route.class_names == ["a", "b"]
        and route.actions == {"infer"}
    )
    assert ("ensure_loaded", "ds/1", "key") in gw.calls
    assert await bridge.resolve("ds/1", "key") is route


@pytest.mark.asyncio
async def test_resolve_reauthorizes_every_call_even_when_cached(fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gw = FakeGateway()
    bridge = LegacyModelBridge(gw)
    await bridge.resolve("ds/1", "good")
    del fake_stat["ds/1"]
    with pytest.raises(LookupError):
        await bridge.resolve("ds/1", "other")


@pytest.mark.asyncio
async def test_resolve_core_model_prefix_falls_back_to_static_table(fake_stat):
    gw = FakeGateway()
    route = await LegacyModelBridge(gw).resolve(
        "perception_encoder/PE-Core-B16-224", None
    )
    assert (
        route.task_type == "embedding"
        and route.registry_id == "perception-encoder/PE-Core-B16-224"
    )


@pytest.mark.asyncio
async def test_resolve_applies_sdk_alias(fake_stat):
    fake_stat["coco/3"] = ("object-detection", "infer")
    gw = FakeGateway()
    bridge = LegacyModelBridge(gw)
    route = await bridge.resolve("yolov8n-640", "k")
    assert (
        route.registry_id == "coco/3" and ("ensure_loaded", "coco/3", "k") in gw.calls
    )
    assert "yolov8n-640" in bridge and "coco/3" in bridge


@pytest.mark.asyncio
async def test_alias_resolves_to_canonical_route_keeping_recorded_requests(fake_stat):
    fake_stat["coco/3"] = ("object-detection", "infer")
    gw = FakeGateway()
    bridge = LegacyModelBridge(gw)
    canonical = await bridge.resolve("coco/3", "k")
    bridge.record_request(canonical, "coco/3", "/infer/object_detection")
    alias_route = await bridge.resolve("yolov8n-640", "k")
    assert alias_route is canonical
    bridge.record_request(alias_route, "yolov8n-640", "/yolov8n-640")
    assert set(canonical.requested_at) == {"coco/3", "yolov8n-640"}
    assert {
        path for paths in canonical.request_paths_by_id.values() for path in paths
    } == {"/infer/object_detection", "/yolov8n-640"}


@pytest.mark.asyncio
async def test_concurrent_first_resolutions_share_one_cached_route(fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gw = FakeGateway()
    plain_ensure_loaded = gw.ensure_loaded

    async def suspending_ensure_loaded(model_id, instance="", api_key="", device=""):
        await asyncio.sleep(0)
        return await plain_ensure_loaded(model_id, instance, api_key, device)

    gw.ensure_loaded = suspending_ensure_loaded
    bridge = LegacyModelBridge(gw)
    first, second = await asyncio.gather(
        bridge.resolve("ds/1", None), bridge.resolve("ds/1", None)
    )
    assert first is second
    assert {id(route) for route in bridge._routes.values()} == {id(first)}
    assert await bridge.resolve("ds/1", None) is first


@pytest.mark.asyncio
async def test_resolve_unknown_model_is_404(fake_stat):
    with pytest.raises(LookupError):
        await LegacyModelBridge(FakeGateway()).resolve("nope/1", None)


@pytest.mark.asyncio
async def test_describe_lists_models_loaded_outside_the_bridge(fake_stat):
    gw = FakeGateway(
        model_info={
            "other/2": {"model_mro_names": ["YOLOv8", "ObjectDetectionModel", "object"]}
        }
    )
    await gw.load("other/2")
    routes = await LegacyModelBridge(gw).describe()
    assert [(r.model_id, r.task_type) for r in routes] == [
        ("other/2", "object-detection")
    ]


@pytest.mark.asyncio
async def test_offline_mode_skips_registry_and_uses_mro(fake_stat, monkeypatch):
    monkeypatch.setattr("inference_server.legacy.bridge.OFFLINE_MODE", True)
    gw = FakeGateway(
        model_info={
            "ds/1": {"model_mro_names": ["ClassificationModel"], "class_names": ["a"]}
        }
    )
    route = await LegacyModelBridge(gw).resolve("ds/1", None)
    assert route.task_type == "classification" and fake_stat == {}


@pytest.mark.asyncio
async def test_offline_mode_load_failure_is_404(fake_stat, monkeypatch):
    monkeypatch.setattr("inference_server.legacy.bridge.OFFLINE_MODE", True)
    gw = FakeGateway()
    gw.ensure_results = [("error", 5)]
    with pytest.raises(LegacyHTTPError) as exc:
        await LegacyModelBridge(gw).resolve("ds/1", None)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_metadata_refreshes_after_ttl(fake_stat, monkeypatch):
    monkeypatch.setattr("inference_server.legacy.bridge.LEGACY_ROUTE_METADATA_TTL_S", 0)
    fake_stat["ds/1"] = ("object-detection", "infer")
    gw = FakeGateway(model_info={"ds/1": {"class_names": ["old"]}})
    bridge = LegacyModelBridge(gw)
    assert (await bridge.resolve("ds/1", None)).class_names == ["old"]
    gw.loaded["ds/1"]["class_names"] = ["new"]
    assert (await bridge.resolve("ds/1", None)).class_names == ["new"]


@pytest.mark.asyncio
async def test_eviction_invalidates_cached_metadata(fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gw = FakeGateway(model_info={"ds/1": {"class_names": ["old"]}})
    bridge = LegacyModelBridge(gw)
    assert (await bridge.resolve("ds/1", None)).class_names == ["old"]

    gw.loaded.pop("ds/1")
    gw.model_info["ds/1"] = {"class_names": ["new"]}
    assert await bridge.describe() == []

    stats_reads = []
    plain_stats = gw.stats

    async def counting_stats():
        stats_reads.append(1)
        return await plain_stats()

    gw.stats = counting_stats
    route = await bridge.resolve("ds/1", None)
    assert stats_reads and route.class_names == ["new"]


@pytest.mark.asyncio
async def test_ensure_loaded_polls_until_ready(fake_stat, monkeypatch):
    monkeypatch.setattr("inference_server.legacy.bridge.LEGACY_LOAD_POLL_INTERVAL_S", 0)
    fake_stat["ds/1"] = ("object-detection", "infer")
    gw = FakeGateway()
    gw.ensure_results = [("load_timeout", 10), ("load_timeout", 10)]
    route = await LegacyModelBridge(gw).resolve("ds/1", None)
    assert route.task_type == "object-detection"
    assert sum(1 for c in gw.calls if c[0] == "ensure_loaded") == 3


@pytest.mark.asyncio
async def test_ensure_loaded_times_out_with_503(fake_stat, monkeypatch):
    monkeypatch.setattr("inference_server.legacy.bridge.LEGACY_LOAD_POLL_INTERVAL_S", 0)
    monkeypatch.setattr("inference_server.legacy.bridge.LEGACY_LOAD_TIMEOUT_S", 0)
    fake_stat["ds/1"] = ("object-detection", "infer")
    gw = FakeGateway()
    gw.ensure_results = [("load_timeout", 10)] * 5
    with pytest.raises(LegacyHTTPError) as exc:
        await LegacyModelBridge(gw).resolve("ds/1", None)
    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_infer_fans_out_per_image(fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gw = FakeGateway(predictions={("ds/1", "infer"): lambda img, p: ("pred", img)})
    bridge = LegacyModelBridge(gw)
    route = await bridge.resolve("ds/1", None)
    out = await bridge.infer(
        route,
        None,
        "infer",
        [ImagePayload(b"a", 1, 1), ImagePayload(b"b", 1, 1)],
        {"confidence": 0.5},
    )
    assert out == [("pred", b"a"), ("pred", b"b")]


def _raising(error):
    def _raise(image, params):
        raise error

    return _raise


@pytest.mark.asyncio
async def test_infer_wraps_a_gateway_value_error_as_a_model_input_error(fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gw = FakeGateway(predictions={("ds/1", "infer"): _raising(ValueError("bad shape"))})
    bridge = LegacyModelBridge(gw)
    route = await bridge.resolve("ds/1", None)

    with pytest.raises(ModelInputError) as exc:
        await bridge.infer(route, None, "infer", [ImagePayload(b"a", 1, 1)], {})

    assert str(exc.value) == "bad shape"
    assert exc.value.help_url is None


@pytest.mark.asyncio
async def test_infer_raises_the_model_input_error_a_gateway_value_error_was_caused_by(
    fake_stat,
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    original = ModelInputError("bad", help_url="https://x")
    try:
        raise ValueError(str(original)) from original
    except ValueError as error:
        raised = error
    gw = FakeGateway(predictions={("ds/1", "infer"): _raising(raised)})
    bridge = LegacyModelBridge(gw)
    route = await bridge.resolve("ds/1", None)

    with pytest.raises(ModelInputError) as exc:
        await bridge.infer(route, None, "infer", [ImagePayload(b"a", 1, 1)], {})

    assert exc.value is original


@pytest.mark.asyncio
async def test_infer_leaves_other_gateway_errors_unchanged(fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gw = FakeGateway(predictions={("ds/1", "infer"): _raising(RuntimeError("boom"))})
    bridge = LegacyModelBridge(gw)
    route = await bridge.resolve("ds/1", None)

    with pytest.raises(RuntimeError, match="boom"):
        await bridge.infer(route, None, "infer", [ImagePayload(b"a", 1, 1)], {})


@pytest.mark.asyncio
async def test_unload_drops_routes(fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gw = FakeGateway()
    bridge = LegacyModelBridge(gw)
    await bridge.resolve("ds/1", None)
    await bridge.unload("ds/1")
    assert "ds/1" not in bridge and [r.model_id for r in await bridge.describe()] == []


@pytest.mark.asyncio
async def test_unload_resolves_alias_when_route_is_not_cached(fake_stat):
    gw = FakeGateway()
    await LegacyModelBridge(gw).unload("yolov8n-640")
    assert ("unload", "coco/3") in gw.calls


@pytest.mark.asyncio
async def test_sync_bridge_runs_from_plain_thread_pool_worker(fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    loop = asyncio.get_running_loop()
    sync = SyncLegacyBridge(LegacyModelBridge(FakeGateway()), LoopBridge(loop))
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = await loop.run_in_executor(pool, lambda: sync.resolve("ds/1", None))
    assert result.task_type == "object-detection"
    with pytest.raises(RuntimeError):
        sync.resolve("ds/1", None)


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


@pytest.fixture
def server_loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop, thread
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


def test_sync_bridge_records_the_request_on_the_loop_thread(
    fake_stat, monkeypatch, server_loop
):
    loop, loop_thread = server_loop
    fake_stat["ds/1"] = ("object-detection", "infer")
    recording_threads = []

    def _clock():
        recording_threads.append(threading.get_ident())
        return 10.0

    monkeypatch.setattr(bridge_mod, "_clock", _clock)
    sync = SyncLegacyBridge(LegacyModelBridge(FakeGateway()), LoopBridge(loop))

    def _step():
        route = sync.resolve("ds/1", "k")
        sync.record_request(route, "alias-1", "/workflows/run", alias="ds/1")
        return route, bridge_mod._CURRENT_REQUEST.get()

    route, current = contextvars.copy_context().run(_step)

    assert route.requested_at == {"alias-1": 10.0}
    assert route.request_paths_by_id == {"alias-1": {"/workflows/run": 10.0}}
    assert route.request_aliases_by_id == {"alias-1": {"ds/1": 10.0}}
    assert current == {
        ("ds/1", "alias-1"): (route, "alias-1", "/workflows/run", "ds/1")
    }
    assert recording_threads == [loop_thread.ident]
    assert loop_thread.ident != threading.get_ident()
    assert bridge_mod._CURRENT_REQUEST.get() is None


def test_sync_bridge_ignores_a_request_without_model_id(fake_stat, server_loop):
    loop, _ = server_loop
    fake_stat["ds/1"] = ("object-detection", "infer")
    sync = SyncLegacyBridge(LegacyModelBridge(FakeGateway()), LoopBridge(loop))

    def _step():
        route = sync.resolve("ds/1", "k")
        sync.record_request(route, "", "/workflows/run")
        return route, bridge_mod._CURRENT_REQUEST.get()

    route, current = contextvars.copy_context().run(_step)

    assert route.requested_at == {} and current is None


def test_sync_bridge_row_survives_a_reload_triggered_by_a_later_infer(
    fake_stat, monkeypatch, server_loop
):
    loop, _ = server_loop
    fake_stat["ds/1"] = ("object-detection", "infer")
    clock = {"now": 10.0}
    monkeypatch.setattr(bridge_mod, "_clock", lambda: clock["now"])
    gateway = _StampingGateway(clock, predictions={("ds/1", "infer"): "pred"})
    bridge = LegacyModelBridge(gateway)
    sync = SyncLegacyBridge(bridge, LoopBridge(loop))

    def _step():
        route = sync.resolve("ds/1", "k")
        sync.record_request(route, "ds/1", "/workflows/run")
        gateway.loaded.pop("ds/1")
        clock["now"] = 20.0
        return sync.infer(route, "k", "infer", [ImagePayload(b"a", 1, 1)], {})

    assert contextvars.copy_context().run(_step) == ["pred"]

    routes = asyncio.run_coroutine_threadsafe(bridge.describe(), loop).result(5)
    assert gateway.loaded["ds/1"]["loaded_monotonic"] == 20.0
    assert [
        (route.registry_id, route.requested_at, route.request_paths_by_id)
        for route in routes
    ] == [("ds/1", {"ds/1": 20.0}, {"ds/1": {"/workflows/run": 20.0}})]


def _row(routes, registry_id):
    return next(route for route in routes if route.registry_id == registry_id)


def test_sync_bridge_does_not_refresh_an_earlier_model_when_a_later_one_loads(
    fake_stat, monkeypatch, server_loop
):
    loop, _ = server_loop
    fake_stat["ds/1"] = ("object-detection", "infer")
    fake_stat["ds/2"] = ("object-detection", "infer")
    clock = {"now": 10.0}
    monkeypatch.setattr(bridge_mod, "_clock", lambda: clock["now"])
    gateway = _StampingGateway(clock)
    bridge = LegacyModelBridge(gateway)
    sync = SyncLegacyBridge(bridge, LoopBridge(loop))

    def _step():
        route_a = sync.resolve("ds/1", "k")
        sync.record_request(route_a, "ds/1", "/workflows/run")
        gateway.loaded["ds/1"]["loaded_monotonic"] = 15.0
        clock["now"] = 20.0
        route_b = sync.resolve("ds/2", "k")
        sync.record_request(route_b, "ds/2", "/workflows/run")

    contextvars.copy_context().run(_step)

    routes = asyncio.run_coroutine_threadsafe(bridge.describe(), loop).result(5)
    assert _row(routes, "ds/1").request_paths_by_id.get("ds/1", {}) == {}
    assert _row(routes, "ds/2").request_paths_by_id == {
        "ds/2": {"/workflows/run": 20.0}
    }


def test_sync_bridge_reload_of_an_earlier_model_keeps_its_row_and_leaves_others(
    fake_stat, monkeypatch, server_loop
):
    loop, _ = server_loop
    fake_stat["ds/1"] = ("object-detection", "infer")
    fake_stat["ds/2"] = ("object-detection", "infer")
    clock = {"now": 10.0}
    monkeypatch.setattr(bridge_mod, "_clock", lambda: clock["now"])
    gateway = _StampingGateway(
        clock, predictions={("ds/1", "infer"): "pred", ("ds/2", "infer"): "pred"}
    )
    bridge = LegacyModelBridge(gateway)
    sync = SyncLegacyBridge(bridge, LoopBridge(loop))

    def _step():
        route_a = sync.resolve("ds/1", "k")
        sync.record_request(route_a, "ds/1", "/workflows/run")
        route_b = sync.resolve("ds/2", "k")
        sync.record_request(route_b, "ds/2", "/workflows/run")
        gateway.loaded.pop("ds/1")
        clock["now"] = 20.0
        return sync.infer(route_a, "k", "infer", [ImagePayload(b"a", 1, 1)], {})

    assert contextvars.copy_context().run(_step) == ["pred"]

    routes = asyncio.run_coroutine_threadsafe(bridge.describe(), loop).result(5)
    assert _row(routes, "ds/1").request_paths_by_id == {
        "ds/1": {"/workflows/run": 20.0}
    }
    assert _row(routes, "ds/2").request_paths_by_id == {
        "ds/2": {"/workflows/run": 10.0}
    }


def test_sync_bridge_reload_keeps_every_id_requested_for_the_same_model(
    fake_stat, monkeypatch, server_loop
):
    loop, _ = server_loop
    fake_stat["ds/1"] = ("object-detection", "infer")
    clock = {"now": 10.0}
    monkeypatch.setattr(bridge_mod, "_clock", lambda: clock["now"])
    gateway = _StampingGateway(clock, predictions={("ds/1", "infer"): "pred"})
    bridge = LegacyModelBridge(gateway)
    sync = SyncLegacyBridge(bridge, LoopBridge(loop))

    def _step():
        route = sync.resolve("ds/1", "k")
        sync.record_request(route, "ds/1", "/workflows/run")
        sync.record_request(route, "alias-1", "/infer/object_detection")
        gateway.loaded.pop("ds/1")
        clock["now"] = 20.0
        return sync.infer(route, "k", "infer", [ImagePayload(b"a", 1, 1)], {})

    assert contextvars.copy_context().run(_step) == ["pred"]

    routes = asyncio.run_coroutine_threadsafe(bridge.describe(), loop).result(5)
    assert _row(routes, "ds/1").request_paths_by_id == {
        "ds/1": {"/workflows/run": 20.0},
        "alias-1": {"/infer/object_detection": 20.0},
    }


def test_recorded_requests_are_copy_on_write_across_contexts(fake_stat, server_loop):
    loop, _ = server_loop
    fake_stat["ds/1"] = ("object-detection", "infer")
    fake_stat["ds/2"] = ("object-detection", "infer")
    sync = SyncLegacyBridge(LegacyModelBridge(FakeGateway()), LoopBridge(loop))

    def _step():
        route_a = sync.resolve("ds/1", "k")
        sync.record_request(route_a, "ds/1", "/workflows/run")
        snapshot = contextvars.copy_context()
        route_b = sync.resolve("ds/2", "k")
        sync.record_request(route_b, "ds/2", "/workflows/run")
        return snapshot, bridge_mod._CURRENT_REQUEST.get()

    snapshot, current = contextvars.copy_context().run(_step)

    assert sorted(key for key in current) == [("ds/1", "ds/1"), ("ds/2", "ds/2")]
    assert list(snapshot.run(bridge_mod._CURRENT_REQUEST.get)) == [("ds/1", "ds/1")]


@pytest.mark.asyncio
async def test_resolved_model_for_falls_back_to_registry_id(fake_stat):
    fake_stat["coco/3"] = ("object-detection", "infer")
    bridge = LegacyModelBridge(FakeGateway())
    route = await bridge.resolve("yolov8n-640", "k")
    assert resolved_model_for(route).model_dump(exclude_none=True) == {
        "model_id": "coco/3"
    }
    route.resolved_model = {"model_id": "coco/3", "backend": "onnx"}
    assert resolved_model_for(route).backend == "onnx"


@pytest.mark.asyncio
async def test_route_carries_resolved_model_from_stats(fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gw = FakeGateway(
        model_info={
            "ds/1": {
                "resolved_model": {
                    "model_id": "ds/1",
                    "model_package_id": "pkg",
                    "backend": "trt",
                    "quantization": "fp16",
                }
            }
        }
    )
    route = await LegacyModelBridge(gw).resolve("ds/1", None)
    assert route.resolved_model["backend"] == "trt"


def _recording_stat(monkeypatch, table, calls):
    from inference_models.errors import ModelNotFoundError
    from inference_server.framework import model_stat

    def _metadata(model_id, api_key=None, **_):
        calls.append((model_id, api_key or ""))
        if model_id.startswith("pp_ocr"):
            raise AssertionError(f"synthetic id statted: {model_id}")
        outcome = table.get(model_id)
        if outcome is None:
            raise ModelNotFoundError(message=model_id, help_url="")
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(task_type=outcome[0])

    monkeypatch.setattr(model_stat, "get_one_page_of_model_metadata", _metadata)


@pytest.mark.asyncio
async def test_resolve_pipeline_id_stats_every_stage_with_the_callers_key(monkeypatch):
    calls = []
    _recording_stat(
        monkeypatch,
        {
            "pp-ocrv6-det/small": ("object-detection", "infer"),
            "pp-ocrv6-rec/medium": ("text-only-ocr", "infer"),
        },
        calls,
    )
    route = await LegacyModelBridge(FakeGateway()).resolve(
        "pp_ocr/small-medium", "key-1"
    )
    assert route.task_type == "structured-ocr" and route.action == "infer"
    assert route.registry_id == "pp_ocr/small-medium"
    assert sorted(calls) == [
        ("pp-ocrv6-det/small", "key-1"),
        ("pp-ocrv6-rec/medium", "key-1"),
    ]


@pytest.mark.asyncio
async def test_resolve_pipeline_id_is_lookup_error_when_a_stage_is_missing(monkeypatch):
    calls = []
    _recording_stat(
        monkeypatch, {"pp-ocrv6-det/small": ("object-detection", "infer")}, calls
    )
    with pytest.raises(LookupError):
        await LegacyModelBridge(FakeGateway()).resolve("pp_ocr/small-small", "k")
    assert ("pp-ocrv6-rec/small", "k") in calls


@pytest.mark.asyncio
async def test_resolve_pipeline_id_is_permission_error_when_a_stage_is_denied(
    monkeypatch,
):
    from inference_models.errors import UnauthorizedModelAccessError

    _recording_stat(
        monkeypatch,
        {
            "pp-ocrv6-det/small": ("object-detection", "infer"),
            "pp-ocrv6-rec/small": UnauthorizedModelAccessError(
                message="pp-ocrv6-rec/small", help_url=""
            ),
        },
        [],
    )
    with pytest.raises(PermissionError):
        await LegacyModelBridge(FakeGateway()).resolve("pp_ocr/small-small", "k")


@pytest.mark.asyncio
async def test_resolve_pipeline_id_skips_a_disabled_stage(monkeypatch):
    calls = []
    _recording_stat(
        monkeypatch, {"pp-ocrv6-det/small": ("object-detection", "infer")}, calls
    )
    route = await LegacyModelBridge(FakeGateway()).resolve("pp_ocr/small-none", "k")
    assert route.task_type == "structured-ocr"
    assert calls == [("pp-ocrv6-det/small", "k")]


@pytest.mark.asyncio
async def test_loaded_pipeline_is_reauthorized_on_every_resolve(monkeypatch):
    calls = []
    outcome = [("structured-ocr", "infer")]

    async def _stat(common):
        calls.append((common.model_id, common.api_key))
        if isinstance(outcome[0], Exception):
            raise outcome[0]
        return outcome[0]

    monkeypatch.setattr(
        "inference_server.legacy.bridge.stat_model_while_checking_auth", _stat
    )
    bridge = LegacyModelBridge(FakeGateway())
    await bridge.resolve("pp_ocr/small-small", "good")
    assert "pp_ocr/small-small" in bridge
    outcome[0] = PermissionError("revoked")
    with pytest.raises(PermissionError):
        await bridge.resolve("pp_ocr/small-small", "good")
    assert calls == [
        ("pp_ocr/small-small", "good"),
        ("pp_ocr/small-small", "good"),
    ]


def _rows(routes):
    rows = {
        row_key: (
            sorted(route.request_paths_by_id.get(row_key, {})),
            sorted(route.request_aliases_by_id.get(row_key, {})),
        )
        for route in routes
        for row_key in route.requested_at
    }
    return rows


async def _failed_resolve(bridge, model_id, **row):
    bridge.gateway.ensure_results.append(("error", 3))
    with pytest.raises(LegacyHTTPError):
        await bridge.resolve(model_id, "k", **row)


async def _recorded_resolve(bridge, model_id, path, alias=None):
    route = await bridge.resolve(
        model_id, "k", row_key=model_id, path=path, alias=alias
    )
    bridge.record_request(route, model_id, path, alias=alias)
    return route


@pytest.mark.asyncio
async def test_request_whose_load_failed_alone_creates_no_row(fake_stat):
    fake_stat["coco/3"] = ("object-detection", "infer")
    bridge = LegacyModelBridge(FakeGateway())

    await _failed_resolve(bridge, "coco/3", row_key="coco/3", path="/p1")

    assert await bridge.describe() == []


@pytest.mark.asyncio
async def test_request_whose_load_failed_joins_the_row_of_its_own_key(fake_stat):
    fake_stat["coco/3"] = ("object-detection", "infer")
    bridge = LegacyModelBridge(FakeGateway())

    await _failed_resolve(
        bridge, "yolov8n-640", row_key="yolov8n-640", path="/p1", alias="coco/3"
    )
    await _recorded_resolve(bridge, "yolov8n-640", "/p2")

    assert _rows(await bridge.describe()) == {
        "yolov8n-640": (["/p1", "/p2"], ["coco/3"])
    }


@pytest.mark.asyncio
async def test_request_whose_load_failed_stays_off_another_row_of_the_model(fake_stat):
    fake_stat["coco/3"] = ("object-detection", "infer")
    bridge = LegacyModelBridge(FakeGateway())

    await _failed_resolve(
        bridge, "yolov8n-640", row_key="yolov8n-640", path="/p1", alias="coco/3"
    )
    await _recorded_resolve(bridge, "coco/3", "/p2")

    assert _rows(await bridge.describe()) == {"coco/3": (["/p2"], [])}

    await _recorded_resolve(bridge, "yolov8n-640", "/p3")

    assert _rows(await bridge.describe()) == {
        "coco/3": (["/p2"], []),
        "yolov8n-640": (["/p1", "/p3"], ["coco/3"]),
    }


@pytest.mark.asyncio
async def test_request_failing_the_model_lookup_keeps_the_request(fake_stat):
    bridge = LegacyModelBridge(FakeGateway())

    with pytest.raises(LookupError):
        await bridge.resolve("coco/3", "k", row_key="coco/3", path="/p1")

    assert bridge._pending_requests == {"coco/3": ({"/p1"}, set())}
    assert await bridge.describe() == []

    fake_stat["coco/3"] = ("object-detection", "infer")
    await _recorded_resolve(bridge, "coco/3", "/p2")

    assert _rows(await bridge.describe()) == {"coco/3": (["/p1", "/p2"], [])}


@pytest.mark.asyncio
async def test_request_refused_in_the_model_lookup_keeps_the_request(fake_stat):
    from inference_models.errors import UnauthorizedModelAccessError

    fake_stat["coco/3"] = UnauthorizedModelAccessError(message="coco/3", help_url="")
    bridge = LegacyModelBridge(FakeGateway())

    with pytest.raises(PermissionError):
        await bridge.resolve(
            "coco/3", "k", row_key="coco/3", path="/p1", alias="alias-1"
        )

    assert bridge._pending_requests == {"coco/3": ({"/p1"}, {"alias-1"})}
    assert await bridge.describe() == []

    fake_stat["coco/3"] = ("object-detection", "infer")
    await _recorded_resolve(bridge, "coco/3", "/p2")

    assert _rows(await bridge.describe()) == {"coco/3": (["/p1", "/p2"], ["alias-1"])}


@pytest.mark.asyncio
async def test_request_whose_load_failed_survives_unload_remove_and_unload_all(
    fake_stat,
):
    fake_stat["coco/3"] = ("object-detection", "infer")
    bridge = LegacyModelBridge(FakeGateway())

    await _failed_resolve(bridge, "coco/3", row_key="coco/3", path="/p1")
    await bridge.remove("coco/3")
    await bridge.unload("coco/3")
    await bridge.unload_all()
    await _recorded_resolve(bridge, "coco/3", "/p2")

    assert _rows(await bridge.describe()) == {"coco/3": (["/p1", "/p2"], [])}


@pytest.mark.asyncio
async def test_request_whose_load_failed_leaves_with_the_removed_row(fake_stat):
    fake_stat["coco/3"] = ("object-detection", "infer")
    bridge = LegacyModelBridge(FakeGateway())

    await _failed_resolve(bridge, "coco/3", row_key="coco/3", path="/p1")
    await _recorded_resolve(bridge, "coco/3", "/p2")
    await bridge.remove("coco/3")
    await _recorded_resolve(bridge, "coco/3", "/p3")

    assert _rows(await bridge.describe()) == {"coco/3": (["/p3"], [])}


@pytest.mark.asyncio
async def test_load_failing_in_the_gateway_load_keeps_the_request(fake_stat):
    fake_stat["coco/3"] = ("object-detection", "infer")

    class _FailingLoadGateway(FakeGateway):
        async def load(self, model_id, api_key="", timeout_s=None, pinned=True):
            return ("error", 3)

    bridge = LegacyModelBridge(_FailingLoadGateway())

    with pytest.raises(LegacyHTTPError):
        await bridge.load("coco/3", "k", row_key="coco/3", path="/model/add")

    assert _rows(await bridge.describe()) == {}

    await _recorded_resolve(bridge, "coco/3", "/p2")

    assert _rows(await bridge.describe()) == {"coco/3": (["/model/add", "/p2"], [])}


@pytest.mark.asyncio
async def test_load_failing_in_resolve_keeps_the_request_once(fake_stat):
    fake_stat["coco/3"] = ("object-detection", "infer")
    bridge = LegacyModelBridge(FakeGateway())
    bridge.gateway.ensure_results.append(("error", 3))

    with pytest.raises(LegacyHTTPError):
        await bridge.load("coco/3", "k", row_key="coco/3", path="/model/add")
    route = await bridge.load("coco/3", "k", row_key="coco/3", path="/start/coco/3")
    bridge.record_request(route, "coco/3", "/start/coco/3")

    assert _rows(await bridge.describe()) == {
        "coco/3": (["/model/add", "/start/coco/3"], [])
    }


@pytest.mark.asyncio
async def test_offline_load_failure_keeps_the_request(fake_stat, monkeypatch):
    monkeypatch.setattr("inference_server.legacy.bridge.OFFLINE_MODE", True)
    bridge = LegacyModelBridge(FakeGateway())
    bridge.gateway.ensure_results.append(("error", 5))

    with pytest.raises(LegacyHTTPError) as error:
        await bridge.resolve("ds/1", None, row_key="ds/1", path="/p1")
    await _recorded_resolve(bridge, "ds/1", "/p2")

    assert error.value.status_code == 404
    assert _rows(await bridge.describe()) == {"ds/1": (["/p1", "/p2"], [])}


@pytest.mark.asyncio
async def test_failed_reload_inside_infer_does_not_join_a_failed_request(
    fake_stat, monkeypatch
):
    fake_stat["coco/3"] = ("object-detection", "infer")
    clock = {"now": 10.0}
    monkeypatch.setattr(bridge_mod, "_clock", lambda: clock["now"])
    gateway = FakeGateway(predictions={("coco/3", "infer"): "pred"})
    bridge = LegacyModelBridge(gateway)
    route = await _recorded_resolve(bridge, "coco/3", "/p2")
    await _failed_resolve(bridge, "coco/3", row_key="coco/3", path="/p1")

    clock["now"] = 20.0
    gateway.ensure_results.append(("error", 3))
    with pytest.raises(LegacyHTTPError):
        await bridge.infer(route, "k", "infer", [ImagePayload(b"a", 1, 1)], {})

    assert route.request_paths_by_id == {"coco/3": {"/p2": 20.0}}

    clock["now"] = 30.0
    await _recorded_resolve(bridge, "coco/3", "/p3")

    assert route.request_paths_by_id == {
        "coco/3": {"/p1": 30.0, "/p2": 30.0, "/p3": 30.0}
    }


@pytest.mark.asyncio
async def test_successful_reload_inside_infer_leaves_a_failed_request_pending(
    fake_stat, monkeypatch
):
    fake_stat["coco/3"] = ("object-detection", "infer")
    clock = {"now": 10.0}
    monkeypatch.setattr(bridge_mod, "_clock", lambda: clock["now"])
    gateway = _StampingGateway(clock, predictions={("coco/3", "infer"): "pred"})
    bridge = LegacyModelBridge(gateway)
    route = await _recorded_resolve(bridge, "coco/3", "/p2")
    await _failed_resolve(bridge, "coco/3", row_key="coco/3", path="/p1")
    gateway.loaded.pop("coco/3")

    clock["now"] = 20.0
    await bridge.infer(route, "k", "infer", [ImagePayload(b"a", 1, 1)], {})

    assert bridge._pending_requests == {"coco/3": ({"/p1"}, set())}
    assert _rows(await bridge.describe()) == {"coco/3": (["/p2"], [])}

    await _recorded_resolve(bridge, "coco/3", "/p3")

    assert bridge._pending_requests == {}
    assert _rows(await bridge.describe()) == {"coco/3": (["/p1", "/p2", "/p3"], [])}


def test_workflow_rerecording_a_discarded_route_leaves_the_failed_request_for_the_new_route(
    fake_stat, server_loop
):
    loop, _ = server_loop
    fake_stat["ds/1"] = ("object-detection", "infer")
    gateway = FakeGateway()
    bridge = LegacyModelBridge(gateway)
    sync = SyncLegacyBridge(bridge, LoopBridge(loop))

    def _failed_request():
        gateway.ensure_results.append(("error", 3))
        with pytest.raises(LegacyHTTPError):
            sync.resolve("ds/1", "k", row_key="ds/1", path="/failed")

    def _workflow():
        first = sync.resolve("ds/1", "k", row_key="ds/1", path="/workflows/run")
        sync.record_request(first, "ds/1", "/workflows/run")
        asyncio.run_coroutine_threadsafe(bridge.unload("ds/1"), loop).result(5)
        contextvars.copy_context().run(_failed_request)
        second = sync.resolve("ds/1", "k", row_key="ds/1", path="/workflows/run")
        sync.record_request(second, "ds/1", "/workflows/run")

    contextvars.copy_context().run(_workflow)
    routes = asyncio.run_coroutine_threadsafe(bridge.describe(), loop).result(5)

    assert _rows(routes) == {"ds/1": (["/failed", "/workflows/run"], [])}
    assert bridge._pending_requests == {}


@pytest.mark.asyncio
async def test_pending_request_stays_until_the_current_route_records(fake_stat):
    fake_stat["coco/3"] = ("object-detection", "infer")
    bridge = LegacyModelBridge(FakeGateway())
    await _failed_resolve(bridge, "coco/3", row_key="coco/3", path="/p1")
    first = await bridge.resolve("coco/3", "k")
    await bridge.unload("coco/3")
    second = await bridge.resolve("coco/3", "k")

    bridge.record_request(first, "coco/3", "/p2")

    assert bridge._pending_requests == {"coco/3": ({"/p1"}, set())}

    bridge.record_request(second, "coco/3", "/p3")

    assert bridge._pending_requests == {}
    assert _rows([second]) == {"coco/3": (["/p1", "/p3"], [])}


@pytest.mark.asyncio
async def test_cancelled_resolve_keeps_the_request(fake_stat, monkeypatch):
    fake_stat["coco/3"] = ("object-detection", "infer")
    bridge = LegacyModelBridge(FakeGateway())
    started = asyncio.Event()
    real_stat = bridge._stat

    async def blocking_stat(model_id, registry_id, api_key):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(bridge, "_stat", blocking_stat)
    task = asyncio.ensure_future(
        bridge.resolve("coco/3", "k", row_key="coco/3", path="/cancelled")
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert bridge._pending_requests == {"coco/3": ({"/cancelled"}, set())}

    monkeypatch.setattr(bridge, "_stat", real_stat)
    await _recorded_resolve(bridge, "coco/3", "/p2")

    assert _rows(await bridge.describe()) == {"coco/3": (["/cancelled", "/p2"], [])}


@pytest.mark.asyncio
async def test_cancelled_load_keeps_the_request(fake_stat):
    fake_stat["coco/3"] = ("object-detection", "infer")
    started = asyncio.Event()

    class _BlockingLoadGateway(FakeGateway):
        async def load(self, model_id, api_key="", timeout_s=None, pinned=True):
            started.set()
            await asyncio.Event().wait()

    bridge = LegacyModelBridge(_BlockingLoadGateway())
    task = asyncio.ensure_future(
        bridge.load("coco/3", "k", row_key="coco/3", path="/cancelled")
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert bridge._pending_requests == {"coco/3": ({"/cancelled"}, set())}


def test_workflow_registration_whose_load_failed_joins_the_alias_row(
    fake_stat, server_loop
):
    from inference_server.workflows.models_provider import GatewayModelsProvider

    loop, _ = server_loop
    fake_stat["ds/1"] = ("object-detection", "infer")
    fake_stat["ds/2"] = ("object-detection", "infer")
    gateway = FakeGateway()
    bridge = LegacyModelBridge(gateway)
    sync = SyncLegacyBridge(bridge, LoopBridge(loop))

    def _failed_workflow():
        provider = GatewayModelsProvider(sync, "k", "/workflows/run")
        gateway.ensure_results.append(("error", 3))
        with pytest.raises(LegacyHTTPError):
            provider.add_model("ds/1", "k", model_id_alias="alias-1")
        provider.add_model("ds/2", "k")

    def _workflow():
        provider = GatewayModelsProvider(sync, "k", "/infer/workflows/ws/wf")
        provider.add_model("ds/1", "k", model_id_alias="alias-1")

    contextvars.copy_context().run(_failed_workflow)
    routes = asyncio.run_coroutine_threadsafe(bridge.describe(), loop).result(5)

    assert _rows(routes) == {"ds/2": (["/workflows/run"], [])}

    contextvars.copy_context().run(_workflow)
    routes = asyncio.run_coroutine_threadsafe(bridge.describe(), loop).result(5)

    assert _rows(routes) == {
        "ds/2": (["/workflows/run"], []),
        "alias-1": (["/infer/workflows/ws/wf", "/workflows/run"], ["ds/1"]),
    }
