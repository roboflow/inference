import asyncio
import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from inference_models.errors import (
    ForbiddenModelAccessError,
    ModelInputError,
    ModelNotFoundError,
    ModelPackageAlternativesExhaustedError,
    ModelPackageRestrictedError,
    ModelRetrievalError,
    NoModelPackagesAvailableError,
    PaymentRequiredModelAccessError,
    UnauthorizedModelAccessError,
    UsagePausedModelAccessError,
)

from inference_server.framework.model_stat import ModelStat
from inference_server.gateway import ModelManagerGateway
from inference_server.legacy import bridge as bridge_mod
from inference_server.legacy.bridge import (
    LegacyModelBridge,
    LoopBridge,
    SyncLegacyBridge,
    resolved_model_for,
)
from inference_server.legacy.common import ImagePayload
from inference_server.legacy.errors import LegacyHTTPError
from inference_server.legacy.load_failures import (
    ModelLoadFailedError,
    load_failure_error,
)
from inference_server.usage.request_hook import MODEL_INVOCATIONS
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
async def test_resolve_owlv2_static_fallback_is_object_detection(fake_stat):
    route = await LegacyModelBridge(FakeGateway()).resolve(
        "owlv2/owlv2-base-patch16-ensemble", "k"
    )
    assert (route.task_type, route.action) == ("object-detection", "infer")


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
    monkeypatch.setattr("inference_server.legacy.bridge.LEGACY_OFFLINE_MODE", True)
    gw = FakeGateway(
        model_info={
            "ds/1": {"model_mro_names": ["ClassificationModel"], "class_names": ["a"]}
        }
    )
    route = await LegacyModelBridge(gw).resolve("ds/1", None)
    assert route.task_type == "classification" and fake_stat == {}


@pytest.mark.asyncio
async def test_offline_mode_load_failure_is_404(fake_stat, monkeypatch):
    monkeypatch.setattr("inference_server.legacy.bridge.LEGACY_OFFLINE_MODE", True)
    gw = FakeGateway()
    gw.ensure_results = [("error", 5)]
    with pytest.raises(LegacyHTTPError) as exc:
        await LegacyModelBridge(gw).resolve("ds/1", None)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_model_layer_offline_skips_registry_when_server_setting_is_online(
    fake_stat, monkeypatch
):
    monkeypatch.setattr("inference_server.configuration.OFFLINE_MODE", False)
    monkeypatch.setattr("inference_server.legacy.bridge.LEGACY_OFFLINE_MODE", True)
    fake_stat["ds/1"] = ("object-detection", "infer")
    gw = FakeGateway(
        model_info={
            "ds/1": {"model_mro_names": ["ClassificationModel"], "class_names": ["a"]}
        }
    )

    route = await LegacyModelBridge(gw).resolve("ds/1", None)

    assert route.task_type == "classification"


@pytest.mark.asyncio
async def test_model_layer_online_consults_registry_when_server_setting_is_offline(
    fake_stat, monkeypatch
):
    monkeypatch.setattr("inference_server.configuration.OFFLINE_MODE", True)
    monkeypatch.setattr("inference_server.legacy.bridge.LEGACY_OFFLINE_MODE", False)
    fake_stat["ds/1"] = ("object-detection", "infer")
    gw = FakeGateway(
        model_info={
            "ds/1": {"model_mro_names": ["ClassificationModel"], "class_names": ["a"]}
        }
    )

    route = await LegacyModelBridge(gw).resolve("ds/1", None)

    assert route.task_type == "object-detection"


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
    outcome = [ModelStat("structured-ocr", "infer")]

    async def _stat(common):
        calls.append((common.model_id, common.api_key))
        if isinstance(outcome[0], Exception):
            raise outcome[0]
        return outcome[0]

    monkeypatch.setattr(
        "inference_server.legacy.bridge.stat_model_details_while_checking_auth", _stat
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


def test_pending_request_bounds_are_fixed():
    assert bridge_mod._MAX_PENDING_REQUEST_KEYS == 256
    assert bridge_mod._MAX_PENDING_VALUES_PER_KEY == 64


def test_pending_requests_drop_the_least_recently_held_keys(monkeypatch):
    monkeypatch.setattr(bridge_mod, "_MAX_PENDING_REQUEST_KEYS", 4)
    bridge = LegacyModelBridge(FakeGateway())

    for index in range(4 + 5):
        bridge._hold_pending_request(f"m/{index}", "/p", None)

    assert list(bridge._pending_requests) == [f"m/{index}" for index in range(5, 9)]


def test_pending_request_held_again_moves_to_the_newest_position(monkeypatch):
    monkeypatch.setattr(bridge_mod, "_MAX_PENDING_REQUEST_KEYS", 3)
    bridge = LegacyModelBridge(FakeGateway())
    bridge._hold_pending_request("m/0", "/p1", None)
    bridge._hold_pending_request("m/1", "/p1", None)
    bridge._hold_pending_request("m/2", "/p1", None)

    bridge._hold_pending_request("m/0", "/p2", None)
    bridge._hold_pending_request("m/3", "/p1", None)

    assert list(bridge._pending_requests) == ["m/2", "m/0", "m/3"]
    assert bridge._pending_requests["m/0"] == ({"/p1", "/p2"}, set())


def test_pending_request_values_per_key_are_bounded(monkeypatch):
    monkeypatch.setattr(bridge_mod, "_MAX_PENDING_VALUES_PER_KEY", 3)
    bridge = LegacyModelBridge(FakeGateway())

    for index in range(3 + 5):
        bridge._hold_pending_request("m/0", f"/p{index}", f"a{index}")

    assert bridge._pending_requests == {
        "m/0": ({"/p0", "/p1", "/p2"}, {"a0", "a1", "a2"})
    }


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
    monkeypatch.setattr("inference_server.legacy.bridge.LEGACY_OFFLINE_MODE", True)
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


HELP_URL = "https://help.example/errors"


def _failure(error_type, message="boom", **detail):
    return ("error", 5, {"error_type": error_type, "message": message, **detail})


@pytest.mark.parametrize(
    "error_class,status_code",
    [
        (UnauthorizedModelAccessError, None),
        (ModelNotFoundError, None),
        (PaymentRequiredModelAccessError, 402),
        (ForbiddenModelAccessError, 403),
        (UsagePausedModelAccessError, 423),
        (NoModelPackagesAvailableError, None),
        (ModelPackageRestrictedError, None),
    ],
)
def test_load_failure_error_rebuilds_the_described_class(error_class, status_code):
    error = load_failure_error(
        _failure(
            error_class.__name__,
            "boom",
            help_url=HELP_URL,
            status_code=status_code,
            restricted=False,
        )
    )

    assert type(error) is error_class
    assert error.args == ("boom",)
    assert error.help_url == HELP_URL
    assert str(error) == f"boom - VISIT {HELP_URL} FOR FURTHER SUPPORT"
    assert getattr(error, "status_code", None) == status_code


def test_load_failure_error_keeps_a_status_the_class_does_not_carry():
    error = load_failure_error(_failure("ModelRetrievalError", status_code=403))

    assert type(error) is ModelRetrievalError
    assert error.status_code == 403
    assert str(error) == "boom"


@pytest.mark.parametrize("restricted", [True, False])
def test_load_failure_error_rebuilds_exhausted_alternatives(restricted):
    error = load_failure_error(
        _failure(
            "ModelPackageAlternativesExhaustedError",
            help_url=HELP_URL,
            restricted=restricted,
        )
    )

    assert type(error) is ModelPackageAlternativesExhaustedError
    assert error.help_url == HELP_URL
    assert [type(alternative) for alternative in error.alternatives_errors] == (
        [ModelPackageRestrictedError] if restricted else []
    )


@pytest.mark.parametrize(
    "error_type",
    ["RuntimeError", "PermissionError", "LookupError", "Optional", "List", "x" * 64],
)
def test_load_failure_error_of_an_unknown_kind_is_a_load_failure_named_after_it(
    error_type,
):
    error = load_failure_error(_failure(error_type))

    assert type(error) is not ModelLoadFailedError
    assert isinstance(error, ModelLoadFailedError)
    assert type(error).__name__ == error_type
    assert type(error) is type(load_failure_error(_failure(error_type)))
    assert str(error) == "boom"
    assert not isinstance(error, (RuntimeError, LookupError, PermissionError))


@pytest.mark.parametrize(
    "error_type",
    [None, 7, "", "not an identifier", "a.b", "class", "None", "x" * 65],
)
def test_load_failure_error_with_an_unusable_class_name_is_a_plain_load_failure(
    error_type,
):
    error = load_failure_error(_failure(error_type))

    assert type(error) is ModelLoadFailedError
    assert str(error) == "boom"


def test_load_failure_error_class_names_are_bounded(monkeypatch):
    from inference_server.legacy import load_failures

    monkeypatch.setattr(load_failures, "_LOAD_FAILURE_CLASSES", {})
    monkeypatch.setattr(load_failures, "_MAX_LOAD_FAILURE_CLASSES", 2)

    names = [
        type(load_failure_error(_failure(error_type))).__name__
        for error_type in ["FirstError", "SecondError", "ThirdError", "FirstError"]
    ]

    assert names == ["FirstError", "SecondError", "ModelLoadFailedError", "FirstError"]
    assert sorted(load_failures._LOAD_FAILURE_CLASSES) == ["FirstError", "SecondError"]


@pytest.mark.parametrize(
    "result", [("error",), ("error", 5), ("error", 5, None), ("error", 5, "text")]
)
def test_load_failure_error_without_a_description_is_none(result):
    assert load_failure_error(result) is None


def _route(registry_id="ds/1"):
    return SimpleNamespace(registry_id=registry_id)


def _assert_not_ready(error):
    assert error.status_code == 503
    assert error.message == "Model is temporarily not ready - retry request."
    assert error.extra == {}
    assert error.headers == {"Retry-After": "1"}


@pytest.mark.asyncio
async def test_ensure_loaded_raises_the_error_the_gateway_described():
    gateway = FakeGateway()
    gateway.ensure_results = [_failure("ModelNotFoundError", "missing")]

    with pytest.raises(ModelNotFoundError) as exc:
        await LegacyModelBridge(gateway).ensure_loaded(_route(), None)

    assert str(exc.value) == "missing"


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [("error", 5), ("error", 3), ("error",), ()])
async def test_ensure_loaded_failure_without_a_description_is_a_broken_package(result):
    gateway = FakeGateway()
    gateway.ensure_loaded = lambda *args: _returning(result)

    with pytest.raises(LegacyHTTPError) as exc:
        await LegacyModelBridge(gateway).ensure_loaded(_route(), None)

    assert exc.value.status_code == 500
    assert exc.value.message == "Model package is broken."
    assert exc.value.extra == {}
    assert exc.value.headers == {}


async def _returning(result):
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result", [("error", 6), ("error", 6, {"error_type": "KeyError", "message": "m"})]
)
async def test_ensure_loaded_not_loaded_code_is_not_ready_with_retry_after(result):
    gateway = FakeGateway()
    gateway.ensure_results = [result]

    with pytest.raises(LegacyHTTPError) as exc:
        await LegacyModelBridge(gateway).ensure_loaded(_route(), None)

    _assert_not_ready(exc.value)


@pytest.mark.asyncio
async def test_ensure_loaded_deadline_is_not_ready_with_retry_after(monkeypatch):
    monkeypatch.setattr(bridge_mod, "LEGACY_LOAD_POLL_INTERVAL_S", 0)
    monkeypatch.setattr(bridge_mod, "LEGACY_LOAD_TIMEOUT_S", 0)
    gateway = FakeGateway()
    gateway.ensure_results = [("load_timeout", 10)]

    with pytest.raises(LegacyHTTPError) as exc:
        await LegacyModelBridge(gateway).ensure_loaded(_route(), None)

    _assert_not_ready(exc.value)


@pytest.mark.asyncio
async def test_explicit_load_timeout_is_not_ready_with_retry_after(fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")

    class _TimingOutGateway(FakeGateway):
        async def load(self, model_id, api_key="", timeout_s=None, pinned=True):
            raise asyncio.TimeoutError()

    bridge = LegacyModelBridge(_TimingOutGateway())

    with pytest.raises(LegacyHTTPError) as exc:
        await bridge.load("ds/1", "k", row_key="ds/1", path="/model/add")

    _assert_not_ready(exc.value)
    assert bridge._pending_requests == {"ds/1": ({"/model/add"}, set())}


@pytest.mark.asyncio
async def test_inference_timeout_is_not_turned_into_not_ready(fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")

    def _time_out(image, params):
        raise asyncio.TimeoutError()

    bridge = LegacyModelBridge(FakeGateway(predictions={("ds/1", "infer"): _time_out}))
    route = await bridge.resolve("ds/1", None)

    with pytest.raises(asyncio.TimeoutError):
        await bridge.infer(route, None, "infer", [ImagePayload(b"a", 1, 1)], {})


@pytest.mark.asyncio
async def test_explicit_load_raises_the_error_the_gateway_described(fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")

    class _FailingLoadGateway(FakeGateway):
        async def load(self, model_id, api_key="", timeout_s=None, pinned=True):
            return _failure("PaymentRequiredModelAccessError", "no credits")

    bridge = LegacyModelBridge(_FailingLoadGateway())

    with pytest.raises(PaymentRequiredModelAccessError) as exc:
        await bridge.load("ds/1", "k", row_key="ds/1", path="/model/add")

    assert exc.value.status_code == 402
    assert bridge._pending_requests == {"ds/1": ({"/model/add"}, set())}


class _SlowFailingManager:
    def __init__(self, error, fail_after_s):
        self.error = error
        self.fail_after_s = fail_after_s
        self.load_calls = 0
        self.executor = None

    def __contains__(self, key):
        return False

    def load(self, key, api_key, **kwargs):
        import time

        self.load_calls += 1
        time.sleep(self.fail_after_s)
        raise self.error


@pytest.mark.asyncio
async def test_load_failing_between_two_polls_is_reported_and_not_started_again(
    monkeypatch,
):
    monkeypatch.setattr(bridge_mod, "LEGACY_LOAD_POLL_INTERVAL_S", 0.4)
    monkeypatch.setattr(bridge_mod, "LEGACY_LOAD_TIMEOUT_S", 3)
    manager = _SlowFailingManager(ModelNotFoundError("missing"), fail_after_s=0.2)
    bridge = LegacyModelBridge(ModelManagerGateway(manager, load_wait_s=0.05))

    with pytest.raises(ModelNotFoundError) as exc:
        await bridge.ensure_loaded(_route(), None)

    assert str(exc.value) == "missing"
    assert manager.load_calls == 1


@pytest.mark.asyncio
async def test_poll_ignores_a_gateway_that_reports_no_last_failure(monkeypatch):
    monkeypatch.setattr(bridge_mod, "LEGACY_LOAD_POLL_INTERVAL_S", 0)
    gateway = FakeGateway()
    gateway.ensure_results = [("load_timeout", 10)]
    gateway.last_load_failure = lambda model_id, instance="": None

    await LegacyModelBridge(gateway).ensure_loaded(_route(), None)

    assert [call[0] for call in gateway.calls] == ["ensure_loaded", "ensure_loaded"]


def test_workflow_thread_gets_the_error_the_gateway_described(server_loop):
    loop, _ = server_loop
    manager = _SlowFailingManager(UnauthorizedModelAccessError("denied"), 0)
    bridge = LegacyModelBridge(ModelManagerGateway(manager))
    sync = SyncLegacyBridge(bridge, LoopBridge(loop))

    with pytest.raises(UnauthorizedModelAccessError) as exc:
        sync.ensure_loaded(_route(), "k")

    assert str(exc.value) == "denied"


@pytest.mark.asyncio
async def test_route_carries_the_architecture_and_variant_of_the_registry(fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer", "yolov8", "yolov8-n")

    route = await LegacyModelBridge(FakeGateway()).resolve("ds/1", "k")

    assert (route.model_architecture, route.model_variant) == ("yolov8", "yolov8-n")


@pytest.mark.asyncio
async def test_route_without_registry_labels_carries_none(fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")

    route = await LegacyModelBridge(FakeGateway()).resolve("ds/1", "k")

    assert (route.model_architecture, route.model_variant) == (None, None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_id,architecture,variant",
    [
        ("clip/ViT-B-16", "clip", "ViT-B-16"),
        (
            "grounding_dino/groundingdino_swint_ogc",
            "grounding-dino",
            "groundingdino_swint_ogc",
        ),
        ("yolo_world/l", "yolo-world", "l"),
        (
            "smolvlm2/smolvlm-2.2b-instruct",
            "smolvlm-2.2b-instruct",
            "smolvlm-2.2b-instruct",
        ),
        ("perception_encoder/PE-Core-L14-336", "perception_encoder", "PE-Core-L14-336"),
        ("sam2/hiera_large", "sam2", "hiera_large"),
        ("owlv2/owlv2-base-patch16-ensemble", "owlv2", "owlv2-base-patch16-ensemble"),
    ],
)
async def test_core_model_fallback_reports_the_legacy_architecture_and_variant(
    fake_stat, model_id, architecture, variant
):
    route = await LegacyModelBridge(FakeGateway()).resolve(model_id, "k")

    assert (route.model_architecture, route.model_variant) == (architecture, variant)


@pytest.mark.asyncio
async def test_infer_appends_one_model_invocation_to_the_request_holder(
    fake_stat, monkeypatch
):
    fake_stat["coco/3"] = ("object-detection", "infer", "yolov8", "yolov8-n")
    gw = FakeGateway(
        predictions={("coco/3", "infer"): lambda img, p: ("pred", img)},
        model_info={"coco/3": {"input_height": 640, "input_width": 480}},
    )
    bridge = LegacyModelBridge(gw)
    ticks = iter([10.0, 10.25])
    monkeypatch.setattr(bridge_mod.time, "perf_counter", lambda: next(ticks))
    holder = []
    token = MODEL_INVOCATIONS.set(holder)
    try:
        route = await bridge.resolve("yolov8n-640", "k")
        await bridge.infer(
            route,
            "k",
            "infer",
            [ImagePayload(b"a", 1, 1), ImagePayload(b"b", 1, 1)],
            {},
        )
    finally:
        MODEL_INVOCATIONS.reset(token)

    assert holder == [
        {
            "model_id": "yolov8n-640",
            "model_architecture": "yolov8",
            "model_variant": "yolov8-n",
            "task_type": "object-detection",
            "model_input_height": 640,
            "model_input_width": 480,
            "execution_duration": 0.25,
            "frames": 2,
        }
    ]


@pytest.mark.asyncio
async def test_text_only_call_counts_one_frame_and_omits_unknown_labels(fake_stat):
    gw = FakeGateway(predictions={("clip/ViT-B-16", "embed_text"): [[1.0]]})
    bridge = LegacyModelBridge(gw)
    holder = []
    token = MODEL_INVOCATIONS.set(holder)
    try:
        route = await bridge.resolve("clip/ViT-B-16", "k")
        await bridge.infer_params_only(route, "k", "embed_text", {"texts": ["a"]})
    finally:
        MODEL_INVOCATIONS.reset(token)

    assert len(holder) == 1
    assert holder[0]["frames"] == 1
    assert holder[0]["model_id"] == "clip/ViT-B-16"
    assert "model_input_height" not in holder[0]
    assert "model_input_width" not in holder[0]


@pytest.mark.asyncio
async def test_infer_without_a_request_holder_appends_nothing(fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gw = FakeGateway(predictions={("ds/1", "infer"): lambda img, p: ("pred", img)})
    bridge = LegacyModelBridge(gw)
    route = await bridge.resolve("ds/1", "k")

    assert MODEL_INVOCATIONS.get() is None
    await bridge.infer(route, "k", "infer", [ImagePayload(b"a", 1, 1)], {})
    assert MODEL_INVOCATIONS.get() is None


@pytest.mark.asyncio
async def test_failed_infer_appends_the_attempted_invocation_and_reraises(
    fake_stat, monkeypatch
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gw = FakeGateway(predictions={("ds/1", "infer"): _raising(ValueError("bad shape"))})
    bridge = LegacyModelBridge(gw)
    ticks = iter([10.0, 10.5])
    monkeypatch.setattr(bridge_mod.time, "perf_counter", lambda: next(ticks))
    holder = []
    token = MODEL_INVOCATIONS.set(holder)
    try:
        route = await bridge.resolve("ds/1", "k")
        with pytest.raises(ModelInputError):
            await bridge.infer(
                route,
                "k",
                "infer",
                [ImagePayload(b"a", 1, 1), ImagePayload(b"b", 1, 1)],
                {},
            )
    finally:
        MODEL_INVOCATIONS.reset(token)

    assert holder == [
        {
            "model_id": "ds/1",
            "task_type": "object-detection",
            "execution_duration": 0.5,
            "frames": 2,
        }
    ]


def test_task_type_from_mro_covers_registry():
    from inference_model_manager.registry_defaults import _ACTION_CONFIGS

    unmapped = [
        name
        for name in _ACTION_CONFIGS
        if name not in bridge_mod._TASK_TYPE_BY_MRO
        and name not in bridge_mod._NO_HTTP_ROUTE
    ]

    assert unmapped == []


def test_task_type_from_mro_prefers_the_concrete_class():
    assert (
        bridge_mod._task_type_from_mro(
            ["OWLv2HF", "OpenVocabularyObjectDetectionModel"]
        )
        == "open-vocabulary-object-detection"
    )
    assert bridge_mod._task_type_from_mro(["Qwen35HF", "object"]) == "vlm"
    assert bridge_mod._task_type_from_mro(["L2CSNetOnnx", "object"]) == "gaze-detection"


@pytest.mark.parametrize(
    "mro_names",
    [
        ["Cosmos3EdgeActionRecognition", "ActionRecognitionModel", "ABC", "object"],
        ["SomeFineTune", "ActionRecognitionModel", "ABC", "object"],
    ],
)
def test_task_type_from_mro_maps_action_recognition(mro_names):
    assert bridge_mod._task_type_from_mro(mro_names) == "action-recognition"


@pytest.mark.asyncio
async def test_resolve_fills_video_sampling_from_metadata(fake_stat):
    fake_stat["clips/1"] = ("action-recognition", "infer")
    sampling = {
        "window_seconds": 8.0,
        "sample_fps": 2.0,
        "min_frames": 4,
        "max_frame_side": 720,
        "mode": "sliding_window",
        "max_frames": 16,
    }
    gw = FakeGateway(
        model_info={
            "clips/1": {
                "class_names": None,
                "actions": {"infer": {}},
                "model_class_name": "Cosmos3EdgeActionRecognition",
                "video_sampling": sampling,
            }
        }
    )
    bridge = LegacyModelBridge(gw)

    route = await bridge.resolve("clips/1", "key")

    assert route.task_type == "action-recognition"
    assert route.action == "infer"
    assert route.video_sampling == sampling


@pytest.mark.asyncio
async def test_resolve_leaves_video_sampling_none_for_other_models(fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gw = FakeGateway(
        model_info={"ds/1": {"class_names": ["a"], "actions": {"infer": {}}}}
    )
    bridge = LegacyModelBridge(gw)

    route = await bridge.resolve("ds/1", "key")

    assert route.video_sampling is None


@pytest.mark.asyncio
async def test_offline_stream_only_model_is_400(fake_stat, monkeypatch):
    monkeypatch.setattr("inference_server.legacy.bridge.LEGACY_OFFLINE_MODE", True)
    gw = FakeGateway(
        model_info={
            "sam2-rt/1": {
                "model_mro_names": ["SAM2ForStream", "object"],
                "model_class_name": "SAM2ForStream",
            }
        }
    )
    with pytest.raises(LegacyHTTPError) as exc:
        await LegacyModelBridge(gw).resolve("sam2-rt/1", None)
    assert exc.value.status_code == 400
    assert "streaming" in exc.value.message
