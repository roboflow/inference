import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

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
                "tasks": {"infer": {}},
                "model_class_name": "X",
            }
        }
    )
    bridge = LegacyModelBridge(gw)
    route = await bridge.resolve("ds/1", "key")
    assert (
        route.task_type == "object-detection"
        and route.class_names == ["a", "b"]
        and route.tasks == {"infer"}
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
    assert canonical.request_aliases == {"yolov8n-640"}
    assert canonical.request_paths == {"/infer/object_detection", "/yolov8n-640"}


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
