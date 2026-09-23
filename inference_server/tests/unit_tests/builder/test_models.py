import json
import sys
import time
import types
from collections import defaultdict

import pytest
from starlette.testclient import TestClient

from inference_sdk.http.utils.aliases import REGISTERED_ALIASES
from inference_server import configuration
from inference_server.builder import models

from .conftest import FakeBridge, make_route

_BLOCKS_LOADER_MODULE = (
    "roboflow_workflows.execution_engine.introspection.blocks_loader"
)


def _make_block(identifier, task_types=None, variants=None):
    manifest_class = type(
        "FakeManifest",
        (),
        {
            "get_compatible_task_types": staticmethod(lambda: task_types),
            "get_supported_model_variants": staticmethod(lambda: variants),
            "model_json_schema": staticmethod(lambda: {}),
        },
    )
    return type(
        "FakeBlock",
        (),
        {"manifest_class": manifest_class, "identifier": identifier},
    )()


def _install_blocks(monkeypatch, blocks):
    module = types.ModuleType(_BLOCKS_LOADER_MODULE)
    module.load_workflow_blocks = lambda: blocks
    monkeypatch.setitem(sys.modules, _BLOCKS_LOADER_MODULE, module)


def _install_failing_blocks(monkeypatch, error):
    def _raise():
        raise error

    module = types.ModuleType(_BLOCKS_LOADER_MODULE)
    module.load_workflow_blocks = _raise
    monkeypatch.setitem(sys.modules, _BLOCKS_LOADER_MODULE, module)


@pytest.fixture
def models_cache_dir(tmp_path, monkeypatch):
    cache_dir = tmp_path / "models-cache"
    cache_dir.mkdir()
    monkeypatch.setattr(models, "generate_models_cache_dir", lambda: str(cache_dir))
    return cache_dir


def _write_package(models_cache_dir, model_id, package_id, config):
    from inference_models.models.auto_loaders.model_cache_paths import (
        slugify_model_id_to_os_safe_format,
    )

    package_dir = (
        models_cache_dir
        / slugify_model_id_to_os_safe_format(model_id=model_id)
        / package_id
    )
    package_dir.mkdir(parents=True)
    (package_dir / "model_config.json").write_text(json.dumps(config))
    return package_dir


_DETECTION_BLOCK = "roboflow_core/object_detection_model@v2"
_CLASSIFICATION_BLOCK = "roboflow_core/classification_model@v2"


def _task_blocks():
    return [
        _make_block(_DETECTION_BLOCK, task_types=["object-detection"]),
        _make_block(_CLASSIFICATION_BLOCK, task_types=["classification"]),
    ]


@pytest.mark.asyncio
async def test_lists_model_manager_routes(monkeypatch, models_cache_dir):
    _install_blocks(monkeypatch, _task_blocks())
    bridge = FakeBridge(
        [
            make_route("ws/od/1", "object-detection"),
            make_route("ws/cls/2", "classification"),
        ]
    )

    listed = await models.list_models(bridge)

    by_id = {entry["model_id"]: entry for entry in listed}
    assert set(by_id) == {"ws/od/1", "ws/cls/2"}
    assert by_id["ws/od/1"] == {
        "model_id": "ws/od/1",
        "name": "ws/od/1",
        "task_type": "object-detection",
        "model_architecture": "",
        "is_foundation": False,
        "compatible_block_types": [_DETECTION_BLOCK],
        "aliases": [],
    }
    assert by_id["ws/cls/2"]["compatible_block_types"] == [_CLASSIFICATION_BLOCK]
    assert by_id["ws/cls/2"]["is_foundation"] is False


@pytest.mark.asyncio
async def test_lists_models_from_inference_models_cache(monkeypatch, models_cache_dir):
    _install_blocks(monkeypatch, _task_blocks())
    _write_package(
        models_cache_dir,
        "ws/seg/3",
        "pkga",
        {
            "model_id": "ws/seg/3",
            "task_type": "instance-segmentation",
            "model_architecture": "rfdetr",
        },
    )

    listed = await models.list_models(FakeBridge([]))

    by_id = {entry["model_id"]: entry for entry in listed}
    assert by_id["ws/seg/3"]["task_type"] == "instance-segmentation"
    assert by_id["ws/seg/3"]["model_architecture"] == "rfdetr"
    assert by_id["ws/seg/3"]["is_foundation"] is False
    assert by_id["ws/seg/3"]["name"] == "ws/seg/3"


@pytest.mark.asyncio
async def test_skips_symlinked_package_directory(
    monkeypatch, models_cache_dir, tmp_path
):
    _install_blocks(monkeypatch, [])
    from inference_models.models.auto_loaders.model_cache_paths import (
        slugify_model_id_to_os_safe_format,
    )

    outside_package = tmp_path / "outside-package"
    outside_package.mkdir()
    (outside_package / "model_config.json").write_text(
        json.dumps({"model_id": "ws/linked/4", "task_type": "object-detection"})
    )
    model_root = models_cache_dir / slugify_model_id_to_os_safe_format(
        model_id="ws/linked/4"
    )
    model_root.mkdir()
    (model_root / "pkga").symlink_to(outside_package, target_is_directory=True)

    listed = await models.list_models(FakeBridge([]))

    assert [entry["model_id"] for entry in listed] == []


@pytest.mark.asyncio
async def test_skips_symlinked_model_config_file(
    monkeypatch, models_cache_dir, tmp_path
):
    _install_blocks(monkeypatch, [])
    from inference_models.models.auto_loaders.model_cache_paths import (
        slugify_model_id_to_os_safe_format,
    )

    outside_config = tmp_path / "outside-config.json"
    outside_config.write_text(
        json.dumps({"model_id": "ws/linked-config/6", "task_type": "classification"})
    )
    package_dir = (
        models_cache_dir
        / slugify_model_id_to_os_safe_format(model_id="ws/linked-config/6")
        / "pkga"
    )
    package_dir.mkdir(parents=True)
    (package_dir / "model_config.json").symlink_to(outside_config)

    listed = await models.list_models(FakeBridge([]))

    assert [entry["model_id"] for entry in listed] == []


@pytest.mark.asyncio
async def test_skips_cached_config_with_non_string_task_type(
    monkeypatch, models_cache_dir
):
    _install_blocks(monkeypatch, [])
    _write_package(
        models_cache_dir,
        "ws/broken/7",
        "pkga",
        {"model_id": "ws/broken/7", "task_type": ["classification"]},
    )
    _write_package(
        models_cache_dir,
        "ws/good/8",
        "pkga",
        {"model_id": "ws/good/8", "task_type": "classification"},
    )

    listed = await models.list_models(FakeBridge([]))

    assert [entry["model_id"] for entry in listed] == ["ws/good/8"]


@pytest.mark.asyncio
async def test_drops_model_with_conflicting_cached_metadata(
    monkeypatch, models_cache_dir
):
    _install_blocks(monkeypatch, [])
    _write_package(
        models_cache_dir,
        "ws/conflict/5",
        "pkga",
        {"model_id": "ws/conflict/5", "task_type": "object-detection"},
    )
    from inference_models.models.auto_loaders.model_cache_paths import (
        slugify_model_id_to_os_safe_format,
    )

    second_package = (
        models_cache_dir
        / slugify_model_id_to_os_safe_format(model_id="ws/conflict/5")
        / "pkgb"
    )
    second_package.mkdir()
    (second_package / "model_config.json").write_text(
        json.dumps({"model_id": "ws/conflict/5", "task_type": "classification"})
    )

    listed = await models.list_models(FakeBridge([]))

    assert [entry["model_id"] for entry in listed] == []


@pytest.mark.asyncio
async def test_lists_cached_foundation_model(monkeypatch, models_cache_dir):
    foundation_block = _make_block("roboflow_core/clip@v1", variants=["clip/ViT-B-16"])
    _install_blocks(monkeypatch, [foundation_block])
    _write_package(models_cache_dir, "clip/ViT-B-16", "pkga", {})

    listed = await models.list_models(FakeBridge([]))

    by_id = {entry["model_id"]: entry for entry in listed}
    assert by_id["clip/ViT-B-16"]["is_foundation"] is True
    assert by_id["clip/ViT-B-16"]["task_type"] == ""
    assert by_id["clip/ViT-B-16"]["compatible_block_types"] == ["roboflow_core/clip@v1"]


@pytest.mark.asyncio
async def test_symlinked_foundation_slug_directory_is_not_cached(
    monkeypatch, models_cache_dir, tmp_path
):
    foundation_block = _make_block("roboflow_core/clip@v1", variants=["clip/ViT-B-16"])
    _install_blocks(monkeypatch, [foundation_block])
    from inference_models.models.auto_loaders.model_cache_paths import (
        slugify_model_id_to_os_safe_format,
    )

    outside_model_root = tmp_path / "outside-model-root"
    (outside_model_root / "pkga").mkdir(parents=True)
    (outside_model_root / "pkga" / "model_config.json").write_text("{}")
    (
        models_cache_dir / slugify_model_id_to_os_safe_format(model_id="clip/ViT-B-16")
    ).symlink_to(outside_model_root, target_is_directory=True)

    listed = await models.list_models(FakeBridge([]))

    assert [entry["model_id"] for entry in listed] == []


@pytest.mark.asyncio
async def test_uncached_foundation_model_is_absent(monkeypatch, models_cache_dir):
    foundation_block = _make_block("roboflow_core/clip@v1", variants=["clip/ViT-B-16"])
    _install_blocks(monkeypatch, [foundation_block])

    listed = await models.list_models(FakeBridge([]))

    assert [entry["model_id"] for entry in listed] == []


@pytest.mark.asyncio
async def test_offline_mode_hides_models_absent_from_registry(
    monkeypatch, models_cache_dir
):
    from inference_models.weights_providers import offline_registry

    _install_blocks(monkeypatch, [])
    monkeypatch.setattr(configuration, "OFFLINE_MODE", True)
    record = offline_registry.OfflineModelStatus(
        canonical_model_id="ws/od/1",
        requested_aliases=[],
        source="warmup",
        recorded_at=None,
        proven={},
        packages=[
            offline_registry.OfflinePackageStatus(
                package_id="pkga",
                trusted_source=True,
                presence=offline_registry.OfflinePackagePresence.OK,
            )
        ],
    )
    monkeypatch.setattr(offline_registry, "list_records_status", lambda: [record])
    bridge = FakeBridge(
        [
            make_route("ws/od/1", "object-detection"),
            make_route("ws/cls/2", "classification"),
        ]
    )

    listed = await models.list_models(bridge)

    assert [entry["model_id"] for entry in listed] == ["ws/od/1"]


@pytest.mark.asyncio
async def test_offline_mode_keeps_listing_when_registry_unreadable(
    monkeypatch, models_cache_dir
):
    from inference_models.weights_providers import offline_registry

    _install_blocks(monkeypatch, [])
    monkeypatch.setattr(configuration, "OFFLINE_MODE", True)

    def _raise():
        raise RuntimeError("registry unreadable")

    monkeypatch.setattr(offline_registry, "list_records_status", _raise)
    bridge = FakeBridge([make_route("ws/od/1", "object-detection")])

    listed = await models.list_models(bridge)

    assert [entry["model_id"] for entry in listed] == ["ws/od/1"]


@pytest.mark.asyncio
async def test_listing_survives_block_loader_import_error(
    monkeypatch, models_cache_dir
):
    _install_failing_blocks(monkeypatch, ImportError("no roboflow_workflows"))
    _write_package(
        models_cache_dir,
        "ws/seg/3",
        "pkga",
        {"model_id": "ws/seg/3", "task_type": "instance-segmentation"},
    )
    bridge = FakeBridge([make_route("ws/od/1", "object-detection")])

    listed = await models.list_models(bridge)

    by_id = {entry["model_id"]: entry for entry in listed}
    assert set(by_id) == {"ws/od/1", "ws/seg/3"}
    assert by_id["ws/od/1"]["compatible_block_types"] == []
    assert by_id["ws/seg/3"]["compatible_block_types"] == []


@pytest.mark.asyncio
async def test_aliases_are_attached_and_shortest_alias_becomes_name(
    monkeypatch, models_cache_dir
):
    _install_blocks(monkeypatch, [])
    reverse_aliases = defaultdict(list)
    for alias, canonical in REGISTERED_ALIASES.items():
        reverse_aliases[canonical].append(alias)
    canonical_model_id = REGISTERED_ALIASES["yolov8n-640"]
    expected_aliases = reverse_aliases[canonical_model_id]
    bridge = FakeBridge([make_route(canonical_model_id, "object-detection")])

    listed = await models.list_models(bridge)

    assert listed[0]["aliases"] == expected_aliases
    assert listed[0]["name"] == min(expected_aliases, key=len)


@pytest.mark.asyncio
async def test_alias_used_as_model_id_gets_no_aliases(monkeypatch, models_cache_dir):
    _install_blocks(monkeypatch, [])
    bridge = FakeBridge([make_route("yolov8n-640", "object-detection")])

    listed = await models.list_models(bridge)

    assert listed[0]["model_id"] == "yolov8n-640"
    assert listed[0]["aliases"] == []
    assert listed[0]["name"] == "yolov8n-640"


@pytest.mark.asyncio
async def test_multi_alias_model_uses_shortest_alias_as_name(
    monkeypatch, models_cache_dir
):
    _install_blocks(monkeypatch, [])
    bridge = FakeBridge([make_route("coco-dataset-vdnr1/19", "instance-segmentation")])

    listed = await models.list_models(bridge)

    assert listed[0]["aliases"] == ["yolov11n-seg-640", "yolo11n-seg-640"]
    assert listed[0]["name"] == "yolo11n-seg-640"


def test_models_route_requires_csrf(builder_app):
    client = TestClient(builder_app)

    response = client.get("/build/api/models")

    assert response.status_code == 403


def test_models_route_returns_models_and_caches_them(
    builder_app, builder_env, monkeypatch
):
    calls = {"count": 0}
    payloads = [
        [{"model_id": "first", "name": "first"}],
        [{"model_id": "second", "name": "second"}],
    ]

    async def fake_list_models(bridge):
        result = payloads[min(calls["count"], len(payloads) - 1)]
        calls["count"] += 1
        return result

    monkeypatch.setattr(models, "list_models", fake_list_models)
    real_time = time.time
    offset = {"value": 0.0}
    monkeypatch.setattr(builder_env.time, "time", lambda: real_time() + offset["value"])
    client = TestClient(builder_app)

    first = client.get("/build/api/models", headers={"X-CSRF": builder_env.csrf})
    assert first.status_code == 200
    assert first.json() == {"models": [{"model_id": "first", "name": "first"}]}

    cached = client.get("/build/api/models", headers={"X-CSRF": builder_env.csrf})
    assert cached.json() == {"models": [{"model_id": "first", "name": "first"}]}
    assert calls["count"] == 1

    offset["value"] = 31.0
    refreshed = client.get("/build/api/models", headers={"X-CSRF": builder_env.csrf})
    assert refreshed.json() == {"models": [{"model_id": "second", "name": "second"}]}
    assert calls["count"] == 2
