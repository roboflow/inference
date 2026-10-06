import json

import pytest

from inference_models.models.auto_loaders.model_cache_paths import (
    slugify_model_id_to_os_safe_format,
)
from inference_server import configuration
from tests.unit_tests.legacy.conftest import FakeGateway

CLOUD_ONLY_BLOCK = "roboflow_core/open_ai@v4"
FOUNDATION_BLOCK = "roboflow_core/clip_comparison@v2"
PLAIN_BLOCK = "roboflow_core/dynamic_crop@v1"


@pytest.fixture
def cache_roots(tmp_path, monkeypatch):
    server_root = tmp_path / "server_cache"
    models_home = tmp_path / "models_home"
    server_root.mkdir()
    models_home.mkdir()
    monkeypatch.setattr(configuration, "MODEL_CACHE_DIR", str(server_root))
    monkeypatch.setattr(configuration, "INFERENCE_HOME", str(models_home))
    monkeypatch.setattr(configuration, "ENABLE_BUILDER", True)
    return server_root, models_home


def _air_gapped_info(response, block_type):
    for block in response.json()["blocks"]:
        if block["manifest_type_identifier"] == block_type:
            return (
                block["block_schema"]
                .get("json_schema_extra", {})
                .get("air_gapped_info")
            )
    raise AssertionError(f"{block_type} not described")


def _cache_variant_in_inference_models_layout(root, model_id):
    package_dir = (
        root / "models-cache" / slugify_model_id_to_os_safe_format(model_id) / "pkga"
    )
    package_dir.mkdir(parents=True)
    (package_dir / "model_config.json").write_text(
        json.dumps({"model_id": model_id, "task_type": "embedding"})
    )


def _cache_variant_in_traditional_layout(root, model_id):
    from inference_server.builder import model_cache

    model_dir = root / model_cache.get_model_id_cache_path(
        model_id=model_id, cache_dir_root=str(root)
    )
    model_dir.mkdir(parents=True)
    (model_dir / "textual.onnx").write_bytes(b"x")


@pytest.mark.parametrize("method", ["get", "post"])
def test_air_gapped_describe_reports_availability_per_block(
    legacy_client, cache_roots, method
):
    client = legacy_client(FakeGateway())

    response = getattr(client, method)("/workflows/blocks/describe?air_gapped=true")

    assert response.status_code == 200, response.text
    assert _air_gapped_info(response, CLOUD_ONLY_BLOCK)["available"] is False
    assert _air_gapped_info(response, PLAIN_BLOCK) == {"available": True}
    info = _air_gapped_info(response, FOUNDATION_BLOCK)
    assert info["available"] is False
    assert info["reason"] == "missing_cache_artifacts"
    assert info["model_id"] == "clip/RN101"


def test_air_gapped_describe_leaves_the_response_alone_when_not_requested(
    legacy_client, cache_roots
):
    client = legacy_client(FakeGateway())

    response = client.get("/workflows/blocks/describe")

    assert response.status_code == 200, response.text
    assert _air_gapped_info(response, CLOUD_ONLY_BLOCK) is None
    assert _air_gapped_info(response, FOUNDATION_BLOCK) is None


def test_air_gapped_describe_is_ignored_when_the_builder_is_disabled(
    legacy_client, cache_roots, monkeypatch
):
    monkeypatch.setattr(configuration, "ENABLE_BUILDER", False)
    client = legacy_client(FakeGateway())

    response = client.get("/workflows/blocks/describe?air_gapped=true")

    assert response.status_code == 200, response.text
    assert _air_gapped_info(response, CLOUD_ONLY_BLOCK) is None


def test_air_gapped_describe_sees_a_variant_in_the_inference_models_layout(
    legacy_client, cache_roots
):
    _, models_home = cache_roots
    _cache_variant_in_inference_models_layout(models_home, "clip/RN50")
    client = legacy_client(FakeGateway())

    response = client.get("/workflows/blocks/describe?air_gapped=true")

    info = _air_gapped_info(response, FOUNDATION_BLOCK)
    assert info["available"] is True
    assert "reason" not in info
    assert info["model_id"] == "clip/RN101"


def test_air_gapped_describe_sees_a_variant_in_the_traditional_layout(
    legacy_client, cache_roots
):
    server_root, _ = cache_roots
    _cache_variant_in_traditional_layout(server_root, "clip/RN50")
    client = legacy_client(FakeGateway())

    response = client.get("/workflows/blocks/describe?air_gapped=true")

    assert _air_gapped_info(response, FOUNDATION_BLOCK)["available"] is True
