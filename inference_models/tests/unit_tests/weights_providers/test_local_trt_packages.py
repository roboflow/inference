import hashlib
import json
import os
from typing import Optional
from unittest.mock import patch

import pytest

from inference_models.models.auto_loaders import model_cache_paths
from inference_models.models.auto_loaders.model_cache_paths import (
    generate_model_cache_root_for_model_id,
    generate_model_package_cache_path,
    generate_shared_blobs_path,
)
from inference_models.models.auto_loaders.entities import BackendType
from inference_models.weights_providers.entities import (
    LocalFileArtefactSpecs,
    PackageSourceType,
)
from inference_models.weights_providers.local_trt_constants import (
    LOCAL_TRT_MANIFEST_FILE,
)
from inference_models.weights_providers.local_trt_packages import (
    discover_local_trt_packages,
)
from inference_models.weights_providers.roboflow import get_roboflow_model


def _write_file(path: str, content: bytes) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file:
        file.write(content)
    return hashlib.md5(content).hexdigest()


def _build_local_trt_layout(
    model_id: str,
    package_id: str = "localtrtabc123",
    manifest_overrides: Optional[dict] = None,
    files_overrides: Optional[dict] = None,
) -> dict:
    package_dir = generate_model_package_cache_path(
        model_id=model_id, package_id=package_id
    )
    shared_blobs_dir = generate_shared_blobs_path()
    os.makedirs(package_dir, exist_ok=True)
    os.makedirs(shared_blobs_dir, exist_ok=True)

    engine_md5 = _write_file(os.path.join(package_dir, "engine.plan"), b"engine-bytes")
    config_md5 = _write_file(
        os.path.join(package_dir, "inference_config.json"), b'{"network_input": {}}'
    )
    class_names_md5 = _write_file(
        os.path.join(package_dir, "class_names.txt"), b"class-a\n"
    )
    trt_config_md5 = _write_file(
        os.path.join(package_dir, "trt_config.json"), b'{"static_batch_size": 1}'
    )
    for md5_hash in (engine_md5, config_md5, class_names_md5, trt_config_md5):
        _write_file(
            os.path.join(shared_blobs_dir, md5_hash), b"shared-" + md5_hash.encode()
        )

    package_manifest = {
        "type": "trt-model-package-v1",
        "backendType": "trt",
        "dynamicBatchSize": False,
        "staticBatchSize": 1,
        "quantization": "fp16",
        "cudaDeviceType": "Orin",
        "cudaDeviceCC": "8.7",
        "cudaVersion": "12.2",
        "trtVersion": "8.6.2",
        "sameCCCompatible": True,
        "trtForwardCompatible": False,
        "trtLeanRuntimeExcluded": False,
        "machineType": "jetson",
        "machineSpecs": {
            "type": "jetson-machine-specs-v1",
            "l4tVersion": "36.3",
            "deviceName": "jetson-orin-nano",
            "driverVersion": "540.3",
        },
    }
    if manifest_overrides:
        package_manifest.update(manifest_overrides)

    files = {
        "engine.plan": engine_md5,
        "inference_config.json": config_md5,
        "class_names.txt": class_names_md5,
        "trt_config.json": trt_config_md5,
    }
    if files_overrides:
        files.update(files_overrides)

    manifest = {
        "packageManifest": package_manifest,
        "files": files,
        "modelArchitecture": "rfdetr",
        "taskType": "object-detection",
    }
    with open(
        os.path.join(package_dir, LOCAL_TRT_MANIFEST_FILE), "w", encoding="utf-8"
    ) as f:
        json.dump(manifest, f)

    return {
        "model_id": model_id,
        "package_id": package_id,
        "package_dir": package_dir,
    }


@pytest.fixture
def local_trt_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(model_cache_paths, "INFERENCE_HOME", str(tmp_path))
    return _build_local_trt_layout(model_id="workspace/rfdetr-nano")


def _roboflow_metadata(resolved_model_id: str):
    from inference_models.weights_providers.roboflow import RoboflowModelMetadata

    return RoboflowModelMetadata.model_validate(
        {
            "type": "external-model-metadata-v1",
            "modelId": resolved_model_id,
            "modelArchitecture": "rfdetr",
            "taskType": "object-detection",
            "modelPackages": [],
        }
    )


def test_discover_local_trt_packages_returns_local_cache_metadata(local_trt_layout):
    discovered = discover_local_trt_packages(model_id=local_trt_layout["model_id"])
    assert len(discovered) == 1
    package = discovered[0]
    assert package.package_id == local_trt_layout["package_id"]
    assert package.backend == BackendType.TRT
    assert package.package_source == PackageSourceType.LOCAL_CACHE
    assert all(
        isinstance(artefact, LocalFileArtefactSpecs)
        for artefact in package.package_artefacts
    )


def test_discover_local_trt_packages_marks_untrusted_and_sets_cache_model_id(
    local_trt_layout,
):
    discovered = discover_local_trt_packages(model_id=local_trt_layout["model_id"])
    package = discovered[0]
    assert package.trusted_source is False
    assert package.cache_model_id == local_trt_layout["model_id"]


def test_discover_local_trt_packages_skips_md5_mismatch(local_trt_layout):
    engine_path = os.path.join(local_trt_layout["package_dir"], "engine.plan")
    with open(engine_path, "wb") as f:
        f.write(b"tampered-bytes")
    assert discover_local_trt_packages(model_id=local_trt_layout["model_id"]) == []


def test_discover_local_trt_packages_skips_invalid_md5_format(tmp_path, monkeypatch):
    monkeypatch.setattr(model_cache_paths, "INFERENCE_HOME", str(tmp_path))
    layout = _build_local_trt_layout(
        model_id="workspace/rfdetr-nano",
        files_overrides={"engine.plan": "not-a-valid-md5"},
    )
    assert discover_local_trt_packages(model_id=layout["model_id"]) == []


def test_discover_local_trt_packages_skips_corrupt_manifest(local_trt_layout):
    manifest_path = os.path.join(
        local_trt_layout["package_dir"], LOCAL_TRT_MANIFEST_FILE
    )
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("{not valid json")
    assert discover_local_trt_packages(model_id=local_trt_layout["model_id"]) == []


def test_discover_local_trt_packages_skips_bad_version_without_raising(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(model_cache_paths, "INFERENCE_HOME", str(tmp_path))
    layout = _build_local_trt_layout(
        model_id="workspace/rfdetr-nano",
        manifest_overrides={"trtVersion": "not-a-version"},
    )
    # Must not raise even though the version string is unparseable.
    assert discover_local_trt_packages(model_id=layout["model_id"]) == []


def test_generate_model_cache_root_matches_package_parent(local_trt_layout):
    cache_root = generate_model_cache_root_for_model_id(
        model_id=local_trt_layout["model_id"]
    )
    assert local_trt_layout["package_dir"].startswith(cache_root)


@patch("inference_models.weights_providers.roboflow.get_model_metadata")
def test_get_roboflow_model_merges_discovered_local_packages(
    get_model_metadata_mock, local_trt_layout
):
    get_model_metadata_mock.return_value = _roboflow_metadata(
        local_trt_layout["model_id"]
    )
    metadata = get_roboflow_model(model_id=local_trt_layout["model_id"], api_key="k")
    package_ids = {package.package_id for package in metadata.model_packages}
    assert local_trt_layout["package_id"] in package_ids


@patch("inference_models.weights_providers.roboflow.get_model_metadata")
def test_get_roboflow_model_discovers_local_packages_by_resolved_model_id(
    get_model_metadata_mock, tmp_path, monkeypatch
):
    monkeypatch.setattr(model_cache_paths, "INFERENCE_HOME", str(tmp_path))
    resolved_model_id = "workspace/coco-38"
    layout = _build_local_trt_layout(model_id=resolved_model_id)
    get_model_metadata_mock.return_value = _roboflow_metadata(resolved_model_id)

    metadata = get_roboflow_model(model_id="rfdetr-nano", api_key="k")

    local_packages = [
        package
        for package in metadata.model_packages
        if package.package_source == PackageSourceType.LOCAL_CACHE
    ]
    assert len(local_packages) == 1
    # The discovered package must point loading at the resolved id cache dir so
    # the alias request loads from the correct location.
    assert local_packages[0].cache_model_id == resolved_model_id


@patch("inference_models.weights_providers.roboflow.get_model_metadata")
def test_get_roboflow_model_survives_corrupt_local_cache(
    get_model_metadata_mock, tmp_path, monkeypatch
):
    monkeypatch.setattr(model_cache_paths, "INFERENCE_HOME", str(tmp_path))
    model_id = "workspace/rfdetr-nano"
    # A directory that looks like a local TRT package but has an unsafe id and
    # no manifest must not break model resolution.
    cache_root = generate_model_cache_root_for_model_id(model_id=model_id)
    os.makedirs(os.path.join(cache_root, "localtrt-bad-id"), exist_ok=True)
    get_model_metadata_mock.return_value = _roboflow_metadata(model_id)

    metadata = get_roboflow_model(model_id=model_id, api_key="k")

    assert metadata.model_id == model_id
