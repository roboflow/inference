import hashlib
import json
import os
from unittest.mock import patch

import pytest

from inference_models.models.auto_loaders.core import (
    generate_model_cache_root_for_model_id,
    generate_model_package_cache_path,
    generate_shared_blobs_path,
)
from inference_models.models.auto_loaders.entities import BackendType
from inference_models.weights_providers.entities import (
    LocalFileArtefactSpecs,
    PackageSourceType,
)
from inference_models.weights_providers.local_trt_constants import LOCAL_TRT_MANIFEST_FILE
from inference_models.weights_providers.local_trt_packages import discover_local_trt_packages
from inference_models.weights_providers.roboflow import get_roboflow_model


def _write_file(path: str, content: bytes) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file:
        file.write(content)
    return hashlib.md5(content).hexdigest()


@pytest.fixture
def local_trt_layout(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCE_HOME", str(tmp_path))
    model_id = "workspace/rfdetr-nano"
    package_id = "localtrtabc123"
    package_dir = generate_model_package_cache_path(model_id=model_id, package_id=package_id)
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
        _write_file(os.path.join(shared_blobs_dir, md5_hash), b"shared-" + md5_hash.encode())

    manifest = {
        "packageManifest": {
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
        },
        "files": {
            "engine.plan": engine_md5,
            "inference_config.json": config_md5,
            "class_names.txt": class_names_md5,
            "trt_config.json": trt_config_md5,
        },
        "modelArchitecture": "rfdetr",
        "taskType": "object-detection",
    }
    with open(os.path.join(package_dir, LOCAL_TRT_MANIFEST_FILE), "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    return {
        "model_id": model_id,
        "package_id": package_id,
        "package_dir": package_dir,
    }


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


def test_generate_model_cache_root_matches_package_parent(local_trt_layout):
    cache_root = generate_model_cache_root_for_model_id(model_id=local_trt_layout["model_id"])
    assert local_trt_layout["package_dir"].startswith(cache_root)


@patch("inference_models.weights_providers.roboflow.get_model_metadata")
def test_get_roboflow_model_merges_discovered_local_packages(
    get_model_metadata_mock, local_trt_layout
):
    from inference_models.weights_providers.roboflow import RoboflowModelMetadata

    get_model_metadata_mock.return_value = RoboflowModelMetadata.model_validate(
        {
            "type": "external-model-metadata-v1",
            "modelId": local_trt_layout["model_id"],
            "modelArchitecture": "rfdetr",
            "taskType": "object-detection",
            "modelPackages": [],
        }
    )
    metadata = get_roboflow_model(model_id=local_trt_layout["model_id"], api_key="k")
    package_ids = {package.package_id for package in metadata.model_packages}
    assert local_trt_layout["package_id"] in package_ids
