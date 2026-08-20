import hashlib
import json
import os
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from inference_models.models.auto_loaders import core as auto_loader_core
from inference_models.models.auto_loaders import model_cache_paths
from inference_models.models.auto_loaders.core import (
    initialize_model,
    parse_model_config,
)
from inference_models.models.auto_loaders.entities import BackendType
from inference_models.models.auto_loaders.model_cache_paths import (
    generate_model_cache_root_for_model_id,
    generate_model_package_cache_path,
    generate_shared_blobs_path,
    resolve_existing_model_package_cache_path,
)
from inference_models.weights_providers import local_trt_packages, offline_registry
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
    assert LOCAL_TRT_MANIFEST_FILE in {
        artefact.file_handle for artefact in package.package_artefacts
    }


def test_discovered_local_trt_package_initializes_with_bound_source_manifest(
    local_trt_layout,
):
    package = discover_local_trt_packages(model_id=local_trt_layout["model_id"])[0]
    model_class = MagicMock()
    model_class.from_pretrained.return_value = MagicMock()

    with patch.object(
        auto_loader_core,
        "resolve_model_class",
        return_value=model_class,
    ):
        _, package_dir = initialize_model(
            model_id=local_trt_layout["model_id"],
            model_architecture="rfdetr",
            task_type="object-detection",
            model_package=package,
            model_init_kwargs={},
            auto_resolution_cache=MagicMock(),
            auto_negotiation_hash="a" * 64,
            model_dependencies=[],
            model_dependencies_instances={},
            model_dependencies_directories={},
        )

    manifest = parse_model_config(
        config_path=os.path.join(package_dir, "model_config.json")
    )
    assert LOCAL_TRT_MANIFEST_FILE in {
        artifact["file_handle"] for artifact in manifest.package_artifacts
    }
    model_class.from_pretrained.assert_called_once()


def test_discovered_local_trt_package_reloads_offline_without_cache_writes(
    local_trt_layout,
):
    package = discover_local_trt_packages(model_id=local_trt_layout["model_id"])[0]
    model_class = MagicMock()
    model_class.from_pretrained.return_value = MagicMock()
    auto_resolution_cache = MagicMock()
    initialize_kwargs = {
        "model_id": local_trt_layout["model_id"],
        "model_architecture": "rfdetr",
        "task_type": "object-detection",
        "model_package": package,
        "model_init_kwargs": {},
        "auto_resolution_cache": auto_resolution_cache,
        "auto_negotiation_hash": "a" * 64,
        "model_dependencies": [],
        "model_dependencies_instances": {},
        "model_dependencies_directories": {},
    }

    with patch.object(
        auto_loader_core,
        "resolve_model_class",
        return_value=model_class,
    ):
        initialize_model(**initialize_kwargs)
        auto_resolution_cache.reset_mock()
        with (
            patch.object(auto_loader_core, "OFFLINE_MODE", True),
            patch.object(
                auto_loader_core,
                "FileLock",
                side_effect=AssertionError(
                    "offline initialization attempted to create a lock"
                ),
            ),
            patch.object(
                auto_loader_core,
                "dump_model_config_for_offline_use",
                side_effect=AssertionError(
                    "offline initialization attempted to publish a manifest"
                ),
            ),
        ):
            _, package_dir = initialize_model(**initialize_kwargs)

    assert package_dir == local_trt_layout["package_dir"]
    assert model_class.from_pretrained.call_count == 2
    auto_resolution_cache.register.assert_not_called()
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


def test_local_trt_md5_is_streamed_in_bounded_chunks(tmp_path):
    artefact_path = tmp_path / "engine.plan"
    content = b"a" * (3 * 1024 * 1024 + 17)
    artefact_path.write_bytes(content)
    real_fdopen = os.fdopen
    read_sizes = []

    class TrackingFile:
        def __init__(self, file_handle):
            self._file_handle = file_handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self._file_handle.close()

        def read(self, size):
            read_sizes.append(size)
            return self._file_handle.read(size)

        def fileno(self):
            return self._file_handle.fileno()

    def tracking_fdopen(file_descriptor, mode):
        return TrackingFile(real_fdopen(file_descriptor, mode))

    with patch.object(
        local_trt_packages.os,
        "fdopen",
        side_effect=tracking_fdopen,
    ):
        actual_md5 = local_trt_packages._md5_regular_file(str(artefact_path))

    assert actual_md5 == hashlib.md5(content).hexdigest()
    assert len(read_sizes) > 2
    assert set(read_sizes) == {1024 * 1024}


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


def _install_package_with_cli_installer(model_id: str, compilation_directory: str):
    """Build a package by calling the real CLI installer (installer-faithful layout)."""
    pytest.importorskip(
        "inference_cli",
        reason="the CLI local TRT installer is required for installer-faithful tests",
    )
    from inference_cli.lib.enterprise.inference_compiler.core.entities import (
        TRTConfig,
        TRTModelPackageV1,
    )
    from inference_cli.lib.enterprise.inference_compiler.core.local_trt_install import (
        install_compiled_trt_package,
    )

    os.makedirs(compilation_directory, exist_ok=True)
    engine_path = os.path.join(compilation_directory, "compiled.plan")
    with open(engine_path, "wb") as file:
        file.write(b"engine-bytes")
    inference_config_path = os.path.join(
        compilation_directory, "raw_inference_config.json"
    )
    with open(inference_config_path, "w", encoding="utf-8") as file:
        json.dump({"network_input": {}}, file)
    class_names_path = os.path.join(compilation_directory, "class_names_source.txt")
    with open(class_names_path, "w", encoding="utf-8") as file:
        file.write("class-a\n")
    package_manifest = TRTModelPackageV1.model_validate(
        {
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
    )
    return install_compiled_trt_package(
        model_id=model_id,
        model_architecture="rfdetr",
        task_type="object-detection",
        package_manifest=package_manifest,
        trt_config=TRTConfig(static_batch_size=1),
        engine_path=engine_path,
        inference_config_path=inference_config_path,
        class_names_path=class_names_path,
        compilation_directory=compilation_directory,
    )


@pytest.fixture
def cli_installed_package(tmp_path, monkeypatch):
    monkeypatch.setattr(model_cache_paths, "INFERENCE_HOME", str(tmp_path))
    monkeypatch.setattr(offline_registry, "INFERENCE_HOME", str(tmp_path))
    model_id = "workspace/rfdetr-nano"
    package_id, install_dir = _install_package_with_cli_installer(
        model_id=model_id,
        compilation_directory=str(tmp_path / "compilation"),
    )
    return {
        "model_id": model_id,
        "package_id": package_id,
        "install_dir": install_dir,
    }


def test_cli_installed_package_is_discovered(cli_installed_package):
    # Regression test for the symlink defect: the installer used to materialize
    # artefacts as symlinks to shared blobs, which discovery rejects — every
    # CLI-installed package was silently invisible to the loader.
    discovered = discover_local_trt_packages(
        model_id=cli_installed_package["model_id"]
    )

    assert [package.package_id for package in discovered] == [
        cli_installed_package["package_id"]
    ]
    package = discovered[0]
    assert package.backend == BackendType.TRT
    assert package.package_source == PackageSourceType.LOCAL_CACHE
    assert package.trusted_source is False
    assert package.cache_model_id == cli_installed_package["model_id"]
    assert {artefact.file_handle for artefact in package.package_artefacts} == {
        "engine.plan",
        "inference_config.json",
        "class_names.txt",
        "trt_config.json",
        LOCAL_TRT_MANIFEST_FILE,
    }


def test_cli_installed_package_contains_only_regular_files(cli_installed_package):
    install_dir = cli_installed_package["install_dir"]
    entries = sorted(os.listdir(install_dir))
    assert {
        "engine.plan",
        "inference_config.json",
        "class_names.txt",
        "trt_config.json",
        LOCAL_TRT_MANIFEST_FILE,
    }.issubset(set(entries))
    for entry in entries:
        entry_path = os.path.join(install_dir, entry)
        assert not os.path.islink(entry_path), f"{entry} must not be a symlink"
        assert os.path.isfile(entry_path)


def test_cli_installed_package_is_recorded_in_offline_registry(cli_installed_package):
    record = offline_registry.load_record_raw(
        model_id=cli_installed_package["model_id"]
    )

    assert record is not None
    assert record["source"] == offline_registry.RECORD_SOURCE_CLI_INSTALL
    assert record["source"] == "cli-install"
    assert cli_installed_package["package_id"] in record["proven"]
    recorded_metadata = offline_registry.load_model_metadata(
        model_id=cli_installed_package["model_id"]
    )
    assert recorded_metadata is not None
    assert recorded_metadata.model_architecture == "rfdetr"
    assert recorded_metadata.task_type == "object-detection"
    recorded_package = next(
        package
        for package in recorded_metadata.model_packages
        if package.package_id == cli_installed_package["package_id"]
    )
    discovered_package = discover_local_trt_packages(
        model_id=cli_installed_package["model_id"]
    )[0]
    recorded_artefacts = {
        (artefact.file_handle, artefact.md5_hash)
        for artefact in recorded_package.package_artefacts
    }
    discovered_artefacts = {
        (artefact.file_handle, artefact.md5_hash)
        for artefact in discovered_package.package_artefacts
    }
    assert recorded_artefacts == discovered_artefacts
    assert recorded_package.backend == discovered_package.backend
    assert recorded_package.trusted_source is False
    assert recorded_package.cache_model_id == cli_installed_package["model_id"]
    assert (
        recorded_package.trt_package_details == discovered_package.trt_package_details
    )
    assert (
        recorded_package.environment_requirements
        == discovered_package.environment_requirements
    )
