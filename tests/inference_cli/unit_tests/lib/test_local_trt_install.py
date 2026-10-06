import hashlib
import json
import os

import pytest

pytest.importorskip(
    "inference_models",
    reason="inference_models is required to exercise the local TRT installer",
)

from inference_cli.lib.enterprise.inference_compiler.core.entities import (
    TRTConfig,
    TRTModelPackageV1,
)
from inference_cli.lib.enterprise.inference_compiler.core.local_trt_install import (
    install_compiled_trt_package,
)
from inference_models.models.auto_loaders import model_cache_paths
from inference_models.models.auto_loaders.model_cache_paths import (
    generate_shared_blobs_path,
)
from inference_models.weights_providers import offline_registry
from inference_models.weights_providers.local_trt_constants import (
    LOCAL_TRT_MANIFEST_FILE,
    LOCAL_TRT_PACKAGE_PREFIX,
)

_PACKAGE_MANIFEST_PAYLOAD = {
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

_ENGINE_BYTES = b"engine-bytes"


@pytest.fixture
def inference_home(tmp_path, monkeypatch):
    monkeypatch.setattr(model_cache_paths, "INFERENCE_HOME", str(tmp_path))
    monkeypatch.setattr(offline_registry, "INFERENCE_HOME", str(tmp_path))
    return tmp_path


def _run_installer(
    model_id: str,
    compilation_directory: str,
    with_keypoints_metadata: bool = False,
):
    os.makedirs(compilation_directory, exist_ok=True)
    engine_path = os.path.join(compilation_directory, "compiled.plan")
    with open(engine_path, "wb") as file:
        file.write(_ENGINE_BYTES)
    inference_config_path = os.path.join(
        compilation_directory, "raw_inference_config.json"
    )
    with open(inference_config_path, "w", encoding="utf-8") as file:
        json.dump({"network_input": {}}, file)
    class_names_path = os.path.join(compilation_directory, "class_names_source.txt")
    with open(class_names_path, "w", encoding="utf-8") as file:
        file.write("class-a\n")
    keypoints_metadata_path = None
    if with_keypoints_metadata:
        keypoints_metadata_path = os.path.join(
            compilation_directory, "keypoints_source.json"
        )
        with open(keypoints_metadata_path, "w", encoding="utf-8") as file:
            json.dump({"skeleton": []}, file)
    return install_compiled_trt_package(
        model_id=model_id,
        model_architecture="rfdetr",
        task_type="object-detection",
        package_manifest=TRTModelPackageV1.model_validate(_PACKAGE_MANIFEST_PAYLOAD),
        trt_config=TRTConfig(static_batch_size=1),
        engine_path=engine_path,
        inference_config_path=inference_config_path,
        class_names_path=class_names_path,
        compilation_directory=compilation_directory,
        keypoints_metadata_path=keypoints_metadata_path,
    )


def test_install_materializes_all_artefacts_as_regular_files(
    inference_home, tmp_path
):
    # when
    package_id, install_dir = _run_installer(
        model_id="workspace/rfdetr-nano",
        compilation_directory=str(tmp_path / "compilation"),
        with_keypoints_metadata=True,
    )

    # then
    assert package_id.startswith(LOCAL_TRT_PACKAGE_PREFIX)
    expected_files = {
        "engine.plan",
        "inference_config.json",
        "class_names.txt",
        "trt_config.json",
        "keypoints_metadata.json",
        LOCAL_TRT_MANIFEST_FILE,
    }
    assert expected_files.issubset(set(os.listdir(install_dir)))
    for file_name in expected_files:
        file_path = os.path.join(install_dir, file_name)
        # Symlinked artefacts are rejected by local TRT discovery, which made
        # every CLI-installed package invisible to the loader.
        assert not os.path.islink(file_path), f"{file_name} must not be a symlink"
        assert os.path.isfile(file_path)
    with open(os.path.join(install_dir, "engine.plan"), "rb") as file:
        assert file.read() == _ENGINE_BYTES


def test_install_still_writes_shared_blob_copies(inference_home, tmp_path):
    # when
    _, install_dir = _run_installer(
        model_id="workspace/rfdetr-nano",
        compilation_directory=str(tmp_path / "compilation"),
    )

    # then
    with open(
        os.path.join(install_dir, LOCAL_TRT_MANIFEST_FILE), encoding="utf-8"
    ) as file:
        file_md5 = json.load(file)["files"]
    shared_blobs_dir = generate_shared_blobs_path()
    for handle, md5_hash in file_md5.items():
        blob_path = os.path.join(shared_blobs_dir, md5_hash)
        assert os.path.isfile(blob_path), f"missing shared blob for {handle}"


def test_install_appends_cli_install_record_to_offline_registry(
    inference_home, tmp_path
):
    # when
    package_id, install_dir = _run_installer(
        model_id="workspace/rfdetr-nano",
        compilation_directory=str(tmp_path / "compilation"),
    )

    # then
    record = offline_registry.load_record_raw(model_id="workspace/rfdetr-nano")
    assert record is not None
    assert record["source"] == offline_registry.RECORD_SOURCE_CLI_INSTALL
    assert record["source"] == "cli-install"
    assert record["canonical_model_id"] == "workspace/rfdetr-nano"
    assert package_id in record["proven"]
    assert [package["package_id"] for package in record["packages"]] == [package_id]
    recorded_package = record["packages"][0]
    assert recorded_package["backend"] == "trt"
    assert recorded_package["trusted_source"] is False
    assert recorded_package["cache_model_id"] == "workspace/rfdetr-nano"
    with open(os.path.join(install_dir, LOCAL_TRT_MANIFEST_FILE), "rb") as file:
        manifest_md5 = hashlib.md5(file.read()).hexdigest()
    recorded_artifacts = {
        artefact["file_handle"]: artefact["md5_hash"]
        for artefact in recorded_package["artifacts"]
    }
    assert recorded_artifacts[LOCAL_TRT_MANIFEST_FILE] == manifest_md5
    assert set(recorded_artifacts.keys()) == {
        "engine.plan",
        "inference_config.json",
        "class_names.txt",
        "trt_config.json",
        LOCAL_TRT_MANIFEST_FILE,
    }


def test_install_succeeds_when_offline_registry_append_fails(
    inference_home, tmp_path, monkeypatch
):
    # given
    def broken_record(*args, **kwargs):
        raise RuntimeError("registry on fire")

    monkeypatch.setattr(offline_registry, "record_successful_load", broken_record)

    # when
    package_id, install_dir = _run_installer(
        model_id="workspace/rfdetr-nano",
        compilation_directory=str(tmp_path / "compilation"),
    )

    # then - the install completed despite the registry failure
    assert package_id.startswith(LOCAL_TRT_PACKAGE_PREFIX)
    assert os.path.isfile(os.path.join(install_dir, "engine.plan"))
    assert os.path.isfile(os.path.join(install_dir, LOCAL_TRT_MANIFEST_FILE))
    assert offline_registry.load_record_raw(model_id="workspace/rfdetr-nano") is None


def test_reinstall_of_same_package_is_idempotent(inference_home, tmp_path):
    # when
    first_package_id, first_install_dir = _run_installer(
        model_id="workspace/rfdetr-nano",
        compilation_directory=str(tmp_path / "compilation-1"),
    )
    second_package_id, second_install_dir = _run_installer(
        model_id="workspace/rfdetr-nano",
        compilation_directory=str(tmp_path / "compilation-2"),
    )

    # then
    assert first_package_id == second_package_id
    assert first_install_dir == second_install_dir
    for file_name in ("engine.plan", LOCAL_TRT_MANIFEST_FILE):
        file_path = os.path.join(second_install_dir, file_name)
        assert not os.path.islink(file_path)
        assert os.path.isfile(file_path)
    record = offline_registry.load_record_raw(model_id="workspace/rfdetr-nano")
    assert [package["package_id"] for package in record["packages"]] == [
        first_package_id
    ]
