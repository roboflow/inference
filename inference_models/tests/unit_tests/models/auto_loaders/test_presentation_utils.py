import os
from unittest import mock

import pytest
from packaging.version import Version

from inference_models.models.auto_loaders import core as auto_loaders_core
from inference_models.models.auto_loaders import model_cache_paths
from inference_models.models.auto_loaders.core import AutoModel
from inference_models.models.auto_loaders.entities import BackendType
from inference_models.models.auto_loaders.presentation_utils import (
    calculate_artefacts_size,
    calculate_model_package_size,
    calculate_size_of_all_model_packages_artefacts,
    resolve_local_model_package_dir,
)
from inference_models.weights_providers import offline_registry
from inference_models.weights_providers.entities import (
    FileDownloadSpecs,
    LocalFileArtefactSpecs,
    ModelMetadata,
    ModelPackageMetadata,
    ServerEnvironmentRequirements,
)
from inference_models.weights_providers.roboflow_offline import (
    ROBOFLOW_OFFLINE_WEIGHTS_PROVIDER,
)


@pytest.fixture
def registry_home(tmp_path):
    with mock.patch.object(
        offline_registry, "INFERENCE_HOME", str(tmp_path)
    ), mock.patch.object(model_cache_paths, "INFERENCE_HOME", str(tmp_path)):
        yield tmp_path


def _materialize_package(model_id: str, package_id: str, files) -> str:
    package_dir = model_cache_paths.generate_model_package_cache_path(
        model_id=model_id, package_id=package_id
    )
    os.makedirs(package_dir, exist_ok=True)
    for handle, size in files.items():
        with open(os.path.join(package_dir, handle), "wb") as artefact:
            artefact.write(b"x" * size)
    return package_dir


def test_calculate_artefacts_size_sums_local_files(tmp_path) -> None:
    # given
    for handle, size in {"weights.onnx": 1000, "config.json": 24}.items():
        with open(tmp_path / handle, "wb") as artefact:
            artefact.write(b"x" * size)
    artefacts = [
        LocalFileArtefactSpecs(file_handle="weights.onnx", md5_hash="a" * 32),
        LocalFileArtefactSpecs(file_handle="config.json", md5_hash="b" * 32),
    ]

    # when
    size, success = calculate_artefacts_size(
        package_artefacts=artefacts,
        local_package_dir=str(tmp_path),
    )

    # then
    assert size == 1024
    assert success is True


def test_calculate_artefacts_size_flags_local_files_without_package_dir() -> None:
    # given
    artefacts = [LocalFileArtefactSpecs(file_handle="weights.onnx", md5_hash="a" * 32)]

    # when
    size, success = calculate_artefacts_size(
        package_artefacts=artefacts,
        local_package_dir=None,
    )

    # then
    assert size == 0
    assert success is False


def test_calculate_artefacts_size_flags_missing_local_file(tmp_path) -> None:
    # given
    artefacts = [LocalFileArtefactSpecs(file_handle="weights.onnx", md5_hash="a" * 32)]

    # when
    size, success = calculate_artefacts_size(
        package_artefacts=artefacts,
        local_package_dir=str(tmp_path),
    )

    # then
    assert size == 0
    assert success is False


def test_calculate_artefacts_size_mixes_remote_and_local(tmp_path) -> None:
    # given
    with open(tmp_path / "config.json", "wb") as artefact:
        artefact.write(b"x" * 24)
    artefacts = [
        FileDownloadSpecs(
            download_url="https://signed.example/weights.onnx",
            file_handle="weights.onnx",
            md5_hash="a" * 32,
        ),
        LocalFileArtefactSpecs(file_handle="config.json", md5_hash="b" * 32),
    ]

    # when
    with mock.patch(
        "inference_models.models.auto_loaders.presentation_utils.get_content_length",
        return_value=1000,
    ):
        size, success = calculate_artefacts_size(
            package_artefacts=artefacts,
            local_package_dir=str(tmp_path),
        )

    # then
    assert size == 1024
    assert success is True


def test_resolve_local_model_package_dir_for_materialized_package(
    registry_home,
) -> None:
    # given
    package_dir = _materialize_package(
        model_id="workspace/model/1",
        package_id="pkgonnx",
        files={"weights.onnx": 10},
    )
    package = ModelPackageMetadata(
        package_id="pkgonnx",
        backend=BackendType.ONNX,
        package_artefacts=[
            LocalFileArtefactSpecs(file_handle="weights.onnx", md5_hash="a" * 32)
        ],
    )

    # when / then
    assert (
        resolve_local_model_package_dir(
            model_package=package, model_id="workspace/model/1"
        )
        == package_dir
    )
    assert resolve_local_model_package_dir(model_package=package, model_id=None) is None
    assert (
        resolve_local_model_package_dir(
            model_package=package, model_id="workspace/other/1"
        )
        is None
    )


def test_calculate_size_of_all_model_packages_sizes_local_package(
    registry_home,
) -> None:
    # given
    _materialize_package(
        model_id="workspace/model/1",
        package_id="pkgonnx",
        files={"weights.onnx": 512, "config.json": 512},
    )
    package = ModelPackageMetadata(
        package_id="pkgonnx",
        backend=BackendType.ONNX,
        package_artefacts=[
            LocalFileArtefactSpecs(file_handle="weights.onnx", md5_hash="a" * 32),
            LocalFileArtefactSpecs(file_handle="config.json", md5_hash="b" * 32),
        ],
    )

    # when
    results = calculate_size_of_all_model_packages_artefacts(
        model_packages=[package],
        model_id="workspace/model/1",
    )

    # then
    assert results == [(1024, True)]


def _register_offline_example(package_file_size: int) -> None:
    metadata = ModelMetadata(
        model_id="workspace/model/1",
        model_architecture="rfdetr",
        task_type="object-detection",
        model_packages=[
            ModelPackageMetadata(
                package_id="pkgonnx",
                backend=BackendType.ONNX,
                package_artefacts=[
                    FileDownloadSpecs(
                        download_url="https://signed.example/weights.onnx",
                        file_handle="weights.onnx",
                        md5_hash="a" * 32,
                    ),
                ],
                trusted_source=True,
                environment_requirements=ServerEnvironmentRequirements(
                    cuda_device_cc=Version("8.6"),
                    cuda_device_name="NVIDIA GeForce RTX 3090",
                    driver_version=None,
                    cuda_version=None,
                    trt_version=None,
                    os_version="linux",
                ),
            ),
        ],
    )
    offline_registry.record_successful_load(
        model_metadata=metadata,
        requested_model_id="workspace/model/1",
        proven_package_id="pkgonnx",
    )
    _materialize_package(
        model_id="workspace/model/1",
        package_id="pkgonnx",
        files={"weights.onnx": package_file_size},
    )


def test_describe_model_package_sizes_offline_package(registry_home, capsys) -> None:
    # given: an offline-served package materialized on disk (2 MB)
    _register_offline_example(package_file_size=2 * 1024**2)

    # when
    AutoModel.describe_model_package(
        "workspace/model/1",
        "pkgonnx",
        weights_provider=ROBOFLOW_OFFLINE_WEIGHTS_PROVIDER,
        pull_artefacts_size=True,
    )

    # then: size is the real on-disk size, not the 0.0 MB failure marker
    captured = capsys.readouterr().out
    assert "2.0 MB" in captured
    assert "⚠️" not in captured


def test_describe_provider_swap_announces_offline_registry(
    registry_home, capsys
) -> None:
    # given
    _register_offline_example(package_file_size=1024)

    # when: OFFLINE_MODE with the default provider must swap + announce
    with mock.patch.object(auto_loaders_core, "OFFLINE_MODE", True):
        AutoModel.describe_model("workspace/model/1")

    # then: normalize rich's line wrapping before matching phrases
    captured = " ".join(capsys.readouterr().out.split())
    assert "OFFLINE_MODE is enabled" in captured
    assert "may be incomplete" in captured
    assert ROBOFLOW_OFFLINE_WEIGHTS_PROVIDER in captured


def test_describe_provider_swap_inactive_when_online() -> None:
    # when / then
    assert (
        auto_loaders_core._swap_describe_provider_when_offline(
            weights_provider="roboflow"
        )
        == "roboflow"
    )
    assert (
        auto_loaders_core._swap_describe_provider_when_offline(
            weights_provider="custom-provider"
        )
        == "custom-provider"
    )
