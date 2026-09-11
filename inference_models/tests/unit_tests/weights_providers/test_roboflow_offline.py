from unittest import mock

import pytest
from packaging.version import Version

from inference_models.errors import ModelRetrievalError
from inference_models.models.auto_loaders.entities import BackendType
from inference_models.weights_providers import offline_registry
from inference_models.weights_providers.core import (
    WEIGHTS_PROVIDERS,
    get_model_from_provider,
)
from inference_models.weights_providers.entities import (
    FileDownloadSpecs,
    ModelMetadata,
    ModelPackageMetadata,
    PackageSourceType,
    ServerEnvironmentRequirements,
)
from inference_models.weights_providers.roboflow_offline import (
    ROBOFLOW_OFFLINE_WEIGHTS_PROVIDER,
)


@pytest.fixture
def registry_home(tmp_path):
    from inference_models.models.auto_loaders import model_cache_paths

    with mock.patch.object(
        offline_registry, "INFERENCE_HOME", str(tmp_path)
    ), mock.patch.object(model_cache_paths, "INFERENCE_HOME", str(tmp_path)):
        yield tmp_path


def _materialize_package(model_id: str, package_id: str, handles) -> None:
    import os

    from inference_models.models.auto_loaders import model_cache_paths

    package_dir = model_cache_paths.generate_model_package_cache_path(
        model_id=model_id, package_id=package_id
    )
    os.makedirs(package_dir, exist_ok=True)
    for handle in handles:
        with open(os.path.join(package_dir, handle), "wb") as artefact:
            artefact.write(b"artefact bytes")


def _register_example(model_id: str = "workspace/model/1") -> None:
    metadata = ModelMetadata(
        model_id=model_id,
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
        requested_model_id="alias-name",
        proven_package_id="pkgonnx",
    )


def test_offline_provider_is_registered() -> None:
    assert ROBOFLOW_OFFLINE_WEIGHTS_PROVIDER in WEIGHTS_PROVIDERS


def test_offline_provider_serves_recorded_metadata(registry_home) -> None:
    # given: a recorded package whose artefacts are materialized on disk
    _register_example()
    _materialize_package(
        model_id="workspace/model/1",
        package_id="pkgonnx",
        handles=["weights.onnx"],
    )

    # when
    metadata = get_model_from_provider(
        model_id="workspace/model/1",
        provider=ROBOFLOW_OFFLINE_WEIGHTS_PROVIDER,
    )
    by_alias = get_model_from_provider(
        model_id="alias-name",
        provider=ROBOFLOW_OFFLINE_WEIGHTS_PROVIDER,
    )

    # then
    assert metadata.model_id == "workspace/model/1"
    assert by_alias.model_id == "workspace/model/1"
    package = metadata.model_packages[0]
    assert package.package_source is PackageSourceType.LOCAL_CACHE
    assert package.environment_requirements.cuda_device_cc == Version("8.6")


def test_offline_provider_raises_actionable_error_without_record(
    registry_home,
) -> None:
    # when / then
    with pytest.raises(ModelRetrievalError) as error:
        get_model_from_provider(
            model_id="workspace/unknown/9",
            provider=ROBOFLOW_OFFLINE_WEIGHTS_PROVIDER,
        )
    assert "OFFLINE_MODE_WARM_UP" in str(error.value)


def test_offline_provider_refuses_record_without_materialized_artefacts(
    registry_home,
) -> None:
    """A record whose packages have no files on disk cannot serve - the
    presence pre-filter drops them and the provider fails actionably."""
    # given
    _register_example()

    # when / then
    with pytest.raises(ModelRetrievalError) as error:
        get_model_from_provider(
            model_id="workspace/model/1",
            provider=ROBOFLOW_OFFLINE_WEIGHTS_PROVIDER,
        )
    assert "OFFLINE_MODE_WARM_UP" in str(error.value)


def test_offline_provider_drops_unmaterialized_packages(registry_home) -> None:
    """Recorded-but-never-loaded sibling packages are filtered out so
    negotiation never attempts them."""
    # given: two recorded packages, only one materialized
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
            ),
            ModelPackageMetadata(
                package_id="pkgtorch",
                backend=BackendType.TORCH,
                package_artefacts=[
                    FileDownloadSpecs(
                        download_url="https://signed.example/weights.pt",
                        file_handle="weights.pt",
                        md5_hash="b" * 32,
                    ),
                ],
                trusted_source=True,
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
        handles=["weights.onnx"],
    )

    # when
    served = get_model_from_provider(
        model_id="workspace/model/1",
        provider=ROBOFLOW_OFFLINE_WEIGHTS_PROVIDER,
    )

    # then
    assert [p.package_id for p in served.model_packages] == ["pkgonnx"]
