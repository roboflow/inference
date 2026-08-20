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
    with mock.patch.object(offline_registry, "INFERENCE_HOME", str(tmp_path)):
        yield tmp_path


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
    # given
    _register_example()

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
