import json
import re
import urllib.parse
from typing import Dict, List, Optional, Union
from unittest.mock import patch

import pytest
from packaging.version import Version
from requests import Response
from requests_mock import Mocker

from inference_models.configuration import API_CALLS_MAX_TRIES, ROBOFLOW_API_HOST
from inference_models.errors import (
    AssumptionError,
    ModelMetadataConsistencyError,
    ModelRetrievalError,
    RetryError,
    UnauthorizedModelAccessError,
)
from inference_models.models.auto_loaders.entities import BackendType
from inference_models.weights_providers import roboflow as roboflow_module
from inference_models.weights_providers.entities import (
    FileDownloadSpecs,
    JetsonEnvironmentRequirements,
    ModelPackageMetadata,
    ONNXPackageDetails,
    Quantization,
    ServerEnvironmentRequirements,
    TorchScriptPackageDetails,
    TRTPackageDetails,
)
from inference_models.weights_providers.roboflow import (
    RoboflowModelMetadata,
    RoboflowModelPackageFile,
    RoboflowModelPackageV1,
    as_version,
    get_error_response_payload,
    get_model_metadata,
    get_one_page_of_model_metadata,
    get_roboflow_model,
    handle_response_errors,
    parse_model_package_metadata,
    roboflow_license_server_proxy_url_builder,
)

DUMMY_PROXY_PREFIX = "http://license.local/proxy?url="


def test_as_version_when_valid_version_provided() -> None:
    # when
    result = as_version(value="1.2.3")

    # then
    assert result == Version("1.2.3")


def test_as_version_when_invalid_version_provided() -> None:
    # when
    with pytest.raises(ModelMetadataConsistencyError):
        _ = as_version(value="invalid")


def test_parse_mediapipe_model_package_when_package_is_valid() -> None:
    # given
    metadata = RoboflowModelPackageV1(
        type="external-model-package-v1",
        packageId="my-package-id",
        packageManifest={
            "type": "mediapipe-model-package-v1",
            "backendType": "mediapipe",
        },
        packageFiles=[
            RoboflowModelPackageFile(
                fileHandle="some", downloadUrl="https://dummy.com", md5Hash="some"
            )
        ],
        trustedSource=True,
    )

    # when
    result = parse_model_package_metadata(metadata=metadata)

    # then
    assert result == ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.MEDIAPIPE,
        package_artefacts=[
            FileDownloadSpecs(
                download_url="https://dummy.com", file_handle="some", md5_hash="some"
            ),
        ],
        quantization=Quantization.UNKNOWN,
        trusted_source=True,
    )


def test_parse_mediapipe_model_package_when_package_is_invalid() -> None:
    # given
    metadata = RoboflowModelPackageV1(
        type="external-model-package-v1",
        packageId="my-package-id",
        packageManifest={
            "type": "mediapipe-model-package-v1",
            "backendType": "invalid",
        },
        packageFiles=[
            RoboflowModelPackageFile(
                fileHandle="some", downloadUrl="https://dummy.com", md5Hash="some"
            )
        ],
        trustedSource=True,
    )

    # when
    with pytest.raises(ModelMetadataConsistencyError):
        _ = parse_model_package_metadata(metadata=metadata)


def test_parse_ultralytics_model_package() -> None:
    # given
    metadata = RoboflowModelPackageV1(
        type="external-model-package-v1",
        packageId="my-package-id",
        packageManifest={
            "type": "ultralytics-model-package-v1",
            "backendType": "ultralytics",
        },
        packageFiles=[
            RoboflowModelPackageFile(
                fileHandle="some", downloadUrl="https://dummy.com", md5Hash="some"
            )
        ],
        trustedSource=False,
    )

    # when
    result = parse_model_package_metadata(metadata=metadata)

    # then
    assert result == ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.ULTRALYTICS,
        package_artefacts=[
            FileDownloadSpecs(
                download_url="https://dummy.com", file_handle="some", md5_hash="some"
            ),
        ],
        quantization=Quantization.UNKNOWN,
        trusted_source=False,
    )


def test_parse_hf_model_package_model_package_when_valid_input_provided() -> None:
    # given
    metadata = RoboflowModelPackageV1(
        type="external-model-package-v1",
        packageId="my-package-id",
        packageManifest={
            "type": "hf-model-package-v1",
            "backendType": "hf",
            "quantization": "fp32",
        },
        packageFiles=[
            RoboflowModelPackageFile(
                fileHandle="some", downloadUrl="https://dummy.com", md5Hash="some"
            )
        ],
        trustedSource=True,
    )

    # when
    result = parse_model_package_metadata(metadata=metadata)

    # then
    assert result == ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.HF,
        package_artefacts=[
            FileDownloadSpecs(
                download_url="https://dummy.com", file_handle="some", md5_hash="some"
            ),
        ],
        quantization=Quantization.FP32,
        trusted_source=True,
    )


def test_parse_hf_model_package_model_package_when_invalid_input_provided() -> None:
    # given
    metadata = RoboflowModelPackageV1(
        type="external-model-package-v1",
        packageId="my-package-id",
        packageManifest={"type": "hf-model-package-v1"},
        packageFiles=[
            RoboflowModelPackageFile(
                fileHandle="some", downloadUrl="https://dummy.com", md5Hash="some"
            )
        ],
        trustedSource=True,
    )

    # when
    with pytest.raises(ModelMetadataConsistencyError):
        _ = parse_model_package_metadata(metadata=metadata)


def test_parse_torch_model_package_when_batch_size_missmatch_present() -> None:
    # given
    metadata = RoboflowModelPackageV1(
        type="external-model-package-v1",
        packageId="my-package-id",
        packageManifest={
            "type": "torch-model-package-v1",
            "backendType": "torch",
            "quantization": "fp32",
        },
        packageFiles=[
            RoboflowModelPackageFile(
                fileHandle="some", downloadUrl="https://dummy.com", md5Hash="some"
            )
        ],
        trustedSource=True,
    )

    # when
    with pytest.raises(ModelMetadataConsistencyError):
        _ = parse_model_package_metadata(metadata=metadata)


def test_parse_torch_model_package_when_invalid_manifest_provided() -> None:
    # given
    metadata = RoboflowModelPackageV1(
        type="external-model-package-v1",
        packageId="my-package-id",
        packageManifest={"type": "torch-model-package-v1"},
        packageFiles=[
            RoboflowModelPackageFile(
                fileHandle="some", downloadUrl="https://dummy.com", md5Hash="some"
            )
        ],
        trustedSource=True,
    )

    # when
    with pytest.raises(ModelMetadataConsistencyError):
        _ = parse_model_package_metadata(metadata=metadata)


@pytest.mark.parametrize(
    "dynamic_batch_size, static_batch_size",
    [
        (True, 2),
        (False, 0),
        (False, None),
    ],
)
def test_parse_torch_model_package_when_invalid_manifest_provided_regarding_batch_size(
    dynamic_batch_size: bool,
    static_batch_size: Optional[int],
) -> None:
    # given
    metadata = RoboflowModelPackageV1(
        type="external-model-package-v1",
        packageId="my-package-id",
        packageManifest={
            "type": "torch-model-package-v1",
            "backendType": "torch",
            "quantization": "fp32",
            "dynamicBatchSize": dynamic_batch_size,
            "staticBatchSize": static_batch_size,
        },
        packageFiles=[
            RoboflowModelPackageFile(
                fileHandle="some", downloadUrl="https://dummy.com", md5Hash="some"
            )
        ],
        trustedSource=True,
    )

    # when
    with pytest.raises(ModelMetadataConsistencyError):
        _ = parse_model_package_metadata(metadata=metadata)


def test_parse_torch_model_package_when_valid_manifest_provided() -> None:
    # given
    metadata = RoboflowModelPackageV1(
        type="external-model-package-v1",
        packageId="my-package-id",
        packageManifest={
            "type": "torch-model-package-v1",
            "backendType": "torch",
            "quantization": "fp32",
            "dynamicBatchSize": True,
        },
        packageFiles=[
            RoboflowModelPackageFile(
                fileHandle="some", downloadUrl="https://dummy.com", md5Hash="some"
            )
        ],
        trustedSource=True,
    )

    # when
    result = parse_model_package_metadata(metadata=metadata)

    # then
    assert result == ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.TORCH,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        static_batch_size=None,
        package_artefacts=[
            FileDownloadSpecs(
                download_url="https://dummy.com", file_handle="some", md5_hash="some"
            ),
        ],
        trusted_source=metadata.trusted_source,
    )


def test_parse_trt_model_package_when_invalid_manifest_provided() -> None:
    # given
    metadata = RoboflowModelPackageV1(
        type="external-model-package-v1",
        packageId="my-package-id",
        packageManifest={"type": "trt-model-package-v1"},
        packageFiles=[
            RoboflowModelPackageFile(
                fileHandle="some", downloadUrl="https://dummy.com", md5Hash="some"
            )
        ],
        trustedSource=True,
    )

    with pytest.raises(ModelMetadataConsistencyError):
        _ = parse_model_package_metadata(metadata=metadata)


def test_parse_trt_model_package_when_manifest_with_batch_size_missmatch_provided() -> (
    None
):
    # given
    metadata = RoboflowModelPackageV1(
        type="external-model-package-v1",
        packageId="my-package-id",
        packageManifest={
            "type": "trt-model-package-v1",
            "backendType": "trt",
            "dynamicBatchSize": False,
            "staticBatchSize": None,
            "quantization": "fp16",
            "cudaDeviceType": "orin",
            "cudaDeviceCC": "8.7",
            "cudaVersion": "12.6",
            "trtVersion": "10.3.0.17",
            "sameCCCompatible": False,
            "trtForwardCompatible": False,
            "trtLeanRuntimeExcluded": False,
            "machineType": "jetson",
            "machineSpecs": {
                "type": "jetson-machine-specs-v1",
                "l4tVersion": "36.4.3",
                "deviceName": "jetson-orin-nx",
                "driverVersion": "540.0.1",
            },
        },
        packageFiles=[
            RoboflowModelPackageFile(
                fileHandle="some", downloadUrl="https://dummy.com", md5Hash="some"
            )
        ],
        trustedSource=True,
    )

    # when
    with pytest.raises(ModelMetadataConsistencyError):
        _ = parse_model_package_metadata(metadata=metadata)


def test_parse_trt_model_package_when_manifest_with_dynamic_batch_size_missmatch_provided() -> (
    None
):
    # given
    metadata = RoboflowModelPackageV1(
        type="external-model-package-v1",
        packageId="my-package-id",
        packageManifest={
            "type": "trt-model-package-v1",
            "backendType": "trt",
            "dynamicBatchSize": True,
            "static_batch_size": None,
            "quantization": "fp16",
            "cudaDeviceType": "orin",
            "cudaDeviceCC": "8.7",
            "cudaVersion": "12.6",
            "trtVersion": "10.3.0.17",
            "sameCCCompatible": False,
            "trtForwardCompatible": False,
            "trtLeanRuntimeExcluded": False,
            "machineType": "jetson",
            "machineSpecs": {
                "type": "jetson-machine-specs-v1",
                "l4tVersion": "36.4.3",
                "deviceName": "jetson-orin-nx",
                "driverVersion": "540.0.1",
            },
        },
        packageFiles=[
            RoboflowModelPackageFile(
                fileHandle="some", downloadUrl="https://dummy.com", md5Hash="some"
            )
        ],
        trustedSource=True,
    )

    # when
    with pytest.raises(ModelMetadataConsistencyError):
        _ = parse_model_package_metadata(metadata=metadata)


def test_parse_trt_model_package_when_manifest_with_environment_specification_missmatch_for_gpu_server() -> (
    None
):
    # given
    metadata = RoboflowModelPackageV1(
        type="external-model-package-v1",
        packageId="my-package-id",
        packageManifest={
            "type": "trt-model-package-v1",
            "backendType": "trt",
            "dynamicBatchSize": True,
            "minBatchSize": 1,
            "optBatchSize": 8,
            "maxBatchSize": 16,
            "static_batch_size": None,
            "quantization": "fp16",
            "cudaDeviceType": "orin",
            "cudaDeviceCC": "8.7",
            "cudaVersion": "12.6",
            "trtVersion": "10.3.0.17",
            "sameCCCompatible": False,
            "trtForwardCompatible": False,
            "trtLeanRuntimeExcluded": False,
            "machineType": "gpu-server",
            "machineSpecs": {
                "type": "jetson-machine-specs-v1",
                "l4tVersion": "36.4.3",
                "deviceName": "jetson-orin-nx",
                "driverVersion": "540.0.1",
            },
        },
        packageFiles=[
            RoboflowModelPackageFile(
                fileHandle="some", downloadUrl="https://dummy.com", md5Hash="some"
            )
        ],
        trustedSource=True,
    )

    # when
    with pytest.raises(ModelMetadataConsistencyError):
        _ = parse_model_package_metadata(metadata=metadata)


def test_parse_trt_model_package_when_manifest_with_environment_specification_missmatch_for_jetson() -> (
    None
):
    # given
    metadata = RoboflowModelPackageV1(
        type="external-model-package-v1",
        packageId="my-package-id",
        packageManifest={
            "type": "trt-model-package-v1",
            "backendType": "trt",
            "dynamicBatchSize": True,
            "minBatchSize": 1,
            "optBatchSize": 8,
            "maxBatchSize": 16,
            "static_batch_size": None,
            "quantization": "fp16",
            "cudaDeviceType": "orin",
            "cudaDeviceCC": "8.7",
            "cudaVersion": "12.6",
            "trtVersion": "10.3.0.17",
            "sameCCCompatible": False,
            "trtForwardCompatible": False,
            "trtLeanRuntimeExcluded": False,
            "machineType": "jetson",
            "machineSpecs": {
                "type": "gpu-server-specs-v1",
                "osVersion": "ubuntu-20.04",
                "driverVersion": "540.0.1",
            },
        },
        packageFiles=[
            RoboflowModelPackageFile(
                fileHandle="some", downloadUrl="https://dummy.com", md5Hash="some"
            )
        ],
        trustedSource=True,
    )

    # when
    with pytest.raises(ModelMetadataConsistencyError):
        _ = parse_model_package_metadata(metadata=metadata)


def test_parse_trt_model_package_when_manifest_with_jetson_environment_specification() -> (
    None
):
    # given
    metadata = RoboflowModelPackageV1(
        type="external-model-package-v1",
        packageId="my-package-id",
        packageManifest={
            "type": "trt-model-package-v1",
            "backendType": "trt",
            "dynamicBatchSize": True,
            "static_batch_size": None,
            "minBatchSize": 1,
            "optBatchSize": 8,
            "maxBatchSize": 16,
            "quantization": "fp16",
            "cudaDeviceType": "orin",
            "cudaDeviceCC": "8.7",
            "cudaVersion": "12.6",
            "trtVersion": "10.3.0.17",
            "sameCCCompatible": False,
            "trtForwardCompatible": False,
            "trtLeanRuntimeExcluded": False,
            "machineType": "jetson",
            "machineSpecs": {
                "type": "jetson-machine-specs-v1",
                "l4tVersion": "36.4.3",
                "deviceName": "jetson-orin-nx",
                "driverVersion": "540.0.1",
            },
        },
        packageFiles=[
            RoboflowModelPackageFile(
                fileHandle="some", downloadUrl="https://dummy.com", md5Hash="some"
            )
        ],
        trustedSource=True,
    )

    # when
    result = parse_model_package_metadata(metadata=metadata)

    # then
    environment_requirements = JetsonEnvironmentRequirements(
        cuda_device_cc=Version("8.7"),
        cuda_device_name="orin",
        l4t_version=Version("36.4.3"),
        jetson_product_name="jetson-orin-nx",
        cuda_version=Version("12.6"),
        trt_version=Version("10.3.0.17"),
        driver_version=Version("540.0.1"),
    )
    trt_package_details = TRTPackageDetails(
        min_dynamic_batch_size=1,
        opt_dynamic_batch_size=8,
        max_dynamic_batch_size=16,
        same_cc_compatible=False,
        trt_forward_compatible=False,
        trt_lean_runtime_excluded=False,
    )
    assert result == ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.TRT,
        quantization=Quantization.FP16,
        dynamic_batch_size_supported=True,
        static_batch_size=None,
        trt_package_details=trt_package_details,
        package_artefacts=[
            FileDownloadSpecs(
                download_url="https://dummy.com", file_handle="some", md5_hash="some"
            ),
        ],
        environment_requirements=environment_requirements,
        trusted_source=metadata.trusted_source,
    )


def test_parse_trt_model_package_when_manifest_with_server_environment_specification() -> (
    None
):
    # given
    metadata = RoboflowModelPackageV1(
        type="external-model-package-v1",
        packageId="my-package-id",
        packageManifest={
            "type": "trt-model-package-v1",
            "backendType": "trt",
            "dynamicBatchSize": True,
            "static_batch_size": None,
            "minBatchSize": 1,
            "optBatchSize": 8,
            "maxBatchSize": 16,
            "quantization": "fp16",
            "cudaDeviceType": "orin",
            "cudaDeviceCC": "8.7",
            "cudaVersion": "12.6",
            "trtVersion": "10.3.0.17",
            "sameCCCompatible": False,
            "trtForwardCompatible": False,
            "trtLeanRuntimeExcluded": False,
            "machineType": "gpu-server",
            "machineSpecs": {
                "type": "gpu-server-specs-v1",
                "osVersion": "ubuntu-20.04",
                "driverVersion": "540.0.1",
            },
        },
        packageFiles=[
            RoboflowModelPackageFile(
                fileHandle="some", downloadUrl="https://dummy.com", md5Hash="some"
            )
        ],
        trustedSource=True,
    )

    # when
    result = parse_model_package_metadata(metadata=metadata)

    # then
    environment_requirements = ServerEnvironmentRequirements(
        cuda_device_cc=Version("8.7"),
        cuda_device_name="orin",
        cuda_version=Version("12.6"),
        trt_version=Version("10.3.0.17"),
        driver_version=Version("540.0.1"),
        os_version="ubuntu-20.04",
    )
    trt_package_details = TRTPackageDetails(
        min_dynamic_batch_size=1,
        opt_dynamic_batch_size=8,
        max_dynamic_batch_size=16,
        same_cc_compatible=False,
        trt_forward_compatible=False,
        trt_lean_runtime_excluded=False,
    )
    assert result == ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.TRT,
        quantization=Quantization.FP16,
        dynamic_batch_size_supported=True,
        static_batch_size=None,
        trt_package_details=trt_package_details,
        package_artefacts=[
            FileDownloadSpecs(
                download_url="https://dummy.com", file_handle="some", md5_hash="some"
            ),
        ],
        environment_requirements=environment_requirements,
        trusted_source=metadata.trusted_source,
    )


def test_parse_onnx_model_package_when_invalid_manifest_provided() -> None:
    # given
    metadata = RoboflowModelPackageV1(
        type="external-model-package-v1",
        packageId="my-package-id",
        packageManifest={"type": "onnx-model-package-v1"},
        packageFiles=[
            RoboflowModelPackageFile(
                fileHandle="some", downloadUrl="https://dummy.com", md5Hash="some"
            )
        ],
        trustedSource=True,
    )

    # when
    with pytest.raises(ModelMetadataConsistencyError):
        _ = parse_model_package_metadata(metadata=metadata)


def test_parse_onnx_model_package_when_valid_manifest_with_batch_size_missmatch_provided() -> (
    None
):
    # given
    metadata = RoboflowModelPackageV1(
        type="external-model-package-v1",
        packageId="my-package-id",
        packageManifest={
            "type": "onnx-model-package-v1",
            "backendType": "onnx",
            "quantization": "fp32",
            "opset": 19,
            "incompatibleProviders": ["TRTExecutionProvider"],
        },
        packageFiles=[
            RoboflowModelPackageFile(
                fileHandle="some", downloadUrl="https://dummy.com", md5Hash="some"
            )
        ],
        trustedSource=True,
    )

    with pytest.raises(ModelMetadataConsistencyError):
        _ = parse_model_package_metadata(metadata=metadata)


@pytest.mark.parametrize(
    "dynamic_batch_size, static_batch_size",
    [
        (True, 2),
        (False, 0),
        (False, None),
    ],
)
def test_parse_onnx_model_package_when_invalid_manifest_provided_regarding_batch_size(
    dynamic_batch_size: bool,
    static_batch_size: Optional[int],
) -> None:
    # given
    metadata = RoboflowModelPackageV1(
        type="external-model-package-v1",
        packageId="my-package-id",
        packageManifest={
            "type": "onnx-model-package-v1",
            "backendType": "onnx",
            "quantization": "fp32",
            "dynamicBatchSize": dynamic_batch_size,
            "staticBatchSize": static_batch_size,
            "opset": 19,
            "incompatibleProviders": ["TRTExecutionProvider"],
        },
        packageFiles=[
            RoboflowModelPackageFile(
                fileHandle="some", downloadUrl="https://dummy.com", md5Hash="some"
            )
        ],
        trustedSource=True,
    )

    # when
    with pytest.raises(ModelMetadataConsistencyError):
        _ = parse_model_package_metadata(metadata=metadata)


def test_parse_onnx_model_package_when_valid_manifest_provided() -> None:
    # given
    metadata = RoboflowModelPackageV1(
        type="external-model-package-v1",
        packageId="my-package-id",
        packageManifest={
            "type": "onnx-model-package-v1",
            "backendType": "onnx",
            "quantization": "fp32",
            "dynamicBatchSize": True,
            "opset": 19,
            "incompatibleProviders": ["TRTExecutionProvider"],
        },
        packageFiles=[
            RoboflowModelPackageFile(
                fileHandle="some", downloadUrl="https://dummy.com", md5Hash="some"
            )
        ],
        trustedSource=True,
    )

    # when
    result = parse_model_package_metadata(metadata=metadata)

    # then
    assert result == ModelPackageMetadata(
        package_id=metadata.package_id,
        backend=BackendType.ONNX,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        static_batch_size=None,
        package_artefacts=[
            FileDownloadSpecs(
                download_url="https://dummy.com", file_handle="some", md5_hash="some"
            ),
        ],
        onnx_package_details=ONNXPackageDetails(
            opset=19,
            incompatible_providers=["TRTExecutionProvider"],
        ),
        trusted_source=metadata.trusted_source,
    )


def test_parse_torch_script_model_package_when_package_is_manifest_invalid() -> None:
    # given
    metadata = RoboflowModelPackageV1(
        type="external-model-package-v1",
        packageId="my-package-id",
        packageManifest={
            "type": "torch-script-model-package-v1",
        },
        packageFiles=[
            RoboflowModelPackageFile(
                fileHandle="some", downloadUrl="https://dummy.com", md5Hash="some"
            )
        ],
        trustedSource=True,
    )

    # when
    with pytest.raises(ModelMetadataConsistencyError):
        _ = parse_model_package_metadata(metadata=metadata)


@pytest.mark.parametrize(
    "dynamic_batch_size, static_batch_size",
    [
        (True, 2),
        (False, 0),
        (False, None),
    ],
)
def test_parse_torch_script_model_package_when_package_batch_settings_are_invalid(
    dynamic_batch_size: bool,
    static_batch_size: Optional[int],
) -> None:
    # given
    metadata = RoboflowModelPackageV1(
        type="external-model-package-v1",
        packageId="my-package-id",
        packageManifest={
            "type": "torch-script-model-package-v1",
            "backendType": "torch-script",
            "dynamicBatchSize": dynamic_batch_size,
            "staticBatchSize": static_batch_size,
            "quantization": "fp32",
            "supportedDeviceTypes": ["cuda", "cpu", "mps"],
            "torchVersion": "2.6.0",
            "torchVisionVersion": "0.22.0",
        },
        packageFiles=[
            RoboflowModelPackageFile(
                fileHandle="some", downloadUrl="https://dummy.com", md5Hash="some"
            )
        ],
        trustedSource=True,
    )

    # when
    with pytest.raises(ModelMetadataConsistencyError):
        _ = parse_model_package_metadata(metadata=metadata)


def test_parse_torch_script_model_package_when_package_is_manifest_valid() -> None:
    # given
    metadata = RoboflowModelPackageV1(
        type="external-model-package-v1",
        packageId="my-package-id",
        packageManifest={
            "type": "torch-script-model-package-v1",
            "backendType": "torch-script",
            "dynamicBatchSize": False,
            "staticBatchSize": 2,
            "quantization": "fp32",
            "supportedDeviceTypes": ["cuda", "cpu", "mps"],
            "torchVersion": "2.6.0",
            "torchVisionVersion": "0.22.0",
        },
        packageFiles=[
            RoboflowModelPackageFile(
                fileHandle="some", downloadUrl="https://dummy.com", md5Hash="some"
            )
        ],
        modelFeatures={"nms_fused": True},
        trustedSource=True,
    )

    # when
    result = parse_model_package_metadata(metadata=metadata)

    # then
    assert result == ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.TORCH_SCRIPT,
        dynamic_batch_size_supported=False,
        static_batch_size=2,
        package_artefacts=[
            FileDownloadSpecs(
                download_url="https://dummy.com", file_handle="some", md5_hash="some"
            ),
        ],
        quantization=Quantization.FP32,
        trusted_source=True,
        model_features={"nms_fused": True},
        torch_script_package_details=TorchScriptPackageDetails(
            supported_device_types={"cuda", "cpu", "mps"},
            torch_version=Version("2.6.0"),
            torch_vision_version=Version("0.22.0"),
        ),
    )


def test_parse_model_package_metadata_when_the_package_structure_could_not_be_parsed() -> (
    None
):
    # when
    result = parse_model_package_metadata(
        metadata={"type": "not-supported", "packageId": "some-package-id"},
    )

    # then
    assert result is None


def test_parse_model_package_metadata_when_package_manifest_type_is_to_be_ignored() -> (
    None
):
    # given
    metadata = RoboflowModelPackageV1(
        type="external-model-package-v1",
        packageId="my-package-id",
        packageManifest={
            "type": "tfjs-model-package-v1",
        },
        packageFiles=[
            RoboflowModelPackageFile(
                fileHandle="some", downloadUrl="https://dummy.com", md5Hash="some"
            )
        ],
        trustedSource=True,
    )

    # when
    result = parse_model_package_metadata(metadata=metadata)

    # then
    assert result is None


def test_parse_model_package_metadata_when_package_manifest_type_is_not_recognised() -> (
    None
):
    # given
    metadata = RoboflowModelPackageV1(
        type="external-model-package-v1",
        packageId="my-package-id",
        packageManifest={
            "type": "new-model-package-type-v1",
        },
        packageFiles=[
            RoboflowModelPackageFile(
                fileHandle="some", downloadUrl="https://dummy.com", md5Hash="some"
            )
        ],
        trustedSource=True,
    )

    # when
    result = parse_model_package_metadata(metadata=metadata)

    # then
    assert result is None


def test_parse_model_package_metadata_when_package_manifest_is_known_but_invalid() -> (
    None
):
    # given
    metadata = RoboflowModelPackageV1(
        type="external-model-package-v1",
        packageId="my-package-id",
        packageManifest={
            "type": "onnx-model-package-v1",
            "quantization": "fp32",
            "opset": 19,
            "incompatibleProviders": ["TRTExecutionProvider"],
        },
        packageFiles=[
            RoboflowModelPackageFile(
                fileHandle="some", downloadUrl="https://dummy.com", md5Hash="some"
            )
        ],
        trustedSource=True,
    )

    # when
    with pytest.raises(ModelMetadataConsistencyError):
        _ = parse_model_package_metadata(metadata=metadata)


def test_parse_model_package_metadata_when_package_manifest_is_known_and_valid() -> (
    None
):
    # given
    metadata = RoboflowModelPackageV1(
        type="external-model-package-v1",
        packageId="my-package-id",
        packageManifest={
            "type": "onnx-model-package-v1",
            "backendType": "onnx",
            "quantization": "fp32",
            "dynamicBatchSize": True,
            "opset": 19,
            "incompatibleProviders": ["TRTExecutionProvider"],
        },
        packageFiles=[
            RoboflowModelPackageFile(
                fileHandle="some", downloadUrl="https://dummy.com", md5Hash="some"
            )
        ],
        trustedSource=True,
    )

    # when
    result = parse_model_package_metadata(metadata=metadata)

    # then
    assert result == ModelPackageMetadata(
        package_id=metadata.package_id,
        backend=BackendType.ONNX,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        static_batch_size=None,
        package_artefacts=[
            FileDownloadSpecs(
                download_url="https://dummy.com", file_handle="some", md5_hash="some"
            ),
        ],
        onnx_package_details=ONNXPackageDetails(
            opset=19,
            incompatible_providers=["TRTExecutionProvider"],
        ),
        trusted_source=metadata.trusted_source,
    )


def test_get_error_response_payload_when_response_is_json() -> None:
    # given
    response = Response()
    response.status_code = 404
    response._content = json.dumps({"message": "Not found"}).encode("utf-8")

    # when
    result = get_error_response_payload(response=response)

    # then
    assert json.loads(result) == {"message": "Not found"}


def test_get_error_response_payload_when_response_is_not_json() -> None:
    # given
    response = Response()
    response.status_code = 404
    response._content = b"DUMMY"

    # when
    result = get_error_response_payload(response=response)

    # then
    assert result == "DUMMY"


def test_handle_response_errors_when_status_code_is_non_auth_error() -> None:
    # given
    response = Response()
    response.status_code = 401

    # when
    with pytest.raises(UnauthorizedModelAccessError):
        handle_response_errors(response=response, operation_name="some")


def test_handle_response_errors_when_status_code_is_retryable_error() -> None:
    # given
    response = Response()
    response.status_code = 429

    # when
    with pytest.raises(RetryError):
        handle_response_errors(response=response, operation_name="some")


def test_handle_response_errors_when_status_code_is_non_retryable_error() -> None:
    # given
    response = Response()
    response.status_code = 404

    # when
    with pytest.raises(ModelRetrievalError):
        handle_response_errors(response=response, operation_name="some")


def test_handle_response_errors_when_success_response_provided() -> None:
    # given
    response = Response()
    response.status_code = 200

    # when
    handle_response_errors(response=response, operation_name="some")

    # then - no error


def test_get_one_page_of_model_metadata_when_retry_not_needed_and_parsable_response(
    requests_mock: Mocker,
) -> None:
    requests_mock.get(
        f"{ROBOFLOW_API_HOST}/models/v1/external/weights",
        json={
            "modelMetadata": {
                "type": "external-model-metadata-v1",
                "modelId": "my-model",
                "modelArchitecture": "yolov8",
                "taskType": "object-detection",
                "modelPackages": [
                    {
                        "type": "new-unknown",
                    },
                    {
                        "type": "external-model-package-v1",
                        "packageId": "my-package-id",
                        "packageManifest": {},
                        "packageFiles": [
                            {
                                "fileHandle": "some",
                                "downloadUrl": "https://link.com",
                                "md5Hash": "some",
                            }
                        ],
                    },
                ],
                "nextPage": "some",
            }
        },
    )

    # when
    result = get_one_page_of_model_metadata(
        model_id="my-model", api_key="some", page_size=100, start_after="start"
    )

    # then
    assert result == RoboflowModelMetadata(
        type="external-model-metadata-v1",
        modelId="my-model",
        modelArchitecture="yolov8",
        taskType="object-detection",
        modelPackages=[
            {"type": "new-unknown"},
            RoboflowModelPackageV1(
                type="external-model-package-v1",
                packageId="my-package-id",
                packageManifest={},
                packageFiles=[
                    RoboflowModelPackageFile(
                        fileHandle="some",
                        downloadUrl="https://link.com",
                        md5Hash="some",
                    )
                ],
            ),
        ],
        nextPage="some",
    )
    assert requests_mock.last_request.headers["Authorization"] == "Bearer some"
    parsed_params = urllib.parse.parse_qs(requests_mock.last_request.query)
    assert parsed_params["modelid"][0] == "my-model"
    assert parsed_params["pagesize"][0] == "100"
    assert parsed_params["startafter"][0] == "start"


def test_get_one_page_of_model_metadata_excludes_auth_header_when_local_api_key(
    requests_mock: Mocker,
) -> None:
    requests_mock.get(
        f"{ROBOFLOW_API_HOST}/models/v1/external/weights",
        json={
            "modelMetadata": {
                "type": "external-model-metadata-v1",
                "modelId": "my-model",
                "modelArchitecture": "yolov8",
                "taskType": "object-detection",
                "modelPackages": [],
            }
        },
    )

    # when
    _ = get_one_page_of_model_metadata(model_id="my-model", api_key="local")

    # then
    assert "Authorization" not in requests_mock.last_request.headers


def test_get_one_page_of_model_metadata_when_retry_not_needed_and_not_parsable_response(
    requests_mock: Mocker,
) -> None:
    requests_mock.get(
        f"{ROBOFLOW_API_HOST}/models/v1/external/weights", json={"status": "ok"}
    )

    # when
    with pytest.raises(ModelRetrievalError):
        _ = get_one_page_of_model_metadata(model_id="my-model")


def test_get_one_page_of_model_metadata_when_retry_needed_and_parsable_response(
    requests_mock: Mocker,
) -> None:
    requests_mock.get(
        f"{ROBOFLOW_API_HOST}/models/v1/external/weights",
        [
            {"status_code": 429},
            {
                "status_code": 200,
                "json": {
                    "modelMetadata": {
                        "type": "external-model-metadata-v1",
                        "modelId": "my-model",
                        "modelArchitecture": "yolov8",
                        "taskType": "object-detection",
                        "modelPackages": [
                            {
                                "type": "new-unknown",
                            },
                            {
                                "type": "external-model-package-v1",
                                "packageId": "my-package-id",
                                "packageManifest": {},
                                "packageFiles": [
                                    {
                                        "fileHandle": "some",
                                        "downloadUrl": "https://link.com",
                                        "md5Hash": "some",
                                    }
                                ],
                            },
                        ],
                        "nextPage": "some",
                    }
                },
            },
        ],
    )

    # when
    result = get_one_page_of_model_metadata(
        model_id="my-model", api_key="some", page_size=100, start_after="start"
    )

    # then
    assert result == RoboflowModelMetadata(
        type="external-model-metadata-v1",
        modelId="my-model",
        modelArchitecture="yolov8",
        taskType="object-detection",
        modelPackages=[
            {"type": "new-unknown"},
            RoboflowModelPackageV1(
                type="external-model-package-v1",
                packageId="my-package-id",
                packageManifest={},
                packageFiles=[
                    RoboflowModelPackageFile(
                        fileHandle="some",
                        downloadUrl="https://link.com",
                        md5Hash="some",
                    )
                ],
            ),
        ],
        nextPage="some",
    )
    assert len(requests_mock.request_history) == 2


def test_get_one_page_of_model_metadata_when_retries_exceeded(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        f"{ROBOFLOW_API_HOST}/models/v1/external/weights",
        [
            {"status_code": 429},
        ]
        * API_CALLS_MAX_TRIES,
    )

    # when
    with pytest.raises(RetryError):
        _ = get_one_page_of_model_metadata(
            model_id="my-model", api_key="some", page_size=100, start_after="start"
        )

    # then
    assert len(requests_mock.request_history) == API_CALLS_MAX_TRIES


def test_get_model_metadata_with_pagination(requests_mock: Mocker) -> None:
    requests_mock.get(
        f"{ROBOFLOW_API_HOST}/models/v1/external/weights",
        [
            {
                "status_code": 200,
                "json": {
                    "modelMetadata": {
                        "type": "external-model-metadata-v1",
                        "modelId": "my-model",
                        "modelArchitecture": "yolov8",
                        "taskType": "object-detection",
                        "modelPackages": [
                            {
                                "type": "new-unknown",
                            },
                            {
                                "type": "external-model-package-v1",
                                "packageId": "my-package-id-1",
                                "packageManifest": {},
                                "packageFiles": [
                                    {
                                        "fileHandle": "some",
                                        "downloadUrl": "https://link.com",
                                    }
                                ],
                            },
                        ],
                        "nextPage": "some",
                    }
                },
            },
            {
                "status_code": 200,
                "json": {
                    "modelMetadata": {
                        "type": "external-model-metadata-v1",
                        "modelId": "my-model",
                        "modelArchitecture": "yolov8",
                        "taskType": "object-detection",
                        "modelPackages": [
                            {
                                "type": "external-model-package-v1",
                                "packageId": "my-package-id-2",
                                "packageManifest": {},
                                "packageFiles": [
                                    {
                                        "fileHandle": "some",
                                        "downloadUrl": "https://link.com",
                                    }
                                ],
                            }
                        ],
                    }
                },
            },
        ],
    )

    # when
    result = get_model_metadata(model_id="my-model", api_key="my-api-key")

    # then
    assert result == RoboflowModelMetadata(
        type="external-model-metadata-v1",
        modelId="my-model",
        modelArchitecture="yolov8",
        taskType="object-detection",
        modelPackages=[
            {"type": "new-unknown"},
            RoboflowModelPackageV1(
                type="external-model-package-v1",
                packageId="my-package-id-1",
                packageManifest={},
                packageFiles=[
                    RoboflowModelPackageFile(
                        fileHandle="some", downloadUrl="https://link.com", md5Hash=None
                    )
                ],
            ),
            RoboflowModelPackageV1(
                type="external-model-package-v1",
                packageId="my-package-id-2",
                packageManifest={},
                packageFiles=[
                    RoboflowModelPackageFile(
                        fileHandle="some", downloadUrl="https://link.com", md5Hash=None
                    )
                ],
            ),
        ],
        nextPage=None,
    )


def test_get_roboflow_model(requests_mock: Mocker) -> None:
    requests_mock.get(
        f"{ROBOFLOW_API_HOST}/models/v1/external/weights",
        [
            {
                "status_code": 200,
                "json": {
                    "modelMetadata": {
                        "type": "external-model-metadata-v1",
                        "modelId": "my-model",
                        "modelArchitecture": "yolov8",
                        "modelVariant": "yolov8-n",
                        "taskType": "object-detection",
                        "modelPackages": [
                            {
                                "type": "new-unknown",
                            },
                            {
                                "type": "external-model-package-v1",
                                "packageId": "my-package-id-1",
                                "packageManifest": {
                                    "type": "onnx-model-package-v1",
                                    "backendType": "onnx",
                                    "quantization": "fp32",
                                    "dynamicBatchSize": True,
                                    "opset": 19,
                                    "incompatibleProviders": ["TRTExecutionProvider"],
                                },
                                "packageFiles": [
                                    {
                                        "fileHandle": "some",
                                        "downloadUrl": "https://link.com",
                                    }
                                ],
                            },
                        ],
                        "nextPage": "some",
                    }
                },
            },
            {
                "status_code": 200,
                "json": {
                    "modelMetadata": {
                        "type": "external-model-metadata-v1",
                        "modelId": "my-model",
                        "modelArchitecture": "yolov8",
                        "modelVariant": "yolov8-n",
                        "taskType": "object-detection",
                        "modelPackages": [
                            {
                                "type": "external-model-package-v1",
                                "packageId": "my-package-id-2",
                                "packageManifest": {
                                    "type": "trt-model-package-v1",
                                    "backendType": "trt",
                                    "dynamicBatchSize": True,
                                    "static_batch_size": None,
                                    "minBatchSize": 1,
                                    "optBatchSize": 8,
                                    "maxBatchSize": 16,
                                    "quantization": "fp16",
                                    "cudaDeviceType": "orin",
                                    "cudaDeviceCC": "8.7",
                                    "cudaVersion": "12.6",
                                    "trtVersion": "10.3.0.17",
                                    "sameCCCompatible": False,
                                    "trtForwardCompatible": False,
                                    "trtLeanRuntimeExcluded": False,
                                    "machineType": "jetson",
                                    "machineSpecs": {
                                        "type": "jetson-machine-specs-v1",
                                        "l4tVersion": "36.4.3",
                                        "deviceName": "jetson-orin-nx",
                                        "driverVersion": "540.0.1",
                                    },
                                },
                                "packageFiles": [
                                    {
                                        "fileHandle": "some",
                                        "downloadUrl": "https://link.com",
                                    }
                                ],
                            }
                        ],
                    }
                },
            },
        ],
    )

    # when
    result = get_roboflow_model(model_id="my-model", api_key="my-api-key")

    # then
    assert result.model_id == "my-model"
    assert result.model_architecture == "yolov8"
    assert result.model_variant == "yolov8-n"
    assert result.task_type == "object-detection"
    assert len(result.model_packages) == 2


@patch.object(roboflow_module, "ROBOFLOW_LICENSE_SERVER", "license.local")
def test_get_roboflow_model_with_proxy(requests_mock: Mocker) -> None:
    # given
    requests_mock.register_uri(
        "GET",
        re.compile(r"http://license\.local/proxy"),
        [
            {
                "status_code": 200,
                "json": {
                    "modelMetadata": {
                        "type": "external-model-metadata-v1",
                        "modelId": "my-model",
                        "modelArchitecture": "yolov8",
                        "modelVariant": "yolov8-n",
                        "taskType": "object-detection",
                        "modelPackages": [
                            {
                                "type": "new-unknown",
                            },
                            {
                                "type": "external-model-package-v1",
                                "packageId": "my-package-id-1",
                                "packageManifest": {
                                    "type": "onnx-model-package-v1",
                                    "backendType": "onnx",
                                    "quantization": "fp32",
                                    "dynamicBatchSize": True,
                                    "opset": 19,
                                    "incompatibleProviders": ["TRTExecutionProvider"],
                                },
                                "packageFiles": [
                                    {
                                        "fileHandle": "some",
                                        "downloadUrl": "https://link.com",
                                    }
                                ],
                            },
                        ],
                        "nextPage": "some",
                    }
                },
            },
            {
                "status_code": 200,
                "json": {
                    "modelMetadata": {
                        "type": "external-model-metadata-v1",
                        "modelId": "my-model",
                        "modelArchitecture": "yolov8",
                        "modelVariant": "yolov8-n",
                        "taskType": "object-detection",
                        "modelPackages": [
                            {
                                "type": "external-model-package-v1",
                                "packageId": "my-package-id-2",
                                "packageManifest": {
                                    "type": "trt-model-package-v1",
                                    "backendType": "trt",
                                    "dynamicBatchSize": True,
                                    "static_batch_size": None,
                                    "minBatchSize": 1,
                                    "optBatchSize": 8,
                                    "maxBatchSize": 16,
                                    "quantization": "fp16",
                                    "cudaDeviceType": "orin",
                                    "cudaDeviceCC": "8.7",
                                    "cudaVersion": "12.6",
                                    "trtVersion": "10.3.0.17",
                                    "sameCCCompatible": False,
                                    "trtForwardCompatible": False,
                                    "trtLeanRuntimeExcluded": False,
                                    "machineType": "jetson",
                                    "machineSpecs": {
                                        "type": "jetson-machine-specs-v1",
                                        "l4tVersion": "36.4.3",
                                        "deviceName": "jetson-orin-nx",
                                        "driverVersion": "540.0.1",
                                    },
                                },
                                "packageFiles": [
                                    {
                                        "fileHandle": "some",
                                        "downloadUrl": "https://link.com",
                                    }
                                ],
                            }
                        ],
                    }
                },
            },
        ],
    )

    # when
    result = get_roboflow_model(model_id="my-model", api_key="my-api-key")

    # then — same structural assertions as non-proxy test
    assert result.model_id == "my-model"
    assert result.model_architecture == "yolov8"
    assert result.model_variant == "yolov8-n"
    assert result.task_type == "object-detection"
    assert len(result.model_packages) == 2

    # verify all API calls went through proxy
    assert requests_mock.call_count == 2
    for history_entry in requests_mock.request_history:
        parsed = urllib.parse.urlparse(history_entry.url)
        assert parsed.scheme == "http"
        assert parsed.netloc == "license.local"
        assert parsed.path == "/proxy"
        outer_params = urllib.parse.parse_qs(parsed.query)
        assert "url" in outer_params
        # verify the inner URL points to the real API
        inner_url = outer_params["url"][0]
        assert inner_url.startswith(f"{ROBOFLOW_API_HOST}/models/v1/external/weights")

    # verify download URLs in parsed packages are proxied
    for package in result.model_packages:
        for artefact in package.package_artefacts:
            parsed = urllib.parse.urlparse(artefact.download_url)
            assert parsed.netloc == "license.local"
            assert parsed.path == "/proxy"
            inner_url = urllib.parse.parse_qs(parsed.query)["url"][0]
            assert inner_url == "https://link.com"


@patch.object(roboflow_module, "ROBOFLOW_LICENSE_SERVER", "license.local:8080")
def test_basic_url_no_query():
    result = roboflow_license_server_proxy_url_builder(
        url="https://api.roboflow.com/models/v1/weights",
        query=None,
    )
    outer = _parse_proxy_result(result)
    assert outer.scheme == "http"
    assert outer.netloc == "license.local:8080"
    assert outer.path == "/proxy"

    inner_url = _extract_proxied_url(result)
    inner = urllib.parse.urlparse(inner_url)
    assert inner.scheme == "https"
    assert inner.netloc == "api.roboflow.com"
    assert inner.path == "/models/v1/weights"
    assert inner.query == ""


@patch.object(roboflow_module, "ROBOFLOW_LICENSE_SERVER", "license.local:8080")
def test_query_params_are_embedded_in_proxied_url():
    result = roboflow_license_server_proxy_url_builder(
        url="https://api.roboflow.com/weights",
        query={"modelId": "my-model", "api_key": "abc123"},
    )
    inner_params = _parse_proxied_url_query(result)
    assert inner_params["modelId"] == ["my-model"]
    assert inner_params["api_key"] == ["abc123"]


@patch.object(roboflow_module, "ROBOFLOW_LICENSE_SERVER", "license.local:8080")
def test_query_with_list_values():
    result = roboflow_license_server_proxy_url_builder(
        url="https://api.roboflow.com/weights",
        query={"tag": ["a", "b"]},
    )
    inner_params = _parse_proxied_url_query(result)
    assert inner_params["tag"] == ["a", "b"]


@patch.object(roboflow_module, "ROBOFLOW_LICENSE_SERVER", "license.local:8080")
def test_empty_query_dict_no_params_in_proxied_url():
    result = roboflow_license_server_proxy_url_builder(
        url="https://api.roboflow.com/weights",
        query={},
    )
    inner_url = _extract_proxied_url(result)
    inner = urllib.parse.urlparse(inner_url)
    assert inner.query == ""


@patch.object(roboflow_module, "ROBOFLOW_LICENSE_SERVER", None)
def test_no_license_server_returns_plain_url():
    result = roboflow_license_server_proxy_url_builder(
        url="https://api.roboflow.com/weights",
        query=None,
    )
    parsed = urllib.parse.urlparse(result)
    assert parsed.scheme == "https"
    assert parsed.netloc == "api.roboflow.com"
    assert parsed.path == "/weights"
    assert parsed.query == ""


@patch.object(roboflow_module, "ROBOFLOW_LICENSE_SERVER", None)
def test_no_license_server_with_query_returns_url_with_params():
    result = roboflow_license_server_proxy_url_builder(
        url="https://api.roboflow.com/weights",
        query={"modelId": "my-model"},
    )
    parsed = urllib.parse.urlparse(result)
    assert parsed.scheme == "https"
    assert parsed.netloc == "api.roboflow.com"
    assert parsed.path == "/weights"
    params = urllib.parse.parse_qs(parsed.query)
    assert params["modelId"] == ["my-model"]


@patch.object(roboflow_module, "ROBOFLOW_LICENSE_SERVER", "")
def test_empty_string_license_server_returns_plain_url():
    result = roboflow_license_server_proxy_url_builder(
        url="https://api.roboflow.com/weights",
        query=None,
    )
    parsed = urllib.parse.urlparse(result)
    assert parsed.scheme == "https"
    assert parsed.netloc == "api.roboflow.com"
    assert parsed.query == ""


@patch.object(roboflow_module, "ROBOFLOW_LICENSE_SERVER", "proxy.internal")
def test_url_with_existing_params_preserved_after_proxying():
    result = roboflow_license_server_proxy_url_builder(
        url="https://api.roboflow.com/weights?existing=val&other=123",
        query=None,
    )
    inner_params = _parse_proxied_url_query(result)
    assert inner_params["existing"] == ["val"]
    assert inner_params["other"] == ["123"]


@patch.object(roboflow_module, "ROBOFLOW_LICENSE_SERVER", "proxy.internal:9090")
def test_proxy_url_structure():
    result = roboflow_license_server_proxy_url_builder(
        url="https://api.roboflow.com/weights",
        query=None,
    )
    outer = _parse_proxy_result(result)
    assert outer.scheme == "http"
    assert outer.netloc == "proxy.internal:9090"
    assert outer.path == "/proxy"
    assert "url" in urllib.parse.parse_qs(outer.query)


@patch.object(roboflow_module, "ROBOFLOW_LICENSE_SERVER", "proxy.internal")
def test_proxy_uses_http_not_https():
    result = roboflow_license_server_proxy_url_builder(
        url="https://api.roboflow.com/weights",
        query=None,
    )
    outer = _parse_proxy_result(result)
    assert outer.scheme == "http"


@patch.object(roboflow_module, "ROBOFLOW_LICENSE_SERVER", "proxy.internal")
def test_roundtrip_preserves_query_with_special_characters():
    result = roboflow_license_server_proxy_url_builder(
        url="https://api.roboflow.com/weights",
        query={"modelId": "workspace/model", "api_key": "key=with=equals"},
    )
    inner_params = _parse_proxied_url_query(result)
    assert inner_params["modelId"] == ["workspace/model"]
    assert inner_params["api_key"] == ["key=with=equals"]


@patch.object(roboflow_module, "ROBOFLOW_LICENSE_SERVER", "proxy.internal")
def test_proxy_url_param_does_not_leak_into_inner_url():
    result = roboflow_license_server_proxy_url_builder(
        url="https://api.roboflow.com/weights",
        query={"modelId": "test"},
    )
    outer = _parse_proxy_result(result)
    outer_params = urllib.parse.parse_qs(outer.query)
    # outer should only have "url", not "modelId"
    assert "modelId" not in outer_params
    assert "url" in outer_params
    # inner should have "modelId", not "url"
    inner_params = _parse_proxied_url_query(result)
    assert "modelId" in inner_params


@patch.object(roboflow_module, "ROBOFLOW_LICENSE_SERVER", "proxy.internal")
def test_params_from_url_and_query_are_both_preserved():
    result = roboflow_license_server_proxy_url_builder(
        url="https://api.roboflow.com/weights?existing=from_url&page=1",
        query={"modelId": "my-model", "api_key": "abc123"},
    )
    inner_params = _parse_proxied_url_query(result)
    # params originally in the URL
    assert inner_params["existing"] == ["from_url"]
    assert inner_params["page"] == ["1"]
    # params added via query argument
    assert inner_params["modelId"] == ["my-model"]
    assert inner_params["api_key"] == ["abc123"]


@patch.object(roboflow_module, "ROBOFLOW_LICENSE_SERVER", "proxy.internal")
def test_overlapping_params_from_url_and_query():
    with pytest.raises(AssumptionError):
        _ = roboflow_license_server_proxy_url_builder(
            url="https://api.roboflow.com/weights?key=from_url",
            query={"key": "from_query"},
        )


@patch.object(roboflow_module, "ROBOFLOW_LICENSE_SERVER", None)
def test_params_from_url_and_query_are_both_preserved_without_proxy():
    result = roboflow_license_server_proxy_url_builder(
        url="https://api.roboflow.com/weights?existing=from_url",
        query={"modelId": "my-model"},
    )
    parsed = urllib.parse.urlparse(result)
    params = urllib.parse.parse_qs(parsed.query)
    assert params["existing"] == ["from_url"]
    assert params["modelId"] == ["my-model"]


def test_parse_mediapipe_model_package_when_package_is_valid_with_proxy_builder() -> (
    None
):
    # given
    metadata = RoboflowModelPackageV1(
        type="external-model-package-v1",
        packageId="my-package-id",
        packageManifest={
            "type": "mediapipe-model-package-v1",
            "backendType": "mediapipe",
        },
        packageFiles=[
            RoboflowModelPackageFile(
                fileHandle="some", downloadUrl="https://dummy.com", md5Hash="some"
            )
        ],
        trustedSource=True,
    )

    # when
    result = parse_model_package_metadata(
        metadata=metadata,
        proxy_url_builder=_dummy_proxy_url_builder,
    )

    # then
    assert result == ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.MEDIAPIPE,
        package_artefacts=[
            FileDownloadSpecs(
                download_url=f"{DUMMY_PROXY_PREFIX}https://dummy.com",
                file_handle="some",
                md5_hash="some",
            ),
        ],
        quantization=Quantization.UNKNOWN,
        trusted_source=True,
    )


def test_parse_ultralytics_model_package_with_proxy_builder() -> None:
    # given
    metadata = RoboflowModelPackageV1(
        type="external-model-package-v1",
        packageId="my-package-id",
        packageManifest={
            "type": "ultralytics-model-package-v1",
            "backendType": "ultralytics",
        },
        packageFiles=[
            RoboflowModelPackageFile(
                fileHandle="some", downloadUrl="https://dummy.com", md5Hash="some"
            )
        ],
        trustedSource=False,
    )

    # when
    result = parse_model_package_metadata(
        metadata=metadata,
        proxy_url_builder=_dummy_proxy_url_builder,
    )

    # then
    assert result == ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.ULTRALYTICS,
        package_artefacts=[
            FileDownloadSpecs(
                download_url=f"{DUMMY_PROXY_PREFIX}https://dummy.com",
                file_handle="some",
                md5_hash="some",
            ),
        ],
        quantization=Quantization.UNKNOWN,
        trusted_source=False,
    )


def test_parse_hf_model_package_when_valid_input_provided_with_proxy_builder() -> None:
    # given
    metadata = RoboflowModelPackageV1(
        type="external-model-package-v1",
        packageId="my-package-id",
        packageManifest={
            "type": "hf-model-package-v1",
            "backendType": "hf",
            "quantization": "fp32",
        },
        packageFiles=[
            RoboflowModelPackageFile(
                fileHandle="some", downloadUrl="https://dummy.com", md5Hash="some"
            )
        ],
        trustedSource=True,
    )

    # when
    result = parse_model_package_metadata(
        metadata=metadata,
        proxy_url_builder=_dummy_proxy_url_builder,
    )

    # then
    assert result == ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.HF,
        package_artefacts=[
            FileDownloadSpecs(
                download_url=f"{DUMMY_PROXY_PREFIX}https://dummy.com",
                file_handle="some",
                md5_hash="some",
            ),
        ],
        quantization=Quantization.FP32,
        trusted_source=True,
    )


def test_parse_torch_model_package_when_valid_manifest_provided_with_proxy_builder() -> (
    None
):
    # given
    metadata = RoboflowModelPackageV1(
        type="external-model-package-v1",
        packageId="my-package-id",
        packageManifest={
            "type": "torch-model-package-v1",
            "backendType": "torch",
            "quantization": "fp32",
            "dynamicBatchSize": True,
        },
        packageFiles=[
            RoboflowModelPackageFile(
                fileHandle="some", downloadUrl="https://dummy.com", md5Hash="some"
            )
        ],
        trustedSource=True,
    )

    # when
    result = parse_model_package_metadata(
        metadata=metadata,
        proxy_url_builder=_dummy_proxy_url_builder,
    )

    # then
    assert result == ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.TORCH,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        static_batch_size=None,
        package_artefacts=[
            FileDownloadSpecs(
                download_url=f"{DUMMY_PROXY_PREFIX}https://dummy.com",
                file_handle="some",
                md5_hash="some",
            ),
        ],
        trusted_source=True,
    )


def test_parse_onnx_model_package_when_valid_manifest_provided_with_proxy_builder() -> (
    None
):
    # given
    metadata = RoboflowModelPackageV1(
        type="external-model-package-v1",
        packageId="my-package-id",
        packageManifest={
            "type": "onnx-model-package-v1",
            "backendType": "onnx",
            "quantization": "fp32",
            "dynamicBatchSize": True,
            "opset": 19,
            "incompatibleProviders": ["TRTExecutionProvider"],
        },
        packageFiles=[
            RoboflowModelPackageFile(
                fileHandle="some", downloadUrl="https://dummy.com", md5Hash="some"
            )
        ],
        trustedSource=True,
    )

    # when
    result = parse_model_package_metadata(
        metadata=metadata,
        proxy_url_builder=_dummy_proxy_url_builder,
    )

    # then
    assert result == ModelPackageMetadata(
        package_id=metadata.package_id,
        backend=BackendType.ONNX,
        quantization=Quantization.FP32,
        dynamic_batch_size_supported=True,
        static_batch_size=None,
        package_artefacts=[
            FileDownloadSpecs(
                download_url=f"{DUMMY_PROXY_PREFIX}https://dummy.com",
                file_handle="some",
                md5_hash="some",
            ),
        ],
        onnx_package_details=ONNXPackageDetails(
            opset=19,
            incompatible_providers=["TRTExecutionProvider"],
        ),
        trusted_source=True,
    )


def test_parse_trt_model_package_with_jetson_env_and_proxy_builder() -> None:
    # given
    metadata = RoboflowModelPackageV1(
        type="external-model-package-v1",
        packageId="my-package-id",
        packageManifest={
            "type": "trt-model-package-v1",
            "backendType": "trt",
            "dynamicBatchSize": True,
            "static_batch_size": None,
            "minBatchSize": 1,
            "optBatchSize": 8,
            "maxBatchSize": 16,
            "quantization": "fp16",
            "cudaDeviceType": "orin",
            "cudaDeviceCC": "8.7",
            "cudaVersion": "12.6",
            "trtVersion": "10.3.0.17",
            "sameCCCompatible": False,
            "trtForwardCompatible": False,
            "trtLeanRuntimeExcluded": False,
            "machineType": "jetson",
            "machineSpecs": {
                "type": "jetson-machine-specs-v1",
                "l4tVersion": "36.4.3",
                "deviceName": "jetson-orin-nx",
                "driverVersion": "540.0.1",
            },
        },
        packageFiles=[
            RoboflowModelPackageFile(
                fileHandle="some", downloadUrl="https://dummy.com", md5Hash="some"
            )
        ],
        trustedSource=True,
    )

    # when
    result = parse_model_package_metadata(
        metadata=metadata,
        proxy_url_builder=_dummy_proxy_url_builder,
    )

    # then
    environment_requirements = JetsonEnvironmentRequirements(
        cuda_device_cc=Version("8.7"),
        cuda_device_name="orin",
        l4t_version=Version("36.4.3"),
        jetson_product_name="jetson-orin-nx",
        cuda_version=Version("12.6"),
        trt_version=Version("10.3.0.17"),
        driver_version=Version("540.0.1"),
    )
    trt_package_details = TRTPackageDetails(
        min_dynamic_batch_size=1,
        opt_dynamic_batch_size=8,
        max_dynamic_batch_size=16,
        same_cc_compatible=False,
        trt_forward_compatible=False,
        trt_lean_runtime_excluded=False,
    )
    assert result == ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.TRT,
        quantization=Quantization.FP16,
        dynamic_batch_size_supported=True,
        static_batch_size=None,
        trt_package_details=trt_package_details,
        package_artefacts=[
            FileDownloadSpecs(
                download_url=f"{DUMMY_PROXY_PREFIX}https://dummy.com",
                file_handle="some",
                md5_hash="some",
            ),
        ],
        environment_requirements=environment_requirements,
        trusted_source=True,
    )


def test_parse_trt_model_package_with_server_env_and_proxy_builder() -> None:
    # given
    metadata = RoboflowModelPackageV1(
        type="external-model-package-v1",
        packageId="my-package-id",
        packageManifest={
            "type": "trt-model-package-v1",
            "backendType": "trt",
            "dynamicBatchSize": True,
            "static_batch_size": None,
            "minBatchSize": 1,
            "optBatchSize": 8,
            "maxBatchSize": 16,
            "quantization": "fp16",
            "cudaDeviceType": "orin",
            "cudaDeviceCC": "8.7",
            "cudaVersion": "12.6",
            "trtVersion": "10.3.0.17",
            "sameCCCompatible": False,
            "trtForwardCompatible": False,
            "trtLeanRuntimeExcluded": False,
            "machineType": "gpu-server",
            "machineSpecs": {
                "type": "gpu-server-specs-v1",
                "osVersion": "ubuntu-20.04",
                "driverVersion": "540.0.1",
            },
        },
        packageFiles=[
            RoboflowModelPackageFile(
                fileHandle="some", downloadUrl="https://dummy.com", md5Hash="some"
            )
        ],
        trustedSource=True,
    )

    # when
    result = parse_model_package_metadata(
        metadata=metadata,
        proxy_url_builder=_dummy_proxy_url_builder,
    )

    # then
    environment_requirements = ServerEnvironmentRequirements(
        cuda_device_cc=Version("8.7"),
        cuda_device_name="orin",
        cuda_version=Version("12.6"),
        trt_version=Version("10.3.0.17"),
        driver_version=Version("540.0.1"),
        os_version="ubuntu-20.04",
    )
    trt_package_details = TRTPackageDetails(
        min_dynamic_batch_size=1,
        opt_dynamic_batch_size=8,
        max_dynamic_batch_size=16,
        same_cc_compatible=False,
        trt_forward_compatible=False,
        trt_lean_runtime_excluded=False,
    )
    assert result == ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.TRT,
        quantization=Quantization.FP16,
        dynamic_batch_size_supported=True,
        static_batch_size=None,
        trt_package_details=trt_package_details,
        package_artefacts=[
            FileDownloadSpecs(
                download_url=f"{DUMMY_PROXY_PREFIX}https://dummy.com",
                file_handle="some",
                md5_hash="some",
            ),
        ],
        environment_requirements=environment_requirements,
        trusted_source=True,
    )


def test_parse_torch_script_model_package_when_valid_with_proxy_builder() -> None:
    # given
    metadata = RoboflowModelPackageV1(
        type="external-model-package-v1",
        packageId="my-package-id",
        packageManifest={
            "type": "torch-script-model-package-v1",
            "backendType": "torch-script",
            "dynamicBatchSize": False,
            "staticBatchSize": 2,
            "quantization": "fp32",
            "supportedDeviceTypes": ["cuda", "cpu", "mps"],
            "torchVersion": "2.6.0",
            "torchVisionVersion": "0.22.0",
        },
        packageFiles=[
            RoboflowModelPackageFile(
                fileHandle="some", downloadUrl="https://dummy.com", md5Hash="some"
            )
        ],
        modelFeatures={"nms_fused": True},
        trustedSource=True,
    )

    # when
    result = parse_model_package_metadata(
        metadata=metadata,
        proxy_url_builder=_dummy_proxy_url_builder,
    )

    # then
    assert result == ModelPackageMetadata(
        package_id="my-package-id",
        backend=BackendType.TORCH_SCRIPT,
        dynamic_batch_size_supported=False,
        static_batch_size=2,
        package_artefacts=[
            FileDownloadSpecs(
                download_url=f"{DUMMY_PROXY_PREFIX}https://dummy.com",
                file_handle="some",
                md5_hash="some",
            ),
        ],
        quantization=Quantization.FP32,
        trusted_source=True,
        model_features={"nms_fused": True},
        torch_script_package_details=TorchScriptPackageDetails(
            supported_device_types={"cuda", "cpu", "mps"},
            torch_version=Version("2.6.0"),
            torch_vision_version=Version("0.22.0"),
        ),
    )


def _parse_proxy_result(result: str) -> urllib.parse.ParseResult:
    """Parse the outer proxy URL and return ParseResult."""
    return urllib.parse.urlparse(result)


def _parse_proxied_url_query(result: str) -> dict:
    """Extract and parse query params from the inner proxied URL."""
    inner_url = _extract_proxied_url(result)
    parsed_inner = urllib.parse.urlparse(inner_url)
    return urllib.parse.parse_qs(parsed_inner.query)


def _extract_proxied_url(result: str) -> str:
    """Extract the inner URL from the proxy's ?url= parameter."""
    parsed = _parse_proxy_result(result)
    proxy_params = urllib.parse.parse_qs(parsed.query)
    assert "url" in proxy_params, f"Expected 'url' query param in: {result}"
    assert len(proxy_params["url"]) == 1, f"Expected single 'url' value in: {result}"
    return proxy_params["url"][0]


def _dummy_proxy_url_builder(
    url: str, query: Optional[Dict[str, Union[str, List[str]]]]
) -> str:
    return f"{DUMMY_PROXY_PREFIX}{url}"
