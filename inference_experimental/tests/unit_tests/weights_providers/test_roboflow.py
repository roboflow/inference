import json
import urllib.parse
from typing import Optional

import pytest
from inference_exp.configuration import API_CALLS_MAX_TRIES, ROBOFLOW_API_HOST
from inference_exp.errors import (
    ModelMetadataConsistencyError,
    ModelRetrievalError,
    RetryError,
    UnauthorizedModelAccessError,
)
from inference_exp.weights_providers.entities import (
    BackendType,
    FileDownloadSpecs,
    JetsonEnvironmentRequirements,
    ModelPackageMetadata,
    ONNXPackageDetails,
    Quantization,
    ServerEnvironmentRequirements,
    TorchScriptPackageDetails,
    TRTPackageDetails,
)
from inference_exp.weights_providers.roboflow import (
    RoboflowModelMetadata,
    RoboflowModelPackageFile,
    RoboflowModelPackageV1,
    as_version,
    get_error_response_payload,
    get_model_metadata,
    get_one_page_of_model_metadata,
    get_roboflow_model,
    handle_response_errors,
    parse_hf_model_package,
    parse_model_package_metadata,
    parse_onnx_model_package,
    parse_torch_model_package,
    parse_trt_model_package,
    parse_ultralytics_model_package,
)
from packaging.version import Version
from pydantic import ValidationError
from requests import Response
from requests_mock import Mocker


def test_as_version_when_valid_version_provided() -> None:
    # when
    result = as_version(value="1.2.3")

    # then
    assert result == Version("1.2.3")


def test_as_version_when_invalid_version_provided() -> None:
    # when
    with pytest.raises(ModelMetadataConsistencyError):
        _ = as_version(value="invalid")


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

    parsed_params = urllib.parse.parse_qs(requests_mock.last_request.query)
    assert parsed_params["modelid"][0] == "my-model"
    assert parsed_params["api_key"][0] == "some"
    assert parsed_params["pagesize"][0] == "100"
    assert parsed_params["startafter"][0] == "start"


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
    assert result.task_type == "object-detection"
    assert len(result.model_packages) == 2
