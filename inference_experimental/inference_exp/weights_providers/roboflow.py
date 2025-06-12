import json
from typing import Annotated, Callable, Dict, List, Literal, Optional, Union

import backoff
import requests
from inference_exp.configuration import (
    API_CALLS_MAX_RETRIES,
    API_CALLS_TIMEOUT,
    IDEMPOTENT_API_REQUEST_CODES_TO_RETRY,
    ROBOFLOW_API_HOST,
    ROBOFLOW_API_KEY,
)
from inference_exp.errors import (
    ModelMetadataConsistencyError,
    ModelMetadataHandlerNotImplementedError,
    ModelRetrievalError,
    RetryError,
    UnauthorizedModelAccessError,
)
from inference_exp.logger import logger
from inference_exp.weights_providers.entities import (
    BackendType,
    FileDownloadSpecs,
    JetsonEnvironmentRequirements,
    ModelMetadata,
    ModelPackageMetadata,
    ONNXPackageDetails,
    Quantization,
    ServerEnvironmentRequirements,
    TRTPackageDetails,
)
from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, Discriminator, Field, ValidationError
from requests import Response, Timeout

MAX_MODEL_PACKAGE_PAGES = 10
MODEL_PACKAGES_TO_IGNORE = {
    "oak-model-package-v1",
    "tfjs-model-package-v1",
}


class RoboflowModelPackageFile(BaseModel):
    file_name: str = Field(alias="fileName")
    download_url: str = Field(alias="downloadUrl")


class RoboflowModelPackageV1(BaseModel):
    type: Literal["external-model-package-v1"]
    package_id: str = Field(alias="packageId")
    package_manifest: dict = Field(alias="packageManifest")
    package_files: List[RoboflowModelPackageFile] = Field(alias="packageFiles")


class RoboflowModelMetadata(BaseModel):
    type: Literal["external-model-metadata-v1"]
    model_id: str = Field(alias="modelId")
    model_architecture: str = Field(alias="modelArchitecture")
    task_type: Optional[str] = Field(alias="taskType", default=None)
    model_packages: List[Union[RoboflowModelPackageV1, dict]] = Field(
        alias="modelPackages",
    )
    next_page: Optional[str] = Field(alias="nextPage")


def get_roboflow_model(model_id: str, api_key: Optional[str] = None) -> ModelMetadata:
    model_metadata = get_model_metadata(model_id=model_id, api_key=api_key)
    parsed_model_packages = []
    for model_package in model_metadata.model_packages:
        parsed_model_package = parse_model_package_metadata(metadata=model_package)
        parsed_model_packages.append(parsed_model_package)
    return ModelMetadata(
        model_id=model_metadata.model_id,
        model_architecture=model_metadata.model_architecture,
        model_packages=parsed_model_packages,
        task_type=model_metadata.task_type,
    )


def get_model_metadata(model_id: str, api_key: Optional[str]) -> RoboflowModelMetadata:
    if api_key is None:
        api_key = ROBOFLOW_API_KEY
    fetched_pages = []
    start_after = None
    while len(fetched_pages) < MAX_MODEL_PACKAGE_PAGES:
        pagination_result = get_one_page_of_model_metadata(
            model_id=model_id, api_key=api_key, start_after=start_after
        )
        fetched_pages.append(pagination_result)
        start_after = pagination_result.next_page
        if start_after is None:
            break
    all_model_packages = []
    for page in fetched_pages:
        all_model_packages.extend(page.model_packages)
    if not fetched_pages or not all_model_packages:
        raise ModelRetrievalError(
            f"Could not retrieve model {model_id} from Roboflow API. Backend provided empty list of model "
            f"packages `inference` library could load. Contact Roboflow to solve the problem."
        )
    fetched_pages[-1].model_packages = all_model_packages
    return fetched_pages[-1]


@backoff.on_exception(
    backoff.expo,
    exception=RetryError,
    max_tries=API_CALLS_MAX_RETRIES,
)
def get_one_page_of_model_metadata(
    model_id: str,
    api_key: Optional[str] = None,
    page_size: Optional[int] = None,
    start_after: Optional[str] = None,
) -> RoboflowModelMetadata:
    query = {
        "modelId": model_id,
    }
    if api_key:
        query["api_key"] = api_key
    if page_size:
        query["pageSize"] = page_size
    if start_after:
        query["startAfter"] = start_after
    try:
        response = requests.get(
            f"{ROBOFLOW_API_HOST}/models/v1/external/weights",
            params=query,
            timeout=API_CALLS_TIMEOUT,
        )
    except (OSError, Timeout, requests.exceptions.ConnectionError):
        raise RetryError(f"Connectivity error")
    if response.status_code in IDEMPOTENT_API_REQUEST_CODES_TO_RETRY:
        raise RetryError(f"Roboflow API responded with {response.status_code}")
    handle_response_errors(response=response, operation_name="get model weights")
    try:
        return RoboflowModelMetadata.model_validate(response.json()["modelMetadata"])
    except (ValueError, ValidationError, KeyError) as error:
        raise ModelRetrievalError(
            f"Could not decode Roboflow API response when trying to retrieve model {model_id}. If that problem "
            f"is not ephemeral - contact Roboflow."
        ) from error


def handle_response_errors(response: Response, operation_name: str) -> None:
    if response.status_code in IDEMPOTENT_API_REQUEST_CODES_TO_RETRY:
        raise RetryError(
            f"Roboflow API returned invalid response code for {operation_name} operation "
            f"{response.status_code}. If that problem is not ephemeral - contact Roboflow."
        )
    if response.status_code == 401:
        raise UnauthorizedModelAccessError(
            f"Could not {operation_name}. Request unauthorised. Are you sure you use valid Roboflow API key? "
            "See details here: https://docs.roboflow.com/api-reference/authentication and "
            "export key to `ROBOFLOW_API_KEY` environment variable"
        )
    if response.status_code >= 400:
        response_payload = get_error_response_payload(response=response)
        raise ModelRetrievalError(
            f"Roboflow API returned invalid response code for {operation_name} operation "
            f"{response.status_code}.\n\nResponse:\n{response_payload}"
        )


def get_error_response_payload(response: Response) -> str:
    try:
        return json.dumps(response.json(), indent=4)
    except ValueError:
        return response.text


def parse_model_package_metadata(
    metadata: Union[Union[RoboflowModelPackageV1, dict]]
) -> Optional[ModelPackageMetadata]:
    if isinstance(metadata, dict):
        print(metadata)
        metadata_type = metadata.get("type", "unknown")
        model_package_id = metadata.get("packageId", "unknown")
        logger.warning(
            "Roboflow API returned entity describing model package which cannot be parsed. This may indicate that "
            f"your `inference` package is outdated. "
            f"Debug info - entity type: `{metadata_type}`, model package id: {model_package_id}"
        )
        return None
    manifest_type = metadata.package_manifest.get("type", "unknown")
    if manifest_type in MODEL_PACKAGES_TO_IGNORE:
        return None
    if manifest_type not in MODEL_PACKAGE_PARSERS:
        logger.warning(
            "Roboflow API returned entity describing model package which cannot be parsed. This may indicate that "
            f"your `inference` package is outdated. "
            f"Debug info - package manifest type: `{manifest_type}`."
        )
        return None
    try:
        return MODEL_PACKAGE_PARSERS[manifest_type](metadata)
    except Exception as error:
        raise ModelMetadataConsistencyError(
            "Roboflow API returned model package metadata which cannot be parsed. Contact Roboflow to "
            f"solve the problem. Error details: {error}. Error type: {error.__class__.__name__}"
        ) from error


class OnnxModelPackageV1(BaseModel):
    type: Literal["onnx-model-package-v1"]
    backend_type: Literal["onnx"] = Field(alias="backendType")
    dynamic_batch_size: bool = Field(alias="dynamicBatchSize", default=False)
    static_batch_size: Optional[int] = Field(alias="staticBatchSize", default=None)
    quantization: Quantization
    opset: int
    incompatible_providers: Optional[List[str]] = Field(alias="incompatibleProviders", default=None)


def parse_onnx_model_package(metadata: RoboflowModelPackageV1) -> ModelPackageMetadata:
    parsed_manifest = OnnxModelPackageV1.model_validate(metadata.package_manifest)
    if (
        not parsed_manifest.dynamic_batch_size
        and parsed_manifest.static_batch_size is None
    ):
        raise ModelMetadataConsistencyError(
            "While downloading model weights, Roboflow API provided inconsistent metadata "
            "describing model package - ONNX package declared not to support dynamic batch size and "
            "supported static batch size not provided. Contact Roboflow to solve the problem."
        )
    package_artefacts = parse_package_artefacts(
        package_artefacts=metadata.package_files
    )
    return ModelPackageMetadata(
        package_id=metadata.package_id,
        backend=BackendType.ONNX,
        quantization=parsed_manifest.quantization,
        dynamic_batch_size_supported=parsed_manifest.dynamic_batch_size,
        static_batch_size=parsed_manifest.static_batch_size,
        package_artefacts=package_artefacts,
        onnx_package_details=ONNXPackageDetails(
            opset=parsed_manifest.opset,
            incompatible_providers=parsed_manifest.incompatible_providers,
        ),
    )


class JetsonMachineSpecsV1(BaseModel):
    type: Literal["jetson-machine-specs-v1"]
    l4t_version: str = Field(alias="l4tVersion")
    device_name: str = Field(alias="deviceName")
    driver_version: str = Field(alias="driverVersion")


class GPUServerSpecsV1(BaseModel):
    type: Literal["gpu-server-specs-v1"]
    driver_version: str = Field(alias="driverVersion")
    os_version: str = Field(alias="osVersion")


class TrtModelPackageV1(BaseModel):
    type: Literal["trt-model-package-v1"]
    backend_type: Literal["trt"] = Field(alias="backendType")
    dynamic_batch_size: bool = Field(alias="dynamicBatchSize", default=False)
    static_batch_size: Optional[int] = Field(alias="staticBatchSize", default=None)
    min_batch_size: Optional[int] = Field(alias="minBatchSize", default=None)
    opt_batch_size: Optional[int] = Field(alias="optBatchSize", default=None)
    max_batch_size: Optional[int] = Field(alias="maxBatchSize", default=None)
    quantization: Quantization
    cuda_device_type: str = Field(alias="cudaDeviceType")
    cuda_device_cc: str = Field(alias="cudaDeviceCC")
    cuda_version: str = Field(alias="cudaVersion")
    trt_version: str = Field(alias="trtVersion")
    same_cc_compatible: bool = Field(alias="sameCCCompatible", default=False)
    machine_type: Literal["gpu-server", "jetson"] = Field(alias="machineType")
    machine_specs: Annotated[
        Union[JetsonMachineSpecsV1, GPUServerSpecsV1],
        Discriminator(discriminator="type"),
    ] = Field(alias="machineSpecs")


def parse_trt_model_package(metadata: RoboflowModelPackageV1) -> ModelPackageMetadata:
    parsed_manifest = TrtModelPackageV1.model_validate(metadata.package_manifest)
    if (
        not parsed_manifest.dynamic_batch_size
        and parsed_manifest.static_batch_size is None
    ):
        raise ModelMetadataConsistencyError(
            "While downloading model weights, Roboflow API provided inconsistent metadata "
            "describing model package - ONNX package declared not to support dynamic batch size and "
            "supported static batch size not provided. Contact Roboflow to solve the problem."
        )
    if parsed_manifest.machine_type == "gpu-server":
        if not isinstance(parsed_manifest.machine_specs, GPUServerSpecsV1):
            raise ModelMetadataConsistencyError(
                "While downloading model weights, Roboflow API provided inconsistent metadata "
                "describing model package - expected GPU Server specification for TRT model package registered as "
                "compiled on gpu-server. Contact Roboflow to solve the problem."
            )
        environment_requirements = ServerEnvironmentRequirements(
            cuda_device_cc=as_version(parsed_manifest.cuda_device_cc),
            cuda_device_name=parsed_manifest.cuda_device_type,
            driver_version=as_version(parsed_manifest.machine_specs.driver_version),
            cuda_version=as_version(parsed_manifest.cuda_version),
            trt_version=as_version(parsed_manifest.trt_version),
            os_version=parsed_manifest.machine_specs.os_version,
        )
    elif parsed_manifest.machine_type == "jetson":
        if not isinstance(parsed_manifest.machine_specs, JetsonMachineSpecsV1):
            raise ModelMetadataConsistencyError(
                "While downloading model weights, Roboflow API provided inconsistent metadata "
                "describing model package - expected Jetson Device specification for TRT model package registered as "
                "compiled on Jetson. Contact Roboflow to solve the problem."
            )
        environment_requirements = JetsonEnvironmentRequirements(
            cuda_device_cc=as_version(parsed_manifest.cuda_device_cc),
            cuda_device_name=parsed_manifest.cuda_device_type,
            l4t_version=as_version(parsed_manifest.machine_specs.l4t_version),
            jetson_product_name=parsed_manifest.machine_specs.device_name,
            cuda_version=as_version(parsed_manifest.cuda_version),
            trt_version=as_version(parsed_manifest.trt_version),
            driver_version=as_version(parsed_manifest.machine_specs.driver_version),
        )
    else:
        raise ModelMetadataHandlerNotImplementedError(
            "While downloading model weights, Roboflow API provided metadata which are not handled by current version "
            "of inference detected while parsing TRT model package. This problem may indicate that your inference "
            "package is outdated. Try to upgrade - if that does not help, contact Roboflow to solve the problem."
        )
    package_artefacts = parse_package_artefacts(
        package_artefacts=metadata.package_files
    )
    trt_package_details = TRTPackageDetails(
        min_dynamic_batch_size=parsed_manifest.min_batch_size,
        opt_dynamic_batch_size=parsed_manifest.opt_batch_size,
        max_dynamic_batch_size=parsed_manifest.max_batch_size,
        same_cc_compatible=parsed_manifest.same_cc_compatible,
    )
    return ModelPackageMetadata(
        package_id=metadata.package_id,
        backend=BackendType.TRT,
        quantization=parsed_manifest.quantization,
        dynamic_batch_size_supported=parsed_manifest.dynamic_batch_size,
        static_batch_size=parsed_manifest.static_batch_size,
        trt_package_details=trt_package_details,
        package_artefacts=package_artefacts,
        environment_requirements=environment_requirements,
    )


class TorchModelPackageV1(BaseModel):
    type: Literal["torch-model-package-v1"]
    backend_type: Literal["torch"] = Field(alias="backendType")
    dynamic_batch_size: bool = Field(alias="dynamicBatchSize", default=False)
    static_batch_size: Optional[int] = Field(alias="staticBatchSize", default=None)
    quantization: Quantization


def parse_torch_model_package(metadata: RoboflowModelPackageV1) -> ModelPackageMetadata:
    parsed_manifest = TorchModelPackageV1.model_validate(metadata.package_manifest)
    if (
        not parsed_manifest.dynamic_batch_size
        and parsed_manifest.static_batch_size is None
    ):
        raise ModelMetadataConsistencyError(
            "While downloading model weights, Roboflow API provided inconsistent metadata "
            "describing model package - ONNX package declared not to support dynamic batch size and "
            "supported static batch size not provided. Contact Roboflow to solve the problem."
        )
    package_artefacts = parse_package_artefacts(
        package_artefacts=metadata.package_files
    )
    return ModelPackageMetadata(
        package_id=metadata.package_id,
        backend=BackendType.TORCH,
        quantization=parsed_manifest.quantization,
        dynamic_batch_size_supported=parsed_manifest.dynamic_batch_size,
        static_batch_size=parsed_manifest.static_batch_size,
        package_artefacts=package_artefacts,
    )


class HFModelPackageV1(BaseModel):
    type: Literal["hf-model-package-v1"]
    backend_type: Literal["hf"] = Field(alias="backendType")
    quantization: Quantization


def parse_hf_model_package(metadata: RoboflowModelPackageV1) -> ModelPackageMetadata:
    parsed_manifest = HFModelPackageV1.model_validate(metadata.package_manifest)
    package_artefacts = parse_package_artefacts(
        package_artefacts=metadata.package_files
    )
    return ModelPackageMetadata(
        package_id=metadata.package_id,
        backend=BackendType.HF,
        quantization=parsed_manifest.quantization,
        package_artefacts=package_artefacts,
    )


def parse_ultralytics_model_package(
    metadata: RoboflowModelPackageV1,
) -> ModelPackageMetadata:
    package_artefacts = parse_package_artefacts(
        package_artefacts=metadata.package_files
    )
    return ModelPackageMetadata(
        package_id=metadata.package_id,
        backend=BackendType.ULTRALYTICS,
        package_artefacts=package_artefacts,
        quantization=Quantization.UNKNOWN,
    )


def parse_package_artefacts(
    package_artefacts: List[RoboflowModelPackageFile],
) -> List[FileDownloadSpecs]:
    return [
        FileDownloadSpecs(download_url=f.download_url, file_name=f.file_name)
        for f in package_artefacts
    ]


MODEL_PACKAGE_PARSERS: Dict[
    str, Callable[[RoboflowModelPackageV1], ModelPackageMetadata]
] = {
    "onnx-model-package-v1": parse_onnx_model_package,
    "trt-model-package-v1": parse_trt_model_package,
    "torch-model-package-v1": parse_torch_model_package,
    "hf-model-package-v1": parse_hf_model_package,
    "ultralytics-model-package-v1": parse_ultralytics_model_package,
}


def as_version(value: str) -> Version:
    try:
        return Version(value)
    except InvalidVersion as error:
        raise ModelMetadataConsistencyError(
            "Roboflow API returned model package manifest that is expected to provide valid version specification for "
            "one of the field of package manifest, but instead provides value that cannot be parsed. This is most "
            "likely Roboflow API bug - contact Roboflow to solve the problem."
        ) from error
