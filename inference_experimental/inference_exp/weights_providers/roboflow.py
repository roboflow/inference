import json
from typing import Annotated, Callable, Dict, List, Literal, Optional, Union

import backoff
import requests
from inference_exp.configuration import (
    API_CALLS_MAX_TRIES,
    API_CALLS_TIMEOUT,
    IDEMPOTENT_API_REQUEST_CODES_TO_RETRY,
    ROBOFLOW_API_HOST,
    ROBOFLOW_API_KEY,
)
from inference_exp.errors import (
    BaseInferenceError,
    ModelMetadataConsistencyError,
    ModelMetadataHandlerNotImplementedError,
    ModelRetrievalError,
    RetryError,
    UnauthorizedModelAccessError,
)
from inference_exp.logger import LOGGER
from inference_exp.weights_providers.entities import (
    BackendType,
    FileDownloadSpecs,
    JetsonEnvironmentRequirements,
    ModelMetadata,
    ModelPackageMetadata,
    ONNXPackageDetails,
    Quantization,
    ServerEnvironmentRequirements,
    TorchScriptPackageDetails,
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
    file_handle: str = Field(alias="fileHandle")
    download_url: str = Field(alias="downloadUrl")
    md5_hash: Optional[str] = Field(alias="md5Hash", default=None)


class RoboflowModelPackageV1(BaseModel):
    type: Literal["external-model-package-v1"]
    package_id: str = Field(alias="packageId")
    package_manifest: dict = Field(alias="packageManifest")
    model_features: Optional[dict] = Field(alias="modelFeatures", default=None)
    package_files: List[RoboflowModelPackageFile] = Field(alias="packageFiles")
    trusted_source: bool = Field(alias="trustedSource", default=False)


class RoboflowModelMetadata(BaseModel):
    type: Literal["external-model-metadata-v1"]
    model_id: str = Field(alias="modelId")
    model_architecture: str = Field(alias="modelArchitecture")
    task_type: Optional[str] = Field(alias="taskType", default=None)
    model_packages: List[Union[RoboflowModelPackageV1, dict]] = Field(
        alias="modelPackages",
    )
    next_page: Optional[str] = Field(alias="nextPage", default=None)


def get_roboflow_model(model_id: str, api_key: Optional[str] = None) -> ModelMetadata:
    model_metadata = get_model_metadata(model_id=model_id, api_key=api_key)
    parsed_model_packages = []
    for model_package in model_metadata.model_packages:
        parsed_model_package = parse_model_package_metadata(metadata=model_package)
        if parsed_model_package is None:
            continue
        parsed_model_packages.append(parsed_model_package)
    return ModelMetadata(
        model_id=model_metadata.model_id,
        model_architecture=model_metadata.model_architecture,
        model_packages=parsed_model_packages,
        task_type=model_metadata.task_type,
    )


def get_model_metadata(
    model_id: str,
    api_key: Optional[str],
    max_pages: int = MAX_MODEL_PACKAGE_PAGES,
) -> RoboflowModelMetadata:
    if api_key is None:
        api_key = ROBOFLOW_API_KEY
    fetched_pages = []
    start_after = None
    while len(fetched_pages) < max_pages:
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
            message=f"Could not retrieve model {model_id} from Roboflow API. Backend provided empty list of model "
            f"packages `inference-exp` library could load. Contact Roboflow to solve the problem.",
            help_url="https://todo",
        )
    fetched_pages[-1].model_packages = all_model_packages
    return fetched_pages[-1]


@backoff.on_exception(
    backoff.expo,
    exception=RetryError,
    max_tries=API_CALLS_MAX_TRIES,
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
        raise RetryError(
            message=f"Connectivity error",
            help_url="https://todo",
        )
    handle_response_errors(response=response, operation_name="get model weights")
    try:
        return RoboflowModelMetadata.model_validate(response.json()["modelMetadata"])
    except (ValueError, ValidationError, KeyError) as error:
        # TODO: either handle here or fix API, which return 200 with content {error: "endpoint not found"} id endpoint isnt available
        raise ModelRetrievalError(
            message=f"Could not decode Roboflow API response when trying to retrieve model {model_id}. If that problem "
            f"is not ephemeral - contact Roboflow.",
            help_url="https://todo",
        ) from error


def handle_response_errors(response: Response, operation_name: str) -> None:
    if response.status_code == 401 or response.status_code == 403:
        raise UnauthorizedModelAccessError(
            message=f"Could not {operation_name}. Request unauthorised. Are you sure you use valid Roboflow API key? "
            "See details here: https://docs.roboflow.com/api-reference/authentication and "
            "export key to `ROBOFLOW_API_KEY` environment variable",
            help_url="https://todo",
        )
    if response.status_code in IDEMPOTENT_API_REQUEST_CODES_TO_RETRY:
        raise RetryError(
            message=f"Roboflow API returned invalid response code for {operation_name} operation "
            f"{response.status_code}. If that problem is not ephemeral - contact Roboflow.",
            help_url="https://todo",
        )
    if response.status_code >= 400:
        response_payload = get_error_response_payload(response=response)
        raise ModelRetrievalError(
            message=f"Roboflow API returned invalid response code for {operation_name} operation "
            f"{response.status_code}.\n\nResponse:\n{response_payload}",
            help_url="https://todo",
        )


def get_error_response_payload(response: Response) -> str:
    try:
        return json.dumps(response.json(), indent=4)
    except ValueError:
        return response.text


def parse_model_package_metadata(
    metadata: Union[RoboflowModelPackageV1, dict],
) -> Optional[ModelPackageMetadata]:
    if isinstance(metadata, dict):
        metadata_type = metadata.get("type", "unknown")
        model_package_id = metadata.get("packageId", "unknown")
        LOGGER.warning(
            "Roboflow API returned entity describing model package which cannot be parsed. This may indicate that "
            f"your `inference-exp` package is outdated. "
            f"Debug info - entity type: `{metadata_type}`, model package id: {model_package_id}"
        )
        return None
    manifest_type = metadata.package_manifest.get("type", "unknown")
    if manifest_type in MODEL_PACKAGES_TO_IGNORE:
        LOGGER.debug(
            "Ignoring model package with manifest incompatible with inference."
            f"Debug info - model package id: {metadata.package_id}, manifest type: {manifest_type}."
        )
        return None
    if manifest_type not in MODEL_PACKAGE_PARSERS:
        LOGGER.warning(
            "Roboflow API returned entity describing model package which cannot be parsed. This may indicate that "
            f"your `inference-exp` package is outdated. "
            f"Debug info - package manifest type: `{manifest_type}`."
        )
        return None
    try:
        return MODEL_PACKAGE_PARSERS[manifest_type](metadata)
    except BaseInferenceError as error:
        raise error
    except Exception as error:
        raise ModelMetadataConsistencyError(
            message="Roboflow API returned model package metadata which cannot be parsed. Contact Roboflow to "
            f"solve the problem. Error details: {error}. Error type: {error.__class__.__name__}",
            help_url="https://todo",
        ) from error


class OnnxModelPackageV1(BaseModel):
    type: Literal["onnx-model-package-v1"]
    backend_type: Literal["onnx"] = Field(alias="backendType")
    dynamic_batch_size: bool = Field(alias="dynamicBatchSize", default=False)
    static_batch_size: Optional[int] = Field(alias="staticBatchSize", default=None)
    quantization: Quantization
    opset: int
    incompatible_providers: Optional[List[str]] = Field(
        alias="incompatibleProviders", default=None
    )


def parse_onnx_model_package(metadata: RoboflowModelPackageV1) -> ModelPackageMetadata:
    parsed_manifest = OnnxModelPackageV1.model_validate(metadata.package_manifest)
    validate_batch_settings(
        dynamic_batch_size=parsed_manifest.dynamic_batch_size,
        static_batch_size=parsed_manifest.static_batch_size,
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
        trusted_source=metadata.trusted_source,
        model_features=metadata.model_features,
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
    trt_forward_compatible: bool = Field(alias="trtForwardCompatible", default=False)
    trt_lean_runtime_excluded: bool = Field(
        alias="trtLeanRuntimeExcluded", default=False
    )
    machine_type: Literal["gpu-server", "jetson"] = Field(alias="machineType")
    machine_specs: Annotated[
        Union[JetsonMachineSpecsV1, GPUServerSpecsV1],
        Discriminator(discriminator="type"),
    ] = Field(alias="machineSpecs")


def parse_trt_model_package(metadata: RoboflowModelPackageV1) -> ModelPackageMetadata:
    parsed_manifest = TrtModelPackageV1.model_validate(metadata.package_manifest)
    validate_batch_settings(
        dynamic_batch_size=parsed_manifest.dynamic_batch_size,
        static_batch_size=parsed_manifest.static_batch_size,
    )
    if parsed_manifest.dynamic_batch_size is True and any(
        e is None
        for e in [
            parsed_manifest.min_batch_size,
            parsed_manifest.opt_batch_size,
            parsed_manifest.max_batch_size,
        ]
    ):
        raise ModelMetadataConsistencyError(
            message="While downloading model weights, Roboflow API provided inconsistent metadata "
            "describing model package - TRT package declared support for dynamic batch size, but did not "
            "specify min / opt / max batch size supported which is required.",
            help_url="https://todo",
        )
    if parsed_manifest.machine_type == "gpu-server":
        if not isinstance(parsed_manifest.machine_specs, GPUServerSpecsV1):
            raise ModelMetadataConsistencyError(
                message="While downloading model weights, Roboflow API provided inconsistent metadata "
                "describing model package - expected GPU Server specification for TRT model package registered as "
                "compiled on gpu-server. Contact Roboflow to solve the problem.",
                help_url="https://todo",
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
                message="While downloading model weights, Roboflow API provided inconsistent metadata "
                "describing model package - expected Jetson Device specification for TRT model package registered as "
                "compiled on Jetson. Contact Roboflow to solve the problem.",
                help_url="https://todo",
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
            message="While downloading model weights, Roboflow API provided metadata which are not handled by current version "
            "of inference detected while parsing TRT model package. This problem may indicate that your inference "
            "package is outdated. Try to upgrade - if that does not help, contact Roboflow to solve the problem.",
            help_url="https://todo",
        )
    package_artefacts = parse_package_artefacts(
        package_artefacts=metadata.package_files
    )
    trt_package_details = TRTPackageDetails(
        min_dynamic_batch_size=parsed_manifest.min_batch_size,
        opt_dynamic_batch_size=parsed_manifest.opt_batch_size,
        max_dynamic_batch_size=parsed_manifest.max_batch_size,
        same_cc_compatible=parsed_manifest.same_cc_compatible,
        trt_forward_compatible=parsed_manifest.trt_forward_compatible,
        trt_lean_runtime_excluded=parsed_manifest.trt_lean_runtime_excluded,
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
        trusted_source=metadata.trusted_source,
        model_features=metadata.model_features,
    )


class TorchModelPackageV1(BaseModel):
    type: Literal["torch-model-package-v1"]
    backend_type: Literal["torch"] = Field(alias="backendType")
    dynamic_batch_size: bool = Field(alias="dynamicBatchSize", default=False)
    static_batch_size: Optional[int] = Field(alias="staticBatchSize", default=None)
    quantization: Quantization


def parse_torch_model_package(metadata: RoboflowModelPackageV1) -> ModelPackageMetadata:
    parsed_manifest = TorchModelPackageV1.model_validate(metadata.package_manifest)
    validate_batch_settings(
        dynamic_batch_size=parsed_manifest.dynamic_batch_size,
        static_batch_size=parsed_manifest.static_batch_size,
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
        trusted_source=metadata.trusted_source,
        model_features=metadata.model_features,
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
        trusted_source=metadata.trusted_source,
        model_features=metadata.model_features,
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
        trusted_source=metadata.trusted_source,
        model_features=metadata.model_features,
    )


class TorchScriptModelPackageV1(BaseModel):
    type: Literal["torch-script-model-package-v1"]
    backend_type: Literal["torch-script"] = Field(alias="backendType")
    dynamic_batch_size: bool = Field(alias="dynamicBatchSize", default=False)
    static_batch_size: Optional[int] = Field(alias="staticBatchSize", default=None)
    quantization: Quantization
    supported_device_types: List[str] = Field(alias="supportedDeviceTypes")
    torch_version: str = Field(alias="torchVersion")
    torch_vision_version: Optional[str] = Field(
        alias="torchVisionVersion", default=None
    )


def parse_torch_script_model_package(
    metadata: RoboflowModelPackageV1,
) -> ModelPackageMetadata:
    parsed_manifest = TorchScriptModelPackageV1.model_validate(
        metadata.package_manifest
    )
    validate_batch_settings(
        dynamic_batch_size=parsed_manifest.dynamic_batch_size,
        static_batch_size=parsed_manifest.static_batch_size,
    )
    package_artefacts = parse_package_artefacts(
        package_artefacts=metadata.package_files
    )
    torch_vision_version = None
    if parsed_manifest.torch_vision_version is not None:
        torch_vision_version = as_version(parsed_manifest.torch_vision_version)
    torch_script_package_details = TorchScriptPackageDetails(
        supported_device_types=set(parsed_manifest.supported_device_types),
        torch_version=as_version(parsed_manifest.torch_version),
        torch_vision_version=torch_vision_version,
    )
    return ModelPackageMetadata(
        package_id=metadata.package_id,
        backend=BackendType.TORCH_SCRIPT,
        dynamic_batch_size_supported=parsed_manifest.dynamic_batch_size,
        static_batch_size=parsed_manifest.static_batch_size,
        package_artefacts=package_artefacts,
        quantization=parsed_manifest.quantization,
        trusted_source=metadata.trusted_source,
        model_features=metadata.model_features,
        torch_script_package_details=torch_script_package_details,
    )


def validate_batch_settings(
    dynamic_batch_size: bool, static_batch_size: Optional[int]
) -> None:
    if not dynamic_batch_size and (static_batch_size is None or static_batch_size <= 0):
        raise ModelMetadataConsistencyError(
            message="While downloading model weights, Roboflow API provided inconsistent metadata "
            "describing model package - model package declared not to support dynamic batch size and "
            "supported static batch size not provided. Contact Roboflow to solve the problem.",
            help_url="https://todo",
        )
    if dynamic_batch_size and static_batch_size is not None:
        raise ModelMetadataConsistencyError(
            message="While downloading model weights, Roboflow API provided inconsistent metadata "
            "describing model package - model package declared not to support dynamic batch size and "
            "supported static batch size not provided. Contact Roboflow to solve the problem.",
            help_url="https://todo",
        )


def parse_package_artefacts(
    package_artefacts: List[RoboflowModelPackageFile],
) -> List[FileDownloadSpecs]:
    return [
        FileDownloadSpecs(
            download_url=f.download_url, file_handle=f.file_handle, md5_hash=f.md5_hash
        )
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
    "torch-script-model-package-v1": parse_torch_script_model_package,
}


def as_version(value: str) -> Version:
    try:
        return Version(value)
    except InvalidVersion as error:
        raise ModelMetadataConsistencyError(
            message="Roboflow API returned model package manifest that is expected to provide valid version specification for "
            "one of the field of package manifest, but instead provides value that cannot be parsed. This is most "
            "likely Roboflow API bug - contact Roboflow to solve the problem.",
            help_url="https://todo",
        ) from error
