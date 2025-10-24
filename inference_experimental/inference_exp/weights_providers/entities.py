from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set, Tuple, Union

from packaging.version import Version


class BackendType(str, Enum):
    TORCH = "torch"
    TORCH_SCRIPT = "torch-script"
    ONNX = "onnx"
    TRT = "trt"
    HF = "hugging-face"
    ULTRALYTICS = "ultralytics"
    CUSTOM = "custom"


class Quantization(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FileDownloadSpecs:
    download_url: str
    file_handle: str
    md5_hash: Optional[str] = field(default=None)


@dataclass(frozen=True)
class ServerEnvironmentRequirements:
    cuda_device_cc: Version
    cuda_device_name: str
    driver_version: Optional[Version]
    cuda_version: Optional[Version]
    trt_version: Optional[Version]
    os_version: Optional[str]

    def __str__(self) -> str:
        return (
            f"Server(device={self.cuda_device_name}, cc={self.cuda_device_cc}, "
            f"os={self.os_version}, driver={self.driver_version}, cu={self.cuda_version}, "
            f"trt={self.trt_version})"
        )


@dataclass(frozen=True)
class JetsonEnvironmentRequirements:
    cuda_device_cc: Version
    cuda_device_name: str
    l4t_version: Version
    jetson_product_name: Optional[str]
    cuda_version: Optional[Version]
    trt_version: Optional[Version]
    driver_version: Optional[Version]

    def __str__(self) -> str:
        return (
            f"Jetson(device={self.cuda_device_name}, cc={self.cuda_device_cc}, "
            f"l4t={self.l4t_version}, product={self.jetson_product_name}, cu={self.cuda_version}, "
            f"trt={self.trt_version}, driver={self.driver_version})"
        )


@dataclass(frozen=True)
class TRTPackageDetails:
    min_dynamic_batch_size: Optional[int] = field(default=None)
    opt_dynamic_batch_size: Optional[int] = field(default=None)
    max_dynamic_batch_size: Optional[int] = field(default=None)
    same_cc_compatible: bool = field(default=False)
    trt_forward_compatible: bool = field(default=False)
    trt_lean_runtime_excluded: bool = field(default=False)

    def __str__(self):
        return (
            f"TRTPackageDetails("
            f"dynamic_batch=({self.min_dynamic_batch_size}/{self.opt_dynamic_batch_size}/{self.max_dynamic_batch_size}), "
            f"same_cc_compatible={self.same_cc_compatible}, "
            f"trt_forward_compatible={self.trt_forward_compatible}, "
            f"trt_lean_runtime_excluded={self.trt_lean_runtime_excluded})"
        )


@dataclass(frozen=True)
class ONNXPackageDetails:
    opset: int
    incompatible_providers: Optional[List[str]] = field(default=None)


@dataclass(frozen=True)
class TorchScriptPackageDetails:
    supported_device_types: Set[str]
    torch_version: Version
    torch_vision_version: Optional[Version] = field(default=None)


@dataclass(frozen=True)
class ModelPackageMetadata:
    package_id: str
    backend: BackendType
    package_artefacts: List[FileDownloadSpecs]
    quantization: Optional[Quantization] = field(default=None)
    dynamic_batch_size_supported: Optional[bool] = field(default=None)
    static_batch_size: Optional[int] = field(default=None)
    trt_package_details: Optional[TRTPackageDetails] = field(default=None)
    onnx_package_details: Optional[ONNXPackageDetails] = field(default=None)
    torch_script_package_details: Optional[TorchScriptPackageDetails] = field(
        default=None
    )
    trusted_source: bool = field(default=False)
    environment_requirements: Optional[
        Union[ServerEnvironmentRequirements, JetsonEnvironmentRequirements]
    ] = field(default=None)
    model_features: Optional[dict] = field(default=None)

    def get_summary(self) -> str:
        return (
            f"ModelPackageMetadata(package_id={self.package_id}, backend={self.backend.value}, quantization={self.quantization} "
            f"dynamic_batch_size_supported={self.dynamic_batch_size_supported}, "
            f"static_batch_size={self.static_batch_size}, trt_package_details={self.trt_package_details}, "
            f"environment_requirements={self.environment_requirements}, model_features={self.model_features})"
        )

    def get_dynamic_batch_boundaries(self) -> Tuple[int, int]:
        if not self.specifies_dynamic_batch_boundaries():
            raise RuntimeError(
                "Requested dynamic batch boundaries from model package that does not support dynamic batches."
            )
        values = []
        if self.trt_package_details.min_dynamic_batch_size is not None:
            values.append(self.trt_package_details.min_dynamic_batch_size)
        if self.trt_package_details.opt_dynamic_batch_size is not None:
            values.append(self.trt_package_details.opt_dynamic_batch_size)
        if self.trt_package_details.max_dynamic_batch_size is not None:
            values.append(self.trt_package_details.max_dynamic_batch_size)
        return min(values), max(values)

    def specifies_dynamic_batch_boundaries(self) -> bool:
        if not self.dynamic_batch_size_supported:
            return False
        if self.trt_package_details is None:
            return False
        return (
            self.trt_package_details.min_dynamic_batch_size is not None
            or self.trt_package_details.opt_dynamic_batch_size is not None
            or self.trt_package_details.max_dynamic_batch_size is not None
        )


@dataclass(frozen=True)
class ModelMetadata:
    model_id: str
    model_architecture: str
    model_packages: List[ModelPackageMetadata]
    task_type: Optional[str] = field(default=None)
    model_variant: Optional[str] = field(default=None)
