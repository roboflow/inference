from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Union

from packaging.version import Version


class BackendType(str, Enum):
    TORCH = "torch"
    ONNX = "onnx"
    TRT = "trt"
    HF = "hugging-face"
    ULTRALYTICS = "ultralytics"
    CUSTOM = "custom"


class Quantization(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FileDownloadSpecs:
    download_url: str
    file_name: str


@dataclass(frozen=True)
class ServerEnvironmentRequirements:
    gpu_type: str
    driver_version: Version
    cuda_version: Version
    trt_version: Optional[Version]
    os_version: Optional[str]

    def __str__(self) -> str:
        return (
            f"gpu={self.gpu_type} driver={self.driver_version} trt={self.trt_version}"
            f"cu={self.cuda_version} os={self.os_version}"
        )


@dataclass(frozen=True)
class JetsonEnvironmentRequirements:
    jetson_type: str
    jetpack_version: str
    cuda_version: Version
    trt_version: Optional[Version]

    def __str__(self) -> str:
        return (
            f"jetson={self.jetson_type} jetpack={self.jetpack_version} trt={self.trt_version}"
            f"cu={self.cuda_version}"
        )


@dataclass(frozen=True)
class ModelPackageMetadata:
    package_id: str
    backend: BackendType
    package_artefacts: List[FileDownloadSpecs]
    quantization: Optional[Quantization] = field(default=None)
    dynamic_batch_size_supported: Optional[bool] = field(default=None)
    static_batch_size: Optional[int] = field(default=None)
    min_dynamic_batch_size: Optional[int] = field(default=None)
    opt_dynamic_batch_size: Optional[int] = field(default=None)
    max_dynamic_batch_size: Optional[int] = field(default=None)
    environment_requirements: Optional[
        Union[ServerEnvironmentRequirements, JetsonEnvironmentRequirements]
    ] = field(default=None)

    def get_summary(self) -> str:
        return (
            f"package_id={self.package_id}, backend={self.backend.value} quantization={self.quantization} "
            f"dynamic_batch_size_supported={self.dynamic_batch_size_supported} "
            f"static_batch_size={self.static_batch_size} dynamic_batch_size[min/opt/max]={self.min_dynamic_batch_size}/"
            f"{self.opt_dynamic_batch_size}/{self.max_dynamic_batch_size} "
            f"environment_requirements: {self.environment_requirements}"
        )

    def get_dynamic_batch_boundaries(self) -> Tuple[int, int]:
        if not self.specifies_dynamic_batch_boundaries():
            raise RuntimeError(
                "Requested dynamic batch boundaries from model package that does not support dynamic batches."
            )
        values = []
        if self.min_dynamic_batch_size is not None:
            values.append(self.min_dynamic_batch_size)
        if self.opt_dynamic_batch_size is not None:
            values.append(self.opt_dynamic_batch_size)
        if self.max_dynamic_batch_size is not None:
            values.append(self.max_dynamic_batch_size)
        return min(values), max(values)

    def specifies_dynamic_batch_boundaries(self) -> bool:
        if not self.dynamic_batch_size_supported:
            return False
        return (
            self.min_dynamic_batch_size is not None
            or self.opt_dynamic_batch_size is not None
            or self.max_dynamic_batch_size is not None
        )


@dataclass(frozen=True)
class ModelMetadata:
    model_id: str
    model_architecture: str
    model_packages: List[ModelPackageMetadata]
    task_type: Optional[str] = field(default=None)
