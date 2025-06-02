from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union

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
    CUSTOM = "custom"


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


@dataclass(frozen=True)
class JetsonEnvironmentRequirements:
    jetson_type: str
    jetpack_version: str
    cuda_version: Version
    trt_version: Optional[Version]


@dataclass(frozen=True)
class ModelPackageMetadata:
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


@dataclass(frozen=True)
class ModelMetadata:
    model_id: str
    model_architecture: str
    model_packages: List[ModelPackageMetadata]
    task_type: Optional[str] = field(default=None)
