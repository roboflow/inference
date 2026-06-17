from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from inference_cli.lib.enterprise.inference_compiler.constants import (
    DEEP_LAB_MODELS_MAX_DYNAMIC_BATCH_SIZE,
    DEEP_LAB_MODELS_MIN_DYNAMIC_BATCH_SIZE,
    DEEP_LAB_MODELS_OPT_DYNAMIC_BATCH_SIZE,
    DEEP_LAB_MODELS_WORKSPACE_SIZE,
    RESNET_MODELS_MAX_DYNAMIC_BATCH_SIZE,
    RESNET_MODELS_MIN_DYNAMIC_BATCH_SIZE,
    RESNET_MODELS_OPT_DYNAMIC_BATCH_SIZE,
    RESNET_MODELS_WORKSPACE_SIZE,
    RFDETR_MODELS_MAX_DYNAMIC_BATCH_SIZE,
    RFDETR_MODELS_MIN_DYNAMIC_BATCH_SIZE,
    RFDETR_MODELS_OPT_DYNAMIC_BATCH_SIZE,
    RFDETR_MODELS_WORKSPACE_SIZE,
    VIT_MODELS_MAX_DYNAMIC_BATCH_SIZE,
    VIT_MODELS_MIN_DYNAMIC_BATCH_SIZE,
    VIT_MODELS_OPT_DYNAMIC_BATCH_SIZE,
    VIT_MODELS_WORKSPACE_SIZE,
    YOLO_MODELS_MAX_DYNAMIC_BATCH_SIZE,
    YOLO_MODELS_MIN_DYNAMIC_BATCH_SIZE,
    YOLO_MODELS_OPT_DYNAMIC_BATCH_SIZE,
    YOLO_MODELS_WORKSPACE_SIZE,
)


class CompilationConfig(BaseModel):
    workspace_size_gb: int
    min_batch_size: int
    opt_batch_size: int
    max_batch_size: int
    verify_model: Optional[Callable[[str], None]] = Field(default=None)

    @classmethod
    def for_yolo_models(
        cls, verify_model: Optional[Callable[[str], None]] = None
    ) -> "CompilationConfig":
        return cls(
            workspace_size_gb=YOLO_MODELS_WORKSPACE_SIZE,
            min_batch_size=YOLO_MODELS_MIN_DYNAMIC_BATCH_SIZE,
            opt_batch_size=YOLO_MODELS_OPT_DYNAMIC_BATCH_SIZE,
            max_batch_size=YOLO_MODELS_MAX_DYNAMIC_BATCH_SIZE,
            verify_model=verify_model,
        )

    @classmethod
    def for_rfdetr_models(
        cls, verify_model: Optional[Callable[[str], None]] = None
    ) -> "CompilationConfig":
        return cls(
            workspace_size_gb=RFDETR_MODELS_WORKSPACE_SIZE,
            min_batch_size=RFDETR_MODELS_MIN_DYNAMIC_BATCH_SIZE,
            opt_batch_size=RFDETR_MODELS_OPT_DYNAMIC_BATCH_SIZE,
            max_batch_size=RFDETR_MODELS_MAX_DYNAMIC_BATCH_SIZE,
            verify_model=verify_model,
        )

    @classmethod
    def for_resnet_models(
        cls, verify_model: Optional[Callable[[str], None]] = None
    ) -> "CompilationConfig":
        return cls(
            workspace_size_gb=RESNET_MODELS_WORKSPACE_SIZE,
            min_batch_size=RESNET_MODELS_MIN_DYNAMIC_BATCH_SIZE,
            opt_batch_size=RESNET_MODELS_OPT_DYNAMIC_BATCH_SIZE,
            max_batch_size=RESNET_MODELS_MAX_DYNAMIC_BATCH_SIZE,
            verify_model=verify_model,
        )

    @classmethod
    def for_vit_models(
        cls, verify_model: Optional[Callable[[str], None]] = None
    ) -> "CompilationConfig":
        return cls(
            workspace_size_gb=VIT_MODELS_WORKSPACE_SIZE,
            min_batch_size=VIT_MODELS_MIN_DYNAMIC_BATCH_SIZE,
            opt_batch_size=VIT_MODELS_OPT_DYNAMIC_BATCH_SIZE,
            max_batch_size=VIT_MODELS_MAX_DYNAMIC_BATCH_SIZE,
            verify_model=verify_model,
        )

    @classmethod
    def for_deep_lab_models(
        cls, verify_model: Optional[Callable[[str], None]] = None
    ) -> "CompilationConfig":
        return cls(
            workspace_size_gb=DEEP_LAB_MODELS_WORKSPACE_SIZE,
            min_batch_size=DEEP_LAB_MODELS_MIN_DYNAMIC_BATCH_SIZE,
            opt_batch_size=DEEP_LAB_MODELS_OPT_DYNAMIC_BATCH_SIZE,
            max_batch_size=DEEP_LAB_MODELS_MAX_DYNAMIC_BATCH_SIZE,
            verify_model=verify_model,
        )


class GPUServerSpecsV1(BaseModel):
    type: Literal["gpu-server-specs-v1"] = Field(default="gpu-server-specs-v1")
    driver_version: str = Field(alias="driverVersion")
    os_version: str = Field(alias="osVersion")

    class Config:
        populate_by_name = True


class JetsonMachineSpecsV1(BaseModel):
    type: Literal["jetson-machine-specs-v1"] = Field(default="jetson-machine-specs-v1")
    l4t_version: str = Field(alias="l4tVersion")
    device_name: str = Field(alias="deviceName")
    driver_version: str = Field(alias="driverVersion")

    class Config:
        populate_by_name = True


class Quantization(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    UNKNOWN = "unknown"


class TRTMachineType(str, Enum):
    GPU_SERVER = "gpu-server"
    JETSON = "jetson"


class TRTModelPackageV1(BaseModel):
    type: Literal["trt-model-package-v1"] = Field(default="trt-model-package-v1")
    backend_type: Literal["trt"] = Field(default="trt", alias="backendType")
    dynamic_batch_size: bool = Field(alias="dynamicBatchSize")
    static_batch_size: Optional[int] = Field(alias="staticBatchSize", default=None)
    min_batch_size: Optional[int] = Field(alias="minBatchSize", default=None)
    opt_batch_size: Optional[int] = Field(alias="optBatchSize", default=None)
    max_batch_size: Optional[int] = Field(alias="maxBatchSize", default=None)
    quantization: Quantization
    cuda_device_type: str = Field(alias="cudaDeviceType")
    cuda_device_cc: str = Field(alias="cudaDeviceCC")
    cuda_version: str = Field(alias="cudaVersion")
    trt_version: str = Field(alias="trtVersion")
    same_cc_compatible: Optional[bool] = Field(alias="sameCCCompatible", default=None)
    trt_forward_compatible: Optional[bool] = Field(
        alias="trtForwardCompatible", default=None
    )
    trt_lean_runtime_excluded: Optional[bool] = Field(
        alias="trtLeanRuntimeExcluded", default=False
    )
    machine_type: TRTMachineType = Field(
        alias="machineType", default=TRTMachineType.GPU_SERVER
    )
    machine_specs: Union[GPUServerSpecsV1, JetsonMachineSpecsV1] = Field(
        alias="machineSpecs", discriminator="type"
    )

    class Config:
        populate_by_name = True


class TRTConfig(BaseModel):
    static_batch_size: Optional[int] = Field(default=None)
    dynamic_batch_size_min: Optional[int] = Field(default=None)
    dynamic_batch_size_opt: Optional[int] = Field(default=None)
    dynamic_batch_size_max: Optional[int] = Field(default=None)


class PlatformRegistrationPolicy(str, Enum):
    """Whether platform register/upload must succeed for the pipeline to fail."""

    REQUIRED = "required"
    OPTIONAL = "optional"


@dataclass
class CompilationVariantOutcome:
    precision: str
    dynamic_batch: bool
    compiled: bool = False
    installed_local: bool = False
    local_package_id: Optional[str] = None
    registered_platform: bool = False
    uploaded_sealed: bool = False
    compile_error: Optional[str] = None
    register_error: Optional[str] = None
    backend: str = "onnx_cuda"
    reason: str = ""


class CompilationPipelineResult:
    """Outcome of compile → install → register/upload."""

    def __init__(
        self,
        model_id: str,
        model_architecture: str,
        *,
        compiled: bool = False,
        installed_local: bool = False,
        local_package_id: Optional[str] = None,
        local_install_path: Optional[str] = None,
        registered_platform: bool = False,
        uploaded_sealed: bool = False,
        compile_error: Optional[str] = None,
        register_error: Optional[str] = None,
        backend: str = "onnx_cuda",
        reason: str = "",
        variant_outcomes: Optional[List[CompilationVariantOutcome]] = None,
    ):
        self.model_id = model_id
        self.model_architecture = model_architecture
        self.compiled = compiled
        self.installed_local = installed_local
        self.local_package_id = local_package_id
        self.local_install_path = local_install_path
        self.registered_platform = registered_platform
        self.uploaded_sealed = uploaded_sealed
        self.compile_error = compile_error
        self.register_error = register_error
        self.backend = backend
        self.reason = reason
        self.variant_outcomes = variant_outcomes or []

    def as_log_metadata(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_architecture": self.model_architecture,
            "compiled": self.compiled,
            "installed_local": self.installed_local,
            "local_package_id": self.local_package_id,
            "local_install_path": self.local_install_path,
            "registered_platform": self.registered_platform,
            "uploaded_sealed": self.uploaded_sealed,
            "compile_error": self.compile_error,
            "register_error": self.register_error,
            "backend": self.backend,
            "reason": self.reason,
            "variant_outcomes": [
                {
                    "precision": variant.precision,
                    "dynamic_batch": variant.dynamic_batch,
                    "compiled": variant.compiled,
                    "installed_local": variant.installed_local,
                    "local_package_id": variant.local_package_id,
                    "registered_platform": variant.registered_platform,
                    "uploaded_sealed": variant.uploaded_sealed,
                    "compile_error": variant.compile_error,
                    "register_error": variant.register_error,
                    "backend": variant.backend,
                    "reason": variant.reason,
                }
                for variant in self.variant_outcomes
            ],
        }


def aggregate_compilation_variant_outcomes(
    model_id: str,
    model_architecture: str,
    variant_outcomes: List[CompilationVariantOutcome],
) -> CompilationPipelineResult:
    preferred = next(
        (
            variant
            for variant in reversed(variant_outcomes)
            if variant.installed_local and variant.backend == "trt"
        ),
        None,
    )
    if preferred is None:
        preferred = variant_outcomes[-1] if variant_outcomes else None
    if preferred is None:
        return CompilationPipelineResult(
            model_id=model_id,
            model_architecture=model_architecture,
            reason="no compilation variants attempted",
            variant_outcomes=variant_outcomes,
        )
    return CompilationPipelineResult(
        model_id=model_id,
        model_architecture=model_architecture,
        compiled=preferred.compiled,
        installed_local=preferred.installed_local,
        local_package_id=preferred.local_package_id,
        registered_platform=preferred.registered_platform,
        uploaded_sealed=preferred.uploaded_sealed,
        compile_error=preferred.compile_error,
        register_error=preferred.register_error,
        backend=preferred.backend,
        reason=preferred.reason,
        variant_outcomes=variant_outcomes,
    )
