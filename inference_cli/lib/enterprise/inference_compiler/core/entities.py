from enum import Enum
from typing import Callable, Literal, Optional

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
    machine_specs: GPUServerSpecsV1 = Field(alias="machineSpecs")

    class Config:
        populate_by_name = True


class TRTConfig(BaseModel):
    static_batch_size: Optional[int] = Field(default=None)
    dynamic_batch_size_min: Optional[int] = Field(default=None)
    dynamic_batch_size_opt: Optional[int] = Field(default=None)
    dynamic_batch_size_max: Optional[int] = Field(default=None)
