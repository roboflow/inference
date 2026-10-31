"""TRT model-package manifest schema.

Shared by the Roboflow weights provider and local TRT discovery. Kept free of
provider/auto-loader imports so both can depend on it without cycles.
"""

from typing import Annotated, Literal, Optional, Union

from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, Discriminator, Field

from inference_models.errors import ModelMetadataConsistencyError
from inference_models.weights_providers.entities import Quantization


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


def as_version(value: str) -> Version:
    try:
        return Version(value)
    except InvalidVersion as error:
        raise ModelMetadataConsistencyError(
            message="Roboflow API returned model package manifest that is expected to provide valid version specification for "
            "one of the field of package manifest, but instead provides value that cannot be parsed. This is most "
            "likely Roboflow API bug - contact Roboflow to solve the problem.",
            help_url="https://inference-models.roboflow.com/errors/model-retrieval/#modelmetadataconsistencyerror",
        ) from error
