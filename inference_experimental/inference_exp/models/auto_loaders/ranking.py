from typing import List, Optional

import torch
from inference_exp.models.auto_loaders.utils import (
    filter_available_devices_with_selected_device,
)
from inference_exp.runtime_introspection.core import x_ray_runtime_environment
from inference_exp.weights_providers.entities import (
    BackendType,
    JetsonEnvironmentRequirements,
    ModelPackageMetadata,
    Quantization,
    ServerEnvironmentRequirements,
)

BACKEND_PRIORITY = {
    BackendType.TRT: 5,
    BackendType.TORCH: 4,
    BackendType.HF: 3,
    BackendType.ONNX: 2,
    BackendType.ULTRALYTICS: 1,
}
QUANTIZATION_PRIORITY = {
    Quantization.INT8: 4,
    Quantization.FP16: 3,
    Quantization.FP32: 2,
    Quantization.UNKNOWN: 1,
}
DYNAMIC_BATCH_SIZE_KEY = "dynamic"
STATIC_BATCH_SIZE_KEY = "static"
BATCH_SIZE_PRIORITY = {
    DYNAMIC_BATCH_SIZE_KEY: 2,
    STATIC_BATCH_SIZE_KEY: 1,
}


def rank_model_packages(
    model_packages: List[ModelPackageMetadata],
) -> List[ModelPackageMetadata]:
    sorting_features = []
    for model_package in model_packages:
        batch_mode = (
            DYNAMIC_BATCH_SIZE_KEY
            if model_package.dynamic_batch_size_supported
            else STATIC_BATCH_SIZE_KEY
        )
        sorting_features.append(
            (
                BACKEND_PRIORITY.get(model_package.backend, 0),
                QUANTIZATION_PRIORITY.get(model_package.quantization, 0),
                BATCH_SIZE_PRIORITY[batch_mode],
                retrieve_onnx_opset(model_package),  # the higher opset, the better
                retrieve_cuda_device_match(
                    model_package
                ),  # we like more direct matches
                model_package,
            )
        )
    sorted_features = sorted(sorting_features, key=lambda x: x[:5], reverse=True)
    return [f[-1] for f in sorted_features]


def retrieve_onnx_opset(model_package: ModelPackageMetadata) -> int:
    if model_package.onnx_package_details is None:
        return -1
    return model_package.onnx_package_details.opset


def retrieve_cuda_device_match(
    model_package: ModelPackageMetadata,
    selected_device: Optional[torch.device] = None,
) -> int:
    if model_package.environment_requirements is None:
        return 0
    runtime_x_ray = x_ray_runtime_environment()
    all_available_cuda_devices, _ = filter_available_devices_with_selected_device(
        selected_device=selected_device,
        all_available_cuda_devices=runtime_x_ray.gpu_devices,
        all_available_devices_cc=runtime_x_ray.gpu_devices_cc,
    )
    if not isinstance(
        model_package.environment_requirements,
        (JetsonEnvironmentRequirements, ServerEnvironmentRequirements),
    ):
        return 0
    compilation_device = model_package.environment_requirements.cuda_device_name
    return int(any(dev == compilation_device for dev in all_available_cuda_devices))
