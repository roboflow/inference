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
    # ordering TRT and Cu versions from older to newest -
    # with the assumption that incompatible versions are eliminated earlier, and
    # ranking implicitly attempts to match version closes to the current -
    # in come cases we would rank high versions below current one, but for that
    # it is assumed such versions are compatible, otherwise should be
    # discarded in the previous stage.
    cuda_ranking = rank_cuda_versions(model_packages=model_packages)
    trt_ranking = rank_trt_versions(model_packages=model_packages)
    for model_package, cu_rank, trt_rank in zip(
        model_packages, cuda_ranking, trt_ranking
    ):
        batch_mode = (
            DYNAMIC_BATCH_SIZE_KEY
            if model_package.dynamic_batch_size_supported
            else STATIC_BATCH_SIZE_KEY
        )
        static_batch_size_score = (
            0
            if model_package.static_batch_size is None
            else -1 * model_package.static_batch_size
        )
        sorting_features.append(
            (
                BACKEND_PRIORITY.get(model_package.backend, 0),
                QUANTIZATION_PRIORITY.get(model_package.quantization, 0),
                BATCH_SIZE_PRIORITY[batch_mode],
                static_batch_size_score,  # the bigger statis batch size, the worse - requires padding
                retrieve_onnx_opset(model_package),  # the higher opset, the better
                retrieve_trt_forward_compatible_match(
                    model_package
                ),  # exact matches first
                retrieve_same_trt_cc_compatibility(model_package),
                retrieve_cuda_device_match(
                    model_package
                ),  # we like more direct matches
                cu_rank,
                trt_rank,
                model_package,
            )
        )
    sorted_features = sorted(sorting_features, key=lambda x: x[:10], reverse=True)
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


def retrieve_same_trt_cc_compatibility(model_package: ModelPackageMetadata) -> int:
    if model_package.trt_package_details is None:
        return 1
    return int(not model_package.trt_package_details.same_cc_compatible)


def retrieve_trt_forward_compatible_match(model_package: ModelPackageMetadata) -> int:
    if model_package.trt_package_details is None:
        return 1
    return int(not model_package.trt_package_details.trt_forward_compatible)


def rank_cuda_versions(model_packages: List[ModelPackageMetadata]) -> List[int]:
    cuda_versions = []
    package_id_to_cuda_version = {}
    last_ranking = -len(model_packages)
    for package in model_packages:
        if isinstance(package.environment_requirements, ServerEnvironmentRequirements):
            cuda_versions.append(package.environment_requirements.cuda_version)
            package_id_to_cuda_version[package.package_id] = (
                package.environment_requirements.cuda_version
            )
        elif isinstance(
            package.environment_requirements, ServerEnvironmentRequirements
        ):
            cuda_versions.append(package.environment_requirements.cuda_version)
            package_id_to_cuda_version[package.package_id] = (
                package.environment_requirements.cuda_version
            )
    cuda_versions = sorted(set(cuda_versions))
    cuda_versions_ranking = {version: -idx for idx, version in enumerate(cuda_versions)}
    results = []
    for package in model_packages:
        package_cu_version = package_id_to_cuda_version.get(package.package_id)
        result = cuda_versions_ranking.get(package_cu_version, last_ranking)
        results.append(result)
    return results


def rank_trt_versions(model_packages: List[ModelPackageMetadata]) -> List[int]:
    trt_versions = []
    package_id_to_trt_version = {}
    last_ranking = -len(model_packages)
    for package in model_packages:
        if isinstance(package.environment_requirements, ServerEnvironmentRequirements):
            trt_versions.append(package.environment_requirements.trt_version)
            package_id_to_trt_version[package.package_id] = (
                package.environment_requirements.trt_version
            )
        elif isinstance(
            package.environment_requirements, ServerEnvironmentRequirements
        ):
            trt_versions.append(package.environment_requirements.trt_version)
            package_id_to_trt_version[package.package_id] = (
                package.environment_requirements.trt_version
            )
    trt_versions = sorted(set(trt_versions))
    trt_versions_ranking = {version: -idx for idx, version in enumerate(trt_versions)}
    results = []
    for package in model_packages:
        package_trt_version = package_id_to_trt_version.get(package.package_id)
        result = trt_versions_ranking.get(package_trt_version, last_ranking)
        results.append(result)
    return results
