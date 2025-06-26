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
    selected_device: Optional[torch.device] = None,
) -> List[ModelPackageMetadata]:
    # I feel like this will be the biggest liability of new inference :)
    # Some dimensions are just hard to rank arbitrarily and reasonably
    sorting_features = []
    # ordering TRT and Cu versions from older to newest -
    # with the assumption that incompatible versions are eliminated earlier, and
    # ranking implicitly attempts to match version closes to the current -
    # in come cases we would rank high versions below current one, but for that
    # it is assumed such versions are compatible, otherwise should be
    # discarded in the previous stage.
    cuda_ranking = rank_cuda_versions(model_packages=model_packages)
    trt_ranking = rank_trt_versions(model_packages=model_packages)
    for model_package, package_cu_rank, package_trt_rank in zip(
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
                retrieve_onnx_opset_score(
                    model_package
                ),  # the higher opset, the better
                retrieve_trt_forward_compatible_match_score(
                    model_package
                ),  # exact matches first
                retrieve_same_trt_cc_compatibility_score(model_package),
                retrieve_cuda_device_match_score(
                    model_package, selected_device
                ),  # we like more direct matches
                package_cu_rank,
                package_trt_rank,
                retrieve_onnx_incompatible_providers_score(model_package),
                retrieve_trt_dynamic_batch_size_score(model_package),
                retrieve_trt_lean_runtime_excluded_score(model_package),
                retrieve_jetson_device_name_match_score(model_package),
                retrieve_os_version_match_score(model_package),
                retrieve_l4t_version_match_score(model_package),
                retrieve_driver_version_match_score(model_package),
                model_package,
            )
        )
    sorted_features = sorted(sorting_features, key=lambda x: x[:17], reverse=True)
    return [f[-1] for f in sorted_features]


def retrieve_onnx_opset_score(model_package: ModelPackageMetadata) -> int:
    if model_package.onnx_package_details is None:
        return -1
    return model_package.onnx_package_details.opset


def retrieve_cuda_device_match_score(
    model_package: ModelPackageMetadata,
    selected_device: Optional[torch.device] = None,
) -> int:
    if model_package.backend is not BackendType.TRT:
        return 0
    if model_package.environment_requirements is None:
        return 0
    if not isinstance(
        model_package.environment_requirements,
        (JetsonEnvironmentRequirements, ServerEnvironmentRequirements),
    ):
        return 0
    runtime_x_ray = x_ray_runtime_environment()
    all_available_cuda_devices, _ = filter_available_devices_with_selected_device(
        selected_device=selected_device,
        all_available_cuda_devices=runtime_x_ray.gpu_devices,
        all_available_devices_cc=runtime_x_ray.gpu_devices_cc,
    )
    compilation_device = model_package.environment_requirements.cuda_device_name
    return sum(dev == compilation_device for dev in all_available_cuda_devices)


def retrieve_same_trt_cc_compatibility_score(
    model_package: ModelPackageMetadata,
) -> int:
    if model_package.trt_package_details is None:
        return 1
    return int(not model_package.trt_package_details.same_cc_compatible)


def retrieve_trt_forward_compatible_match_score(
    model_package: ModelPackageMetadata,
) -> int:
    if model_package.trt_package_details is None:
        return 1
    return int(not model_package.trt_package_details.trt_forward_compatible)


def retrieve_onnx_incompatible_providers_score(
    model_package: ModelPackageMetadata,
) -> int:
    if model_package.onnx_package_details is None:
        return 0
    if not model_package.onnx_package_details.incompatible_providers:
        return 0
    runtime_x_ray = x_ray_runtime_environment()
    available_onnx_execution_providers = set(
        runtime_x_ray.available_onnx_execution_providers or []
    )
    return -len(
        available_onnx_execution_providers.intersection(
            model_package.onnx_package_details.incompatible_providers
        )
    )


def retrieve_trt_dynamic_batch_size_score(model_package: ModelPackageMetadata) -> int:
    if model_package.trt_package_details is None:
        return 0
    if any(
        bs is None
        for bs in [
            model_package.trt_package_details.min_dynamic_batch_size,
            model_package.trt_package_details.max_dynamic_batch_size,
        ]
    ):
        return 0
    return (
        model_package.trt_package_details.max_dynamic_batch_size
        - model_package.trt_package_details.min_dynamic_batch_size
    )


def retrieve_trt_lean_runtime_excluded_score(
    model_package: ModelPackageMetadata,
) -> int:
    if model_package.trt_package_details is None:
        return 0
    return int(not model_package.trt_package_details.trt_lean_runtime_excluded)


def retrieve_os_version_match_score(model_package: ModelPackageMetadata) -> int:
    if model_package.backend is not BackendType.TRT:
        # irrelevant for not trt
        return 0
    if model_package.environment_requirements is None:
        return 0
    if not isinstance(
        model_package.environment_requirements, ServerEnvironmentRequirements
    ):
        return 0
    if not model_package.environment_requirements.os_version:
        return 0
    runtime_x_ray = x_ray_runtime_environment()
    return int(
        runtime_x_ray.os_version == model_package.environment_requirements.os_version
    )


def retrieve_l4t_version_match_score(model_package: ModelPackageMetadata) -> int:
    if model_package.backend is not BackendType.TRT:
        # irrelevant for not trt
        return 0
    if model_package.environment_requirements is None:
        return 0
    if not isinstance(
        model_package.environment_requirements, JetsonEnvironmentRequirements
    ):
        return 0
    runtime_x_ray = x_ray_runtime_environment()
    return int(
        runtime_x_ray.l4t_version == model_package.environment_requirements.l4t_version
    )


def retrieve_driver_version_match_score(model_package: ModelPackageMetadata) -> int:
    if model_package.trt_package_details is None:
        # irrelevant for not trt
        return 0
    if model_package.environment_requirements is None:
        return 0
    if not isinstance(
        model_package.environment_requirements, JetsonEnvironmentRequirements
    ) and not isinstance(
        model_package.environment_requirements, ServerEnvironmentRequirements
    ):
        return 0
    if not model_package.environment_requirements.driver_version:
        return 0
    runtime_x_ray = x_ray_runtime_environment()
    return int(
        runtime_x_ray.driver_version
        == model_package.environment_requirements.driver_version
    )


def retrieve_jetson_device_name_match_score(model_package: ModelPackageMetadata) -> int:
    if model_package.trt_package_details is None:
        # irrelevant for not trt
        return 0
    if model_package.environment_requirements is None:
        return 0
    if not isinstance(
        model_package.environment_requirements, JetsonEnvironmentRequirements
    ):
        return 0
    runtime_x_ray = x_ray_runtime_environment()
    return int(
        runtime_x_ray.jetson_type
        == model_package.environment_requirements.jetson_product_name
    )


def rank_cuda_versions(model_packages: List[ModelPackageMetadata]) -> List[int]:
    cuda_versions = []
    package_id_to_cuda_version = {}
    last_ranking = -len(model_packages) + 1
    for package in model_packages:
        if isinstance(package.environment_requirements, ServerEnvironmentRequirements):
            cuda_versions.append(package.environment_requirements.cuda_version)
            package_id_to_cuda_version[package.package_id] = (
                package.environment_requirements.cuda_version
            )
        elif isinstance(
            package.environment_requirements, JetsonEnvironmentRequirements
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
    last_ranking = -len(model_packages) + 1
    for package in model_packages:
        if isinstance(package.environment_requirements, ServerEnvironmentRequirements):
            trt_versions.append(package.environment_requirements.trt_version)
            package_id_to_trt_version[package.package_id] = (
                package.environment_requirements.trt_version
            )
        elif isinstance(
            package.environment_requirements, JetsonEnvironmentRequirements
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
