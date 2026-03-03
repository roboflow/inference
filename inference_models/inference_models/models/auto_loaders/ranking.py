from typing import List, Optional, Union

import torch

from inference_models.models.auto_loaders.constants import (
    NMS_CLASS_AGNOSTIC_KEY,
    NMS_CONFIDENCE_THRESHOLD_KEY,
    NMS_FUSED_FEATURE,
    NMS_IOU_THRESHOLD_KEY,
    NMS_MAX_DETECTIONS_KEY,
)
from inference_models.models.auto_loaders.entities import BackendType
from inference_models.models.auto_loaders.utils import (
    filter_available_devices_with_selected_device,
)
from inference_models.runtime_introspection.core import x_ray_runtime_environment
from inference_models.weights_providers.entities import (
    JetsonEnvironmentRequirements,
    ModelPackageMetadata,
    Quantization,
    ServerEnvironmentRequirements,
)

BACKEND_PRIORITY = {
    BackendType.TRT: 7,
    BackendType.ONNX: 6,
    BackendType.TORCH: 5,
    BackendType.HF: 4,
    BackendType.TORCH_SCRIPT: 3,
    BackendType.MEDIAPIPE: 2,
    BackendType.ULTRALYTICS: 1,
    BackendType.CUSTOM: 0,
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
    nms_fusion_preferences: Optional[Union[bool, dict]] = None,
) -> List[ModelPackageMetadata]:
    # I feel like this will be the biggest liability of new inference :))
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
    # this is to ensure determinism when other methods fail
    identifiers_ranking = rank_packages_ids(model_packages=model_packages)
    for model_package, package_cu_rank, package_trt_rank, package_id_rank in zip(
        model_packages, cuda_ranking, trt_ranking, identifiers_ranking
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
                model_package.trusted_source,
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
                retrieve_fused_nms_rank(
                    model_package, nms_fusion_preferences=nms_fusion_preferences
                ),
                retrieve_trt_lean_runtime_excluded_score(model_package),
                retrieve_jetson_device_name_match_score(model_package),
                retrieve_os_version_match_score(model_package),
                retrieve_l4t_version_match_score(model_package),
                retrieve_driver_version_match_score(model_package),
                package_id_rank,
                model_package,
            )
        )
    sorted_features = sorted(sorting_features, key=lambda x: x[:-1], reverse=True)
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


def retrieve_fused_nms_rank(
    model_package: ModelPackageMetadata,
    nms_fusion_preferences: Optional[Union[bool, dict]],
) -> Union[float, int]:
    if nms_fusion_preferences is None or nms_fusion_preferences is False:
        return 0
    if not model_package.model_features:
        return 0
    nms_fused = model_package.model_features.get(NMS_FUSED_FEATURE)
    if not isinstance(nms_fused, dict):
        return 0
    if nms_fusion_preferences is True:
        # default values should be passed by filter, so here we treat every package equally good
        return 1
    actual_max_detections = nms_fused[NMS_MAX_DETECTIONS_KEY]
    actual_confidence_threshold = nms_fused[NMS_CONFIDENCE_THRESHOLD_KEY]
    actual_iou_threshold = nms_fused[NMS_IOU_THRESHOLD_KEY]
    actual_class_agnostic = nms_fused[NMS_CLASS_AGNOSTIC_KEY]
    final_score = 0.0
    if NMS_MAX_DETECTIONS_KEY in nms_fusion_preferences:
        requested_max_detections = nms_fusion_preferences[NMS_MAX_DETECTIONS_KEY]
        if isinstance(requested_max_detections, (list, tuple)):
            min_detections, max_detections = requested_max_detections
        else:
            min_detections, max_detections = (
                requested_max_detections,
                requested_max_detections,
            )
        final_score += score_distance_from_mean(
            min_value=min_detections,
            max_value=max_detections,
            examined_value=actual_max_detections,
        )
    if NMS_CONFIDENCE_THRESHOLD_KEY in nms_fusion_preferences:
        requested_confidence = nms_fusion_preferences[NMS_CONFIDENCE_THRESHOLD_KEY]
        if isinstance(requested_confidence, (list, tuple)):
            min_confidence, max_confidence = requested_confidence
        else:
            min_confidence, max_confidence = (
                requested_confidence,
                requested_confidence,
            )
        final_score += score_distance_from_mean(
            min_value=min_confidence,
            max_value=max_confidence,
            examined_value=actual_confidence_threshold,
        )
    if NMS_IOU_THRESHOLD_KEY in nms_fusion_preferences:
        requested_iou_threshold = nms_fusion_preferences[NMS_IOU_THRESHOLD_KEY]
        if isinstance(requested_iou_threshold, (list, tuple)):
            min_iou_threshold, max_iou_threshold = requested_iou_threshold
        else:
            min_iou_threshold, max_iou_threshold = (
                requested_iou_threshold,
                requested_iou_threshold,
            )
        final_score += score_distance_from_mean(
            min_value=min_iou_threshold,
            max_value=max_iou_threshold,
            examined_value=actual_iou_threshold,
        )
    if NMS_CLASS_AGNOSTIC_KEY in nms_fusion_preferences:
        final_score += float(
            actual_class_agnostic == nms_fusion_preferences[NMS_CLASS_AGNOSTIC_KEY]
        )
    return final_score


def score_distance_from_mean(
    min_value: float, max_value: float, examined_value: float
) -> float:
    min_value, max_value = min(min_value, max_value), max(min_value, max_value)
    if min_value == max_value:
        return float(abs(examined_value - max_value) < 1e-5)
    if examined_value < min_value or examined_value > max_value:
        return 0.0
    span = max_value - min_value
    examined_value_scaled = min(max((examined_value - min_value) / (span + 1e-6), 0), 1)
    return 1.0 - abs(0.5 - examined_value_scaled)


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


def rank_packages_ids(model_packages: List[ModelPackageMetadata]) -> List[int]:
    package_ids = [p.package_id for p in model_packages]
    return sorted(range(len(package_ids)), key=lambda i: package_ids[i])
