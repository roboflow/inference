from typing import List

from inference_exp.weights_providers.entities import (
    BackendType,
    ModelPackageMetadata,
    Quantization,
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
                model_package,
            )
        )
    sorted_features = sorted(sorting_features, key=lambda x: x[:3], reverse=True)
    return [f[3] for f in sorted_features]
