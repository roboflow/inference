from typing import List

from inference.v1.weights_providers.entities import (
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
    # Sort model_packages in-place based on backend, quantization, batch size priorities (descending)
    return sorted(
        model_packages,
        key=lambda model_package: (
            BACKEND_PRIORITY.get(model_package.backend, 0),
            QUANTIZATION_PRIORITY.get(model_package.quantization, 0),
            BATCH_SIZE_PRIORITY[
                (
                    DYNAMIC_BATCH_SIZE_KEY
                    if model_package.dynamic_batch_size_supported
                    else STATIC_BATCH_SIZE_KEY
                )
            ],
        ),
        reverse=True,
    )
