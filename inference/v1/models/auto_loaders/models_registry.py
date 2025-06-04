from typing import Dict, Optional, Tuple

from inference.v1.errors import ModelImplementationLoaderError
from inference.v1.utils.imports import LazyClass
from inference.v1.weights_providers.entities import BackendType

ModelArchitecture = str
TaskType = Optional[str]


REGISTERED_MODELS: Dict[Tuple[ModelArchitecture, TaskType, BackendType], LazyClass] = {
    ("yolov8", "object-detection", BackendType.ONNX): LazyClass(
        module_name="inference.v1.models.yolov8.yolov8_object_detection_onnx",
        class_name="YOLOv8ForObjectDetectionOnnx",
    ),
    ("yolov8", "object-detection", BackendType.TRT): LazyClass(
        module_name="inference.v1.models.yolov8.yolov8_object_detection_trt",
        class_name="YOLOv8ForObjectDetectionTRT",
    ),
    ("paligemma", "vlm", BackendType.HF): LazyClass(
        module_name="inference.v1.models.paligemma.paligemma_hf",
        class_name="PaliGemmaHF",
    ),
}


def resolve_model_class(
    model_architecture: ModelArchitecture,
    task_type: TaskType,
    backend: BackendType,
) -> type:
    if not model_implementation_exists(
        model_architecture=model_architecture,
        task_type=task_type,
        backend=backend,
    ):
        raise ModelImplementationLoaderError(
            f"Did not find implementation for model with architecture: {model_architecture}, "
            f"task type: {task_type} and backend: {backend}"
        )
    return REGISTERED_MODELS[(model_architecture, task_type, backend)].resolve()


def model_implementation_exists(
    model_architecture: ModelArchitecture,
    task_type: TaskType,
    backend: BackendType,
) -> bool:
    lookup_key = (model_architecture, task_type, backend)
    return lookup_key in REGISTERED_MODELS
