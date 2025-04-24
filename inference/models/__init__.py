import importlib
from typing import Any

from inference.core.env import (
    CORE_MODEL_CLIP_ENABLED,
    CORE_MODEL_DOCTR_ENABLED,
    CORE_MODEL_GAZE_ENABLED,
    CORE_MODEL_GROUNDINGDINO_ENABLED,
    CORE_MODEL_SAM2_ENABLED,
    CORE_MODEL_SAM_ENABLED,
    CORE_MODEL_YOLO_WORLD_ENABLED,
    CORE_MODELS_ENABLED,
    DEPTH_ESTIMATION_ENABLED,
)

_MODEL_REGISTRY: dict[str, Any] = {}

CORE_MODELS = {
    "Clip": ("inference.models.clip", CORE_MODEL_CLIP_ENABLED),
    "Gaze": ("inference.models.gaze", CORE_MODEL_GAZE_ENABLED),
    "SegmentAnything": ("inference.models.sam", CORE_MODEL_SAM_ENABLED),
    "SegmentAnything2": ("inference.models.sam2", CORE_MODEL_SAM2_ENABLED),
    "DocTR": ("inference.models.doctr", CORE_MODEL_DOCTR_ENABLED),
    "GroundingDINO": (
        "inference.models.grounding_dino",
        CORE_MODEL_GROUNDINGDINO_ENABLED,
    ),
    "YOLOWorld": ("inference.models.yolo_world", CORE_MODEL_YOLO_WORLD_ENABLED),
    "DepthEstimator": (
        "inference.models.depth_estimation.depthestimation",
        DEPTH_ESTIMATION_ENABLED,
    ),
}

OPTIONAL_MODELS = {
    "PaliGemma": "inference.models.paligemma",
    "LoRAPaliGemma": "inference.models.paligemma",
    "Florence2": "inference.models.florence2",
    "LoRAFlorence2": "inference.models.florence2",
    "Qwen25VL": "inference.models.qwen25vl",
    "LoRAQwen25VL": "inference.models.qwen25vl",
    "TrOCR": "inference.models.trocr",
    "SmolVLM": "inference.models.smolvlm",
    "Moondream2": "inference.models.moondream2",
}

STANDARD_MODELS = {
    "ResNetClassification": "inference.models.resnet",
    "RFDETRObjectDetection": "inference.models.rfdetr",
    "VitClassification": "inference.models.vit",
    "YOLACT": "inference.models.yolact",
    "YOLONASObjectDetection": "inference.models.yolonas",
    "YOLOv5InstanceSegmentation": "inference.models.yolov5",
    "YOLOv5ObjectDetection": "inference.models.yolov5",
    "YOLOv7InstanceSegmentation": "inference.models.yolov7",
    "YOLOv8Classification": "inference.models.yolov8",
    "YOLOv8InstanceSegmentation": "inference.models.yolov8",
    "YOLOv8KeypointsDetection": "inference.models.yolov8",
    "YOLOv8ObjectDetection": "inference.models.yolov8",
    "YOLOv9ObjectDetection": "inference.models.yolov9",
    "YOLOv10ObjectDetection": "inference.models.yolov10",
    "YOLOv11InstanceSegmentation": "inference.models.yolov11",
    "YOLOv11KeypointsDetection": "inference.models.yolov11",
    "YOLOv11ObjectDetection": "inference.models.yolov11",
    "YOLOv12ObjectDetection": "inference.models.yolov12",
}


def get_model_class(name: str) -> Any:
    """Lazily import and return a model class."""
    if name in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[name]

    # Handle core models with environment flags
    if CORE_MODELS_ENABLED and name in CORE_MODELS:
        module_path, flag_enabled = CORE_MODELS[name]
        if flag_enabled:
            module = importlib.import_module(module_path)
            _MODEL_REGISTRY[name] = getattr(module, name)

    # Handle optional models
    elif name in OPTIONAL_MODELS:
        module_path = OPTIONAL_MODELS[name]
        module = importlib.import_module(module_path)
        _MODEL_REGISTRY[name] = getattr(module, name)

    # Handle standard models
    elif name in STANDARD_MODELS:
        module_path = STANDARD_MODELS[name]
        module = importlib.import_module(module_path)
        _MODEL_REGISTRY[name] = getattr(module, name)

    return _MODEL_REGISTRY.get(name)


def __getattr__(name: str) -> Any:
    """Implement lazy loading for model classes."""
    if name in __all__:
        return get_model_class(name)
    raise AttributeError(f"module 'inference.models' has no attribute '{name}'")


__all__ = (
    list(CORE_MODELS.keys())
    + list(OPTIONAL_MODELS.keys())
    + list(STANDARD_MODELS.keys())
)
