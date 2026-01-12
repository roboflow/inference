import importlib
from typing import Any

# Preinit nvdiffrast for SAM3D as it breaks if any flash attn model is loaded in first
try:
    import torch

    if torch.cuda.is_available():
        import utils3d.torch

        _nvdiffrast_ctx = utils3d.torch.RastContext(backend="cuda")
        _dummy_verts = torch.zeros(1, 3, 3, device="cuda")
        _dummy_faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")
        _ = utils3d.torch.rasterize_triangle_faces(
            _nvdiffrast_ctx, _dummy_verts, _dummy_faces, 64, 64
        )
        del _dummy_verts, _dummy_faces, _
except:
    pass

from inference.core.env import (
    CORE_MODEL_CLIP_ENABLED,
    CORE_MODEL_DOCTR_ENABLED,
    CORE_MODEL_EASYOCR_ENABLED,
    CORE_MODEL_GAZE_ENABLED,
    CORE_MODEL_GROUNDINGDINO_ENABLED,
    CORE_MODEL_SAM2_ENABLED,
    CORE_MODEL_SAM3_ENABLED,
    CORE_MODEL_SAM_ENABLED,
    CORE_MODEL_YOLO_WORLD_ENABLED,
    CORE_MODELS_ENABLED,
    DEPTH_ESTIMATION_ENABLED,
    SAM3_3D_OBJECTS_ENABLED,
)

_MODEL_REGISTRY: dict[str, Any] = {}

CORE_MODELS = {
    "Clip": ("inference.models.clip", CORE_MODEL_CLIP_ENABLED),
    "Gaze": ("inference.models.gaze", CORE_MODEL_GAZE_ENABLED),
    "SegmentAnything": ("inference.models.sam", CORE_MODEL_SAM_ENABLED),
    "SegmentAnything2": ("inference.models.sam2", CORE_MODEL_SAM2_ENABLED),
    "SegmentAnything3": ("inference.models.sam3", CORE_MODEL_SAM3_ENABLED),
    "SegmentAnything3_3D_Objects": (
        "inference.models.sam3_3d",
        SAM3_3D_OBJECTS_ENABLED,
    ),
    "Sam3ForInteractiveImageSegmentation": (
        "inference.models.sam3",
        CORE_MODEL_SAM3_ENABLED,
    ),
    "DocTR": ("inference.models.doctr", CORE_MODEL_DOCTR_ENABLED),
    "EasyOCR": ("inference.models.easy_ocr", CORE_MODEL_EASYOCR_ENABLED),
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
    "Qwen3VL": "inference.models.qwen3vl",
    "LoRAQwen3VL": "inference.models.qwen3vl",
    "TrOCR": "inference.models.trocr",
    "SmolVLM": "inference.models.smolvlm",
    "LoRASmolVLM": "inference.models.smolvlm",
    "Moondream2": "inference.models.moondream2",
    "PerceptionEncoder": "inference.models.perception_encoder",
    "EasyOCR": "inference.models.easy_ocr",
}

STANDARD_MODELS = {
    "ResNetClassification": "inference.models.resnet",
    "RFDETRObjectDetection": "inference.models.rfdetr",
    "RFDETRInstanceSegmentation": "inference.models.rfdetr",
    "VitClassification": "inference.models.vit",
    "DinoV3Classification": "inference.models.dinov3",
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
    cls = get_model_class(name)
    if cls is not None:
        return cls
    raise AttributeError(f"module 'inference.models' has no attribute '{name}'")


__all__ = (
    list(CORE_MODELS.keys())
    + list(OPTIONAL_MODELS.keys())
    + list(STANDARD_MODELS.keys())
)
