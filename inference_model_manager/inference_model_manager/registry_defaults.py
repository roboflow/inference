"""Lazy model registry — zero heavy imports at module load.

Registration happens per-class when a model is first loaded.
Config is a static dict keyed by class name string — no imports needed
until actual registration. Validators and serializers are pure Python
(no torch/transformers/peft deps).

Usage:
    from inference_model_manager.registry_defaults import registry, lazy_register
    lazy_register(model_class)  # registers if base class has config
"""

from __future__ import annotations

from inference_model_manager.registry import ModelRegistry

# Module-level singleton.
registry = ModelRegistry()

# Track which classes we've already registered (by id) to avoid re-scanning MRO.
_registered_classes: set[int] = set()


# ---------------------------------------------------------------------------
# Static config — keyed by class NAME string, matched against MRO.
# No imports happen here. Pure data.
# ---------------------------------------------------------------------------

# Each entry: (task_name, method, default, params, validator_name, serializer_name, response_type)
# validator_name/serializer_name are looked up from the modules at registration time.

_TASK_CONFIGS: dict[str, list[tuple[str, str, bool, list[str], str, str, str]]] = {
    # --- Object Detection ---
    "ObjectDetectionModel": [
        ("infer", "infer", True, ["images"],
         "validate_images_required", "serialize_detections_compact",
         "roboflow-object-detection-compact-v1"),
    ],
    "OpenVocabularyObjectDetectionModel": [
        ("infer", "infer", True, ["images", "classes"],
         "validate_images_and_classes", "serialize_detections_compact",
         "roboflow-object-detection-compact-v1"),
    ],
    # --- Classification ---
    "ClassificationModel": [
        ("infer", "infer", True, ["images"],
         "validate_images_required", "serialize_classification_compact",
         "roboflow-classification-compact-v1"),
    ],
    "MultiLabelClassificationModel": [
        ("infer", "infer", True, ["images"],
         "validate_images_required", "serialize_multilabel_classification_compact",
         "roboflow-classification-compact-v1"),
    ],
    # --- Instance Segmentation ---
    "InstanceSegmentationModel": [
        ("infer", "infer", True, ["images"],
         "validate_images_required", "serialize_instance_segmentation_compact",
         "roboflow-instance-segmentation-compact-v1"),
    ],
    # --- Semantic Segmentation ---
    "SemanticSegmentationModel": [
        ("infer", "infer", True, ["images"],
         "validate_images_required", "serialize_semantic_segmentation_compact",
         "roboflow-semantic-segmentation-compact-v1"),
    ],
    # --- Keypoints ---
    "KeyPointsDetectionModel": [
        ("infer", "infer", True, ["images"],
         "validate_images_required", "serialize_keypoints_compact",
         "roboflow-keypoints-compact-v1"),
    ],
    # --- Depth Estimation ---
    "DepthEstimationModel": [
        ("infer", "infer", True, ["images"],
         "validate_images_required", "serialize_depth_compact",
         "roboflow-depth-compact-v1"),
    ],
    # --- Documents / OCR ---
    "StructuredOCRModel": [
        ("infer", "infer", True, ["images"],
         "validate_images_required", "serialize_text",
         "roboflow-text-v1"),
    ],
    # --- Embeddings ---
    "TextImageEmbeddingModel": [
        ("embed_images", "embed_images", True, ["images"],
         "validate_images_required", "serialize_embeddings",
         "roboflow-embeddings-compact-v1"),
        ("embed_text", "embed_text", False, ["texts"],
         "validate_texts_required", "serialize_embeddings",
         "roboflow-embeddings-compact-v1"),
    ],
    # --- Gaze ---
    "L2CSNetOnnx": [
        ("infer", "infer", True, ["images"],
         "validate_images_required", "serialize_detections_compact",
         "roboflow-object-detection-compact-v1"),
    ],
    # --- VLM / Prompt models ---
    "PaliGemmaHF": [
        ("prompt", "prompt", True, ["images", "prompt"],
         "validate_images_and_prompt", "serialize_text",
         "roboflow-text-v1"),
    ],
    "Gemma4HF": [
        ("prompt", "prompt", True, ["images", "prompt"],
         "validate_images_and_prompt", "serialize_text",
         "roboflow-text-v1"),
    ],
    "Qwen25VLHF": [
        ("prompt", "prompt", True, ["images", "prompt"],
         "validate_images_and_prompt", "serialize_text",
         "roboflow-text-v1"),
    ],
    "Qwen3VLHF": [
        ("prompt", "prompt", True, ["images", "prompt"],
         "validate_images_and_prompt", "serialize_text",
         "roboflow-text-v1"),
    ],
    "Qwen35HF": [
        ("prompt", "prompt", True, ["images", "prompt"],
         "validate_images_and_prompt", "serialize_text",
         "roboflow-text-v1"),
    ],
    "SmolVLMHF": [
        ("prompt", "prompt", True, ["images", "prompt"],
         "validate_images_and_prompt", "serialize_text",
         "roboflow-text-v1"),
    ],
    # --- Florence2 ---
    "Florence2HF": [
        ("caption", "caption_image", True, ["images"],
         "validate_images_required", "serialize_text", "roboflow-text-v1"),
        ("detect", "detect_objects", False, ["images"],
         "validate_images_required", "serialize_text", "roboflow-text-v1"),
        ("ocr", "ocr_image", False, ["images"],
         "validate_images_required", "serialize_text", "roboflow-text-v1"),
        ("parse_document", "parse_document", False, ["images"],
         "validate_images_required", "serialize_text", "roboflow-text-v1"),
        ("prompt", "prompt", False, ["images", "prompt"],
         "validate_images_and_prompt", "serialize_text", "roboflow-text-v1"),
        ("segment_phrase", "segment_phrase", False, ["images", "prompt"],
         "validate_images_and_prompt", "serialize_text", "roboflow-text-v1"),
        ("ground_phrase", "ground_phrase", False, ["images", "prompt"],
         "validate_images_and_prompt", "serialize_text", "roboflow-text-v1"),
        ("classify_region", "classify_image_region", False, ["images", "prompt"],
         "validate_images_and_prompt", "serialize_text", "roboflow-text-v1"),
        ("caption_region", "caption_image_region", False, ["images", "prompt"],
         "validate_images_and_prompt", "serialize_text", "roboflow-text-v1"),
        ("ocr_region", "ocr_image_region", False, ["images", "prompt"],
         "validate_images_and_prompt", "serialize_text", "roboflow-text-v1"),
        ("segment_region", "segment_region", False, ["images", "prompt"],
         "validate_images_and_prompt", "serialize_text", "roboflow-text-v1"),
    ],
    # --- SAM ---
    "SAMTorch": [
        ("embed", "embed_images", True, ["images"],
         "validate_images_required", "serialize_embeddings",
         "roboflow-embeddings-compact-v1"),
        ("segment", "segment_images", False, ["embeddings", "points"],
         "validate_passthrough", "serialize_instance_segmentation_compact",
         "roboflow-instance-segmentation-compact-v1"),
    ],
    "SAM2Torch": [
        ("embed", "embed_images", True, ["images"],
         "validate_images_required", "serialize_embeddings",
         "roboflow-embeddings-compact-v1"),
        ("segment", "segment_images", False, ["embeddings", "points"],
         "validate_passthrough", "serialize_instance_segmentation_compact",
         "roboflow-instance-segmentation-compact-v1"),
    ],
    # --- Moondream2 ---
    "MoonDream2HF": [
        ("caption", "caption", True, ["images"],
         "validate_images_required", "serialize_text", "roboflow-text-v1"),
        ("detect", "detect", False, ["images"],
         "validate_images_required", "serialize_detections_compact",
         "roboflow-object-detection-compact-v1"),
        ("query", "query", False, ["images", "prompt"],
         "validate_images_and_prompt", "serialize_text", "roboflow-text-v1"),
        ("point", "point", False, ["images", "prompt"],
         "validate_images_and_prompt", "serialize_detections_compact",
         "roboflow-object-detection-compact-v1"),
        ("encode", "encode_images", False, ["images"],
         "validate_images_required", "serialize_embeddings",
         "roboflow-embeddings-compact-v1"),
    ],
    # --- GlmOCR ---
    "GlmOcrHF": [
        ("recognize_text", "recognize_text", True, ["images"],
         "validate_images_required", "serialize_text", "roboflow-text-v1"),
        ("recognize_table", "recognize_table", False, ["images"],
         "validate_images_required", "serialize_text", "roboflow-text-v1"),
        ("recognize_formula", "recognize_formula", False, ["images"],
         "validate_images_required", "serialize_text", "roboflow-text-v1"),
        ("prompt", "prompt", False, ["images", "prompt"],
         "validate_images_and_prompt", "serialize_text", "roboflow-text-v1"),
    ],
    # --- SAM2 RT (streaming) ---
    "SAM2ForStream": [
        ("prompt", "prompt", True, ["images", "prompt"],
         "validate_images_and_prompt", "serialize_text",
         "roboflow-text-v1"),
        ("track", "track", False, ["images"],
         "validate_images_required", "serialize_text",
         "roboflow-text-v1"),
    ],
    # --- Text-only OCR ---
    "TextOnlyOCRModel": [
        ("infer", "infer", True, ["images"],
         "validate_images_required", "serialize_text",
         "roboflow-text-v1"),
    ],
    # --- Passthrough (benchmark) ---
    "PassthroughModel": [
        ("infer", "infer", True, ["images"],
         "validate_passthrough", "serialize_detections_compact",
         "roboflow-object-detection-compact-v1"),
    ],
}


# ---------------------------------------------------------------------------
# Lazy registration
# ---------------------------------------------------------------------------

def _resolve_validator(name: str):
    """Import validator by name from validators module."""
    from inference_model_manager import validators
    return getattr(validators, name)


def _resolve_serializer(name: str):
    """Import serializer by name from serializers_typed module."""
    from inference_model_manager import serializers_typed
    return getattr(serializers_typed, name)


def lazy_register(model_class: type) -> None:
    """Register tasks for model_class if any MRO ancestor has config.

    Called once per class. Walks MRO, checks class names against
    _TASK_CONFIGS. Imports validators/serializers only when needed
    (pure Python, no heavy deps).
    """
    cls_id = id(model_class)
    if cls_id in _registered_classes:
        return
    _registered_classes.add(cls_id)

    for cls in model_class.__mro__:
        _register_from_config(cls)


def lazy_register_by_names(mro_names: list[str]) -> None:
    """Register tasks using MRO class name strings (subprocess path).

    Worker sends class names in READY pipe — no actual class objects needed.
    Creates lightweight placeholder classes for registry storage. Lookup
    uses get_entry_by_mro_names() which matches by class name string.
    """
    key = ",".join(mro_names)
    if key in _registered_name_keys:
        return
    _registered_name_keys.add(key)

    for name in mro_names:
        config = _TASK_CONFIGS.get(name)
        if config is None:
            continue
        placeholder = type(name, (), {})
        for task_name, method, default, params, val_name, ser_name, resp_type in config:
            registry.register(
                placeholder,
                task_name,
                method=method,
                default=default,
                params=params,
                validator=_resolve_validator(val_name),
                serializer=_resolve_serializer(ser_name),
                response_type=resp_type,
            )


_registered_name_keys: set[str] = set()


def _register_from_config(cls: type) -> None:
    """Register tasks for a single class if it has config."""
    config = _TASK_CONFIGS.get(cls.__name__)
    if config is None:
        return
    for task_name, method, default, params, val_name, ser_name, resp_type in config:
        if registry.get_entry_for_class(cls, task_name) is not None:
            continue
        registry.register(
            cls,
            task_name,
            method=method,
            default=default,
            params=params,
            validator=_resolve_validator(val_name),
            serializer=_resolve_serializer(ser_name),
            response_type=resp_type,
        )


