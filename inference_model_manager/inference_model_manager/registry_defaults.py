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
#
# params: dict[str, dict] — each param has "type", "required", and optionally "default".
# Base class entries provide a fallback. Concrete class entries (matched first via MRO)
# can override with richer params. To add a new model: add an entry keyed by its class name.

# --- Reusable param fragments ---

_P_IMAGES = {"images": {"type": "image", "required": True}}
_P_IMAGES_CLASSES = {
    "images": {"type": "image", "required": True},
    "classes": {"type": "list[str]", "required": True},
}
_P_IMAGES_PROMPT = {
    "images": {"type": "image", "required": True},
    "prompt": {"type": "str", "required": True},
}
_P_GEN_FLAGS = {
    "do_sample": {"type": "bool", "required": False},
    "skip_special_tokens": {"type": "bool", "required": False},
}
_P_VLM_PROMPT_BASIC = {
    "images": {"type": "image", "required": True},
    "prompt": {"type": "str", "required": True},
    "max_new_tokens": {"type": "int", "required": False},
}
_P_VLM_PROMPT = {**_P_VLM_PROMPT_BASIC, **_P_GEN_FLAGS}
_P_FLORENCE2_PHRASE = {
    **_P_VLM_PROMPT_BASIC,
    "do_sample": {"type": "bool", "required": False},
}
_P_FLORENCE2_REGION = {
    "images": {"type": "image", "required": True},
    "xyxy": {"type": "list", "required": True},
    "max_new_tokens": {"type": "int", "required": False},
}
_P_TEXTS = {"texts": {"type": "list[str]", "required": True}}
_P_SAM3_VISUAL_PROMPTS = {
    "images": {"type": "image", "required": False},
    "embeddings": {"type": "tensor", "required": False},
    "point_coordinates": {"type": "list", "required": False},
    "point_labels": {"type": "list", "required": False},
    "boxes": {"type": "list", "required": False},
    "mask_input": {"type": "list", "required": False},
    "multi_mask_output": {"type": "bool", "required": False, "default": True},
    "return_logits": {"type": "bool", "required": False, "default": False},
    "image_hashes": {"type": "list[str]", "required": False},
    "use_embeddings_cache": {"type": "bool", "required": False, "default": True},
    "load_from_mask_input_cache": {"type": "bool", "required": False, "default": False},
    "save_to_mask_input_cache": {"type": "bool", "required": False, "default": False},
    "mask_format": {"type": "str", "required": False, "default": "rle"},
}
_P_SAM3_TEXT_PROMPTS = {
    "images": {"type": "image", "required": True},
    "prompts": {"type": "list", "required": True},
    "output_prob_thresh": {"type": "float", "required": False, "default": 0.5},
    "mask_format": {"type": "str", "required": False, "default": "rle"},
}
_P_SAM3_EMBED = {
    "images": {"type": "image", "required": True},
    "image_hashes": {"type": "list[str]", "required": False},
    "use_embeddings_cache": {"type": "bool", "required": False, "default": True},
}
_P_OWLV2_REFERENCE_EXAMPLES = {
    "reference_examples": {"type": "list", "required": True},
    "confidence": {"type": "float", "required": False},
    "iou_threshold": {"type": "float", "required": False},
    "max_detections": {"type": "int", "required": False},
}

# Common kwargs for object detection models
_K_OD = {
    "confidence": {"type": "float", "required": False},
    "iou_threshold": {"type": "float", "required": False},
    "max_detections": {"type": "int", "required": False},
    "class_agnostic_nms": {"type": "bool", "required": False},
}

_P_MASK_FORMAT = {"mask_format": {"type": "str", "required": False}}

# Instance segmentation adds mask params
_K_ISEG = {
    **_K_OD,
    "masks_smoothing_enabled": {"type": "bool", "required": False},
    "masks_binarization_threshold": {"type": "float", "required": False},
    **_P_MASK_FORMAT,
}

# Keypoints adds threshold
_K_KP = {
    **_K_OD,
    "key_points_threshold": {"type": "float", "required": False},
}


def _p(*dicts: dict) -> dict:
    """Merge param dicts."""
    r: dict = {}
    for d in dicts:
        r.update(d)
    return r


_P_MAX_NEW_TOKENS = {"max_new_tokens": {"type": "int", "required": False}}
_P_IMAGES_CLASSES_GEN = _p(
    _P_IMAGES,
    {"classes": {"type": "list[str]", "required": True}},
    _P_MAX_NEW_TOKENS,
)
_P_VLM_IMAGE_GEN = _p(_P_IMAGES, _P_MAX_NEW_TOKENS, _P_GEN_FLAGS)

_P_SAM_EMBED = {
    "images": {"type": "image", "required": True},
    "image_hashes": {"type": "list[str]", "required": False},
    "use_embeddings_cache": {"type": "bool", "required": False, "default": True},
}
_P_SAM_SEGMENT_COMMON = {
    "images": {"type": "image", "required": False},
    "embeddings": {"type": "tensor", "required": False},
    "image_hashes": {"type": "list[str]", "required": False},
    "point_coordinates": {"type": "list", "required": False},
    "point_labels": {"type": "list", "required": False},
    "boxes": {"type": "list", "required": False},
    "mask_input": {"type": "list", "required": False},
    "multi_mask_output": {"type": "bool", "required": False, "default": True},
    "return_logits": {"type": "bool", "required": False, "default": False},
    "mask_threshold": {"type": "float", "required": False},
    "use_embeddings_cache": {"type": "bool", "required": False, "default": True},
}
_P_SAM_SEGMENT = _p(
    _P_SAM_SEGMENT_COMMON,
    {
        "enforce_mask_input": {"type": "bool", "required": False, "default": False},
        "use_mask_input_cache": {"type": "bool", "required": False, "default": True},
    },
)
_P_SAM2_SEGMENT = _p(
    _P_SAM_SEGMENT_COMMON,
    {
        "load_from_mask_input_cache": {"type": "bool", "required": False, "default": False},
        "save_to_mask_input_cache": {"type": "bool", "required": False, "default": False},
    },
)

_E_OD_CONF_ONLY = [
    (
        "infer",
        "infer",
        True,
        _p(_P_IMAGES, {"confidence": {"type": "float", "required": False}}),
        "validate_images_required",
        "serialize_detections_compact",
        "roboflow-object-detection-compact-v1",
    ),
]
_E_OD_CONF_MAXDET = [
    (
        "infer",
        "infer",
        True,
        _p(
            _P_IMAGES,
            {
                "confidence": {"type": "float", "required": False},
                "max_detections": {"type": "int", "required": False},
            },
        ),
        "validate_images_required",
        "serialize_detections_compact",
        "roboflow-object-detection-compact-v1",
    ),
]
_E_OD_IMAGES_ONLY = [
    (
        "infer",
        "infer",
        True,
        _p(_P_IMAGES),
        "validate_images_required",
        "serialize_detections_compact",
        "roboflow-object-detection-compact-v1",
    ),
]
_E_ISEG_NO_SMOOTHING = [
    (
        "infer",
        "infer",
        True,
        _p(
            _P_IMAGES,
            {"confidence": {"type": "float", "required": False}},
            _P_MASK_FORMAT,
        ),
        "validate_images_required",
        "serialize_instance_segmentation_compact",
        "roboflow-instance-segmentation-compact-v1",
    ),
]
_E_ISEG_CONF_MAXDET = [
    (
        "infer",
        "infer",
        True,
        _p(
            _P_IMAGES,
            {
                "confidence": {"type": "float", "required": False},
                "max_detections": {"type": "int", "required": False},
            },
            _P_MASK_FORMAT,
        ),
        "validate_images_required",
        "serialize_instance_segmentation_compact",
        "roboflow-instance-segmentation-compact-v1",
    ),
]
_E_ISEG_NMS_NO_SMOOTHING = [
    (
        "infer",
        "infer",
        True,
        _p(
            _P_IMAGES,
            {
                "confidence": {"type": "float", "required": False},
                "iou_threshold": {"type": "float", "required": False},
                "max_detections": {"type": "int", "required": False},
                "class_agnostic_nms": {"type": "bool", "required": False},
            },
            _P_MASK_FORMAT,
        ),
        "validate_images_required",
        "serialize_instance_segmentation_compact",
        "roboflow-instance-segmentation-compact-v1",
    ),
]
_E_KP_CONF_THRESH = [
    (
        "infer",
        "infer",
        True,
        _p(
            _P_IMAGES,
            {
                "confidence": {"type": "float", "required": False},
                "key_points_threshold": {"type": "float", "required": False},
            },
        ),
        "validate_images_required",
        "serialize_keypoints_compact",
        "roboflow-keypoints-compact-v1",
    ),
]


def _unpack_config(cfg: tuple) -> tuple:
    task_name, method, default, params, val_name, ser_name, resp_type = cfg[:7]
    aliases = cfg[7] if len(cfg) > 7 else {}
    return task_name, method, default, params, val_name, ser_name, resp_type, aliases


_TASK_CONFIGS: dict[str, list[tuple[str, str, bool, dict, str, str, str]]] = {
    # --- Object Detection (base — fallback for all OD models) ---
    "ObjectDetectionModel": [
        (
            "infer",
            "infer",
            True,
            _p(_P_IMAGES, _K_OD),
            "validate_images_required",
            "serialize_detections_compact",
            "roboflow-object-detection-compact-v1",
        ),
    ],
    "OpenVocabularyObjectDetectionModel": [
        (
            "infer",
            "infer",
            True,
            _p(_P_IMAGES_CLASSES, _K_OD),
            "validate_images_and_classes",
            "serialize_detections_compact",
            "roboflow-object-detection-compact-v1",
        ),
    ],
    "OWLv2HF": [
        (
            "infer_with_reference_examples",
            "infer_with_reference_examples",
            False,
            _p(_P_IMAGES, _P_OWLV2_REFERENCE_EXAMPLES),
            "validate_images_required",
            "serialize_detections_compact",
            "roboflow-object-detection-compact-v1",
        ),
    ],
    "RFDetrForObjectDetectionTorch": _E_OD_CONF_ONLY,
    "RFDetrForObjectDetectionONNX": _E_OD_CONF_ONLY,
    "RFDetrForObjectDetectionTRT": _E_OD_CONF_ONLY,
    "YOLO26ForObjectDetectionOnnx": _E_OD_CONF_ONLY,
    "YOLO26ForObjectDetectionTorchScript": _E_OD_CONF_ONLY,
    "YOLO26ForObjectDetectionTRT": _E_OD_CONF_ONLY,
    "YOLOv10ForObjectDetectionOnnx": _E_OD_CONF_MAXDET,
    "YOLOv10ForObjectDetectionTRT": _E_OD_CONF_MAXDET,
    "PPOCRv6DetectionOnnx": _E_OD_IMAGES_ONLY,
    "RoboflowInstantHF": [
        (
            "infer",
            "infer",
            True,
            _p(
                _P_IMAGES,
                {
                    "confidence": {"type": "float", "required": False},
                    "iou_threshold": {"type": "float", "required": False},
                    "max_detections": {"type": "int", "required": False},
                },
            ),
            "validate_images_required",
            "serialize_detections_compact",
            "roboflow-object-detection-compact-v1",
        ),
    ],
    "GroundingDinoForObjectDetectionTorch": [
        (
            "infer",
            "infer",
            True,
            _p(
                _P_IMAGES_CLASSES,
                {
                    "box_confidence": {"type": "float", "required": False},
                    "text_confidence": {"type": "float", "required": False},
                    "iou_threshold": {"type": "float", "required": False},
                    "max_detections": {"type": "int", "required": False},
                    "class_agnostic_nms": {"type": "bool", "required": False},
                },
            ),
            "validate_images_and_classes",
            "serialize_detections_compact",
            "roboflow-object-detection-compact-v1",
        ),
    ],
    # --- Classification ---
    "ClassificationModel": [
        (
            "infer",
            "infer",
            True,
            _p(_P_IMAGES),
            "validate_images_required",
            "serialize_classification_compact",
            "roboflow-classification-compact-v1",
        ),
    ],
    "MultiLabelClassificationModel": [
        (
            "infer",
            "infer",
            True,
            _p(_P_IMAGES, {"confidence": {"type": "float", "required": False}}),
            "validate_images_required",
            "serialize_multilabel_classification_compact",
            "roboflow-classification-compact-v1",
        ),
    ],
    # --- Instance Segmentation ---
    "InstanceSegmentationModel": [
        (
            "infer",
            "infer",
            True,
            _p(_P_IMAGES, _K_ISEG),
            "validate_images_required",
            "serialize_instance_segmentation_compact",
            "roboflow-instance-segmentation-compact-v1",
        ),
    ],
    "YOLOv5ForInstanceSegmentationOnnx": _E_ISEG_NMS_NO_SMOOTHING,
    "YOLOv5ForInstanceSegmentationTRT": _E_ISEG_NMS_NO_SMOOTHING,
    "YOLOv7ForInstanceSegmentationOnnx": _E_ISEG_NMS_NO_SMOOTHING,
    "YOLOv7ForInstanceSegmentationTRT": _E_ISEG_NMS_NO_SMOOTHING,
    "YOLOACTForInstanceSegmentationOnnx": _E_ISEG_NMS_NO_SMOOTHING,
    "YOLOACTForInstanceSegmentationTRT": _E_ISEG_NMS_NO_SMOOTHING,
    "RFDetrForInstanceSegmentationTorch": _E_ISEG_CONF_MAXDET,
    "RFDetrForInstanceSegmentationOnnx": _E_ISEG_CONF_MAXDET,
    "RFDetrForInstanceSegmentationTRT": _E_ISEG_CONF_MAXDET,
    "YOLO26ForInstanceSegmentationOnnx": _E_ISEG_NO_SMOOTHING,
    "YOLO26ForInstanceSegmentationTorchScript": _E_ISEG_NO_SMOOTHING,
    "YOLO26ForInstanceSegmentationTRT": _E_ISEG_NO_SMOOTHING,
    # --- Semantic Segmentation ---
    "SemanticSegmentationModel": [
        (
            "infer",
            "infer",
            True,
            _p(_P_IMAGES, {"confidence": {"type": "float", "required": False}}),
            "validate_images_required",
            "serialize_semantic_segmentation_compact",
            "roboflow-semantic-segmentation-compact-v1",
        ),
    ],
    # --- Keypoints ---
    "KeyPointsDetectionModel": [
        (
            "infer",
            "infer",
            True,
            _p(_P_IMAGES, _K_KP),
            "validate_images_required",
            "serialize_keypoints_compact",
            "roboflow-keypoints-compact-v1",
        ),
    ],
    "RFDetrForKeyPointsONNX": _E_KP_CONF_THRESH,
    "YOLO26ForKeyPointsDetectionOnnx": _E_KP_CONF_THRESH,
    "YOLO26ForKeyPointsDetectionTorchScript": _E_KP_CONF_THRESH,
    "YOLO26ForKeyPointsDetectionTRT": _E_KP_CONF_THRESH,
    # --- Depth Estimation ---
    "DepthEstimationModel": [
        (
            "infer",
            "infer",
            True,
            _p(_P_IMAGES),
            "validate_images_required",
            "serialize_depth_compact",
            "roboflow-depth-compact-v1",
        ),
    ],
    # --- Documents / OCR ---
    "StructuredOCRModel": [
        (
            "infer",
            "infer",
            True,
            _p(_P_IMAGES),
            "validate_images_required",
            "serialize_structured_ocr_compact",
            "roboflow-structured-ocr-compact-v1",
        ),
    ],
    "EasyOCRTorch": [
        (
            "infer",
            "infer",
            True,
            _p(
                _P_IMAGES,
                {
                    "confidence": {"type": "float", "required": False},
                    "text_regions_separator": {"type": "str", "required": False},
                },
            ),
            "validate_images_required",
            "serialize_structured_ocr_compact",
            "roboflow-structured-ocr-compact-v1",
        ),
    ],
    # --- Embeddings ---
    "TextImageEmbeddingModel": [
        (
            "embed_images",
            "embed_images",
            True,
            _p(_P_IMAGES),
            "validate_images_required",
            "serialize_embeddings",
            "roboflow-embeddings-compact-v1",
        ),
        (
            "embed_text",
            "embed_text",
            False,
            _p(_P_TEXTS),
            "validate_texts_required",
            "serialize_embeddings",
            "roboflow-embeddings-compact-v1",
        ),
    ],
    # --- Gaze ---
    "L2CSNetOnnx": [
        (
            "infer",
            "infer",
            True,
            _p(_P_IMAGES),
            "validate_images_required",
            "serialize_gaze_compact",
            "roboflow-gaze-compact-v1",
        ),
    ],
    # --- VLM / Prompt models ---
    "PaliGemmaHF": [
        (
            "prompt",
            "prompt",
            True,
            _p(_P_VLM_PROMPT),
            "validate_images_and_prompt",
            "serialize_text",
            "roboflow-text-v1",
        ),
    ],
    "Gemma4HF": [
        (
            "prompt",
            "prompt",
            True,
            _p(_P_VLM_PROMPT, {"enable_thinking": {"type": "bool", "required": False}}),
            "validate_images_and_prompt",
            "serialize_text",
            "roboflow-text-v1",
        ),
    ],
    "Qwen25VLHF": [
        (
            "prompt",
            "prompt",
            True,
            _p(_P_VLM_PROMPT),
            "validate_images_and_prompt",
            "serialize_text",
            "roboflow-text-v1",
        ),
    ],
    "Qwen3VLHF": [
        (
            "prompt",
            "prompt",
            True,
            _p(_P_VLM_PROMPT),
            "validate_images_and_prompt",
            "serialize_text",
            "roboflow-text-v1",
        ),
    ],
    "Qwen35HF": [
        (
            "prompt",
            "prompt",
            True,
            _p(
                _P_VLM_PROMPT,
                {"enable_thinking": {"type": "bool", "required": False, "default": False}},
            ),
            "validate_images_and_prompt",
            "serialize_text",
            "roboflow-text-v1",
        ),
    ],
    "SmolVLMHF": [
        (
            "prompt",
            "prompt",
            True,
            _p(
                _P_VLM_PROMPT,
                {"images_to_single_prompt": {"type": "bool", "required": False, "default": True}},
            ),
            "validate_images_and_prompt",
            "serialize_text",
            "roboflow-text-v1",
        ),
    ],
    "Cosmos3EdgeReasoner": [
        (
            "prompt",
            "prompt",
            True,
            _p(_P_VLM_PROMPT, {"return_thinking": {"type": "bool", "required": False}}),
            "validate_images_and_prompt",
            "serialize_text",
            "roboflow-text-v1",
        ),
    ],
    # --- Florence2 ---
    "Florence2HF": [
        (
            "caption",
            "caption_image",
            True,
            _p(_P_IMAGES, {"granularity": {"type": "str", "required": False}}, _P_MAX_NEW_TOKENS),
            "validate_images_required",
            "serialize_text",
            "roboflow-text-v1",
        ),
        (
            "detect",
            "detect_objects",
            False,
            _p(
                _P_IMAGES,
                {
                    "labels_mode": {"type": "str", "required": False},
                    "classes": {"type": "list[str]", "required": False},
                },
                _P_MAX_NEW_TOKENS,
            ),
            "validate_images_required",
            "serialize_detections_compact",
            "roboflow-object-detection-compact-v1",
        ),
        (
            "ocr",
            "ocr_image",
            False,
            _p(_P_IMAGES, _P_MAX_NEW_TOKENS),
            "validate_images_required",
            "serialize_text",
            "roboflow-text-v1",
        ),
        (
            "parse_document",
            "parse_document",
            False,
            _p(_P_IMAGES, _P_MAX_NEW_TOKENS),
            "validate_images_required",
            "serialize_text",
            "roboflow-text-v1",
        ),
        (
            "prompt",
            "prompt",
            False,
            _p(_P_VLM_PROMPT),
            "validate_images_and_prompt",
            "serialize_text",
            "roboflow-text-v1",
        ),
        (
            "segment_phrase",
            "segment_phrase",
            False,
            _p(_P_FLORENCE2_PHRASE),
            "validate_images_and_prompt",
            "serialize_instance_segmentation_compact",
            "roboflow-instance-segmentation-compact-v1",
            {"prompt": "phrase"},
        ),
        (
            "ground_phrase",
            "ground_phrase",
            False,
            _p(_P_FLORENCE2_PHRASE),
            "validate_images_and_prompt",
            "serialize_detections_compact",
            "roboflow-object-detection-compact-v1",
            {"prompt": "phrase"},
        ),
        (
            "classify_region",
            "classify_image_region",
            False,
            _p(_P_FLORENCE2_REGION),
            "validate_images_required",
            "serialize_text",
            "roboflow-text-v1",
        ),
        (
            "caption_region",
            "caption_image_region",
            False,
            _p(_P_FLORENCE2_REGION),
            "validate_images_required",
            "serialize_text",
            "roboflow-text-v1",
        ),
        (
            "ocr_region",
            "ocr_image_region",
            False,
            _p(_P_FLORENCE2_REGION),
            "validate_images_required",
            "serialize_text",
            "roboflow-text-v1",
        ),
        (
            "segment_region",
            "segment_region",
            False,
            _p(_P_FLORENCE2_REGION),
            "validate_images_required",
            "serialize_text",
            "roboflow-text-v1",
        ),
    ],
    # --- SAM ---
    "SAMTorch": [
        (
            "embed",
            "embed_images",
            True,
            _p(_P_SAM_EMBED),
            "validate_images_required",
            "serialize_passthrough",
            "roboflow-sam-embeddings-v1",
        ),
        (
            "segment",
            "segment_images",
            False,
            _p(_P_SAM_SEGMENT),
            "validate_sam_segment",
            "serialize_sam_segmentation_compact",
            "roboflow-sam-segmentation-compact-v1",
        ),
    ],
    "SAM2Torch": [
        (
            "embed",
            "embed_images",
            True,
            _p(_P_SAM_EMBED),
            "validate_images_required",
            "serialize_passthrough",
            "roboflow-sam-embeddings-v1",
        ),
        (
            "segment",
            "segment_images",
            False,
            _p(_P_SAM2_SEGMENT),
            "validate_sam_segment",
            "serialize_sam_segmentation_compact",
            "roboflow-sam-segmentation-compact-v1",
        ),
    ],
    # --- Moondream2 ---
    "MoonDream2HF": [
        (
            "caption",
            "caption",
            True,
            _p(_P_IMAGES, {"length": {"type": "str", "required": False}}, _P_MAX_NEW_TOKENS),
            "validate_images_required",
            "serialize_text",
            "roboflow-text-v1",
        ),
        (
            "detect",
            "detect",
            False,
            _P_IMAGES_CLASSES_GEN,
            "validate_images_and_classes",
            "serialize_detections_compact",
            "roboflow-object-detection-compact-v1",
        ),
        (
            "query",
            "query",
            False,
            _p(_P_VLM_PROMPT_BASIC),
            "validate_images_and_prompt",
            "serialize_text",
            "roboflow-text-v1",
            {"prompt": "question"},
        ),
        (
            "point",
            "point",
            False,
            _P_IMAGES_CLASSES_GEN,
            "validate_images_and_classes",
            "serialize_keypoints_compact",
            "roboflow-keypoints-compact-v1",
        ),
        (
            "encode",
            "encode_images",
            False,
            _p(_P_IMAGES),
            "validate_images_required",
            "serialize_embeddings",
            "roboflow-embeddings-compact-v1",
        ),
    ],
    # --- GlmOCR ---
    "GlmOcrHF": [
        (
            "recognize_text",
            "recognize_text",
            True,
            _P_VLM_IMAGE_GEN,
            "validate_images_required",
            "serialize_text",
            "roboflow-text-v1",
        ),
        (
            "recognize_table",
            "recognize_table",
            False,
            _P_VLM_IMAGE_GEN,
            "validate_images_required",
            "serialize_text",
            "roboflow-text-v1",
        ),
        (
            "recognize_formula",
            "recognize_formula",
            False,
            _P_VLM_IMAGE_GEN,
            "validate_images_required",
            "serialize_text",
            "roboflow-text-v1",
        ),
        (
            "prompt",
            "prompt",
            False,
            _p(_P_VLM_PROMPT),
            "validate_images_and_prompt",
            "serialize_text",
            "roboflow-text-v1",
        ),
    ],
    # --- SAM2 RT (streaming) ---
    "SAM2ForStream": [
        (
            "prompt",
            "prompt",
            True,
            {
                "image": {"type": "image", "required": True},
                "bboxes": {"type": "list", "required": True},
                "state_dict": {"type": "dict", "required": False},
                "clear_old_points": {"type": "bool", "required": False, "default": True},
                "normalize_coords": {"type": "bool", "required": False, "default": True},
                "frame_idx": {"type": "int", "required": False, "default": 0},
            },
            "validate_passthrough",
            "serialize_passthrough",
            "roboflow-generic-v1",
        ),
        (
            "track",
            "track",
            False,
            {
                "image": {"type": "image", "required": True},
                "state_dict": {"type": "dict", "required": False},
            },
            "validate_passthrough",
            "serialize_passthrough",
            "roboflow-generic-v1",
        ),
    ],
    # --- SAM3 ---
    "SAM3Torch": [
        (
            "embed_images",
            "embed_images",
            True,
            _p(_P_SAM3_EMBED),
            "validate_images_required",
            "serialize_passthrough",
            "roboflow-sam3-embeddings-v1",
        ),
        (
            "segment_with_visual_prompts",
            "segment_with_visual_prompts",
            False,
            _p(_P_SAM3_VISUAL_PROMPTS),
            "validate_sam_segment",
            "serialize_passthrough",
            "roboflow-sam3-segmentation-v1",
        ),
        (
            "segment_with_text_prompts",
            "segment_with_text_prompts",
            False,
            _p(_P_SAM3_TEXT_PROMPTS),
            "validate_images_required",
            "serialize_passthrough",
            "roboflow-sam3-segmentation-v1",
        ),
    ],
    # --- Text-only OCR ---
    "TextOnlyOCRModel": [
        (
            "infer",
            "infer",
            True,
            _p(_P_IMAGES),
            "validate_images_required",
            "serialize_text",
            "roboflow-text-v1",
        ),
    ],
    # --- Passthrough (benchmark) ---
    "PassthroughModel": [
        (
            "infer",
            "infer",
            True,
            _p(_P_IMAGES),
            "validate_passthrough",
            "serialize_detections_compact",
            "roboflow-object-detection-compact-v1",
        ),
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
    # Under registry._lock so a second thread first-loading the same class
    # blocks until registration is complete, instead of returning early on the
    # dedup check and serving before the entries exist.
    with registry._lock:
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
    with registry._lock:
        key = ",".join(mro_names)
        if key in _registered_name_keys:
            return
        _registered_name_keys.add(key)

        for name in mro_names:
            config = _TASK_CONFIGS.get(name)
            if config is None:
                continue
            placeholder = type(name, (), {})
            for cfg in config:
                task_name, method, default, params, val_name, ser_name, resp_type, aliases = (
                    _unpack_config(cfg)
                )
                registry.register(
                    placeholder,
                    task_name,
                    method=method,
                    default=default,
                    params=params,
                    validator=_resolve_validator(val_name),
                    serializer=_resolve_serializer(ser_name),
                    response_type=resp_type,
                    param_aliases=aliases,
                )


_registered_name_keys: set[str] = set()


def _register_from_config(cls: type) -> None:
    """Register tasks for a single class if it has config."""
    config = _TASK_CONFIGS.get(cls.__name__)
    if config is None:
        return
    existing = set(registry.registered_tasks(cls))
    for cfg in config:
        task_name, method, default, params, val_name, ser_name, resp_type, aliases = (
            _unpack_config(cfg)
        )
        if task_name in existing:
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
            param_aliases=aliases,
        )
