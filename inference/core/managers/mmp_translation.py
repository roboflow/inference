"""Legacy <-> new-world translation for ModelManagerAdapter.

Everything the adapter needs to speak both dialects: the static mirror of the
new-world handler registry, request forwarding (image -> manager input), per-task
param mapping, native-prediction -> legacy-response repack, and error
translation. Imports from the new packages are lazy so the legacy stack never
pays for them while the gate is off.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pybase64

from inference.core.entities.responses.clip import (
    ClipCompareResponse,
    ClipEmbeddingResponse,
)
from inference.core.entities.responses.inference import (
    ClassificationInferenceResponse,
    InferenceResponseImage,
    InstanceSegmentationInferenceResponse,
    InstanceSegmentationPrediction,
    InstanceSegmentationRLEPrediction,
    Keypoint,
    KeypointsDetectionInferenceResponse,
    KeypointsPrediction,
    LMMInferenceResponse,
    MultiLabelClassificationInferenceResponse,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
    Point,
    SemanticSegmentationInferenceResponse,
    SemanticSegmentationPrediction,
)
from inference.core.entities.responses.ocr import OCRInferenceResponse
from inference.core.entities.responses.perception_encoder import (
    PerceptionEncoderCompareResponse,
    PerceptionEncoderEmbeddingResponse,
)
from inference.core.entities.responses.sam import (
    SamEmbeddingResponse,
    SamSegmentationResponse,
)
from inference.core.entities.responses.sam2 import (
    Sam2EmbeddingResponse,
    Sam2SegmentationPrediction,
    Sam2SegmentationResponse,
)
from inference.core.entities.responses.sam3 import (
    Sam3EmbeddingResponse,
    Sam3PromptEcho,
    Sam3PromptResult,
    Sam3SegmentationPrediction,
    Sam3SegmentationResponse,
)
from inference.core.env import CLIP_MAX_BATCH_SIZE, LEGACY_MMP_ADAPTER_MODE
from inference.core.exceptions import (
    InferenceModelNotFound,
    InferencePayloadTooLargeError,
    InputImageLoadError,
    InvalidImageTypeDeclared,
    InvalidModelIDError,
    ModelArtefactError,
    ModelDeploymentNotSupportedError,
    ModelManagerLockAcquisitionError,
    PostProcessingError,
    RoboflowAPIConnectionError,
    RoboflowAPINotAuthorizedError,
    RoboflowAPINotNotFoundError,
    RoboflowAPITimeoutError,
)
from inference.core.managers import mmp_florence2
from inference.core.registries.roboflow import GENERIC_MODELS
from inference.core.utils.image_utils import (
    BASE64_DATA_TYPE_PATTERN,
    convert_gray_image_to_bgr,
    encode_image_to_jpeg_bytes,
    fetch_image_bytes_from_url,
    load_image_from_numpy_object,
    load_image_from_numpy_str,
    load_image_rgb,
)
from inference.core.utils.postprocess import (
    cosine_similarity,
    mask2poly,
    masks2multipoly,
    masks2poly,
)
from inference.core.utils.roboflow import get_model_id_chunks
from inference.core.utils.visualisation import draw_detection_predictions
from inference_models.utils.performance import performance_profiler

# Static mirror of the (model_type, action) pairs registered by
# inference_server/handlers/*/description.py. Kept as literals so the legacy
# side never imports the handler modules; a parity test asserts no drift.
NEW_WORLD_HANDLERS = frozenset(
    [
        ("classification", "infer"),
        ("depth-estimation", "infer"),
        ("embedding", "compare"),
        ("embedding", "embed_images"),
        ("embedding", "embed_text"),
        ("gaze-detection", "infer"),
        ("instance-segmentation", "infer"),
        ("interactive-instance-segmentation", "embed"),
        ("interactive-instance-segmentation", "embed_images"),
        ("interactive-instance-segmentation", "segment"),
        ("interactive-instance-segmentation", "segment_with_text_prompts"),
        ("interactive-instance-segmentation", "segment_with_visual_prompts"),
        ("keypoint-detection", "infer"),
        ("multi-label-classification", "infer"),
        ("object-detection", "infer"),
        ("open-vocabulary-object-detection", "infer"),
        ("open-vocabulary-object-detection", "infer_with_reference_examples"),
        ("passthrough", "infer"),
        ("semantic-segmentation", "infer"),
        ("structured-ocr", "infer"),
        ("text-only-ocr", "infer"),
        ("vlm", "caption"),
        ("vlm", "caption_image"),
        ("vlm", "caption_image_region"),
        ("vlm", "caption_region"),
        ("vlm", "classify_region"),
        ("vlm", "detect"),
        ("vlm", "detect_objects"),
        ("vlm", "encode"),
        ("vlm", "encode_images"),
        ("vlm", "ground_phrase"),
        ("vlm", "ocr"),
        ("vlm", "ocr_image"),
        ("vlm", "ocr_region"),
        ("vlm", "parse_document"),
        ("vlm", "point"),
        ("vlm", "prompt"),
        ("vlm", "query"),
        ("vlm", "recognize_formula"),
        ("vlm", "recognize_table"),
        ("vlm", "recognize_text"),
        ("vlm", "segment_phrase"),
    ]
)

# Pairs the adapter can actually translate today; grows per rollout phase.
IMPLEMENTED_ROUTES = frozenset(
    [
        ("classification", "infer"),
        ("depth-estimation", "infer"),
        ("embedding", "compare"),
        ("embedding", "embed_images"),
        ("embedding", "embed_text"),
        ("instance-segmentation", "infer"),
        ("keypoint-detection", "infer"),
        ("multi-label-classification", "infer"),
        ("interactive-instance-segmentation", "embed"),
        ("interactive-instance-segmentation", "embed_images"),
        ("interactive-instance-segmentation", "segment"),
        ("interactive-instance-segmentation", "segment_with_text_prompts"),
        ("interactive-instance-segmentation", "segment_with_visual_prompts"),
        ("object-detection", "infer"),
        ("open-vocabulary-object-detection", "infer"),
        ("semantic-segmentation", "infer"),
        ("structured-ocr", "infer"),
        ("text-only-ocr", "infer"),
        ("vlm", "detect"),
        ("vlm", "prompt"),
    ]
)

# VLM model classes whose legacy response contract the adapter cannot satisfy.
VLM_UNSUPPORTED_MODEL_CLASSES = frozenset()

OWLV2_BACKED_MODEL_CLASSES = frozenset(["OWLv2HF", "RoboflowInstantHF"])

# Moondream2 registers typed actions (caption/detect/query/point) and no
# generic prompt action; the legacy /infer/lmm surface routes every request
# through predict -> detect, so the adapter mirrors that single path.
MOONDREAM_BACKED_MODEL_CLASSES = frozenset(["MoonDream2HF"])

IMPLEMENTED_TASK_TYPES = frozenset(task_type for task_type, _ in IMPLEMENTED_ROUTES)


def implemented_actions(task_type: str) -> frozenset:
    return frozenset(action for t, action in IMPLEMENTED_ROUTES if t == task_type)


# Explicit foundation endpoints supply a concrete action via the request type,
# overriding the task type's default action. Candidates are checked against
# the model's registered tasks in order.
_ACTION_CANDIDATES_BY_REQUEST_TYPE: Dict[str, Tuple[str, ...]] = {
    "SamEmbeddingRequest": ("embed",),
    "SamSegmentationRequest": ("segment",),
    "Sam2EmbeddingRequest": ("embed", "embed_images"),
    "Sam2SegmentationRequest": ("segment_with_visual_prompts", "segment"),
    "Sam3SegmentationRequest": ("segment_with_text_prompts",),
    "ClipImageEmbeddingRequest": ("embed_images",),
    "ClipTextEmbeddingRequest": ("embed_text",),
    "ClipCompareRequest": ("compare",),
    "PerceptionEncoderImageEmbeddingRequest": ("embed_images",),
    "PerceptionEncoderTextEmbeddingRequest": ("embed_text",),
    "PerceptionEncoderCompareRequest": ("compare",),
}


def resolve_request_action(route: dict, request: Any) -> str:
    if _is_moondream_backed(route) and "detect" in (route.get("tasks") or set()):
        return "detect"
    candidates = _ACTION_CANDIDATES_BY_REQUEST_TYPE.get(type(request).__name__)
    if not candidates:
        return route["action"]
    tasks = route.get("tasks") or set()
    for candidate in candidates:
        if candidate in tasks:
            return candidate
    return candidates[0]


# Error codes returned by the MMP in ("error", code) lifecycle tuples;
# values mirror inference_model_manager.model_manager_process.
_MMP_ERR_POOL_FULL = 1
_MMP_ERR_NO_BACKEND = 2
_MMP_ERR_STALE = 3
_MMP_ERR_BACKEND = 4
_MMP_ERR_LOAD_FAILED = 5
_MMP_ERR_NOT_LOADED = 6
_MMP_ERR_SERVER_FULL = 7

_RETRYABLE_ERR_CODES = {_MMP_ERR_POOL_FULL, _MMP_ERR_STALE, _MMP_ERR_SERVER_FULL}

_DISABLE_PREPROC_FIELDS = (
    "disable_preproc_auto_orient",
    "disable_preproc_contrast",
    "disable_preproc_grayscale",
    "disable_preproc_static_crop",
)
_OD_MAX_CANDIDATES_DEFAULT = 3000


_GENERIC_MODEL_TASK_TYPES: Dict[Tuple[str, str], str] = {
    ("embed", "clip"): "embedding",
    ("embed", "perception_encoder"): "embedding",
    ("embed", "sam"): "interactive-instance-segmentation",
    ("embed", "sam2"): "interactive-instance-segmentation",
    ("ocr", "doctr"): "structured-ocr",
    ("ocr", "easy_ocr"): "structured-ocr",
    ("ocr", "trocr"): "text-only-ocr",
}

_PLATFORM_TASK_TYPE_ALIASES: Dict[str, str] = {}

_MMP_CORE_DATASET_ALIASES: Dict[str, str] = {
    "perception_encoder": "perception-encoder",
}

_LEGACY_CORE_DATASET_BY_MMP = {
    canonical: legacy for legacy, canonical in _MMP_CORE_DATASET_ALIASES.items()
}


def canonical_mmp_model_id(model_id: str) -> str:
    dataset_id, separator, rest = model_id.partition("/")
    canonical = _MMP_CORE_DATASET_ALIASES.get(dataset_id)
    if canonical is None:
        return model_id
    return f"{canonical}{separator}{rest}"


_DEFAULT_ACTION_BY_TASK_TYPE: Dict[str, str] = {
    "embedding": "embed_images",
    "interactive-instance-segmentation": "embed",
}


def _default_action_for(task_type: str) -> str:
    return _DEFAULT_ACTION_BY_TASK_TYPE.get(task_type, "infer")


def _generic_model_task_type(model_id: str) -> Optional[str]:
    dataset_id, separator, rest = model_id.partition("/")
    legacy_dataset = _LEGACY_CORE_DATASET_BY_MMP.get(dataset_id)
    if legacy_dataset is not None:
        model_id = f"{legacy_dataset}{separator}{rest}"
    entry = GENERIC_MODELS.get(model_id)
    if entry is None:
        try:
            dataset_id, _ = get_model_id_chunks(model_id=model_id)
        except InvalidModelIDError:
            return None
        entry = GENERIC_MODELS.get(dataset_id)
    if entry is None:
        return None
    return _GENERIC_MODEL_TASK_TYPES.get(entry)


async def stat_model(model_id: str, api_key: str) -> Tuple[str, str]:
    """Resolve (task_type, default_action): generic core ids locally, the
    rest via the new world's stat + auth."""
    generic_task_type = _generic_model_task_type(model_id)
    if generic_task_type is not None:
        return generic_task_type, _default_action_for(generic_task_type)
    from inference_server.framework.entities import CommonRequestParams
    from inference_server.framework.model_stat import stat_model_while_checking_auth

    try:
        task_type, default_action = await stat_model_while_checking_auth(
            CommonRequestParams(model_id=model_id, api_key=api_key)
        )
    except PermissionError as error:
        raise RoboflowAPINotAuthorizedError(str(error)) from error
    except LookupError as error:
        raise RoboflowAPINotNotFoundError(str(error)) from error
    except RuntimeError as error:
        raise RoboflowAPIConnectionError(str(error)) from error
    canonical_task_type = _PLATFORM_TASK_TYPE_ALIASES.get(task_type)
    if canonical_task_type is not None:
        return canonical_task_type, _default_action_for(canonical_task_type)
    return task_type, default_action


def raise_for_lifecycle_result(result: tuple, model_id: str) -> None:
    """Translate MMP load/ensure_loaded status tuples into legacy exceptions."""
    kind = result[0]
    if kind in ("ok", "model_ready"):
        return
    if kind == "load_timeout":
        raise InferenceModelNotFound(
            f"Model {model_id} is still loading - retry request."
        )
    if kind == "error":
        code = result[1] if len(result) > 1 else None
        if code in _RETRYABLE_ERR_CODES:
            raise ModelManagerLockAcquisitionError(
                f"Inference backend is busy for model {model_id} (code {code})."
            )
        if code == _MMP_ERR_NO_BACKEND:
            raise ModelDeploymentNotSupportedError(
                f"No inference backend can serve model {model_id}."
            )
        if code == _MMP_ERR_NOT_LOADED:
            raise InferenceModelNotFound(f"Model with id {model_id} not loaded.")
        raise ModelArtefactError(
            f"Inference backend failed for model {model_id} (code {code})."
        )
    raise ModelArtefactError(
        f"Unexpected inference backend response for model {model_id}: {result!r}."
    )


def translate_infer_error(error: Exception, model_id: str) -> Exception:
    """Map new-world infer exceptions to legacy ones; returns input if unmapped.

    Matched by class name so the legacy stack never imports inference_server
    just to classify an exception.
    """
    if isinstance(error, asyncio.TimeoutError):
        return RoboflowAPITimeoutError(
            f"Timed out waiting for inference result for model {model_id}."
        )
    name = type(error).__name__
    if name == "ServerBusyError":
        return ModelManagerLockAcquisitionError(str(error))
    if name == "PayloadTooLargeError":
        return InferencePayloadTooLargeError(str(error))
    return error


def ensure_request_supported(
    model_id: str, request: Any, route: Optional[dict] = None
) -> None:
    """Reject fidelity-breaking legacy-only params instead of silently drifting."""
    if route is not None and mmp_florence2.is_florence2_route(route):
        mmp_florence2.ensure_image_input_supported(request)
        return
    for field in _DISABLE_PREPROC_FIELDS:
        if getattr(request, field, False):
            raise ModelDeploymentNotSupportedError(
                f"{field} is not supported for model '{model_id}' on the MMP path."
            )
    max_candidates = getattr(request, "max_candidates", None)
    if max_candidates is not None and max_candidates != _OD_MAX_CANDIDATES_DEFAULT:
        raise ModelDeploymentNotSupportedError(
            f"max_candidates is not supported for model '{model_id}' on the MMP path."
        )
    mask_decode_mode = getattr(request, "mask_decode_mode", None)
    if mask_decode_mode is not None and mask_decode_mode != "accurate":
        raise ModelDeploymentNotSupportedError(
            f"mask_decode_mode={mask_decode_mode!r} is not supported for model "
            f"'{model_id}' on the MMP path."
        )
    tradeoff_factor = getattr(request, "tradeoff_factor", None)
    if tradeoff_factor:
        raise ModelDeploymentNotSupportedError(
            f"tradeoff_factor is not supported for model '{model_id}' on the MMP path."
        )


def _numeric_confidence(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelDeploymentNotSupportedError(
            f"confidence={value!r} is not supported on the MMP path."
        )
    return float(value)


def _roboflow_confidence(value: Any) -> Optional[Union[float, str]]:
    if isinstance(value, str) and value in ("best", "default"):
        return value
    return _numeric_confidence(value)


def build_task_params(
    task_type: str, action: str, request: Any, route: Optional[dict] = None
) -> dict:
    params: dict = {}
    if task_type == "interactive-instance-segmentation":
        return _build_interactive_segmentation_params(action, request)
    if task_type == "vlm":
        if route is not None and mmp_florence2.is_florence2_route(route):
            return mmp_florence2.build_prompt_params(request)
        if route is not None and _is_moondream_backed(route):
            return {"classes": [getattr(request, "prompt", None)]}
        return _build_vlm_params(request)
    if task_type == "structured-ocr":
        _ensure_ocr_request_supported(request)
        return params
    if task_type == "text-only-ocr":
        return params
    if task_type == "open-vocabulary-object-detection":
        return _build_open_vocabulary_params(request)
    if task_type == "depth-estimation":
        return params
    confidence = _roboflow_confidence(getattr(request, "confidence", None))
    if confidence is not None:
        params["confidence"] = confidence
    if task_type in (
        "classification",
        "multi-label-classification",
        "semantic-segmentation",
    ):
        return params
    iou_threshold = getattr(request, "iou_threshold", None)
    if iou_threshold is not None:
        params["iou_threshold"] = float(iou_threshold)
    max_detections = getattr(request, "max_detections", None)
    if max_detections is not None:
        params["max_detections"] = int(max_detections)
    class_agnostic_nms = getattr(request, "class_agnostic_nms", None)
    if class_agnostic_nms is not None:
        params["class_agnostic_nms"] = bool(class_agnostic_nms)
    if task_type == "keypoint-detection":
        keypoint_confidence = getattr(request, "keypoint_confidence", None)
        if keypoint_confidence is not None:
            params["key_points_threshold"] = float(keypoint_confidence)
    return params


def _namespaced_client_hash(image_id: str, request: Any) -> str:
    from inference_model_manager.hash_namespacing import namespace_client_hash_id

    return namespace_client_hash_id(image_id, getattr(request, "api_key", None))


def _strip_client_hash_namespace(hash_id: str, request: Any) -> str:
    from inference_model_manager.hash_namespacing import strip_tenant_namespace

    return strip_tenant_namespace(hash_id, getattr(request, "api_key", None))


def _build_interactive_segmentation_params(action: str, request: Any) -> dict:
    if action in ("embed", "embed_images"):
        params: dict = {}
        image_id = getattr(request, "image_id", None)
        if image_id:
            params["image_hashes"] = [_namespaced_client_hash(image_id, request)]
        if action == "embed_images":
            # SAM3 processor state is far larger than an SHM slot; the legacy
            # response only echoes the image id, so leave embeddings in the
            # worker-side cache.
            params["return_embeddings"] = False
        return params
    if action == "segment":
        if type(request).__name__ == "SamSegmentationRequest":
            return _build_sam_segment_params(request)
        return _build_sam2_segment_params(request)
    if action == "segment_with_visual_prompts":
        return _build_visual_prompt_params(request)
    if action == "segment_with_text_prompts":
        return _build_text_prompt_params(request)
    raise ModelDeploymentNotSupportedError(
        f"SAM action '{action}' is not supported on the MMP path."
    )


def _build_sam_segment_params(request: Any) -> dict:
    if getattr(request, "embeddings", None):
        raise ModelDeploymentNotSupportedError(
            "embeddings input is not supported on the MMP path."
        )
    image = getattr(request, "image", None)
    image_id = getattr(request, "image_id", None)
    if not image and not image_id:
        raise ValueError("Must provide either image, cached image_id, or embeddings")
    params: dict = {"multi_mask_output": False}
    if getattr(request, "has_mask_input", False):
        if getattr(request, "mask_input", None) is not None:
            raise ModelDeploymentNotSupportedError(
                "mask_input is not supported on the MMP path."
            )
        if not getattr(request, "use_mask_input_cache", True):
            raise ModelDeploymentNotSupportedError(
                "has_mask_input without use_mask_input_cache is not supported on "
                "the MMP path."
            )
        if not image_id:
            raise ValueError("Must provide either mask_input or cached image_id")
        params["enforce_mask_input"] = True
    point_coords = getattr(request, "point_coords", None)
    if point_coords is not None:
        params["point_coordinates"] = [[list(point) for point in point_coords]]
    point_labels = getattr(request, "point_labels", None)
    if point_labels is not None:
        params["point_labels"] = [list(point_labels)]
    if image_id:
        params["image_hashes"] = [_namespaced_client_hash(image_id, request)]
    response_format = getattr(request, "format", None)
    if response_format == "binary":
        raise ModelDeploymentNotSupportedError(
            "format='binary' is not supported on the MMP path."
        )
    if response_format != "json":
        raise ValueError(f"Invalid format {response_format}")
    return params


def _build_sam2_segment_params(request: Any) -> dict:
    response_format = getattr(request, "format", None)
    if response_format not in ("json", "rle", "binary"):
        raise ValueError(f"Invalid format {response_format}")
    if response_format == "binary":
        raise ModelDeploymentNotSupportedError(
            "format='binary' is not supported on the MMP path."
        )
    params = _build_visual_prompt_params(request)
    if not any(key in params for key in ("point_coordinates", "point_labels", "boxes")):
        params["point_coordinates"] = [[[0, 0]]]
        params["point_labels"] = [[-1]]
    params["return_logits"] = True
    return params


def _build_visual_prompt_params(request: Any) -> dict:
    if getattr(request, "mask_input", None) is not None or getattr(
        request, "has_mask_input", False
    ):
        raise ModelDeploymentNotSupportedError(
            "mask_input is not supported on the MMP path."
        )
    for field in ("save_logits_to_cache", "load_logits_from_cache"):
        if getattr(request, field, False):
            raise ModelDeploymentNotSupportedError(
                f"{field} is not supported on the MMP path."
            )
    prompts = getattr(request, "prompts", None)
    if prompts is not None:
        args = prompts.to_sam2_inputs()
    else:
        args = {"point_coords": None, "point_labels": None, "box": None}
    point_coords = args.get("point_coords")
    point_labels = args.get("point_labels")
    boxes = args.get("box")
    if point_coords or point_labels:
        point_coords, point_labels = _pad_points(point_coords, point_labels)
    params: dict = {
        "multi_mask_output": bool(getattr(request, "multimask_output", True)),
    }
    if point_coords:
        params["point_coordinates"] = [point_coords]
    if point_labels:
        params["point_labels"] = [point_labels]
    if boxes:
        params["boxes"] = [boxes]
    image_id = getattr(request, "image_id", None)
    if image_id:
        params["image_hashes"] = [_namespaced_client_hash(image_id, request)]
    return params


def _pad_points(
    point_coords: Optional[List[list]], point_labels: Optional[List[list]]
) -> Tuple[Optional[List[list]], Optional[List[list]]]:
    if not point_coords or not point_labels:
        return point_coords, point_labels
    max_len = max(len(coords) for coords in point_coords)
    padded_coords = [
        list(coords) + [[0, 0]] * (max_len - len(coords)) for coords in point_coords
    ]
    padded_labels = [
        list(labels) + [-1] * (max_len - len(labels)) for labels in point_labels
    ]
    return padded_coords, padded_labels


def _build_text_prompt_params(request: Any) -> dict:
    if getattr(request, "nms_iou_threshold", None) is not None:
        raise ModelDeploymentNotSupportedError(
            "nms_iou_threshold is not supported on the MMP path."
        )
    prompts = getattr(request, "prompts", None)
    if not prompts:
        raise ModelDeploymentNotSupportedError(
            "SAM3 concept segmentation requires prompts on the MMP path."
        )
    # The worker applies a single threshold floor; forward the min of the
    # request and per-prompt thresholds so per-prompt refinement in the
    # repack still has the masks to filter.
    threshold = float(getattr(request, "output_prob_thresh", None) or 0.5)
    for prompt in prompts:
        prompt_threshold = getattr(prompt, "output_prob_thresh", None)
        if prompt_threshold is not None:
            threshold = min(threshold, float(prompt_threshold))
    return {
        "prompts": [prompt.dict() for prompt in prompts],
        "output_prob_thresh": threshold,
    }


def _build_vlm_params(request: Any) -> dict:
    prompt = getattr(request, "prompt", None)
    if not prompt:
        raise ModelDeploymentNotSupportedError(
            "VLM inference requires a prompt on the MMP path."
        )
    params: dict = {"prompt": prompt}
    max_new_tokens = getattr(request, "max_new_tokens", None)
    if max_new_tokens is not None:
        params["max_new_tokens"] = int(max_new_tokens)
    if getattr(request, "enable_thinking", False):
        params["enable_thinking"] = True
    return params


def _ensure_ocr_request_supported(request: Any) -> None:
    language_codes = getattr(request, "language_codes", None)
    if language_codes is not None and list(language_codes) != ["en"]:
        raise ModelDeploymentNotSupportedError(
            "language_codes other than ['en'] are not supported on the MMP path."
        )
    if getattr(request, "quantize", False):
        raise ModelDeploymentNotSupportedError(
            "quantize is not supported on the MMP path."
        )


def _build_open_vocabulary_params(request: Any) -> dict:
    classes = getattr(request, "text", None) or getattr(request, "classes", None)
    if getattr(request, "training_data", None) is not None:
        raise ModelDeploymentNotSupportedError(
            "Few-shot detection with training_data is not supported on the MMP path."
        )
    if not classes:
        raise ModelDeploymentNotSupportedError(
            "Open-vocabulary detection requires a list of classes on the MMP path."
        )
    for field, default in (("box_threshold", 0.5), ("text_threshold", 0.5)):
        value = getattr(request, field, None)
        if value is not None and value != default:
            raise ModelDeploymentNotSupportedError(
                f"{field} is not supported on the MMP path."
            )
    params: dict = {"classes": [str(c) for c in classes]}
    confidence = _numeric_confidence(getattr(request, "confidence", None))
    if confidence is not None:
        params["confidence"] = confidence
    class_agnostic_nms = getattr(request, "class_agnostic_nms", None)
    if class_agnostic_nms is not None:
        params["class_agnostic_nms"] = bool(class_agnostic_nms)
    return params


def forward_image(
    image: Any,
) -> Tuple[Union[bytes, np.ndarray], Tuple[int, int]]:
    """InferenceRequestImage -> (manager input, (width, height)).

    Bundled mode preserves a ``numpy_object`` as a contiguous BGR array.
    External MMP receives the same array as ``.npy`` bytes over SHM.
    """
    started = performance_profiler.start()
    performance_profiler.increment("adapter.image_forward.calls")
    try:
        if isinstance(image, dict):
            image_type = image.get("type")
            value = image.get("value")
        else:
            image_type = getattr(image, "type", None)
            value = getattr(image, "value", None)
        performance_profiler.set_metadata("adapter.image_type", image_type)
        if image_type == "base64":
            data = _decode_base64_payload(value)
            dims = _dims_from_header(data)
            _record_forwarded_image(data, dims)
            return data, dims
        if image_type == "url":
            data = fetch_image_bytes_from_url(value=value)
            dims = _dims_from_header(data)
            _record_forwarded_image(data, dims)
            return data, dims
        if image_type == "numpy":
            if isinstance(value, np.ndarray):
                array = load_image_from_numpy_object(value)
            else:
                array = load_image_from_numpy_str(value)
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            data = buffer.getvalue()
            dims = (int(array.shape[1]), int(array.shape[0]))
            _record_forwarded_image(data, dims)
            return data, dims
        if image_type == "numpy_object":
            array = load_image_from_numpy_object(value)
            performance_profiler.record(
                "adapter.image.input_bytes", array.nbytes, "bytes"
            )
            array = np.ascontiguousarray(convert_gray_image_to_bgr(array))
            dims = (int(array.shape[1]), int(array.shape[0]))
            if LEGACY_MMP_ADAPTER_MODE == "bundled":
                performance_profiler.record(
                    "adapter.image.pixels", dims[0] * dims[1], "pixels"
                )
                return array, dims
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            data = buffer.getvalue()
            _record_forwarded_image(data, dims)
            return data, dims
        raise InvalidImageTypeDeclared(
            message=f"Image type '{image_type}' is not supported on the MMP path.",
            public_message=f"Image type '{image_type}' is not supported on the MMP path.",
        )
    finally:
        performance_profiler.stop("adapter.image_forward", started)


def _record_forwarded_image(data: bytes, dims: Tuple[int, int]) -> None:
    performance_profiler.record("adapter.image.encoded_bytes", len(data), "bytes")
    performance_profiler.record("adapter.image.pixels", dims[0] * dims[1], "pixels")


_EMBEDDING_RESPONSE_CLASSES = {
    "ClipImageEmbeddingRequest": ClipEmbeddingResponse,
    "ClipTextEmbeddingRequest": ClipEmbeddingResponse,
    "ClipCompareRequest": ClipCompareResponse,
    "PerceptionEncoderImageEmbeddingRequest": PerceptionEncoderEmbeddingResponse,
    "PerceptionEncoderTextEmbeddingRequest": PerceptionEncoderEmbeddingResponse,
    "PerceptionEncoderCompareRequest": PerceptionEncoderCompareResponse,
}


def _embedding_response_class(request: Any):
    response_class = _EMBEDDING_RESPONSE_CLASSES.get(type(request).__name__)
    if response_class is None:
        raise ModelDeploymentNotSupportedError(
            f"Request type {type(request).__name__} is not supported for embedding "
            f"models on the MMP path."
        )
    return response_class


def _embed_image_call(image: Any) -> dict:
    data, _ = forward_image(image)
    return {"task": "embed_images", "image": data, "params": {}}


def _embed_text_call(texts: List[str]) -> dict:
    return {"task": "embed_text", "image": None, "params": {"texts": list(texts)}}


def build_embedding_calls(
    action: str, request: Any
) -> Tuple[List[dict], Optional[List[str]]]:
    """Embedding request -> ordered MMP calls; compare puts the subject first.

    Returns (calls, prompt_keys); prompt_keys is set only for compare requests
    with a named-prompt mapping, mirroring the legacy dict-similarity variant.
    """
    _embedding_response_class(request)
    if action == "embed_images":
        if isinstance(request.image, list):
            if len(request.image) > CLIP_MAX_BATCH_SIZE:
                raise ValueError(
                    f"The maximum number of images that can be embedded at once is "
                    f"{CLIP_MAX_BATCH_SIZE}"
                )
            images = request.image
        else:
            images = [request.image]
        return [_embed_image_call(image) for image in images], None
    if action == "embed_text":
        texts = request.text if isinstance(request.text, list) else [request.text]
        return [_embed_text_call(texts)], None
    if action != "compare":
        raise ModelDeploymentNotSupportedError(
            f"Embedding action '{action}' is not supported on the MMP path."
        )
    if request.subject_type not in ("image", "text"):
        raise ValueError("subject_type must be either 'image' or 'text'")
    prompt = request.prompt
    prompt_keys = None
    if isinstance(prompt, dict) and not ("type" in prompt and "value" in prompt):
        prompt_keys = list(prompt.keys())
        prompt = [prompt[key] for key in prompt_keys]
    elif not isinstance(prompt, list):
        prompt = [prompt]
    if len(prompt) > CLIP_MAX_BATCH_SIZE:
        raise ValueError(
            f"The maximum number of prompts that can be compared at once is "
            f"{CLIP_MAX_BATCH_SIZE}"
        )
    if request.subject_type == "image":
        calls = [_embed_image_call(request.subject)]
    else:
        calls = [_embed_text_call([request.subject])]
    if request.prompt_type == "image":
        calls.extend(_embed_image_call(image) for image in prompt)
    elif request.prompt_type == "text":
        calls.append(_embed_text_call(prompt))
    else:
        raise ValueError("prompt_type must be either 'image' or 'text'")
    return calls, prompt_keys


def repack_embedding_response(
    action: str,
    request: Any,
    results: List[Any],
    prompt_keys: Optional[List[str]] = None,
):
    response_class = _embedding_response_class(request)
    if action in ("embed_images", "embed_text"):
        return response_class(embeddings=_stack_embeddings(results).tolist())
    subject = _stack_embeddings(results[:1]).reshape(-1)
    prompts = _stack_embeddings(results[1:])
    similarities = [float(cosine_similarity(subject, row)) for row in prompts]
    if prompt_keys is not None:
        similarities = dict(zip(prompt_keys, similarities))
    return response_class(similarity=similarities)


def _stack_embeddings(results: List[Any]) -> np.ndarray:
    arrays = []
    for result in results:
        array = np.asarray(result, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        arrays.append(array)
    return np.concatenate(arrays, axis=0)


def repack_prediction(
    task_type: str,
    action: str,
    prediction: Any,
    dims: Tuple[int, int],
    route: dict,
    request: Any,
):
    class_names = route.get("class_names")
    if task_type == "object-detection":
        return repack_object_detection_response(prediction, dims, class_names, request)
    if task_type == "open-vocabulary-object-detection":
        requested_classes = [
            str(c)
            for c in (
                getattr(request, "text", None)
                or getattr(request, "classes", None)
                or []
            )
        ]
        return repack_object_detection_response(
            prediction, dims, requested_classes, request
        )
    if task_type == "instance-segmentation":
        return repack_instance_segmentation_response(
            prediction, dims, class_names, request
        )
    if task_type == "keypoint-detection":
        return repack_keypoints_response(
            prediction, dims, class_names, route.get("key_points_classes"), request
        )
    if task_type == "classification":
        return repack_classification_response(prediction, dims, class_names, request)
    if task_type == "multi-label-classification":
        return repack_multi_label_classification_response(
            prediction, dims, class_names, request
        )
    if task_type == "semantic-segmentation":
        return repack_semantic_segmentation_response(
            prediction, dims, class_names, request
        )
    if task_type == "depth-estimation":
        return repack_depth_estimation_response(prediction)
    if task_type == "structured-ocr":
        return repack_structured_ocr_response(prediction, dims, class_names, request)
    if task_type == "text-only-ocr":
        return repack_text_ocr_response(prediction, dims)
    if task_type == "interactive-instance-segmentation":
        return repack_interactive_segmentation_response(action, prediction, request)
    if task_type == "vlm":
        if mmp_florence2.is_florence2_route(route):
            return mmp_florence2.repack_response(
                _unwrap_single_prediction(prediction), request, dims
            )
        if _is_moondream_backed(route):
            return _repack_moondream_detection(prediction, request, dims)
        return repack_vlm_response(prediction, dims)
    raise ModelDeploymentNotSupportedError(
        f"No response translation for task type '{task_type}' on the MMP path."
    )


_DETECTION_VISUALIZATION_TASK_TYPES = frozenset(
    ["object-detection", "instance-segmentation", "keypoint-detection"]
)


def render_visualization(
    task_type: str, request: Any, response: Any, route: dict
) -> bytes:
    if (
        task_type in _DETECTION_VISUALIZATION_TASK_TYPES
        or task_type == "open-vocabulary-object-detection"
    ):
        if _is_owlv2_backed(route):
            colors = _class_color_mapping(
                sorted({p.class_name for p in response.predictions})
            )
        else:
            colors = _model_class_colors(route, request)
        return draw_detection_predictions(
            inference_request=request,
            inference_response=response,
            colors=colors,
        )
    if task_type in ("classification", "multi-label-classification"):
        return _draw_classification_predictions(
            request, response, _model_class_colors(route, request)
        )
    raise ModelDeploymentNotSupportedError(
        f"visualize_predictions / format=image is not supported for task type "
        f"'{task_type}' on the MMP path."
    )


def _is_owlv2_backed(route: dict) -> bool:
    if OWLV2_BACKED_MODEL_CLASSES.intersection(route.get("model_mro_names") or []):
        return True
    return route.get("model_class_name") in OWLV2_BACKED_MODEL_CLASSES


def _is_moondream_backed(route: dict) -> bool:
    if MOONDREAM_BACKED_MODEL_CLASSES.intersection(route.get("model_mro_names") or []):
        return True
    return route.get("model_class_name") in MOONDREAM_BACKED_MODEL_CLASSES


def _repack_moondream_detection(
    prediction: Any, request: Any, dims: Tuple[int, int]
) -> ObjectDetectionInferenceResponse:
    detections = _unwrap_single_prediction(prediction)
    xyxy = np.asarray(detections.xyxy, dtype=float).reshape(-1, 4)
    prompt = getattr(request, "prompt", None)
    predictions: List[ObjectDetectionPrediction] = []
    for x1, y1, x2, y2 in xyxy:
        predictions.append(
            ObjectDetectionPrediction(
                x=(float(x1) + float(x2)) / 2.0,
                y=(float(y1) + float(y2)) / 2.0,
                width=float(x2) - float(x1),
                height=float(y2) - float(y1),
                confidence=1.0,
                **{"class": prompt if prompt is not None else ""},
                class_id=0,
            )
        )
    width, height = dims
    return ObjectDetectionInferenceResponse(
        predictions=predictions,
        image=InferenceResponseImage(width=width, height=height),
    )


def _class_color_mapping(class_names: Optional[List[str]]) -> Dict[str, str]:
    from inference.core.models.roboflow import get_color_mapping_from_environment

    return get_color_mapping_from_environment(
        environment=None, class_names=class_names or []
    )


def _model_class_colors(route: dict, request: Any) -> Dict[str, str]:
    colors = route.get("class_colors")
    if colors is None:
        from inference.core.models.roboflow import get_color_mapping_from_environment

        colors = get_color_mapping_from_environment(
            environment=_fetch_color_environment(
                route.get("mmp_model_id"), getattr(request, "api_key", None)
            ),
            class_names=route.get("class_names") or [],
        )
        route["class_colors"] = colors
    return colors


def _fetch_color_environment(
    model_id: Optional[str], api_key: Optional[str]
) -> Optional[dict]:
    if not model_id:
        return None
    try:
        from inference.core.devices.utils import GLOBAL_DEVICE_ID
        from inference.core.registries.roboflow import ModelEndpointType
        from inference.core.roboflow_api import (
            get_roboflow_instant_model_data,
            get_roboflow_model_data,
        )

        _, version_id = get_model_id_chunks(model_id=model_id)
        if version_id is not None:
            api_data = (
                get_roboflow_model_data(
                    api_key=api_key,
                    model_id=model_id,
                    endpoint_type=ModelEndpointType.ORT,
                    device_id=GLOBAL_DEVICE_ID,
                ).get("ort")
                or {}
            )
        else:
            api_data = (
                get_roboflow_instant_model_data(api_key=api_key, model_id=model_id)
                or {}
            )
        colors = api_data.get("colors")
        if isinstance(colors, dict):
            return {"COLORS": colors}
        return None
    except Exception:
        return None


def _draw_classification_predictions(
    request: Any, response: Any, colors: Dict[str, str]
) -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(load_image_rgb(request.image))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    if isinstance(response.predictions, list):
        prediction = response.predictions[0]
        color = colors.get(prediction.class_name, "#4892EA")
        draw.rectangle(
            [0, 0, image.size[1], image.size[0]],
            outline=color,
            width=request.visualization_stroke_width,
        )
        text = (
            f"{prediction.class_id} - {prediction.class_name} "
            f"{prediction.confidence:.2f}"
        )
        text_size = font.getbbox(text)
        button_size = (text_size[2] + 20, text_size[3] + 20)
        button_img = Image.new("RGBA", button_size, color)
        button_draw = ImageDraw.Draw(button_img)
        button_draw.text((10, 10), text, font=font, fill=(255, 255, 255, 255))
        image.paste(button_img, (0, 0))
    else:
        if len(response.predictions) > 0:
            draw.rectangle(
                [0, 0, image.size[1], image.size[0]],
                outline="#4892EA",
                width=request.visualization_stroke_width,
            )
        row = 0
        predictions = sorted(
            response.predictions.items(),
            key=lambda x: x[1].confidence,
            reverse=True,
        )
        for cls_name, prediction in predictions:
            color = colors.get(cls_name, "#4892EA")
            text = f"{cls_name} {prediction.confidence:.2f}"
            text_size = font.getbbox(text)
            button_size = (text_size[2] + 20, text_size[3] + 20)
            button_img = Image.new("RGBA", button_size, color)
            button_draw = ImageDraw.Draw(button_img)
            button_draw.text((10, 10), text, font=font, fill=(255, 255, 255, 255))
            image.paste(button_img, (0, row))
            row += button_size[1]
    buffered = io.BytesIO()
    image = image.convert("RGB")
    image.save(buffered, format="JPEG")
    return buffered.getvalue()


def repack_object_detection_response(
    prediction: Any,
    dims: Tuple[int, int],
    class_names: Optional[List[str]],
    request: Any,
) -> ObjectDetectionInferenceResponse:
    detections = _unwrap_single_prediction(prediction)
    xyxy = np.asarray(detections.xyxy, dtype=float).reshape(-1, 4)
    confidences = np.asarray(detections.confidence, dtype=float).reshape(-1)
    class_ids = np.asarray(detections.class_id).reshape(-1)
    class_filter = getattr(request, "class_filter", None)

    predictions: List[ObjectDetectionPrediction] = []
    for (x1, y1, x2, y2), confidence, class_id in zip(xyxy, confidences, class_ids):
        class_id_int = int(class_id)
        class_name = (
            class_names[class_id_int]
            if class_names and 0 <= class_id_int < len(class_names)
            else str(class_id_int)
        )
        if class_filter and class_name not in class_filter:
            continue
        predictions.append(
            ObjectDetectionPrediction(
                x=(float(x1) + float(x2)) / 2.0,
                y=(float(y1) + float(y2)) / 2.0,
                width=float(x2) - float(x1),
                height=float(y2) - float(y1),
                confidence=float(confidence),
                **{"class": class_name},
                class_id=class_id_int,
            )
        )
    width, height = dims
    return ObjectDetectionInferenceResponse(
        predictions=predictions,
        image=InferenceResponseImage(width=width, height=height),
    )


def repack_instance_segmentation_response(
    prediction: Any,
    dims: Tuple[int, int],
    class_names: Optional[List[str]],
    request: Any,
) -> InstanceSegmentationInferenceResponse:
    detections = _unwrap_single_prediction(prediction)
    return_in_rle = getattr(request, "response_mask_format", "polygon") == "rle"
    mask = detections.mask
    if hasattr(mask, "to_coco_rle_masks"):
        if return_in_rle:
            polys_or_rles = mask.to_coco_rle_masks()
        else:
            polys_or_rles = _rle_masks_to_polygons(mask)
    else:
        masks = np.asarray(mask)
        if return_in_rle:
            polys_or_rles = [_dense_mask_to_coco_rle(m) for m in masks]
        else:
            polys_or_rles = masks2poly(masks)

    xyxy = np.asarray(detections.xyxy, dtype=float).reshape(-1, 4)
    confidences = np.asarray(detections.confidence, dtype=float).reshape(-1)
    class_ids = np.asarray(detections.class_id).reshape(-1)
    class_filter = getattr(request, "class_filter", None)

    predictions = []
    for (x1, y1, x2, y2), mask_as_poly_or_rle, confidence, class_id in zip(
        xyxy, polys_or_rles, confidences, class_ids
    ):
        class_id_int = int(class_id)
        class_name = (
            class_names[class_id_int]
            if class_names and 0 <= class_id_int < len(class_names)
            else str(class_id_int)
        )
        if class_filter and class_name not in class_filter:
            continue
        common = dict(
            x=(float(x1) + float(x2)) / 2.0,
            y=(float(y1) + float(y2)) / 2.0,
            width=float(x2) - float(x1),
            height=float(y2) - float(y1),
            confidence=float(confidence),
            class_id=class_id_int,
        )
        if return_in_rle:
            if isinstance(mask_as_poly_or_rle["counts"], bytes):
                mask_as_poly_or_rle["counts"] = mask_as_poly_or_rle["counts"].decode(
                    "ascii"
                )
            predictions.append(
                InstanceSegmentationRLEPrediction(
                    rle=mask_as_poly_or_rle, **{"class": class_name}, **common
                )
            )
        else:
            predictions.append(
                InstanceSegmentationPrediction(
                    points=[
                        Point(x=float(point[0]), y=float(point[1]))
                        for point in mask_as_poly_or_rle
                    ],
                    **{"class": class_name},
                    **common,
                )
            )
    width, height = dims
    return InstanceSegmentationInferenceResponse(
        predictions=predictions,
        image=InferenceResponseImage(width=width, height=height),
    )


def repack_keypoints_response(
    prediction: Any,
    dims: Tuple[int, int],
    class_names: Optional[List[str]],
    key_points_classes: Optional[List[List[str]]],
    request: Any,
) -> KeypointsDetectionInferenceResponse:
    keypoints_obj, detections = _split_keypoints_prediction(prediction)
    if key_points_classes is None:
        raise ModelArtefactError(
            "Keypoint class names are not available from the inference backend."
        )
    xyxy = np.asarray(detections.xyxy, dtype=float).reshape(-1, 4)
    confidences = np.asarray(detections.confidence, dtype=float).reshape(-1)
    class_ids = np.asarray(detections.class_id).reshape(-1)
    keypoints_xy = np.asarray(keypoints_obj.xy, dtype=float).tolist()
    keypoints_class_id = np.asarray(keypoints_obj.class_id).reshape(-1).tolist()
    keypoints_confidence = np.asarray(keypoints_obj.confidence, dtype=float).tolist()
    class_filter = getattr(request, "class_filter", None)

    predictions: List[KeypointsPrediction] = []
    for (
        (x1, y1, x2, y2),
        confidence,
        class_id,
        instance_keypoints_xy,
        instance_keypoints_class_id,
        instance_keypoints_confidence,
    ) in zip(
        xyxy,
        confidences,
        class_ids,
        keypoints_xy,
        keypoints_class_id,
        keypoints_confidence,
    ):
        class_id_int = int(class_id)
        class_name = (
            class_names[class_id_int]
            if class_names and 0 <= class_id_int < len(class_names)
            else str(class_id_int)
        )
        if class_filter and class_name not in class_filter:
            continue
        predictions.append(
            KeypointsPrediction(
                x=(float(x1) + float(x2)) / 2.0,
                y=(float(y1) + float(y2)) / 2.0,
                width=float(x2) - float(x1),
                height=float(y2) - float(y1),
                confidence=float(confidence),
                **{"class": class_name},
                class_id=class_id_int,
                keypoints=_instance_keypoints_to_response(
                    instance_keypoints_xy=instance_keypoints_xy,
                    instance_keypoints_confidence=instance_keypoints_confidence,
                    instance_keypoints_class_id=int(instance_keypoints_class_id),
                    key_points_classes=key_points_classes,
                ),
            )
        )
    width, height = dims
    return KeypointsDetectionInferenceResponse(
        predictions=predictions,
        image=InferenceResponseImage(width=width, height=height),
    )


def _instance_keypoints_to_response(
    instance_keypoints_xy: List[List[float]],
    instance_keypoints_confidence: List[float],
    instance_keypoints_class_id: int,
    key_points_classes: List[List[str]],
) -> List[Keypoint]:
    keypoint_classes = key_points_classes[instance_keypoints_class_id]
    results = []
    for keypoint_class_id, ((x, y), confidence, keypoint_class_name) in enumerate(
        zip(instance_keypoints_xy, instance_keypoints_confidence, keypoint_classes)
    ):
        if confidence <= 0.0:
            continue
        results.append(
            Keypoint(
                x=x,
                y=y,
                confidence=confidence,
                class_id=keypoint_class_id,
                **{"class": keypoint_class_name},
            )
        )
    return results


def repack_classification_response(
    prediction: Any,
    dims: Tuple[int, int],
    class_names: Optional[List[str]],
    request: Any,
) -> ClassificationInferenceResponse:
    predicted = _unwrap_single_prediction(prediction)
    confidences = _classification_confidence_vector(predicted.confidence, class_names)
    raw_confidence = getattr(request, "confidence", None)
    confidence_threshold = (
        raw_confidence
        if isinstance(raw_confidence, (int, float))
        and not isinstance(raw_confidence, bool)
        else 0.5
    )
    class_predictions = []
    for class_id, class_name in enumerate(class_names):
        class_score = float(confidences[class_id])
        if class_score < confidence_threshold:
            continue
        class_predictions.append(
            {
                "class_id": class_id,
                "class": class_name,
                "confidence": round(class_score, 4),
            }
        )
    class_predictions = sorted(
        class_predictions, key=lambda x: x["confidence"], reverse=True
    )
    width, height = dims
    return ClassificationInferenceResponse(
        image=InferenceResponseImage(width=width, height=height),
        predictions=class_predictions,
        top=class_predictions[0]["class"] if class_predictions else "",
        confidence=class_predictions[0]["confidence"] if class_predictions else 0.0,
    )


def repack_multi_label_classification_response(
    prediction: Any,
    dims: Tuple[int, int],
    class_names: Optional[List[str]],
    request: Any,
) -> MultiLabelClassificationInferenceResponse:
    predicted = _unwrap_single_prediction(prediction)
    confidences = _classification_confidence_vector(predicted.confidence, class_names)
    image_predictions = {
        class_names[class_id]: {"confidence": float(confidence), "class_id": class_id}
        for class_id, confidence in enumerate(confidences)
    }
    predicted_classes = [
        class_names[int(class_id)]
        for class_id in np.asarray(predicted.class_ids).reshape(-1).tolist()
    ]
    width, height = dims
    return MultiLabelClassificationInferenceResponse(
        predictions=image_predictions,
        predicted_classes=predicted_classes,
        image=InferenceResponseImage(width=width, height=height),
    )


def _classification_confidence_vector(
    confidence: Any, class_names: Optional[List[str]]
) -> List[float]:
    confidences = np.asarray(confidence, dtype=float).reshape(-1)
    if not class_names or len(confidences) != len(class_names):
        raise PostProcessingError(
            f"Classification model output contains {len(confidences)} confidence "
            f"score(s), but class names metadata expects "
            f"{len(class_names) if class_names else 0}."
        )
    return confidences.tolist()


def repack_semantic_segmentation_response(
    prediction: Any,
    dims: Tuple[int, int],
    class_names: Optional[List[str]],
    request: Any,
) -> SemanticSegmentationInferenceResponse:
    segmentation = _unwrap_single_prediction(prediction)
    segmentation_map = np.asarray(segmentation.segmentation_map).astype(np.uint8)
    confidence_map = (np.asarray(segmentation.confidence, dtype=float) * 255).astype(
        np.uint8
    )
    class_map = {str(i): name for i, name in enumerate(class_names or [])}
    width, height = dims
    response_image = InferenceResponseImage(width=width, height=height)
    response_predictions = SemanticSegmentationPrediction(
        segmentation_mask=_png_b64(segmentation_map),
        confidence_mask=_png_b64(confidence_map),
        class_map=class_map,
        image=dict(response_image),
    )
    return SemanticSegmentationInferenceResponse(
        predictions=response_predictions,
        image=response_image,
    )


@dataclass
class _DepthImage:
    base64_image: str


@dataclass
class _DepthAdapterResponse:
    response: Dict[str, Any]
    time: Optional[float] = None
    inference_id: Optional[str] = None


def repack_depth_estimation_response(prediction: Any) -> _DepthAdapterResponse:
    depth_map = np.asarray(_unwrap_single_prediction(prediction), dtype=np.float32)
    depth_min = float(depth_map.min())
    depth_max = float(depth_map.max())
    if depth_max == depth_min:
        raise ModelArtefactError("Depth map has no variation (min equals max)")
    normalized_depth = (depth_map - depth_min) / (depth_max - depth_min)
    depth_for_viz = (normalized_depth * 255.0).astype(np.uint8)
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("viridis")
    colored_depth = (cmap(depth_for_viz)[:, :, :3] * 255).astype(np.uint8)
    encoded = base64.b64encode(
        encode_image_to_jpeg_bytes(colored_depth, jpeg_quality=95)
    ).decode("ascii")
    return _DepthAdapterResponse(
        response={
            "normalized_depth": normalized_depth,
            "image": _DepthImage(base64_image=encoded),
        }
    )


def repack_structured_ocr_response(
    prediction: Any,
    dims: Tuple[int, int],
    class_names: Optional[List[str]],
    request: Any,
) -> OCRInferenceResponse:
    if not (isinstance(prediction, tuple) and len(prediction) == 2):
        raise ModelArtefactError(
            "Unexpected structured OCR prediction shape from the inference backend."
        )
    texts, detections = prediction
    text = _unwrap_single_prediction(texts)
    width, height = dims
    response = OCRInferenceResponse(
        result=text if isinstance(text, str) else str(text),
        image=InferenceResponseImage(width=width, height=height),
        time=0.0,
    )
    if getattr(request, "generate_bounding_boxes", False):
        boxes = repack_object_detection_response(
            _unwrap_single_prediction(detections), dims, class_names, request
        )
        response.predictions = boxes.predictions
    return response


def repack_text_ocr_response(
    prediction: Any, dims: Tuple[int, int]
) -> OCRInferenceResponse:
    text = _unwrap_single_prediction(prediction)
    width, height = dims
    return OCRInferenceResponse(
        result=text if isinstance(text, str) else str(text),
        image=InferenceResponseImage(width=width, height=height),
        time=0.0,
    )


def repack_vlm_response(prediction: Any, dims: Tuple[int, int]) -> LMMInferenceResponse:
    response = _unwrap_single_prediction(prediction)
    if not isinstance(response, (str, dict)):
        response = str(response)
    width, height = dims
    return LMMInferenceResponse(
        response=response,
        image=InferenceResponseImage(width=width, height=height),
    )


def repack_interactive_segmentation_response(
    action: str, prediction: Any, request: Any
):
    if action in ("embed", "embed_images"):
        return _repack_sam_embeddings(action, prediction, request)
    if action == "segment":
        if type(request).__name__ == "SamSegmentationRequest":
            return _repack_sam_segmentation(prediction)
        return _repack_sam2_segmentation(prediction, request)
    if action == "segment_with_visual_prompts":
        return _repack_visual_segmentation(prediction, request)
    if action == "segment_with_text_prompts":
        return _repack_text_segmentation(prediction, request)
    raise ModelDeploymentNotSupportedError(
        f"No response translation for SAM action '{action}' on the MMP path."
    )


def _repack_sam2_segmentation(
    prediction: Any, request: Any
) -> Sam2SegmentationResponse:
    result = _unwrap_single_prediction(prediction)
    masks, scores = _choose_most_confident_sam_masks(result.masks, result.scores)
    masks = np.asarray(masks) >= 0.0
    predictions = _sam_masks_to_predictions(
        masks, scores, getattr(request, "format", "json"), Sam2SegmentationPrediction
    )
    return Sam2SegmentationResponse(predictions=predictions, time=0.0)


def _repack_sam_segmentation(prediction: Any) -> SamSegmentationResponse:
    result = _unwrap_single_prediction(prediction)
    masks = np.asarray(result.masks)
    if masks.dtype != np.bool_:
        masks = masks > 0.0
    low_res_masks = np.asarray(result.logits) > 0.0
    return SamSegmentationResponse(
        masks=[polygon.tolist() for polygon in masks2poly(masks)],
        low_res_masks=[polygon.tolist() for polygon in masks2poly(low_res_masks)],
        time=0.0,
    )


def _repack_sam_embeddings(action: str, prediction: Any, request: Any):
    embeddings_obj = _unwrap_single_prediction(prediction)
    if type(request).__name__ == "SamEmbeddingRequest":
        embeddings = np.asarray(embeddings_obj.embeddings)
        if getattr(request, "format", "json") == "binary":
            buffer = io.BytesIO()
            np.save(buffer, embeddings)
            return SamEmbeddingResponse(embeddings=buffer.getvalue(), time=0.0)
        return SamEmbeddingResponse(embeddings=embeddings.tolist(), time=0.0)
    image_id = getattr(request, "image_id", None)
    if not image_id:
        image_id = getattr(embeddings_obj, "image_hash", None)
        if image_id:
            image_id = _strip_client_hash_namespace(image_id, request)
    if action == "embed_images":
        return Sam3EmbeddingResponse(image_id=image_id, time=0.0)
    return Sam2EmbeddingResponse(image_id=image_id, time=0.0)


def _decode_coco_rle_masks(mask_dicts: List[dict]) -> np.ndarray:
    from pycocotools import mask as mask_utils

    decoded = []
    for mask_dict in mask_dicts:
        counts = mask_dict["counts"]
        if isinstance(counts, str):
            counts = counts.encode("utf-8")
        decoded.append(
            mask_utils.decode({"size": mask_dict["size"], "counts": counts}).astype(
                bool
            )
        )
    if not decoded:
        return np.zeros((0, 0, 0), dtype=bool)
    return np.stack(decoded)


def _repack_visual_segmentation(
    prediction: Any, request: Any
) -> Sam2SegmentationResponse:
    result = _unwrap_single_prediction(prediction)
    if isinstance(result, dict):
        # mask_format=rle wire shape: {"masks": [coco-rle dicts], "scores": [...]},
        # already reduced to the most confident mask per prompt worker-side.
        masks = _decode_coco_rle_masks(result.get("masks") or [])
        scores = [float(score) for score in result.get("scores") or []]
    else:
        masks, scores = _choose_most_confident_sam_masks(result.masks, result.scores)
    predictions = _sam_masks_to_predictions(
        masks, scores, getattr(request, "format", "polygon"), Sam2SegmentationPrediction
    )
    return Sam2SegmentationResponse(predictions=predictions, time=0.0)


def _repack_text_segmentation(
    prediction: Any, request: Any
) -> Sam3SegmentationResponse:
    # The per-image worker result IS the per-prompt list — a single-prompt
    # request arrives as a one-element list that must not be unwrapped.
    prompt_outputs = prediction
    if isinstance(prompt_outputs, dict):
        prompt_outputs = [prompt_outputs]
    if not isinstance(prompt_outputs, list) or not all(
        isinstance(output, dict) for output in prompt_outputs
    ):
        raise ModelArtefactError(
            "Unexpected SAM3 text-prompt prediction shape from the inference backend."
        )
    prompts = list(getattr(request, "prompts", None) or [])
    response_format = getattr(request, "format", "polygon")
    prompt_results = []
    for output in prompt_outputs:
        index = int(output.get("prompt_index", len(prompt_results)))
        prompt = prompts[index] if index < len(prompts) else None
        raw_masks = output.get("masks")
        if raw_masks is None:
            raw_masks = []
        if isinstance(raw_masks, list) and raw_masks and isinstance(raw_masks[0], dict):
            masks = _decode_coco_rle_masks(raw_masks)
        else:
            masks = np.asarray(raw_masks)
        scores = [float(score) for score in output.get("scores", [])]
        prompt_threshold = getattr(prompt, "output_prob_thresh", None)
        if prompt_threshold is not None:
            kept = [i for i, score in enumerate(scores) if score >= prompt_threshold]
            masks = masks[kept] if len(kept) else masks[:0]
            scores = [scores[i] for i in kept]
        has_visual = bool(getattr(prompt, "boxes", None))
        echo = Sam3PromptEcho(
            prompt_index=index,
            type="visual" if has_visual else "text",
            text=getattr(prompt, "text", None),
            num_boxes=len(getattr(prompt, "boxes", None) or []) if has_visual else 0,
        )
        predictions = _sam_masks_to_predictions(
            masks, scores, response_format, Sam3SegmentationPrediction
        )
        prompt_results.append(
            Sam3PromptResult(prompt_index=index, echo=echo, predictions=predictions)
        )
    return Sam3SegmentationResponse(prompt_results=prompt_results, time=0.0)


def _choose_most_confident_sam_masks(
    masks: Any, scores: Any
) -> Tuple[np.ndarray, List[float]]:
    masks = np.asarray(masks)
    scores = np.asarray(scores, dtype=float)
    if masks.ndim == 3:
        masks = masks[None]
        scores = scores.reshape(1, -1)
    selected_masks = []
    selected_scores = []
    for prompt_masks, prompt_scores in zip(masks, scores):
        best = int(np.argmax(prompt_scores))
        selected_masks.append(prompt_masks[best])
        selected_scores.append(float(prompt_scores[best]))
    return np.asarray(selected_masks), selected_scores


def _sam_masks_to_predictions(
    masks: np.ndarray, scores: List[float], response_format: Any, prediction_cls
) -> list:
    response_format = response_format or "polygon"
    if response_format in ("polygon", "json"):
        polygons = masks2multipoly((np.asarray(masks) > 0).astype(np.uint8))
        return [
            prediction_cls(
                masks=[polygon.tolist() for polygon in mask_polygons],
                confidence=float(score),
                format="polygon",
            )
            for mask_polygons, score in zip(polygons, scores)
        ]
    if response_format == "rle":
        from pycocotools import mask as mask_utils

        predictions = []
        for mask, score in zip(np.asarray(masks), scores):
            rle = mask_utils.encode(np.asfortranarray((mask > 0).astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("utf-8")
            predictions.append(
                prediction_cls(masks=rle, confidence=float(score), format="rle")
            )
        return predictions
    raise ModelDeploymentNotSupportedError(
        f"format={response_format!r} is not supported on the MMP path."
    )


def _split_keypoints_prediction(prediction: Any) -> Tuple[Any, Any]:
    if isinstance(prediction, tuple) and len(prediction) == 2:
        keypoints, detections = prediction
        keypoints = _unwrap_single_prediction(keypoints)
        detections = _unwrap_single_prediction(detections)
        if keypoints is None or detections is None:
            raise ModelArtefactError(
                "Keypoints prediction from the inference backend is incomplete."
            )
        return keypoints, detections
    raise ModelArtefactError(
        "Unexpected keypoints prediction shape from the inference backend."
    )


def _dense_mask_to_coco_rle(mask: np.ndarray) -> dict:
    from pycocotools import mask as mask_utils

    return mask_utils.encode(np.asfortranarray(np.asarray(mask).astype(np.uint8)))


def _rle_masks_to_polygons(masks: Any) -> List[np.ndarray]:
    from pycocotools import mask as mask_utils

    height, width = masks.image_size
    segments = []
    for counts in masks.masks:
        decoded = np.ascontiguousarray(
            mask_utils.decode({"size": [height, width], "counts": counts})
        )
        if not np.any(decoded):
            segments.append(np.zeros((0, 2), dtype=np.float32))
        else:
            segments.append(mask2poly(decoded))
    return segments


def _png_b64(image: np.ndarray) -> str:
    from PIL import Image

    buffered = io.BytesIO()
    Image.fromarray(image).save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("ascii")


def _unwrap_single_prediction(prediction: Any) -> Any:
    if isinstance(prediction, list):
        if len(prediction) != 1:
            raise ModelArtefactError(
                f"Expected a single prediction from the inference backend, "
                f"got {len(prediction)}."
            )
        return prediction[0]
    return prediction


def _decode_base64_payload(value: Any) -> bytes:
    if not isinstance(value, str):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            raise InputImageLoadError(
                message="Could not decode image bytes as base64 string.",
                public_message="Invalid base64 input: the image payload contains "
                "raw bytes instead of a base64-encoded string.",
            )
    value = BASE64_DATA_TYPE_PATTERN.sub("", value)
    try:
        data = pybase64.b64decode(value)
    except binascii.Error as error:
        raise InputImageLoadError(
            message="Could not load valid image from base64 string.",
            public_message="Malformed base64 input image.",
        ) from error
    if len(data) == 0:
        raise InputImageLoadError(
            message="Could not load valid image from base64 string.",
            public_message="Empty image payload.",
        )
    return data


def _dims_from_header(data: bytes) -> Tuple[int, int]:
    dims = _read_image_dims(data)
    if dims is None:
        raise InputImageLoadError(
            message="Could not read image dimensions from the image header.",
            public_message="Could not decode input image.",
        )
    return int(dims[0]), int(dims[1])


def _read_image_dims(data: bytes) -> Optional[Tuple[int, int]]:
    from inference_model_manager.backends.utils.image_headers import image_dims

    return image_dims(data)
