"""Legacy <-> new-world translation for ModelManagerAdapter.

Everything the adapter needs to speak both dialects: the static mirror of the
new-world handler registry, request forwarding (image -> bytes), per-task
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pybase64

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
from inference.core.entities.responses.sam import SamEmbeddingResponse
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
from inference.core.exceptions import (
    InferenceModelNotFound,
    InferencePayloadTooLargeError,
    InputImageLoadError,
    InvalidImageTypeDeclared,
    ModelArtefactError,
    ModelDeploymentNotSupportedError,
    ModelManagerLockAcquisitionError,
    PostProcessingError,
    RoboflowAPIConnectionError,
    RoboflowAPINotAuthorizedError,
    RoboflowAPINotNotFoundError,
    RoboflowAPITimeoutError,
)
from inference.core.utils.image_utils import (
    BASE64_DATA_TYPE_PATTERN,
    encode_image_to_jpeg_bytes,
    fetch_image_bytes_from_url,
    load_image_from_numpy_object,
    load_image_from_numpy_str,
)
from inference.core.utils.postprocess import mask2poly, masks2multipoly, masks2poly

# Static mirror of the (model_type, action) pairs registered by
# inference_server/handlers/*/description.py. Kept as literals so the legacy
# side never imports the handler modules; a parity test asserts no drift.
NEW_WORLD_HANDLERS = frozenset(
    [
        ("classification", "infer"),
        ("depth-estimation", "infer"),
        ("instance-segmentation", "infer"),
        ("interactive-instance-segmentation", "embed"),
        ("interactive-instance-segmentation", "embed_images"),
        ("interactive-instance-segmentation", "prompt"),
        ("interactive-instance-segmentation", "segment_with_text_prompts"),
        ("interactive-instance-segmentation", "segment_with_visual_prompts"),
        ("interactive-instance-segmentation", "track"),
        ("keypoint-detection", "infer"),
        ("multi-label-classification", "infer"),
        ("object-detection", "infer"),
        ("open-vocabulary-object-detection", "infer"),
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
        ("instance-segmentation", "infer"),
        ("keypoint-detection", "infer"),
        ("multi-label-classification", "infer"),
        ("interactive-instance-segmentation", "embed"),
        ("interactive-instance-segmentation", "embed_images"),
        ("interactive-instance-segmentation", "segment_with_text_prompts"),
        ("interactive-instance-segmentation", "segment_with_visual_prompts"),
        ("object-detection", "infer"),
        ("open-vocabulary-object-detection", "infer"),
        ("semantic-segmentation", "infer"),
        ("structured-ocr", "infer"),
        ("text-only-ocr", "infer"),
        ("vlm", "prompt"),
    ]
)

# VLM model classes whose legacy response contract the adapter cannot satisfy:
# legacy Florence2 returns a dict keyed by task token built by the HF
# processor's post_process_generation; the new-world prompt action returns the
# raw decoded string.
VLM_UNSUPPORTED_MODEL_CLASSES = frozenset(["Florence2HF"])

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
}


def resolve_request_action(route: dict, request: Any) -> str:
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


async def stat_model(model_id: str, api_key: str) -> Tuple[str, str]:
    """Resolve (task_type, default_action) via the new world's stat + auth."""
    from inference_server.framework.entities import CommonRequestParams
    from inference_server.framework.model_stat import stat_model_while_checking_auth

    try:
        return await stat_model_while_checking_auth(
            CommonRequestParams(model_id=model_id, api_key=api_key)
        )
    except PermissionError as error:
        raise RoboflowAPINotAuthorizedError(str(error)) from error
    except LookupError as error:
        raise RoboflowAPINotNotFoundError(str(error)) from error
    except RuntimeError as error:
        raise RoboflowAPIConnectionError(str(error)) from error


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


def ensure_request_supported(model_id: str, request: Any) -> None:
    """Reject fidelity-breaking legacy-only params instead of silently drifting."""
    if getattr(request, "visualize_predictions", False):
        raise ModelDeploymentNotSupportedError(
            f"visualize_predictions / format=image is not supported for model "
            f"'{model_id}' on the MMP path."
        )
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


def build_task_params(task_type: str, action: str, request: Any) -> dict:
    params: dict = {}
    if task_type == "interactive-instance-segmentation":
        return _build_interactive_segmentation_params(action, request)
    if task_type == "vlm":
        return _build_vlm_params(request)
    if task_type == "structured-ocr":
        _ensure_ocr_request_supported(request)
        return params
    if task_type == "text-only-ocr":
        return params
    if task_type == "open-vocabulary-object-detection":
        return _build_open_vocabulary_params(request)
    if task_type in ("semantic-segmentation", "depth-estimation"):
        return params
    confidence = _numeric_confidence(getattr(request, "confidence", None))
    if confidence is not None:
        params["confidence"] = confidence
    if task_type in ("classification", "multi-label-classification"):
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


def _build_interactive_segmentation_params(action: str, request: Any) -> dict:
    if action == "embed":
        return {}
    if action == "embed_images":
        params: dict = {}
        image_id = getattr(request, "image_id", None)
        if image_id:
            params["image_hashes"] = [image_id]
        return params
    if action == "segment_with_visual_prompts":
        return _build_visual_prompt_params(request)
    if action == "segment_with_text_prompts":
        return _build_text_prompt_params(request)
    raise ModelDeploymentNotSupportedError(
        f"SAM action '{action}' is not supported on the MMP path."
    )


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
    if not point_coords and not point_labels and not boxes:
        point_coords, point_labels = [[0, 0]], [-1]
    params: dict = {
        "multi_mask_output": bool(getattr(request, "multimask_output", True)),
    }
    if point_coords:
        params["point_coordinates"] = point_coords
    if point_labels:
        params["point_labels"] = point_labels
    if boxes:
        params["boxes"] = boxes
    image_id = getattr(request, "image_id", None)
    if image_id:
        params["image_hashes"] = [image_id]
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
    return {
        "prompts": [prompt.dict() for prompt in prompts],
        "output_prob_thresh": float(
            getattr(request, "output_prob_thresh", None) or 0.5
        ),
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


def forward_image(image: Any) -> Tuple[bytes, Tuple[int, int]]:
    """InferenceRequestImage -> (compressed bytes for the SHM slot, (width, height)).

    Never decodes pixels adapter-side: dims come from the ndarray shape or a
    header-only read; the MMP worker does the real decode.
    """
    image_type = getattr(image, "type", None)
    value = getattr(image, "value", None)
    if image_type == "base64":
        data = _decode_base64_payload(value)
        return data, _dims_from_header(data)
    if image_type == "url":
        data = fetch_image_bytes_from_url(value=value)
        return data, _dims_from_header(data)
    if image_type == "numpy":
        if isinstance(value, np.ndarray):
            array = load_image_from_numpy_object(value)
        else:
            array = load_image_from_numpy_str(value)
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        return buffer.getvalue(), (int(array.shape[1]), int(array.shape[0]))
    raise InvalidImageTypeDeclared(
        message=f"Image type '{image_type}' is not supported on the MMP path.",
        public_message=f"Image type '{image_type}' is not supported on the MMP path.",
    )


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
        return repack_vlm_response(prediction, dims)
    raise ModelDeploymentNotSupportedError(
        f"No response translation for task type '{task_type}' on the MMP path."
    )


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
    if action == "segment_with_visual_prompts":
        return _repack_visual_segmentation(prediction, request)
    if action == "segment_with_text_prompts":
        return _repack_text_segmentation(prediction, request)
    raise ModelDeploymentNotSupportedError(
        f"No response translation for SAM action '{action}' on the MMP path."
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
    image_id = getattr(request, "image_id", None) or getattr(
        embeddings_obj, "image_hash", None
    )
    if action == "embed_images":
        return Sam3EmbeddingResponse(image_id=image_id, time=0.0)
    return Sam2EmbeddingResponse(image_id=image_id, time=0.0)


def _repack_visual_segmentation(
    prediction: Any, request: Any
) -> Sam2SegmentationResponse:
    result = _unwrap_single_prediction(prediction)
    masks, scores = _choose_most_confident_sam_masks(result.masks, result.scores)
    predictions = _sam_masks_to_predictions(
        masks, scores, getattr(request, "format", "polygon"), Sam2SegmentationPrediction
    )
    return Sam2SegmentationResponse(predictions=predictions, time=0.0)


def _repack_text_segmentation(
    prediction: Any, request: Any
) -> Sam3SegmentationResponse:
    prompt_outputs = _unwrap_single_prediction(prediction)
    if not isinstance(prompt_outputs, list):
        raise ModelArtefactError(
            "Unexpected SAM3 text-prompt prediction shape from the inference backend."
        )
    prompts = list(getattr(request, "prompts", None) or [])
    response_format = getattr(request, "format", "polygon")
    prompt_results = []
    for output in prompt_outputs:
        index = int(output.get("prompt_index", len(prompt_results)))
        prompt = prompts[index] if index < len(prompts) else None
        masks = np.asarray(output.get("masks"))
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
