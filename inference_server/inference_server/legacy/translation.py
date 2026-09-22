from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel

from inference_model_manager.hash_namespacing import (
    namespace_client_hash_id,
    strip_tenant_namespace,
)
from inference_server import configuration
from inference_server.legacy.bridge import Route
from inference_server.legacy.entities import (
    ClassificationInferenceResponse,
    ClipCompareResponse,
    ClipEmbeddingResponse,
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
    OCRInferenceResponse,
    PerceptionEncoderCompareResponse,
    PerceptionEncoderEmbeddingResponse,
    Point,
    Sam2EmbeddingResponse,
    Sam2SegmentationPrediction,
    Sam2SegmentationResponse,
    Sam3EmbeddingResponse,
    Sam3PromptEcho,
    Sam3PromptResult,
    Sam3SegmentationPrediction,
    Sam3SegmentationResponse,
    SamEmbeddingResponse,
    SamSegmentationResponse,
    SemanticSegmentationInferenceResponse,
    SemanticSegmentationPrediction,
)
from inference_server.legacy.errors import LegacyHTTPError

_DISABLE_PREPROC_FIELDS = (
    "disable_preproc_auto_orient",
    "disable_preproc_contrast",
    "disable_preproc_grayscale",
    "disable_preproc_static_crop",
)
_OD_MAX_CANDIDATES_DEFAULT = 3000
_CONFIDENCE_ONLY_TASK_TYPES = frozenset(
    [
        "classification",
        "multi-label-classification",
        "semantic-segmentation",
    ]
)


def ensure_request_supported(model_id: str, request: Any, route: Route) -> None:
    for field in _DISABLE_PREPROC_FIELDS:
        if getattr(request, field, False):
            raise LegacyHTTPError(
                501, f"{field} is not supported for model '{model_id}'."
            )
    max_candidates = getattr(request, "max_candidates", None)
    if max_candidates is not None and max_candidates != _OD_MAX_CANDIDATES_DEFAULT:
        raise LegacyHTTPError(
            501, f"max_candidates is not supported for model '{model_id}'."
        )
    mask_decode_mode = getattr(request, "mask_decode_mode", None)
    if mask_decode_mode is not None and mask_decode_mode != "accurate":
        raise LegacyHTTPError(
            501,
            f"mask_decode_mode={mask_decode_mode!r} is not supported for model "
            f"'{model_id}'.",
        )
    tradeoff_factor = getattr(request, "tradeoff_factor", None)
    if tradeoff_factor:
        raise LegacyHTTPError(
            501, f"tradeoff_factor is not supported for model '{model_id}'."
        )


def _numeric_confidence(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LegacyHTTPError(501, f"confidence={value!r} is not supported.")
    return float(value)


def roboflow_confidence(value: Any) -> Optional[Union[float, str]]:
    if isinstance(value, str) and value in ("best", "default"):
        return value
    return _numeric_confidence(value)


def build_task_params(task_type: str, action: str, request: Any, route: Route) -> dict:
    params: dict = {}
    confidence = roboflow_confidence(getattr(request, "confidence", None))
    if confidence is not None:
        params["confidence"] = confidence
    if task_type in _CONFIDENCE_ONLY_TASK_TYPES:
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


def repack_prediction(
    task_type: str,
    action: str,
    prediction: Any,
    dims: Tuple[int, int],
    route: Route,
    request: Any,
) -> BaseModel:
    class_names = route.class_names
    if task_type == "object-detection":
        return repack_object_detection_response(prediction, dims, class_names, request)
    if task_type == "instance-segmentation":
        return repack_instance_segmentation_response(
            prediction, dims, class_names, request
        )
    if task_type == "keypoint-detection":
        return repack_keypoints_response(
            prediction, dims, class_names, route.key_points_classes, request
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
    raise LegacyHTTPError(501, f"No response translation for task type '{task_type}'.")


def repack_object_detection_response(
    prediction: Any,
    dims: Tuple[int, int],
    class_names: Optional[List[str]],
    request: Any,
) -> ObjectDetectionInferenceResponse:
    detections = unwrap_single_prediction(prediction)
    xyxy = np.asarray(detections.xyxy, dtype=float).reshape(-1, 4)
    confidences = np.asarray(detections.confidence, dtype=float).reshape(-1)
    class_ids = np.asarray(detections.class_id).reshape(-1)
    class_filter = getattr(request, "class_filter", None)

    predictions: List[ObjectDetectionPrediction] = []
    for (x1, y1, x2, y2), confidence, class_id in zip(xyxy, confidences, class_ids):
        class_id_int = int(class_id)
        class_name = _class_name(class_names, class_id_int)
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
    detections = unwrap_single_prediction(prediction)
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
        class_name = _class_name(class_names, class_id_int)
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
        raise LegacyHTTPError(
            500, "Keypoint class names are not available from the inference backend."
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
        class_name = _class_name(class_names, class_id_int)
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
    predicted = unwrap_single_prediction(prediction)
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
    predicted = unwrap_single_prediction(prediction)
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
        raise LegacyHTTPError(
            500,
            f"Classification model output contains {len(confidences)} confidence "
            f"score(s), but class names metadata expects "
            f"{len(class_names) if class_names else 0}.",
        )
    return confidences.tolist()


def repack_semantic_segmentation_response(
    prediction: Any,
    dims: Tuple[int, int],
    class_names: Optional[List[str]],
    request: Any,
) -> SemanticSegmentationInferenceResponse:
    segmentation = unwrap_single_prediction(prediction)
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
        present_class_ids=np.unique(segmentation_map).astype(int).tolist(),
    )
    return SemanticSegmentationInferenceResponse(
        predictions=response_predictions,
        image=response_image,
    )


def masks2poly(masks: np.ndarray) -> List[np.ndarray]:
    segments = []
    for mask in masks:
        binary_mask = _as_binary_uint8(mask)
        if not np.any(binary_mask):
            segments.append(np.zeros((0, 2), dtype=np.float32))
            continue
        segments.append(mask2poly(binary_mask))
    return segments


def masks2multipoly(masks: np.ndarray) -> List[List[np.ndarray]]:
    segments = []
    for mask in masks:
        binary_mask = _as_binary_uint8(mask)
        if not np.any(binary_mask):
            segments.append([np.zeros((0, 2), dtype=np.float32)])
            continue
        segments.append(_mask2multipoly(binary_mask))
    return segments


def mask2poly(mask: np.ndarray) -> np.ndarray:
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if contours:
        contours = np.array(
            contours[np.array([len(x) for x in contours]).argmax()]
        ).reshape(-1, 2)
    else:
        contours = np.zeros((0, 2))
    return contours.astype("float32")


def _mask2multipoly(mask: np.ndarray) -> List[np.ndarray]:
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if contours:
        return [contour.reshape(-1, 2).astype("float32") for contour in contours]
    return [np.zeros((0, 2)).astype("float32")]


def _as_binary_uint8(mask: np.ndarray) -> np.ndarray:
    if mask.dtype == np.bool_:
        binary_mask = mask
        if not binary_mask.flags.c_contiguous:
            binary_mask = np.ascontiguousarray(binary_mask)
        return binary_mask.view(np.uint8)
    if mask.dtype == np.uint8:
        return mask if mask.flags.c_contiguous else np.ascontiguousarray(mask)
    binary_mask = mask > 0
    if not binary_mask.flags.c_contiguous:
        binary_mask = np.ascontiguousarray(binary_mask)
    return binary_mask.view(np.uint8)


def unwrap_single_prediction(prediction: Any) -> Any:
    if isinstance(prediction, list):
        if len(prediction) != 1:
            raise LegacyHTTPError(
                500,
                f"Expected a single prediction from the inference backend, "
                f"got {len(prediction)}.",
            )
        return prediction[0]
    return prediction


def _split_keypoints_prediction(prediction: Any) -> Tuple[Any, Any]:
    if isinstance(prediction, tuple) and len(prediction) == 2:
        keypoints, detections = prediction
        keypoints = unwrap_single_prediction(keypoints)
        detections = unwrap_single_prediction(detections)
        if keypoints is None or detections is None:
            raise LegacyHTTPError(
                500, "Keypoints prediction from the inference backend is incomplete."
            )
        return keypoints, detections
    raise LegacyHTTPError(
        500, "Unexpected keypoints prediction shape from the inference backend."
    )


def _class_name(class_names: Optional[List[str]], class_id: int) -> str:
    if class_names and 0 <= class_id < len(class_names):
        return class_names[class_id]
    return str(class_id)


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
    buffered = io.BytesIO()
    Image.fromarray(image).save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("ascii")


ACTION_CANDIDATES_BY_REQUEST_TYPE: Dict[str, Tuple[str, ...]] = {
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

MOONDREAM_MODEL_CLASS = "MoonDream2HF"

_EMBEDDING_RESPONSE_CLASSES = {
    "ClipImageEmbeddingRequest": ClipEmbeddingResponse,
    "ClipTextEmbeddingRequest": ClipEmbeddingResponse,
    "ClipCompareRequest": ClipCompareResponse,
    "PerceptionEncoderImageEmbeddingRequest": PerceptionEncoderEmbeddingResponse,
    "PerceptionEncoderTextEmbeddingRequest": PerceptionEncoderEmbeddingResponse,
    "PerceptionEncoderCompareRequest": PerceptionEncoderCompareResponse,
}

_MAX_VALUE_BY_DTYPE = {np.dtype(np.uint8): 255, np.dtype(np.uint16): 65535}
_DEPTH_JPEG_QUALITY = 95


def is_moondream_backed(route: Route) -> bool:
    return route.model_class_name == MOONDREAM_MODEL_CLASS


def resolve_request_action(route: Route, request: Any) -> str:
    if is_moondream_backed(route) and "detect" in (route.tasks or set()):
        return "detect"
    candidates = ACTION_CANDIDATES_BY_REQUEST_TYPE.get(type(request).__name__)
    if not candidates:
        return route.action
    tasks = route.tasks or set()
    for candidate in candidates:
        if candidate in tasks:
            return candidate
    return route.action


def build_vlm_params(request: Any) -> dict:
    prompt = getattr(request, "prompt", None)
    if not prompt:
        raise LegacyHTTPError(501, "VLM inference requires a prompt.")
    params: dict = {"prompt": prompt}
    max_new_tokens = getattr(request, "max_new_tokens", None)
    if max_new_tokens is not None:
        params["max_new_tokens"] = int(max_new_tokens)
    if getattr(request, "enable_thinking", False):
        params["enable_thinking"] = True
    return params


def ensure_ocr_request_supported(request: Any) -> None:
    language_codes = getattr(request, "language_codes", None)
    if language_codes is not None and list(language_codes) != ["en"]:
        raise LegacyHTTPError(
            501, "language_codes other than ['en'] are not supported."
        )
    if getattr(request, "quantize", False):
        raise LegacyHTTPError(501, "quantize is not supported.")


def build_open_vocabulary_params(request: Any) -> dict:
    classes = getattr(request, "text", None) or getattr(request, "classes", None)
    if getattr(request, "training_data", None) is not None:
        raise LegacyHTTPError(
            501, "Few-shot detection with training_data is not supported."
        )
    if not classes:
        raise LegacyHTTPError(
            501, "Open-vocabulary detection requires a list of classes."
        )
    for field, default in (("box_threshold", 0.5), ("text_threshold", 0.5)):
        value = getattr(request, field, None)
        if value is not None and value != default:
            raise LegacyHTTPError(501, f"{field} is not supported.")
    params: dict = {"classes": [str(name) for name in classes]}
    confidence = _numeric_confidence(getattr(request, "confidence", None))
    if confidence is not None:
        params["confidence"] = confidence
    class_agnostic_nms = getattr(request, "class_agnostic_nms", None)
    if class_agnostic_nms is not None:
        params["class_agnostic_nms"] = bool(class_agnostic_nms)
    return params


def requested_open_vocabulary_classes(request: Any) -> List[str]:
    classes = getattr(request, "text", None) or getattr(request, "classes", None) or []
    return [str(name) for name in classes]


def _embedding_response_class(request: Any):
    response_class = _EMBEDDING_RESPONSE_CLASSES.get(type(request).__name__)
    if response_class is None:
        raise LegacyHTTPError(
            501,
            f"Request type {type(request).__name__} is not supported for embedding "
            f"models.",
        )
    return response_class


def _embed_image_call(image: Any) -> dict:
    return {"task": "embed_images", "image": image, "params": {}}


def _embed_text_call(texts: List[str]) -> dict:
    return {"task": "embed_text", "image": None, "params": {"texts": list(texts)}}


def build_embedding_calls(
    action: str, request: Any
) -> Tuple[List[dict], Optional[List[str]]]:
    _embedding_response_class(request)
    max_batch_size = configuration.CLIP_MAX_BATCH_SIZE
    if action == "embed_images":
        if isinstance(request.image, list):
            if len(request.image) > max_batch_size:
                raise ValueError(
                    f"The maximum number of images that can be embedded at once is "
                    f"{max_batch_size}"
                )
            images = request.image
        else:
            images = [request.image]
        return [_embed_image_call(image) for image in images], None
    if action == "embed_text":
        texts = request.text if isinstance(request.text, list) else [request.text]
        return [_embed_text_call(texts)], None
    if action != "compare":
        raise LegacyHTTPError(501, f"Embedding action '{action}' is not supported.")
    if request.subject_type not in ("image", "text"):
        raise ValueError("subject_type must be either 'image' or 'text'")
    prompt = request.prompt
    prompt_keys = None
    if isinstance(prompt, dict) and not ("type" in prompt and "value" in prompt):
        prompt_keys = list(prompt.keys())
        prompt = [prompt[key] for key in prompt_keys]
    elif not isinstance(prompt, list):
        prompt = [prompt]
    if len(prompt) > max_batch_size:
        raise ValueError(
            f"The maximum number of prompts that can be compared at once is "
            f"{max_batch_size}"
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
) -> BaseModel:
    response_class = _embedding_response_class(request)
    if action in ("embed_images", "embed_text"):
        return response_class(embeddings=_stack_embeddings(results).tolist())
    subject = _stack_embeddings(results[:1]).reshape(-1)
    prompts = _stack_embeddings(results[1:])
    similarities = [_cosine_similarity(subject, row) for row in prompts]
    if prompt_keys is not None:
        return response_class(similarity=dict(zip(prompt_keys, similarities)))
    return response_class(similarity=similarities)


def _stack_embeddings(results: List[Any]) -> np.ndarray:
    arrays = []
    for result in results:
        array = np.asarray(result, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        arrays.append(array)
    return np.concatenate(arrays, axis=0)


def _cosine_similarity(subject: np.ndarray, prompt: np.ndarray) -> float:
    denominator = float(np.linalg.norm(subject) * np.linalg.norm(prompt))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(subject, prompt) / denominator)


def repack_vlm_response(prediction: Any, dims: Tuple[int, int]) -> LMMInferenceResponse:
    response = unwrap_single_prediction(prediction)
    if not isinstance(response, (str, dict)):
        response = str(response)
    width, height = dims
    return LMMInferenceResponse(
        response=response,
        image=InferenceResponseImage(width=width, height=height),
    )


def repack_moondream_detection(
    prediction: Any, request: Any, dims: Tuple[int, int]
) -> ObjectDetectionInferenceResponse:
    detections = unwrap_single_prediction(prediction)
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


def repack_depth_estimation(prediction: Any) -> dict:
    depth_map = np.asarray(unwrap_single_prediction(prediction), dtype=np.float32)
    depth_min = float(depth_map.min())
    depth_max = float(depth_map.max())
    if depth_max == depth_min:
        raise LegacyHTTPError(500, "Depth map has no variation (min equals max)")
    normalized_depth = (depth_map - depth_min) / (depth_max - depth_min)
    colored_depth = cv2.applyColorMap(
        (normalized_depth * 255.0).astype(np.uint8), cv2.COLORMAP_VIRIDIS
    )
    success, buffer = cv2.imencode(
        ".jpg", colored_depth, [int(cv2.IMWRITE_JPEG_QUALITY), _DEPTH_JPEG_QUALITY]
    )
    if not success:
        raise LegacyHTTPError(500, "Could not encode depth map visualization as JPEG")
    return {
        "normalized_depth": normalized_depth,
        "image": {"base64_image": base64.b64encode(buffer.tobytes()).decode("ascii")},
    }


def _encode_normalized_depth_to_png(
    normalized_depth: np.ndarray, dtype: np.dtype
) -> str:
    depth = np.asarray(normalized_depth, dtype=np.float32)
    max_value = _MAX_VALUE_BY_DTYPE[np.dtype(dtype)]
    quantized = np.round(np.clip(depth, 0.0, 1.0) * max_value).astype(dtype)
    success, buffer = cv2.imencode(".png", quantized)
    if not success:
        raise LegacyHTTPError(500, "Could not encode normalized depth map as PNG")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def encode_normalized_depth_to_png16(normalized_depth: np.ndarray) -> str:
    return _encode_normalized_depth_to_png(normalized_depth, np.uint16)


def encode_normalized_depth_to_png8(normalized_depth: np.ndarray) -> str:
    return _encode_normalized_depth_to_png(normalized_depth, np.uint8)


def repack_structured_ocr_response(
    prediction: Any,
    dims: Tuple[int, int],
    class_names: Optional[List[str]],
    request: Any,
) -> OCRInferenceResponse:
    if not (isinstance(prediction, tuple) and len(prediction) == 2):
        raise LegacyHTTPError(
            500,
            "Unexpected structured OCR prediction shape from the inference backend.",
        )
    texts, detections = prediction
    text = unwrap_single_prediction(texts)
    width, height = dims
    response = OCRInferenceResponse(
        result=text if isinstance(text, str) else str(text),
        time=0.0,
    )
    if getattr(request, "generate_bounding_boxes", False):
        boxes = repack_object_detection_response(
            unwrap_single_prediction(detections), dims, class_names, request
        )
        response.predictions = boxes.predictions
        response.image = InferenceResponseImage(width=width, height=height)
    return response


def repack_text_ocr_response(
    prediction: Any, dims: Tuple[int, int]
) -> OCRInferenceResponse:
    text = unwrap_single_prediction(prediction)
    return OCRInferenceResponse(
        result=text if isinstance(text, str) else str(text),
        time=0.0,
    )


BINARY_FORMAT_UNSUPPORTED_MESSAGE = (
    "format='binary' is not supported on inference_server."
)
_SEGMENT_ACTIONS = (
    "segment",
    "segment_with_visual_prompts",
    "segment_with_text_prompts",
)
_MASK_INPUT_UNSUPPORTED_MESSAGE = "mask_input is not supported on inference_server."
_EMBEDDINGS_INPUT_UNSUPPORTED_MESSAGE = (
    "embeddings input is not supported on inference_server."
)


def build_interactive_segmentation_params(
    action: str, request: Any, api_key: Optional[str]
) -> dict:
    if action in ("embed", "embed_images"):
        params: dict = {}
        image_id = getattr(request, "image_id", None)
        if image_id:
            params["image_hashes"] = [namespace_client_hash_id(image_id, api_key)]
        if action == "embed_images":
            params["return_embeddings"] = False
        return params
    if action in _SEGMENT_ACTIONS:
        if getattr(request, "format", None) == "binary":
            raise LegacyHTTPError(501, BINARY_FORMAT_UNSUPPORTED_MESSAGE)
    if action == "segment":
        if type(request).__name__ == "SamSegmentationRequest":
            return _build_sam_segment_params(request, api_key)
        return _build_sam2_segment_params(request, api_key)
    if action == "segment_with_visual_prompts":
        return _build_visual_prompt_params(request, api_key)
    if action == "segment_with_text_prompts":
        return _build_text_prompt_params(request)
    raise LegacyHTTPError(
        501, f"SAM action '{action}' is not supported on inference_server."
    )


def _build_sam_segment_params(request: Any, api_key: Optional[str]) -> dict:
    if getattr(request, "embeddings", None):
        raise LegacyHTTPError(501, _EMBEDDINGS_INPUT_UNSUPPORTED_MESSAGE)
    image = getattr(request, "image", None)
    image_id = getattr(request, "image_id", None)
    if not image and not image_id:
        raise ValueError("Must provide either image, cached image_id, or embeddings")
    params: dict = {"multi_mask_output": False}
    if getattr(request, "has_mask_input", False):
        if getattr(request, "mask_input", None) is not None:
            raise LegacyHTTPError(501, _MASK_INPUT_UNSUPPORTED_MESSAGE)
        if not getattr(request, "use_mask_input_cache", True):
            raise LegacyHTTPError(
                501,
                "has_mask_input without use_mask_input_cache is not supported on "
                "inference_server.",
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
        params["image_hashes"] = [namespace_client_hash_id(image_id, api_key)]
    response_format = getattr(request, "format", None)
    if response_format != "json":
        raise ValueError(f"Invalid format {response_format}")
    return params


def _build_sam2_segment_params(request: Any, api_key: Optional[str]) -> dict:
    response_format = getattr(request, "format", None)
    if response_format not in ("json", "rle"):
        raise ValueError(f"Invalid format {response_format}")
    params = _build_visual_prompt_params(request, api_key)
    if not any(key in params for key in ("point_coordinates", "point_labels", "boxes")):
        params["point_coordinates"] = [[[0, 0]]]
        params["point_labels"] = [[-1]]
    params["return_logits"] = True
    return params


def _build_visual_prompt_params(request: Any, api_key: Optional[str]) -> dict:
    if getattr(request, "mask_input", None) is not None or getattr(
        request, "has_mask_input", False
    ):
        raise LegacyHTTPError(501, _MASK_INPUT_UNSUPPORTED_MESSAGE)
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
        params["image_hashes"] = [namespace_client_hash_id(image_id, api_key)]
    if getattr(request, "load_logits_from_cache", False):
        params["load_from_mask_input_cache"] = (
            not configuration.DISABLE_SAM3_LOGITS_CACHE
        )
    if getattr(request, "save_logits_to_cache", False):
        params["save_to_mask_input_cache"] = not configuration.DISABLE_SAM3_LOGITS_CACHE
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
    prompts = getattr(request, "prompts", None)
    if not prompts:
        raise LegacyHTTPError(
            501, "SAM3 concept segmentation requires prompts on inference_server."
        )
    threshold = float(getattr(request, "output_prob_thresh", None) or 0.5)
    for prompt in prompts:
        prompt_threshold = getattr(prompt, "output_prob_thresh", None)
        if prompt_threshold is not None:
            threshold = min(threshold, float(prompt_threshold))
    return {
        "prompts": [prompt.model_dump() for prompt in prompts],
        "output_prob_thresh": threshold,
    }


def repack_interactive_segmentation_response(
    action: str, prediction: Any, request: Any, api_key: Optional[str]
) -> BaseModel:
    if action in ("embed", "embed_images"):
        return _repack_sam_embeddings(action, prediction, request, api_key)
    if action == "segment":
        if type(request).__name__ == "SamSegmentationRequest":
            return _repack_sam_segmentation(prediction)
        return _repack_sam2_segmentation(prediction, request)
    if action == "segment_with_visual_prompts":
        return _repack_visual_segmentation(prediction, request)
    if action == "segment_with_text_prompts":
        return _repack_text_segmentation(prediction, request)
    raise LegacyHTTPError(
        501,
        f"No response translation for SAM action '{action}' on inference_server.",
    )


def _repack_sam_embeddings(
    action: str, prediction: Any, request: Any, api_key: Optional[str]
) -> BaseModel:
    embeddings_obj = unwrap_single_prediction(prediction)
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
            image_id = strip_tenant_namespace(image_id, api_key)
    if action == "embed_images":
        return Sam3EmbeddingResponse(image_id=image_id, time=0.0)
    return Sam2EmbeddingResponse(image_id=image_id, time=0.0)


def _repack_sam_segmentation(prediction: Any) -> SamSegmentationResponse:
    result = unwrap_single_prediction(prediction)
    masks = np.asarray(result.masks)
    if masks.dtype != np.bool_:
        masks = masks > 0.0
    low_res_masks = np.asarray(result.logits) > 0.0
    return SamSegmentationResponse(
        masks=[polygon.tolist() for polygon in masks2poly(masks)],
        low_res_masks=[polygon.tolist() for polygon in masks2poly(low_res_masks)],
        time=0.0,
    )


def _repack_sam2_segmentation(
    prediction: Any, request: Any
) -> Sam2SegmentationResponse:
    result = unwrap_single_prediction(prediction)
    masks, scores = _choose_most_confident_sam_masks(result.masks, result.scores)
    masks = np.asarray(masks) >= 0.0
    predictions = _sam_masks_to_predictions(
        masks, scores, getattr(request, "format", "json"), Sam2SegmentationPrediction
    )
    return Sam2SegmentationResponse(predictions=predictions, time=0.0)


def _repack_visual_segmentation(
    prediction: Any, request: Any
) -> Sam2SegmentationResponse:
    result = unwrap_single_prediction(prediction)
    if isinstance(result, dict):
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
    prompt_outputs = prediction
    if isinstance(prompt_outputs, dict):
        prompt_outputs = [prompt_outputs]
    if not isinstance(prompt_outputs, list) or not all(
        isinstance(output, dict) for output in prompt_outputs
    ):
        raise LegacyHTTPError(
            500,
            "Unexpected SAM3 text-prompt prediction shape from the inference backend.",
        )
    prompts = list(getattr(request, "prompts", None) or [])
    response_format = getattr(request, "format", "polygon")
    decoded: List[Tuple[int, np.ndarray, List[float]]] = []
    for output in prompt_outputs:
        index = int(output.get("prompt_index", len(decoded)))
        raw_masks = output.get("masks")
        if raw_masks is None:
            raw_masks = []
        if isinstance(raw_masks, list) and raw_masks and isinstance(raw_masks[0], dict):
            masks = _decode_coco_rle_masks(raw_masks)
        else:
            masks = np.asarray(raw_masks)
        scores = [float(score) for score in output.get("scores", [])]
        decoded.append((index, masks, scores))

    nms_iou_threshold = getattr(request, "nms_iou_threshold", None)
    if nms_iou_threshold is not None and prompts:
        return _repack_text_segmentation_with_nms(
            decoded, prompts, request, response_format, float(nms_iou_threshold)
        )

    prompt_results = []
    for index, masks, scores in decoded:
        prompt = prompts[index] if index < len(prompts) else None
        prompt_threshold = getattr(prompt, "output_prob_thresh", None)
        if prompt_threshold is not None:
            kept = [i for i, score in enumerate(scores) if score >= prompt_threshold]
            masks = masks[kept] if len(kept) else masks[:0]
            scores = [scores[i] for i in kept]
        prompt_results.append(
            Sam3PromptResult(
                prompt_index=index,
                echo=_sam3_prompt_echo(index, prompt),
                predictions=_sam_masks_to_predictions(
                    masks, scores, response_format, Sam3SegmentationPrediction
                ),
            )
        )
    return Sam3SegmentationResponse(prompt_results=prompt_results, time=0.0)


def _repack_text_segmentation_with_nms(
    decoded: List[Tuple[int, np.ndarray, List[float]]],
    prompts: List[Any],
    request: Any,
    response_format: Any,
    nms_iou_threshold: float,
) -> Sam3SegmentationResponse:
    default_threshold = float(getattr(request, "output_prob_thresh", None) or 0.5)
    collected: List[Tuple[int, np.ndarray, float]] = []
    for index, masks, scores in decoded:
        prompt = prompts[index] if index < len(prompts) else None
        prompt_threshold = getattr(prompt, "output_prob_thresh", None)
        if prompt_threshold is None:
            prompt_threshold = default_threshold
        if masks.ndim != 3 or 0 in masks.shape:
            continue
        for mask, score in zip(masks, scores):
            if score >= prompt_threshold:
                collected.append((index, mask, float(score)))
    collected = _apply_cross_prompt_nms(collected, nms_iou_threshold)
    regrouped: Dict[int, List[Tuple[np.ndarray, float]]] = {
        i: [] for i in range(len(prompts))
    }
    for index, mask, score in collected:
        regrouped[index].append((mask, score))
    prompt_results = []
    for index, prompt in enumerate(prompts):
        bucket = regrouped.get(index, [])
        if bucket:
            masks = np.stack([mask for mask, _ in bucket], axis=0)
            scores = [score for _, score in bucket]
        else:
            masks = np.zeros((0, 0, 0), dtype=np.uint8)
            scores = []
        prompt_results.append(
            Sam3PromptResult(
                prompt_index=index,
                echo=_sam3_prompt_echo(index, prompt),
                predictions=_sam_masks_to_predictions(
                    masks, scores, response_format, Sam3SegmentationPrediction
                ),
            )
        )
    return Sam3SegmentationResponse(prompt_results=prompt_results, time=0.0)


def _sam3_prompt_echo(index: int, prompt: Any) -> Sam3PromptEcho:
    has_visual = bool(getattr(prompt, "boxes", None))
    return Sam3PromptEcho(
        prompt_index=index,
        type="visual" if has_visual else "text",
        text=getattr(prompt, "text", None),
        num_boxes=len(getattr(prompt, "boxes", None) or []) if has_visual else 0,
    )


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


def _nms_greedy_pycocotools_rles(
    rles: List[dict], confidences: np.ndarray, iou_threshold: float
) -> np.ndarray:
    from pycocotools import mask as mask_utils

    num_detections = len(rles)
    if num_detections == 0:
        return np.array([], dtype=bool)
    sort_index = np.argsort(confidences)[::-1]
    sorted_rles = [rles[i] for i in sort_index]
    ious = mask_utils.iou(sorted_rles, sorted_rles, [0] * num_detections)
    keep = np.ones(num_detections, dtype=bool)
    for i in range(num_detections):
        if keep[i]:
            condition = ious[i, :] > iou_threshold
            keep[i + 1 :] = np.where(condition[i + 1 :], False, keep[i + 1 :])
    return keep[np.argsort(sort_index)]


def _apply_cross_prompt_nms(
    collected: List[Tuple[int, np.ndarray, float]], iou_threshold: float
) -> List[Tuple[int, np.ndarray, float]]:
    from pycocotools import mask as mask_utils

    if not collected:
        return collected
    rles = [
        mask_utils.encode(np.asfortranarray((mask > 0).astype(np.uint8)))
        for _, mask, _ in collected
    ]
    confidences = np.array([score for _, _, score in collected])
    keep = _nms_greedy_pycocotools_rles(rles, confidences, iou_threshold)
    return [collected[i] for i in range(len(collected)) if keep[i]]


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
        predictions = []
        for mask, score in zip(np.asarray(masks), scores):
            rle = _dense_mask_to_coco_rle(mask > 0)
            rle["counts"] = rle["counts"].decode("utf-8")
            predictions.append(
                prediction_cls(masks=rle, confidence=float(score), format="rle")
            )
        return predictions
    raise LegacyHTTPError(
        501, f"format={response_format!r} is not supported on inference_server."
    )
