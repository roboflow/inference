from __future__ import annotations

import base64
import io
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel

from inference_server.legacy.bridge import Route
from inference_server.legacy.entities import (
    ClassificationInferenceResponse,
    InferenceResponseImage,
    InstanceSegmentationInferenceResponse,
    InstanceSegmentationPrediction,
    InstanceSegmentationRLEPrediction,
    Keypoint,
    KeypointsDetectionInferenceResponse,
    KeypointsPrediction,
    MultiLabelClassificationInferenceResponse,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
    Point,
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
