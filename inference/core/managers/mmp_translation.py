"""Legacy <-> new-world translation for ModelManagerAdapter.

Everything the adapter needs to speak both dialects: the static mirror of the
new-world handler registry, request forwarding (image -> bytes), per-task
param mapping, native-prediction -> legacy-response repack, and error
translation. Imports from the new packages are lazy so the legacy stack never
pays for them while the gate is off.
"""

from __future__ import annotations

import asyncio
import binascii
import io
from typing import Any, List, Optional, Tuple

import numpy as np
import pybase64

from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
)
from inference.core.exceptions import (
    InferenceModelNotFound,
    InferencePayloadTooLargeError,
    InputImageLoadError,
    InvalidImageTypeDeclared,
    ModelArtefactError,
    ModelDeploymentNotSupportedError,
    ModelManagerLockAcquisitionError,
    RoboflowAPIConnectionError,
    RoboflowAPINotAuthorizedError,
    RoboflowAPINotNotFoundError,
    RoboflowAPITimeoutError,
)
from inference.core.utils.image_utils import (
    BASE64_DATA_TYPE_PATTERN,
    fetch_image_bytes_from_url,
    load_image_from_numpy_object,
    load_image_from_numpy_str,
)

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
        ("object-detection", "infer"),
    ]
)

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
    if isinstance(getattr(request, "image", None), list):
        raise ModelDeploymentNotSupportedError(
            f"Batch requests are not supported for model '{model_id}' on the MMP path."
        )


def build_object_detection_params(request: Any) -> dict:
    params: dict = {}
    confidence = getattr(request, "confidence", None)
    if confidence is not None:
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ModelDeploymentNotSupportedError(
                f"confidence={confidence!r} is not supported on the MMP path."
            )
        params["confidence"] = float(confidence)
    iou_threshold = getattr(request, "iou_threshold", None)
    if iou_threshold is not None:
        params["iou_threshold"] = float(iou_threshold)
    max_detections = getattr(request, "max_detections", None)
    if max_detections is not None:
        params["max_detections"] = int(max_detections)
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
