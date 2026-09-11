"""Representation-agnostic readers for decoded VLM ``predictions``.

``common/vlm_decoding`` hands a VLM block ``sv.Detections`` / a classification
dict with the numpy representation and the tensor-native
``inference_models`` carriers under ``ENABLE_TENSOR_DATA_REPRESENTATION`` (see
``vlm_decoding/tensor_native.py``). The per-block suites assert the DECODED
VALUES, which are identical in both modes, so they read them through these
helpers and stay green in both CI lanes. Tests that pin one representation
(e.g. the parity check against the deprecated tensor formatter) skip on the
flag instead.
"""

from typing import Any, List

import supervision as sv
from supervision.config import CLASS_NAME_DATA_FIELD

from inference.core.workflows.execution_engine.constants import (
    CLASS_NAME_KEY,
    CLASS_NAMES_KEY,
    INFERENCE_ID_KEY,
)


def is_detection_prediction(predictions: Any) -> bool:
    """Whether the value is a detection prediction in either representation."""
    if isinstance(predictions, sv.Detections):
        return True
    return hasattr(predictions, "xyxy") and hasattr(predictions, "image_metadata")


def is_classification_prediction(predictions: Any) -> bool:
    """Whether the value is a classification prediction in either representation."""
    if isinstance(predictions, dict):
        return "predictions" in predictions
    return hasattr(predictions, "confidence") and (
        hasattr(predictions, "images_metadata")
        or hasattr(predictions, "image_metadata")
    )


def detection_count(predictions: Any) -> int:
    if isinstance(predictions, sv.Detections):
        return len(predictions)
    return int(predictions.xyxy.shape[0])


def detection_boxes(predictions: Any) -> List[List[float]]:
    if isinstance(predictions, sv.Detections):
        return predictions.xyxy.tolist()
    return predictions.xyxy.detach().cpu().tolist()


def detection_class_ids(predictions: Any) -> List[int]:
    if isinstance(predictions, sv.Detections):
        return predictions.class_id.tolist()
    return predictions.class_id.detach().cpu().tolist()


def detection_class_names(predictions: Any) -> List[str]:
    if isinstance(predictions, sv.Detections):
        return predictions.data[CLASS_NAME_DATA_FIELD].tolist()
    return [entry[CLASS_NAME_KEY] for entry in predictions.bboxes_metadata or []]


def detection_confidences(predictions: Any) -> List[float]:
    if isinstance(predictions, sv.Detections):
        return predictions.confidence.tolist()
    return predictions.confidence.detach().cpu().tolist()


def detection_inference_ids(predictions: Any) -> List[str]:
    """The inference id of every box - per-box data numpy-side, a single
    ``image_metadata`` entry tensor-side."""
    if isinstance(predictions, sv.Detections):
        return predictions.data[INFERENCE_ID_KEY].tolist()
    return [predictions.image_metadata[INFERENCE_ID_KEY]] * detection_count(predictions)


def _classification_image_metadata(predictions: Any) -> dict:
    images_metadata = getattr(predictions, "images_metadata", None)
    if images_metadata is not None:
        return images_metadata[0]
    return predictions.image_metadata


def classification_top_class(predictions: Any) -> str:
    if isinstance(predictions, dict):
        return predictions["top"]
    image_metadata = _classification_image_metadata(predictions)
    top_class_id = int(predictions.class_id.detach().cpu().reshape(-1)[0])
    return image_metadata[CLASS_NAMES_KEY][top_class_id]


def classification_top_confidence(predictions: Any) -> float:
    if isinstance(predictions, dict):
        return predictions["confidence"]
    top_class_id = int(predictions.class_id.detach().cpu().reshape(-1)[0])
    confidence_vector = predictions.confidence.detach().cpu().reshape(-1).tolist()
    return confidence_vector[top_class_id]


def classification_predicted_classes(predictions: Any) -> List[str]:
    """Predicted class names of a multi-label prediction."""
    if isinstance(predictions, dict):
        return predictions["predicted_classes"]
    image_metadata = _classification_image_metadata(predictions)
    return [
        image_metadata[CLASS_NAMES_KEY][int(class_id)]
        for class_id in predictions.class_ids.detach().cpu().tolist()
    ]


def classification_inference_id(predictions: Any) -> str:
    if isinstance(predictions, dict):
        return predictions[INFERENCE_ID_KEY]
    return _classification_image_metadata(predictions)[INFERENCE_ID_KEY]
