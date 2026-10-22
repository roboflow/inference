"""Shared helpers for the anomaly detection (PatchCore, FoundAD) model blocks.

The inference server answers an anomaly model through the classification route
with an ``AnomalyDetectionResponse``: the usual classification fields plus
``anomaly_score`` (raw, unbounded, larger means more anomalous),
``anomaly_threshold`` (the decision boundary saved with the model),
``anomaly_scale`` (the anomalous confidence is
``sigmoid((anomaly_score - anomaly_threshold) / anomaly_scale)``),
``is_anomalous`` (the anomalous confidence reached the request confidence) and,
on request, ``anomaly_map`` - the local evidence at the
network input resolution, serialised as ``{"shape": [h, w], "dtype":
"float32", "data": <base64 of row-major little-endian float32>}``.

``roboflow_workflows`` cannot import ``inference.core``, so the payload decoder
is duplicated here (it mirrors ``inference_sdk.http.utils.anomaly_maps``).
"""

import base64
import binascii
import math
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

ANOMALY_SCORE_KEY = "anomaly_score"
ANOMALY_THRESHOLD_KEY = "anomaly_threshold"
ANOMALY_SCALE_KEY = "anomaly_scale"
ANOMALY_DECISION_THRESHOLD_KEY = "anomaly_decision_threshold"
IS_ANOMALOUS_KEY = "is_anomalous"
ANOMALY_MAP_KEY = "anomaly_map"
ANOMALY_METADATA_KEYS = (
    ANOMALY_SCORE_KEY,
    ANOMALY_THRESHOLD_KEY,
    ANOMALY_SCALE_KEY,
    IS_ANOMALOUS_KEY,
)

ANOMALY_MAP_DTYPE = "float32"
_ANOMALY_MAP_NUMPY_DTYPE = np.dtype("<f4")

NORMAL_CLASS_NAME = "normal"
ANOMALOUS_CLASS_NAME = "anomalous"
# Mirrors inference_models: the sigmoid saturates far before this margin, clipping
# only avoids overflow.
MAX_CONFIDENCE_MARGIN = 60.0


def decode_anomaly_map(payload: Mapping[str, Any]) -> np.ndarray:
    """Decode the ``anomaly_map`` payload of an anomaly detection response.

    Args:
        payload (Mapping[str, Any]): Mapping with ``shape`` (``[height, width]``),
            ``dtype`` (``"float32"``) and ``data`` (base64 of row-major
            little-endian float32 bytes), as the server serialises it.

    Returns:
        Writable float32 array of shape ``(height, width)`` at the network input
        resolution; stretch it to the image size to overlay it.

    Raises:
        ValueError: If the dtype is not float32, the shape is not two positive
            ints, the data is not valid base64, or its length does not match
            the shape.
    """
    if payload.get("dtype") != ANOMALY_MAP_DTYPE:
        raise ValueError(
            f"Expected anomaly map dtype {ANOMALY_MAP_DTYPE!r}, got "
            f"{payload.get('dtype')!r}"
        )
    shape = payload.get("shape")
    if (
        not isinstance(shape, (list, tuple))
        or len(shape) != 2
        or not all(isinstance(side, int) and side > 0 for side in shape)
    ):
        raise ValueError(f"Anomaly map shape must be two positive ints, got {shape!r}")
    try:
        data = base64.b64decode(payload.get("data", ""), validate=True)
    except (binascii.Error, ValueError, TypeError) as error:
        raise ValueError("Anomaly map payload is not valid base64") from error
    expected_bytes = shape[0] * shape[1] * _ANOMALY_MAP_NUMPY_DTYPE.itemsize
    if len(data) != expected_bytes:
        raise ValueError(
            f"Anomaly map payload holds {len(data)} bytes, expected {expected_bytes} "
            f"for shape {tuple(shape)}"
        )
    values = np.frombuffer(data, dtype=_ANOMALY_MAP_NUMPY_DTYPE).reshape(shape)
    return values.astype(np.float32, copy=True)


def coerce_anomaly_map(value: Any) -> Optional[np.ndarray]:
    """The map as a float32 ``(h, w)`` array, whatever form the response carried.

    Args:
        value (Any): ``None`` (map not requested), the serialised payload
            mapping (local execution dumps the response as is), an ``np.ndarray``
            (``inference_sdk`` decodes the payload on the remote path) or a
            nested float list from an older server.

    Returns:
        ``None`` when ``value`` is ``None``, else a float32 ``(h, w)`` array.

    Raises:
        ValueError: If the value is not one of the forms above or is not 2D.
    """
    if value is None:
        return None
    if isinstance(value, Mapping):
        return decode_anomaly_map(value)
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Anomaly map must be a payload mapping, an array or a nested list, "
            f"got {type(value).__name__}"
        ) from error
    if array.ndim != 2:
        raise ValueError(f"Anomaly map must be 2D, got shape {tuple(array.shape)}")
    return array


def require_anomaly_response(prediction: Dict[str, Any], model_id: str) -> None:
    """Fail loudly when a classification response is not an anomaly response.

    Args:
        prediction (Dict[str, Any]): The dumped classification response.
        model_id (str): The model that produced it, for the error message.

    Raises:
        ValueError: If ``anomaly_score``, ``anomaly_threshold``,
            ``anomaly_scale`` or ``is_anomalous`` is missing - the model is a
            plain classifier, not a PatchCore or FoundAD anomaly model.
    """
    missing = [key for key in ANOMALY_METADATA_KEYS if key not in prediction]
    if missing:
        raise ValueError(
            f"Model {model_id!r} did not return an anomaly detection response "
            f"(missing {missing}). The Anomaly Detection Model block needs a "
            f"PatchCore or FoundAD model trained on an anomaly detection project; "
            f"use the Single-Label Classification Model block for other classifiers."
        )


def validate_threshold_override(threshold: Optional[float]) -> Optional[float]:
    """A finite float, or ``None`` to keep the threshold saved with the model.

    Raises:
        ValueError: If ``threshold`` is not a finite number.
    """
    if threshold is None:
        return None
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError(
            f"anomaly_threshold must be a number, got {type(threshold).__name__}"
        )
    value = float(threshold)
    if not math.isfinite(value):
        raise ValueError(f"anomaly_threshold must be finite, got {value}")
    return value


def decision_threshold(threshold: float, scale: float, confidence: float) -> float:
    """The raw score at which the anomalous confidence equals ``confidence``.

    ``threshold + scale * logit(confidence)``, so ``threshold`` itself at 0.5. The
    logit is clipped like the sigmoid, keeping the result finite at 0 and 1.
    """
    if confidence <= 0:
        margin = -MAX_CONFIDENCE_MARGIN
    elif confidence >= 1:
        margin = MAX_CONFIDENCE_MARGIN
    else:
        margin = math.log(confidence / (1 - confidence))
        margin = max(-MAX_CONFIDENCE_MARGIN, min(MAX_CONFIDENCE_MARGIN, margin))
    return threshold + scale * margin


def anomalous_confidence(score: float, threshold: float, scale: float) -> float:
    """``sigmoid((score - threshold) / scale)``, exactly 0.5 at ``threshold``."""
    margin = (score - threshold) / scale
    margin = max(-MAX_CONFIDENCE_MARGIN, min(MAX_CONFIDENCE_MARGIN, margin))
    return 1.0 / (1.0 + math.exp(-margin))


def redecide_at_threshold(
    verdict: Dict[str, Any], threshold: float, confidence: float
) -> float:
    """Re-decide a verdict against a caller-supplied threshold, in place.

    The rule is the server's with the override in place of the saved threshold:
    ``is_anomalous`` becomes ``sigmoid((anomaly_score - threshold) /
    anomaly_scale) >= confidence`` and ``anomaly_threshold`` reports the
    override.

    Args:
        verdict (Dict[str, Any]): Holds ``anomaly_score`` and ``anomaly_scale``.
        threshold (float): The override, already validated.
        confidence (float): The anomalous confidence an image must reach.

    Returns:
        The anomalous confidence relative to the override.
    """
    anomalous = anomalous_confidence(
        score=float(verdict[ANOMALY_SCORE_KEY]),
        threshold=threshold,
        scale=float(verdict[ANOMALY_SCALE_KEY]),
    )
    verdict[ANOMALY_THRESHOLD_KEY] = threshold
    verdict[IS_ANOMALOUS_KEY] = anomalous >= confidence
    return anomalous


def apply_threshold_override(
    prediction: Dict[str, Any], threshold: Optional[float], confidence: float
) -> Dict[str, Any]:
    """Re-decide a dumped anomaly detection response against an override, in place.

    See ``redecide_at_threshold`` for the verdict. ``predictions`` is rebuilt with
    both class confidences relative to the override, the decided class first, and
    ``top`` / ``confidence`` follow it. With ``threshold`` ``None`` the response
    is unchanged: the server already decided against the saved threshold with the
    same ``confidence``.

    Args:
        prediction (Dict[str, Any]): A dumped anomaly detection response.
        threshold (Optional[float]): The override, already validated.
        confidence (float): The anomalous confidence an image must reach.

    Returns:
        The same dict.
    """
    if threshold is None:
        return prediction
    anomalous = redecide_at_threshold(prediction, threshold, confidence)
    class_predictions: List[Dict[str, Any]] = [
        {
            "class_id": 0,
            "class": NORMAL_CLASS_NAME,
            "confidence": round(1 - anomalous, 4),
        },
        {
            "class_id": 1,
            "class": ANOMALOUS_CLASS_NAME,
            "confidence": round(anomalous, 4),
        },
    ]
    if prediction[IS_ANOMALOUS_KEY]:
        class_predictions.reverse()
    prediction["predictions"] = class_predictions
    prediction["top"] = class_predictions[0]["class"]
    prediction["confidence"] = class_predictions[0]["confidence"]
    return prediction
