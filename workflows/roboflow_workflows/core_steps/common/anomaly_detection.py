"""Shared helpers for the anomaly detection (PatchCore, FoundAD) model blocks.

The inference server answers an anomaly model through the classification route
with an ``AnomalyDetectionResponse``: the usual classification fields plus
``anomaly_score`` (raw, unbounded, larger means more anomalous),
``anomaly_threshold`` (the decision boundary saved with the model),
``is_anomalous`` and, on request, ``anomaly_map`` - the local evidence at the
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
IS_ANOMALOUS_KEY = "is_anomalous"
ANOMALY_MAP_KEY = "anomaly_map"
ANOMALY_METADATA_KEYS = (ANOMALY_SCORE_KEY, ANOMALY_THRESHOLD_KEY, IS_ANOMALOUS_KEY)

ANOMALY_MAP_DTYPE = "float32"
_ANOMALY_MAP_NUMPY_DTYPE = np.dtype("<f4")

NORMAL_CLASS_NAME = "normal"
ANOMALOUS_CLASS_NAME = "anomalous"


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
        ValueError: If ``anomaly_score``, ``anomaly_threshold`` or
            ``is_anomalous`` is missing - the model is a plain classifier, not
            a PatchCore or FoundAD anomaly model.
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


def apply_threshold_override(
    prediction: Dict[str, Any], threshold: Optional[float]
) -> Dict[str, Any]:
    """Re-decide the verdict against a caller-supplied threshold, in place.

    ``is_anomalous`` becomes ``anomaly_score >= threshold`` and
    ``anomaly_threshold`` reports the override. ``predictions`` is reordered so
    the decided class comes first, and ``top`` / ``confidence`` follow it. The
    per-class confidences are left as served: they are a sigmoid centred on the
    threshold SAVED with the model, so under an override 0.5 no longer marks the
    decision boundary. With ``threshold`` ``None`` the response is unchanged.

    Args:
        prediction (Dict[str, Any]): A dumped anomaly detection response.
        threshold (Optional[float]): The override, already validated.

    Returns:
        The same dict.
    """
    if threshold is None:
        return prediction
    is_anomalous = float(prediction[ANOMALY_SCORE_KEY]) >= threshold
    prediction[ANOMALY_THRESHOLD_KEY] = threshold
    prediction[IS_ANOMALOUS_KEY] = is_anomalous
    decided_class = ANOMALOUS_CLASS_NAME if is_anomalous else NORMAL_CLASS_NAME
    class_predictions: List[Dict[str, Any]] = list(prediction.get("predictions") or [])
    decided = [p for p in class_predictions if p.get("class") == decided_class]
    others = [p for p in class_predictions if p.get("class") != decided_class]
    if decided:
        prediction["predictions"] = decided + others
        prediction["top"] = decided[0]["class"]
        prediction["confidence"] = decided[0]["confidence"]
    return prediction
