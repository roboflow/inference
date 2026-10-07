"""Wire format for anomaly detection heatmaps (PatchCore, FoundAD).

The map is served at the network input resolution as exact float32 values, so
clients get the model's own evidence and resize it to the image themselves.
Floats carry no structure an entropy coder can exploit, so the payload is a
plain base64 string of little-endian row-major float32 bytes next to its shape.
"""

import base64
import binascii
from typing import Any, Dict, Mapping

import numpy as np

ANOMALY_MAP_DTYPE = "float32"
_ANOMALY_MAP_NUMPY_DTYPE = np.dtype("<f4")


def encode_anomaly_map(anomaly_map: Any) -> Dict[str, Any]:
    """Encode a 2D anomaly map as `{"shape": [height, width], "dtype": "float32",
    "data": <base64 of little-endian row-major float32 bytes>}`."""
    values = np.ascontiguousarray(anomaly_map, dtype=_ANOMALY_MAP_NUMPY_DTYPE)
    if values.ndim != 2:
        raise ValueError(
            f"Anomaly map must be a 2D array, got shape {tuple(values.shape)}"
        )
    return {
        "shape": [int(values.shape[0]), int(values.shape[1])],
        "dtype": ANOMALY_MAP_DTYPE,
        "data": base64.b64encode(values.tobytes()).decode("ascii"),
    }


def decode_anomaly_map(payload: Mapping[str, Any]) -> np.ndarray:
    """Decode a payload produced by `encode_anomaly_map` to a float32 array of
    shape `(height, width)`."""
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
