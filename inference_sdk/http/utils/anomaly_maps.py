"""Decoding of anomaly detection heatmaps (PatchCore, FoundAD) sent by the server.

The map is served at the network input resolution as exact float32 values, so
clients get the model's own evidence and resize it to the image themselves.
Floats carry no structure an entropy coder can exploit, so the payload is a
plain base64 string of little-endian row-major float32 bytes next to its shape.
"""

import base64
import binascii
from typing import Any, List, Mapping, Union

import numpy as np

ANOMALY_MAP_DTYPE = "float32"
_ANOMALY_MAP_NUMPY_DTYPE = np.dtype("<f4")


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


def decode_anomaly_detection_result(
    result: Union[dict, List[dict]],
) -> Union[dict, List[dict]]:
    """Replace a serialized `anomaly_map` payload with a float32 numpy array.

    Results without an `anomaly_map` (the map was not requested, or the model is
    not an anomaly detector) pass through unchanged.
    """
    if isinstance(result, list):
        return [decode_anomaly_detection_result(element) for element in result]
    anomaly_map = result.get("anomaly_map")
    if isinstance(anomaly_map, Mapping):
        result["anomaly_map"] = decode_anomaly_map(anomaly_map)
    return result
