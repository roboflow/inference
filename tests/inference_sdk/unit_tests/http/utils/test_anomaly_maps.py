import base64

import numpy as np
import pytest

from inference_sdk.http.utils.anomaly_maps import (
    decode_anomaly_detection_result,
    decode_anomaly_map,
)


def payload(anomaly_map: np.ndarray) -> dict:
    values = np.ascontiguousarray(anomaly_map, dtype="<f4")
    return {
        "shape": list(values.shape),
        "dtype": "float32",
        "data": base64.b64encode(values.tobytes()).decode("ascii"),
    }


def test_decode_anomaly_map_returns_exact_float32_values():
    anomaly_map = np.random.default_rng(0).random((3, 5), dtype=np.float32)

    decoded = decode_anomaly_map(payload(anomaly_map))

    assert decoded.dtype == np.float32
    assert np.array_equal(decoded, anomaly_map)


def test_decode_anomaly_map_rejects_truncated_payload():
    serialized = payload(np.zeros((2, 2), dtype=np.float32))
    serialized["data"] = serialized["data"][:8]

    with pytest.raises(ValueError, match="bytes"):
        decode_anomaly_map(serialized)


def test_decode_anomaly_detection_result_replaces_payload_with_array():
    anomaly_map = np.ones((2, 2), dtype=np.float32)
    result = {"top": "anomalous", "anomaly_map": payload(anomaly_map)}

    decoded = decode_anomaly_detection_result(result)

    assert decoded["top"] == "anomalous"
    assert np.array_equal(decoded["anomaly_map"], anomaly_map)


def test_decode_anomaly_detection_result_handles_lists_and_passes_others_through():
    results = [
        {"top": "normal", "anomaly_map": None},
        {"top": "A", "predictions": []},
        {
            "top": "anomalous",
            "anomaly_map": payload(np.zeros((1, 1), dtype=np.float32)),
        },
    ]

    decoded = decode_anomaly_detection_result(results)

    assert decoded[0] == {"top": "normal", "anomaly_map": None}
    assert decoded[1] == {"top": "A", "predictions": []}
    assert np.array_equal(decoded[2]["anomaly_map"], np.zeros((1, 1), dtype=np.float32))
