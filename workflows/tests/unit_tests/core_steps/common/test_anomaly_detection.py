import base64

import numpy as np
import pytest
from roboflow_workflows.core_steps.common.anomaly_detection import (
    coerce_anomaly_map,
    decode_anomaly_map,
    require_anomaly_response,
)


def _payload(values: np.ndarray) -> dict:
    array = np.ascontiguousarray(values, dtype="<f4")
    return {
        "shape": [int(array.shape[0]), int(array.shape[1])],
        "dtype": "float32",
        "data": base64.b64encode(array.tobytes()).decode("ascii"),
    }


def test_decode_anomaly_map_round_trips_exact_values() -> None:
    # given
    values = np.array([[0.0, 1.5], [-2.25, 1e6]], dtype=np.float32)

    # when
    decoded = decode_anomaly_map(_payload(values))

    # then
    assert decoded.dtype == np.float32
    assert decoded.flags.writeable
    np.testing.assert_array_equal(decoded, values)


@pytest.mark.parametrize(
    "broken",
    [
        {"shape": [2, 2], "dtype": "float16", "data": ""},
        {"shape": [2], "dtype": "float32", "data": ""},
        {"shape": [2, 2], "dtype": "float32", "data": "not base64!!"},
        {
            "shape": [2, 2],
            "dtype": "float32",
            "data": base64.b64encode(b"\0" * 4).decode(),
        },
    ],
)
def test_decode_anomaly_map_rejects_malformed_payloads(broken: dict) -> None:
    with pytest.raises(ValueError):
        decode_anomaly_map(broken)


def test_coerce_anomaly_map_accepts_every_wire_form() -> None:
    # given
    values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    # when / then
    assert coerce_anomaly_map(None) is None
    np.testing.assert_array_equal(coerce_anomaly_map(_payload(values)), values)
    np.testing.assert_array_equal(coerce_anomaly_map(values), values)
    np.testing.assert_array_equal(coerce_anomaly_map(values.tolist()), values)
    with pytest.raises(ValueError):
        coerce_anomaly_map(np.zeros((2, 2, 2), dtype=np.float32))


def test_require_anomaly_response_names_the_model_and_missing_keys() -> None:
    # given: a plain classifier response (the old anomaly shape had these too)
    plain = {"top": "cat", "confidence": 0.9, "predictions": []}

    # when / then
    with pytest.raises(ValueError, match="my/1") as error:
        require_anomaly_response(plain, model_id="my/1", include_anomaly_map=False)
    assert "anomaly_score" in str(error.value)


def test_require_anomaly_response_accepts_the_score_only_shape() -> None:
    # given: no predictions / top / confidence, as the server now answers
    response = {"anomaly_score": 1.5, "anomaly_threshold": 2.0, "is_anomalous": False}

    # when / then
    require_anomaly_response(response, model_id="my/1", include_anomaly_map=False)


def test_require_anomaly_response_demands_the_map_when_requested() -> None:
    # given
    response = {"anomaly_score": 1.5, "anomaly_threshold": 2.0, "is_anomalous": False}

    # when / then
    with pytest.raises(ValueError, match="anomaly_map"):
        require_anomaly_response(response, model_id="my/1", include_anomaly_map=True)
    require_anomaly_response(
        {**response, "anomaly_map": _payload(np.zeros((2, 2), dtype=np.float32))},
        model_id="my/1",
        include_anomaly_map=True,
    )
