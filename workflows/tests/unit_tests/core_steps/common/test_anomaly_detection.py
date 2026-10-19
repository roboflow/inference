import base64

import numpy as np
import pytest
from roboflow_workflows.core_steps.common.anomaly_detection import (
    apply_threshold_override,
    coerce_anomaly_map,
    decision_threshold,
    decode_anomaly_map,
    require_anomaly_response,
    validate_threshold_override,
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
    with pytest.raises(ValueError, match="my/1") as error:
        require_anomaly_response({"top": "cat", "confidence": 0.9}, model_id="my/1")
    assert "anomaly_score" in str(error.value)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), "0.5", True])
def test_validate_threshold_override_rejects_non_finite_or_non_numbers(invalid) -> None:
    with pytest.raises(ValueError):
        validate_threshold_override(invalid)


def test_validate_threshold_override_keeps_none_and_casts_ints() -> None:
    assert validate_threshold_override(None) is None
    assert validate_threshold_override(2) == 2.0


def _normal_response() -> dict:
    # The model said normal at its saved threshold of 2.0 with scale 0.5.
    return {
        "predictions": [
            {"class": "normal", "class_id": 0, "confidence": 0.7311},
            {"class": "anomalous", "class_id": 1, "confidence": 0.2689},
        ],
        "top": "normal",
        "confidence": 0.7311,
        "anomaly_score": 1.5,
        "anomaly_threshold": 2.0,
        "anomaly_scale": 0.5,
        "is_anomalous": False,
    }


def test_apply_threshold_override_re_decides_and_recomputes_confidences() -> None:
    # given
    response = _normal_response()

    # when: relative to 1.0 the anomalous confidence is sigmoid(1) ~ 0.7311
    result = apply_threshold_override(response, 1.0, confidence=0.5)

    # then
    assert result is response
    assert result["is_anomalous"] is True
    assert result["anomaly_threshold"] == 1.0
    assert result["top"] == "anomalous"
    assert result["confidence"] == 0.7311
    assert result["predictions"] == [
        {"class_id": 1, "class": "anomalous", "confidence": 0.7311},
        {"class_id": 0, "class": "normal", "confidence": 0.2689},
    ]


def test_confidence_applies_relative_to_the_threshold_override() -> None:
    # when: the same override, but the anomalous confidence must reach 0.8
    result = apply_threshold_override(_normal_response(), 1.0, confidence=0.8)

    # then
    assert result["is_anomalous"] is False
    assert result["anomaly_threshold"] == 1.0
    assert result["top"] == "normal"
    assert result["confidence"] == 0.2689


def test_apply_threshold_override_without_override_is_a_no_op() -> None:
    # given
    response = _normal_response()

    # when
    result = apply_threshold_override(_normal_response(), None, confidence=0.1)

    # then
    assert result == response


@pytest.mark.parametrize(
    "confidence, expected",
    [
        (0.5, 2.0),
        (0.8, 2.0 + 0.5 * 1.3862944),
        (0.0, 2.0 - 0.5 * 60),
        (1.0, 2.0 + 0.5 * 60),
    ],
)
def test_decision_threshold_is_where_the_anomalous_confidence_meets_confidence(
    confidence: float, expected: float
) -> None:
    assert decision_threshold(threshold=2.0, scale=0.5, confidence=confidence) == (
        pytest.approx(expected)
    )
