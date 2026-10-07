import base64

import numpy as np
import pytest

from inference.core.utils.anomaly_map_encoding import (
    decode_anomaly_map,
    encode_anomaly_map,
)


def test_round_trip_is_exact_for_float32_values():
    anomaly_map = np.random.default_rng(0).gamma(2.0, 0.05, (28, 36)).astype(np.float32)
    anomaly_map[0, 0] = 0.0
    anomaly_map[1, 1] = np.finfo(np.float32).max
    anomaly_map[2, 2] = np.finfo(np.float32).tiny

    decoded = decode_anomaly_map(encode_anomaly_map(anomaly_map))

    assert decoded.dtype == np.float32
    assert decoded.shape == (28, 36)
    assert np.array_equal(decoded, anomaly_map)
    assert decoded.flags.writeable


def test_payload_is_little_endian_row_major_float32_with_its_shape():
    payload = encode_anomaly_map([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    assert payload["shape"] == [2, 3]
    assert payload["dtype"] == "float32"
    assert base64.b64decode(payload["data"]) == np.arange(1, 7, dtype="<f4").tobytes()


def test_encoding_accepts_array_like_input_of_other_dtypes():
    decoded = decode_anomaly_map(encode_anomaly_map(np.ones((2, 2), dtype=np.float64)))

    assert decoded.dtype == np.float32
    assert np.array_equal(decoded, np.ones((2, 2), dtype=np.float32))


def test_payload_is_a_quarter_of_the_json_float_list():
    anomaly_map = np.random.default_rng(1).random((64, 64), dtype=np.float32)

    payload = encode_anomaly_map(anomaly_map)

    assert len(payload["data"]) < len(str(anomaly_map.tolist())) / 4


@pytest.mark.parametrize("shape", [(5,), (2, 3, 4)])
def test_encoding_rejects_maps_that_are_not_2d(shape):
    with pytest.raises(ValueError, match="2D"):
        encode_anomaly_map(np.zeros(shape, dtype=np.float32))


def test_decoding_rejects_other_dtypes():
    payload = encode_anomaly_map(np.zeros((2, 2), dtype=np.float32))
    payload["dtype"] = "float16"

    with pytest.raises(ValueError, match="dtype"):
        decode_anomaly_map(payload)


@pytest.mark.parametrize("shape", [[4], [2, 2, 1], [0, 4], [2, "2"], "2x2", None])
def test_decoding_rejects_malformed_shapes(shape):
    payload = encode_anomaly_map(np.zeros((2, 2), dtype=np.float32))
    payload["shape"] = shape

    with pytest.raises(ValueError, match="shape"):
        decode_anomaly_map(payload)


def test_decoding_rejects_payload_that_does_not_match_its_shape():
    payload = encode_anomaly_map(np.zeros((2, 2), dtype=np.float32))
    payload["shape"] = [2, 3]

    with pytest.raises(ValueError, match="16 bytes, expected 24"):
        decode_anomaly_map(payload)


def test_decoding_rejects_invalid_base64():
    payload = encode_anomaly_map(np.zeros((2, 2), dtype=np.float32))
    payload["data"] = "not base64!"

    with pytest.raises(ValueError, match="base64"):
        decode_anomaly_map(payload)
