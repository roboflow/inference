import numpy as np
import pytest

from inference.core.utils.depth_encoding import (
    decode_png16_normalized_depth,
    encode_normalized_depth_to_png16,
)


def test_png16_round_trip_preserves_values_within_quantization_step():
    depth = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)

    decoded = decode_png16_normalized_depth(encode_normalized_depth_to_png16(depth))

    assert decoded.shape == (8, 8)
    assert decoded.dtype == np.float32
    assert np.allclose(decoded, depth, atol=1.0 / 65535)


def test_png16_round_trip_is_exact_at_bounds():
    depth = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    decoded = decode_png16_normalized_depth(encode_normalized_depth_to_png16(depth))

    assert np.array_equal(decoded, depth)


def test_png16_encoding_clips_out_of_range_values():
    depth = np.array([[-0.5, 1.5]], dtype=np.float32)

    decoded = decode_png16_normalized_depth(encode_normalized_depth_to_png16(depth))

    assert np.array_equal(decoded, np.array([[0.0, 1.0]], dtype=np.float32))


def test_png16_encoding_accepts_array_like_input():
    decoded = decode_png16_normalized_depth(
        encode_normalized_depth_to_png16([[0.25, 0.75]])
    )

    assert np.allclose(decoded, [[0.25, 0.75]], atol=1.0 / 65535)


def test_png16_payload_is_much_smaller_than_json_for_smooth_maps():
    rows = np.linspace(0.0, 1.0, 512, dtype=np.float32)
    depth = np.tile(rows, (512, 1))

    payload = encode_normalized_depth_to_png16(depth)

    json_size = len(str(depth.tolist()))
    assert len(payload) < json_size / 20


def test_png16_decoding_rejects_invalid_payload():
    with pytest.raises(ValueError):
        decode_png16_normalized_depth("not-a-png")
