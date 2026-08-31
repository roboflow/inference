import numpy as np

from inference_model_manager.errors import INPUT_ERROR_PREFIX
from inference_model_manager.marshalling import (
    model_supports_rle,
    tensors_to_numpy,
    to_bytes,
)


def test_input_error_prefix_value():
    assert INPUT_ERROR_PREFIX == "INPUT_ERROR: "


def test_to_bytes_roundtrips_ndarray_via_npy():
    arr = np.zeros((2, 2, 3), dtype=np.uint8)
    payload = to_bytes(arr)
    assert payload[:6] == b"\x93NUMPY"


def test_to_bytes_passthrough_bytes():
    assert to_bytes(b"abc") == b"abc"


def test_tensors_to_numpy_passthrough_plain_objects():
    obj = {"a": [1, 2], "b": (3,)}
    assert tensors_to_numpy(obj) == obj


class _RleModel:
    supported_mask_formats = {"rle", "dense"}


def test_model_supports_rle():
    assert model_supports_rle(_RleModel()) is True
    assert model_supports_rle(object()) is False
