import numpy as np
import pytest

from inference_model_manager.backends.decode import make_decoder
from inference_model_manager.backends.direct import DirectBackend


def test_encoded_rgb_is_converted_to_bgr_without_changing_raw_numpy():
    imagecodecs = pytest.importorskip("imagecodecs")
    rgb = np.array([[[255, 1, 7], [3, 5, 251]]], dtype=np.uint8)
    encoded = bytes(imagecodecs.png_encode(rgb))
    raw_bgr = rgb[..., ::-1].copy()
    backend = DirectBackend.__new__(DirectBackend)
    backend._decoder_name = "imagecodecs"
    backend._decode = make_decoder("imagecodecs", device="cpu")

    np.testing.assert_array_equal(backend._decode_input(encoded), raw_bgr)
    assert backend._decode_input(raw_bgr) is raw_bgr
