"""Batch decode isolation: one corrupt image must not fail the batch,
and _is_heif must not match arbitrary ISO-BMFF (e.g. MP4) files."""

from __future__ import annotations

import imagecodecs
import numpy as np

from inference_model_manager.backends.decode import _is_heif, make_batch_decoder


def _png_bytes() -> bytes:
    return imagecodecs.png_encode(np.zeros((2, 2, 3), dtype=np.uint8))


def test_corrupt_image_isolated_to_its_index():
    decode = make_batch_decoder("cpu")
    good = memoryview(_png_bytes())
    bad = memoryview(b"\x89PNG garbage not an image")
    out = decode([good, bad, good])
    assert out[0] is not None and tuple(out[0].shape) == (3, 2, 2)
    assert out[1] is None
    assert out[2] is not None


def test_is_heif_rejects_mp4_accepts_heic():
    mp4 = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
    heic = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00heicmif1"
    avif = b"\x00\x00\x00\x18ftypavif\x00\x00\x00\x00avifmif1"
    assert _is_heif(mp4) is False
    assert _is_heif(heic) is True
    assert _is_heif(avif) is True
