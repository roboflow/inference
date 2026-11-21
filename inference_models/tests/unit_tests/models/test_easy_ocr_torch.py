from threading import Lock

import numpy as np

from inference_models.models.easy_ocr.easy_ocr_torch import EasyOCRTorch


class _CapturedFrame(np.ndarray):
    """An np.ndarray subclass, like roboflow-edge's shared-memory frames."""


class _Reader:
    def __init__(self):
        self.images = []

    def readtext(self, image):
        self.images.append(image)
        return []


def test_forward_passes_easyocr_a_plain_ndarray_for_a_subclass() -> None:
    # given
    model = EasyOCRTorch.__new__(EasyOCRTorch)
    model._model, model._lock = _Reader(), Lock()
    image = np.zeros((8, 8, 3), dtype=np.uint8).view(_CapturedFrame)

    # when
    result = model.forward([image])

    # then
    assert result == [[]]
    (passed,) = model._model.images
    # easyocr's reformat_input checks type(image) == np.ndarray
    assert type(passed) is np.ndarray
    assert np.shares_memory(passed, image)
