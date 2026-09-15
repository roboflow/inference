import os.path

import cv2
import numpy as np
import pytest

ASSETS_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "assets",
    )
)


@pytest.fixture(scope="function")
def barcode_image() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "barcodes.png"))


@pytest.fixture(scope="function")
def qr_codes_image() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "qr.png"))
