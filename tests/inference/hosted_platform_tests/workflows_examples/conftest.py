import os

import cv2
import numpy as np
import pytest

ASSETS_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "assets",
        "..",
        "..",
        "..",
        "..",
        "workflows",
        "integration_tests",
        "execution",
        "assets",
    )
)


@pytest.fixture(scope="function")
def crowd_image() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "crowd.jpg"))


@pytest.fixture(scope="function")
def license_plate_image() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "license_plate.jpg"))


@pytest.fixture(scope="function")
def dogs_image() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "dogs.jpg"))


@pytest.fixture(scope="function")
def asl_image() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "asl_image.jpg"))
