import os
import tempfile
from typing import Generator

import cv2
import numpy as np
import pytest

ASSETS_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "integration_tests",
        "execution",
        "assets",
    )
)


@pytest.fixture(scope="function")
def dogs_image() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "dogs.jpg"))


@pytest.fixture(scope="function")
def empty_directory() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir
