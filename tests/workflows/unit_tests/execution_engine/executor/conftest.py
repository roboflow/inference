import os.path
import tempfile
from typing import Generator

import cv2
import numpy as np
import pytest


@pytest.fixture(scope="function")
def example_image_file() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(tmp_dir, "image.jpg")
        cv2.imwrite(file_path, np.zeros((192, 168, 3), dtype=np.uint8))
        yield file_path
