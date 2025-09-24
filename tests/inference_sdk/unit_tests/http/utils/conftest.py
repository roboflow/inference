import os.path
import tempfile
from typing import Generator, Tuple

import cv2
import numpy as np
import pytest


@pytest.fixture(scope="function")
def example_local_image() -> Generator[Tuple[str, np.ndarray], None, None]:
    with tempfile.TemporaryDirectory() as tmp_directory:
        file_path = os.path.join(tmp_directory, "file.jpg")
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        cv2.imwrite(file_path, image)
        yield file_path, image


@pytest.fixture(scope="function")
def example_local_text_file() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as tmp_directory:
        file_path = os.path.join(tmp_directory, "file.txt")
        with open(file_path, "w") as f:
            f.write("For sure not an image :)")
            yield file_path


@pytest.fixture(scope="function")
def example_directory_with_images() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as tmp_directory:
        file_path_1 = os.path.join(tmp_directory, "file_1.jpg")
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        cv2.imwrite(file_path_1, image)
        file_path_2 = os.path.join(tmp_directory, "file_2.png")
        cv2.imwrite(file_path_2, image)
        yield tmp_directory
