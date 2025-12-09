import os

import cv2
import numpy as np
import pytest
import requests
from filelock import FileLock

ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))
DOG_IMAGE_PATH = os.path.join(ASSETS_DIR, "images", "dog.jpeg")
DOG_IMAGE_URL = (
    "https://storage.googleapis.com/roboflow-tests-assets/test-images/dog.jpeg"
)
OCR_TEST_IMAGE_PATH = os.path.join(ASSETS_DIR, "ocr_test_image.png")
OCR_TEST_IMAGE_URL = "https://storage.googleapis.com/roboflow-tests-assets/test-images/ocr_test_image.png"
MAN_IMAGE_PATH = os.path.join(ASSETS_DIR, "man.jpg")
MAN_IMAGE_URL = (
    "https://storage.googleapis.com/roboflow-tests-assets/test-images/man.jpg"
)


@pytest.fixture()
def roboflow_api_key() -> str:
    return os.environ["ROBOFLOW_API_KEY"]


@pytest.fixture(scope="function")
def dog_image_numpy() -> np.ndarray:
    _download_if_not_exists(file_path=DOG_IMAGE_PATH, url=DOG_IMAGE_URL)
    image = cv2.imread(DOG_IMAGE_PATH)
    assert image is not None, "Could not load test image"
    return image


@pytest.fixture(scope="function")
def ocr_test_image_numpy() -> np.ndarray:
    _download_if_not_exists(file_path=OCR_TEST_IMAGE_PATH, url=OCR_TEST_IMAGE_URL)
    image = cv2.imread(OCR_TEST_IMAGE_PATH)
    assert image is not None, "Could not load OCR test image"
    return image


@pytest.fixture(scope="function")
def man_image_numpy() -> np.ndarray:
    _download_if_not_exists(file_path=MAN_IMAGE_PATH, url=MAN_IMAGE_URL)
    image = cv2.imread(MAN_IMAGE_PATH)
    assert image is not None, "Could not load OCR test image"
    return image


def _download_if_not_exists(file_path: str, url: str, lock_timeout: int = 180) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    lock_path = f"{file_path}.lock"
    with FileLock(lock_file=lock_path, timeout=lock_timeout):
        if os.path.exists(file_path):
            return None
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
