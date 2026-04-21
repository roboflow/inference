import os.path
import zipfile

import cv2
import numpy as np
import pytest
import requests
import torch
import torchvision.io
from filelock import FileLock
from PIL import Image

ASSETS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "models", "assets")
)
DOG_IMAGE_PATH = os.path.join(ASSETS_DIR, "dog.jpeg")
DOG_IMAGE_URL = (
    "https://storage.googleapis.com/roboflow-tests-assets/test-images/dog.jpeg"
)
OCR_TEST_IMAGE_PATH = os.path.join(ASSETS_DIR, "ocr_test_image.png")
OCR_TEST_IMAGE_URL = "https://storage.googleapis.com/roboflow-tests-assets/test-images/ocr_test_image.png"
MAN_IMAGE_PATH = os.path.join(ASSETS_DIR, "man.jpg")
MAN_IMAGE_URL = (
    "https://storage.googleapis.com/roboflow-tests-assets/test-images/man.jpg"
)
BIKE_IMAGE_URL = "https://media.roboflow.com/inference/example-input-images/bike.jpg"
BIKE_IMAGE_PATH = os.path.join(ASSETS_DIR, "bike.jpg")
ASL_IMAGE_URL = (
    "https://media.roboflow.com/inference/example-input-images/asl-image.jpg"
)
ASL_IMAGE_PATH = os.path.join(ASSETS_DIR, "asl-image.jpg")
BALLOONS_IMAGE_URL = (
    "https://media.roboflow.com/inference/example-input-images/balloons.jpg"
)
BALLOONS_IMAGE_PATH = os.path.join(ASSETS_DIR, "balloons.jpg")
FLOWERS_IMAGE_URL = (
    "https://media.roboflow.com/inference/example-input-images/flowers.jpg"
)
FLOWERS_IMAGE_PATH = os.path.join(ASSETS_DIR, "flowers.jpg")
CHESS_SET_IMAGE_URL = (
    "https://media.roboflow.com/inference/example-input-images/chess_set.jpg"
)
CHESS_SET_IMAGE_PATH = os.path.join(ASSETS_DIR, "chess_set.jpg")
CHESS_PIECE_IMAGE_URL = (
    "https://media.roboflow.com/inference/example-input-images/chess_piece.jpg"
)
CHESS_PIECE_IMAGE_PATH = os.path.join(ASSETS_DIR, "chess_piece.jpg")

COIN_COUNTING_IMAGE_URL = (
    "https://media.roboflow.com/inference/example-input-images/image-coin-counting.jpg"
)
COIN_COUNTING_IMAGE_PATH = os.path.join(ASSETS_DIR, "image-coin-counting.jpg")
PEOPLE_WALKING_IMAGE_URL = (
    "https://media.roboflow.com/inference/example-input-images/people-walking.jpg"
)
PEOPLE_WALKING_IMAGE_PATH = os.path.join(ASSETS_DIR, "people-walking.jpg")
SNAKE_IMAGE_URL = "https://media.roboflow.com/inference/example-input-images/snake.jpg"
SNAKE_IMAGE_PATH = os.path.join(ASSETS_DIR, "snake.jpg")

# ORIGIN OF THE IMAGE https://github.com/facebookresearch/sam
TRUCK_IMAGE_URL = "https://media.roboflow.com/inference/example-input-images/truck.jpg"
TRUCK_IMAGE_PATH = os.path.join(ASSETS_DIR, "truck.jpg")

SUNFLOWERS_IMAGE_URL = (
    "https://storage.googleapis.com/roboflow-tests-assets/test-images/sunflowers.jpg"
)
SUNFLOWERS_IMAGE_PATH = os.path.join(ASSETS_DIR, "sunflowers.jpg")

BASKETBALL_IMAGE_URL = (
    "https://storage.googleapis.com/roboflow-tests-assets/test-images/basketball.jpg"
)
BASKETBALL_IMAGE_PATH = os.path.join(ASSETS_DIR, "basketball.jpg")


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


@pytest.fixture(scope="function")
def bike_image_numpy() -> np.ndarray:
    _download_if_not_exists(file_path=BIKE_IMAGE_PATH, url=BIKE_IMAGE_URL)
    image = cv2.imread(BIKE_IMAGE_PATH)
    assert image is not None, "Could not load test image"
    return image


@pytest.fixture(scope="function")
def bike_image_torch() -> torch.Tensor:
    _download_if_not_exists(file_path=BIKE_IMAGE_PATH, url=BIKE_IMAGE_URL)
    return torchvision.io.read_image(BIKE_IMAGE_PATH)


@pytest.fixture(scope="function")
def asl_image_numpy() -> np.ndarray:
    _download_if_not_exists(file_path=ASL_IMAGE_PATH, url=ASL_IMAGE_URL)
    image = cv2.imread(ASL_IMAGE_PATH)
    assert image is not None, "Could not load test image"
    return image


@pytest.fixture(scope="function")
def asl_image_torch() -> torch.Tensor:
    _download_if_not_exists(file_path=ASL_IMAGE_PATH, url=ASL_IMAGE_URL)
    return torchvision.io.read_image(ASL_IMAGE_PATH)


@pytest.fixture(scope="function")
def balloons_image_numpy() -> np.ndarray:
    _download_if_not_exists(file_path=BALLOONS_IMAGE_PATH, url=BALLOONS_IMAGE_URL)
    image = cv2.imread(BALLOONS_IMAGE_PATH)
    assert image is not None, "Could not load test image"
    return image


@pytest.fixture(scope="function")
def balloons_image_torch() -> torch.Tensor:
    _download_if_not_exists(file_path=BALLOONS_IMAGE_PATH, url=BALLOONS_IMAGE_URL)
    return torchvision.io.read_image(BALLOONS_IMAGE_PATH)


@pytest.fixture(scope="function")
def chess_set_image_numpy() -> np.ndarray:
    _download_if_not_exists(file_path=CHESS_SET_IMAGE_PATH, url=CHESS_SET_IMAGE_URL)
    image = cv2.imread(CHESS_SET_IMAGE_PATH)
    assert image is not None, "Could not load test image"
    return image


@pytest.fixture(scope="function")
def chess_set_image_torch() -> torch.Tensor:
    _download_if_not_exists(file_path=CHESS_SET_IMAGE_PATH, url=CHESS_SET_IMAGE_URL)
    return torchvision.io.read_image(CHESS_SET_IMAGE_PATH)


@pytest.fixture(scope="function")
def chess_piece_image_numpy() -> np.ndarray:
    _download_if_not_exists(file_path=CHESS_PIECE_IMAGE_PATH, url=CHESS_PIECE_IMAGE_URL)
    image = cv2.imread(CHESS_PIECE_IMAGE_PATH)
    assert image is not None, "Could not load test image"
    return image


@pytest.fixture(scope="function")
def chess_piece_image_torch() -> torch.Tensor:
    _download_if_not_exists(file_path=CHESS_PIECE_IMAGE_PATH, url=CHESS_PIECE_IMAGE_URL)
    return torchvision.io.read_image(CHESS_PIECE_IMAGE_PATH)


@pytest.fixture(scope="function")
def flowers_image_numpy() -> np.ndarray:
    _download_if_not_exists(file_path=FLOWERS_IMAGE_PATH, url=FLOWERS_IMAGE_URL)
    image = cv2.imread(FLOWERS_IMAGE_PATH)
    assert image is not None, "Could not load test image"
    return image


@pytest.fixture(scope="function")
def flowers_image_torch() -> torch.Tensor:
    _download_if_not_exists(file_path=FLOWERS_IMAGE_PATH, url=FLOWERS_IMAGE_URL)
    return torchvision.io.read_image(FLOWERS_IMAGE_PATH)


@pytest.fixture(scope="function")
def coins_counting_image_numpy() -> np.ndarray:
    _download_if_not_exists(
        file_path=COIN_COUNTING_IMAGE_PATH, url=COIN_COUNTING_IMAGE_URL
    )
    image = cv2.imread(COIN_COUNTING_IMAGE_PATH)
    assert image is not None, "Could not load test image"
    return image


@pytest.fixture(scope="function")
def coins_counting_image_torch() -> torch.Tensor:
    _download_if_not_exists(
        file_path=COIN_COUNTING_IMAGE_PATH, url=COIN_COUNTING_IMAGE_URL
    )
    return torchvision.io.read_image(COIN_COUNTING_IMAGE_PATH)


@pytest.fixture(scope="function")
def dog_image_numpy() -> np.ndarray:
    _download_if_not_exists(file_path=DOG_IMAGE_PATH, url=DOG_IMAGE_URL)
    image = cv2.imread(DOG_IMAGE_PATH)
    assert image is not None, "Could not load test image"
    return image


@pytest.fixture(scope="function")
def dog_image_torch() -> torch.Tensor:
    _download_if_not_exists(file_path=DOG_IMAGE_PATH, url=DOG_IMAGE_URL)
    return torchvision.io.read_image(DOG_IMAGE_PATH)


@pytest.fixture(scope="function")
def dog_image_pil() -> Image.Image:
    _download_if_not_exists(file_path=DOG_IMAGE_PATH, url=DOG_IMAGE_URL)
    return Image.open(DOG_IMAGE_PATH)


@pytest.fixture(scope="function")
def ocr_test_image_numpy() -> np.ndarray:
    _download_if_not_exists(file_path=OCR_TEST_IMAGE_PATH, url=OCR_TEST_IMAGE_URL)
    image = cv2.imread(OCR_TEST_IMAGE_PATH)
    assert image is not None, "Could not load OCR test image"
    return image


@pytest.fixture(scope="function")
def ocr_test_image_torch() -> torch.Tensor:
    _download_if_not_exists(file_path=OCR_TEST_IMAGE_PATH, url=OCR_TEST_IMAGE_URL)
    return torchvision.io.read_image(OCR_TEST_IMAGE_PATH)


@pytest.fixture(scope="function")
def man_image_numpy() -> np.ndarray:
    _download_if_not_exists(file_path=MAN_IMAGE_PATH, url=MAN_IMAGE_URL)
    image = cv2.imread(MAN_IMAGE_PATH)
    assert image is not None, "Could not load OCR test image"
    return image


@pytest.fixture(scope="function")
def man_image_torch() -> torch.Tensor:
    _download_if_not_exists(file_path=MAN_IMAGE_PATH, url=MAN_IMAGE_URL)
    return torchvision.io.read_image(MAN_IMAGE_PATH)


@pytest.fixture(scope="function")
def truck_image_numpy() -> np.ndarray:
    _download_if_not_exists(file_path=TRUCK_IMAGE_PATH, url=TRUCK_IMAGE_URL)
    image = cv2.imread(TRUCK_IMAGE_PATH)
    assert image is not None, "Could not load OCR test image"
    return image


@pytest.fixture(scope="function")
def truck_image_torch() -> torch.Tensor:
    _download_if_not_exists(file_path=TRUCK_IMAGE_PATH, url=TRUCK_IMAGE_URL)
    return torchvision.io.read_image(TRUCK_IMAGE_PATH)


@pytest.fixture(scope="function")
def people_walking_image_numpy() -> np.ndarray:
    _download_if_not_exists(
        file_path=PEOPLE_WALKING_IMAGE_PATH, url=PEOPLE_WALKING_IMAGE_URL
    )
    image = cv2.imread(PEOPLE_WALKING_IMAGE_PATH)
    assert image is not None, "Could not load test image"
    return image


@pytest.fixture(scope="function")
def people_walking_image_torch() -> torch.Tensor:
    _download_if_not_exists(
        file_path=PEOPLE_WALKING_IMAGE_PATH, url=PEOPLE_WALKING_IMAGE_URL
    )
    return torchvision.io.read_image(PEOPLE_WALKING_IMAGE_PATH)


@pytest.fixture(scope="function")
def snake_image_numpy() -> np.ndarray:
    _download_if_not_exists(file_path=SNAKE_IMAGE_PATH, url=SNAKE_IMAGE_URL)
    image = cv2.imread(SNAKE_IMAGE_PATH)
    assert image is not None, "Could not load test image"
    return image


@pytest.fixture(scope="function")
def snake_image_torch() -> torch.Tensor:
    _download_if_not_exists(file_path=SNAKE_IMAGE_PATH, url=SNAKE_IMAGE_URL)
    return torchvision.io.read_image(SNAKE_IMAGE_PATH)


@pytest.fixture(scope="function")
def sunflowers_image_numpy() -> np.ndarray:
    _download_if_not_exists(file_path=SUNFLOWERS_IMAGE_PATH, url=SUNFLOWERS_IMAGE_URL)
    image = cv2.imread(SUNFLOWERS_IMAGE_PATH)
    assert image is not None, "Could not load test image"
    return image


@pytest.fixture(scope="function")
def sunflowers_image_torch() -> torch.Tensor:
    _download_if_not_exists(file_path=SUNFLOWERS_IMAGE_PATH, url=SUNFLOWERS_IMAGE_URL)
    return torchvision.io.read_image(SUNFLOWERS_IMAGE_PATH)


@pytest.fixture(scope="function")
def basketball_image_numpy() -> np.ndarray:
    _download_if_not_exists(file_path=BASKETBALL_IMAGE_PATH, url=BASKETBALL_IMAGE_URL)
    image = cv2.imread(BASKETBALL_IMAGE_PATH)
    assert image is not None, "Could not load test image"
    return image


@pytest.fixture(scope="function")
def basketball_image_torch() -> torch.Tensor:
    _download_if_not_exists(file_path=BASKETBALL_IMAGE_PATH, url=BASKETBALL_IMAGE_URL)
    return torchvision.io.read_image(BASKETBALL_IMAGE_PATH)
