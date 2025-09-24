import os.path
from typing import List

import cv2
import numpy as np
import pytest

ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))
ALL_IMAGES_LIST = [os.path.join(ASSETS_DIR, f"{i}.jpg") for i in range(1, 6)]


@pytest.fixture()
def all_images() -> List[np.ndarray]:
    return [cv2.imread(path) for path in ALL_IMAGES_LIST]


@pytest.fixture()
def one_image() -> np.ndarray:
    return cv2.imread(ALL_IMAGES_LIST[0])


@pytest.fixture()
def two_images() -> List[np.ndarray]:
    return [cv2.imread(path) for path in ALL_IMAGES_LIST[:2]]


@pytest.fixture()
def three_images() -> List[np.ndarray]:
    return [cv2.imread(path) for path in ALL_IMAGES_LIST[:3]]


@pytest.fixture()
def four_images() -> List[np.ndarray]:
    return [cv2.imread(path) for path in ALL_IMAGES_LIST[:4]]


@pytest.fixture()
def all_images_tile() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "all_images_tile.png"))


@pytest.fixture()
def all_images_tile_and_custom_colors() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "all_images_tile_and_custom_colors.png"))


@pytest.fixture()
def all_images_tile_and_custom_grid() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "all_images_tile_and_custom_grid.png"))


@pytest.fixture()
def four_images_tile() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "four_images_tile.png"))


@pytest.fixture()
def single_image_tile() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "single_image_tile.png"))


@pytest.fixture()
def single_image_tile_enforced_grid() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "single_image_tile_enforced_grid.png"))


@pytest.fixture()
def three_images_tile() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "three_images_tile.png"))


@pytest.fixture()
def two_images_tile() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "two_images_tile.png"))
