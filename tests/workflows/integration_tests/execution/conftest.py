import os.path
import tempfile
from typing import Generator

import cv2
import numpy as np
import pytest

ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))
ROCK_PAPER_SCISSORS_ASSETS = os.path.join(ASSETS_DIR, "rock_paper_scissors")

DUMMY_SECRET_ENV_VARIABLE = "DUMMY_SECRET"
os.environ[DUMMY_SECRET_ENV_VARIABLE] = "this-is-not-a-real-secret"


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
def red_image() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "red_image.png"))


@pytest.fixture(scope="function")
def fruit_image() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "multi-fruit.jpg"))


@pytest.fixture(scope="function")
def multi_line_text_image() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "multi_line_text.jpg"))


@pytest.fixture(scope="function")
def stitch_left_image() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "stitch", "v_left.jpeg"))


@pytest.fixture(scope="function")
def stitch_right_image() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "stitch", "v_right.jpeg"))


@pytest.fixture(scope="function")
def left_scissors_right_paper() -> np.ndarray:
    return cv2.imread(
        os.path.join(ROCK_PAPER_SCISSORS_ASSETS, "left_scissors_right_paper.jpg")
    )


@pytest.fixture(scope="function")
def left_rock_right_paper() -> np.ndarray:
    return cv2.imread(
        os.path.join(ROCK_PAPER_SCISSORS_ASSETS, "left_rock_right_paper.jpg")
    )


@pytest.fixture(scope="function")
def left_rock_right_rock() -> np.ndarray:
    return cv2.imread(
        os.path.join(ROCK_PAPER_SCISSORS_ASSETS, "left_rock_right_rock.jpg")
    )


@pytest.fixture(scope="function")
def left_scissors_right_scissors() -> np.ndarray:
    return cv2.imread(
        os.path.join(ROCK_PAPER_SCISSORS_ASSETS, "left_scissors_right_scissors.jpg")
    )


@pytest.fixture(scope="function")
def empty_directory() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


def bool_env(val):
    if isinstance(val, bool):
        return val
    return val.lower() in ["true", "1", "t", "y", "yes"]


@pytest.fixture(scope="function")
def face_image() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "face.jpeg"))

