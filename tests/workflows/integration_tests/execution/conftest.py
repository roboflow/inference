import os.path

import cv2
import numpy as np
import pytest

ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))
ROCK_PAPER_SCISSORS_ASSETS = os.path.join(ASSETS_DIR, "rock_paper_scissors")


os.environ["TELEMETRY_OPT_OUT"] = "True"


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
