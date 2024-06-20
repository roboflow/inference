import os.path

import cv2
import numpy as np
import pytest

from inference.core.env import MAX_ACTIVE_MODELS
from inference.core.managers.base import ModelManager
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.models.utils import ROBOFLOW_MODEL_TYPES

ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))
ROCK_PAPER_SCISSORS_ASSETS = os.path.join(ASSETS_DIR, "rock_paper_scissors")


@pytest.fixture(scope="function")
def model_manager() -> ModelManager:
    model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
    model_manager = ModelManager(model_registry=model_registry)
    return WithFixedSizeCache(model_manager, max_size=MAX_ACTIVE_MODELS)


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


@pytest.fixture
def roboflow_api_key() -> str:
    return os.environ["ROBOFLOW_API_KEY"]
