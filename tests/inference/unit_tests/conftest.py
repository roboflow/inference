import pickle

import cv2
import numpy as np
from _pytest.fixtures import fixture


@fixture(scope="function")
def image_as_numpy() -> np.ndarray:
    return np.zeros((128, 128, 3), dtype=np.uint8)


@fixture(scope="function")
def image_as_jpeg_bytes() -> bytes:
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    _, encoded_image = cv2.imencode(".jpg", image)
    return np.array(encoded_image).tobytes()


@fixture(scope="function")
def image_as_png_bytes() -> bytes:
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    _, encoded_image = cv2.imencode(".png", image)
    return np.array(encoded_image).tobytes()


@fixture(scope="function")
def image_as_pickled_bytes() -> bytes:
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    return pickle.dumps(image)
