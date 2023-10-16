import base64
import io
import os.path
import pickle
import tempfile
from typing import Generator

import cv2
import numpy as np
from PIL import Image
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
def image_as_jpeg_base64_bytes() -> bytes:
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    _, encoded_image = cv2.imencode(".jpg", image)
    payload = np.array(encoded_image).tobytes()
    return base64.b64encode(payload)


@fixture(scope="function")
def image_as_jpeg_base64_string() -> str:
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    _, encoded_image = cv2.imencode(".jpg", image)
    payload = np.array(encoded_image).tobytes()
    return base64.b64encode(payload).decode("utf-8")


@fixture(scope="function")
def image_as_buffer() -> Generator[io.BytesIO, None, None]:
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    _, encoded_image = cv2.imencode(".jpg", image)
    payload = np.array(encoded_image).tobytes()
    with io.BytesIO() as buffer:
        buffer.write(payload)
        yield buffer


@fixture(scope="function")
def image_as_pickled_bytes() -> bytes:
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    return pickle.dumps(image)


@fixture(scope="function")
def image_as_pillow() -> Image.Image:
    return Image.new(mode="RGB", size=(128, 128), color=(0, 0, 0))


@fixture(scope="function")
def image_as_local_path() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(tmp_dir, "some_image.jpg")
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        cv2.imwrite(file_path, image)
        yield file_path
